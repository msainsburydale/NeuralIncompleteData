using Flux, NeuralEstimators
using NeuralEstimators: @save

#TODO documentation and add to NeuralEstimators.jl, once methodology is finalized
"""
What we do: it would be ideal computationally to (i) pretrain under the absolute error loss, 
then (ii) freeze the summary network and train under the tanh loss. This is easy to do 
in this circumstance because we have fixed training and validation sets... 
so we can compute the summary statistics and then use these as the training data. 
This would mean we can train for a long time (many epochs) under the 0-1 loss very quickly. 

# Examples
```
using NeuralEstimators, Flux

# Priors μ,σ ~ U(0, 1) and data Zᵢ|μ,σ ~ N(μ, σ²), i = 1,…, m
d = 2    # dimension of the parameter vector θ
n = 1    # dimension of each data replicate Zᵢ
sample(K) = rand(d, K) 
simulate(θ, m = 100) = [ϑ[1] .+ ϑ[2] * randn(n, m) for ϑ ∈ eachcol(θ)]  

# Neural network, based on the DeepSets architecture
w = 128  # width of each hidden layer 
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, d))
summary_network = DeepSet(ψ, ϕ)
inference_network = Chain(Dense(d, w, relu), Dense(w, d))
estimator = MAPEstimator(summary_network, inference_network) 

# Train the estimator
θ_train, θ_val = sample(5000), sample(5000)
Z_train, Z_val = simulate(θ_train), simulate(θ_val)
estimator = train(estimator, θ_train, θ_val, Z_train, Z_val)

# Assess the estimator
θ_test = sample(1000)
Z_test = simulate(θ_test)
assessment = assess(estimator, θ_test, Z_test)
bias(assessment)   
rmse(assessment)   
```
"""
struct MAPEstimator{A, B, C} <: BayesEstimator
    summary_network::A
    inference_network::B
    inference_network_ratio::C
end
MAPEstimator(summary_network, inference_network) = MAPEstimator(summary_network, inference_network, nothing)
(estimator::MAPEstimator)(Z) = estimator.inference_network(estimator.summary_network(Z))

import NeuralEstimators: train
function train(
    estimator::MAPEstimator, 
    θ_train, θ_val, Z_train, Z_val; 
    method = "01", tuning_parameter = 0.1f0,  
    epochs_inference = 1000, batchsize_inference = 10_000, stopping_epochs_inference = 30, 
    kwargs... # keyword arguments passed on to training calls involving the summary network
    )

    summary_network = estimator.summary_network
    inference_network = estimator.inference_network
    inference_network_ratio = estimator.inference_network_ratio

    kwargs = (; kwargs...)
    if haskey(kwargs, :savepath) && !isnothing(kwargs.savepath)
        savepath = kwargs.savepath
        savepath_L1, savepath_L0, savepath_ratio = savepath .* ("_L1", "_L0", "_ratio")
        kwargs = Base.structdiff(kwargs, (; savepath = nothing))
    else
        savepath = savepath_L1 = savepath_L0 = savepath_ratio = nothing
    end

#TODO this is a hack because the R version passes optimizer as kwargs... need to remove it here to get back to default settings of train() called below, since we call train on objects constructed internally (and hence need different optimiser objects). Need to improve the handling of optimiser more generally... might be good to pass a list of optimiser options from which we create the optimiser object... this list could be used by Ensemble, here, elsewhere. I like this idea, think it'll smooth implementation greatly.
    # remove optimiser if present
    if haskey(kwargs, :optimiser)
        kwargs = (; [p => getproperty(kwargs, p) for p in propertynames(kwargs) if p ∉ [:optimiser]]...)
    end

    # Pretrain the point estimator under the mean-absolute-error loss
    @info "Training under the mean-absolute-error loss..."
    point_estimator = Chain(summary_network = summary_network, inference_network = inference_network)
    point_estimator = train(point_estimator, θ_train, θ_val, Z_train, Z_val; loss = Flux.mae, savepath = savepath_L1, kwargs...)
    summary_network = point_estimator[:summary_network]
    inference_network = point_estimator[:inference_network]
    T_train = estimate(summary_network, Z_train)
    T_val = estimate(summary_network, Z_val)

    if method == "posterior"
        @info "Training the ratio estimator..."
        @assert !isnothing(inference_network_ratio) "Training with an approximate posterior requires inference_network_ratio to be specified in the MAPEstimator object"
#TODO should I keep the summary network fixed? Think about this in terms of MI.
        ratio_estimator = RatioEstimator(Chain(summary_network = summary_network, inference_network = inference_network_ratio))
        ratio_estimator = train(ratio_estimator, θ_train, θ_val, Z_train, Z_val; savepath = savepath_ratio, kwargs...)
    end

    # Train the MAP estimator
    #TODO Should document that the subcomponents will have a different structure than the full object... the full object will be saved  with this approach in terms of saving... the resulting network cannot be loaded into the original object, should save the best network at the end of training.
    if method == "01"
        @info "Training under a continuous approximation of the 0-1 loss..."
        tuning_parameter = f32(tuning_parameter)
        loss = (x, y) -> tanhloss(x, y, tuning_parameter)
        inference_network = train(inference_network, θ_train, θ_val, T_train, T_val; use_gpu = false, savepath = savepath_L0, batchsize = batchsize_inference, loss = loss, epochs = epochs_inference, stopping_epochs = stopping_epochs_inference)
    elseif method == "posterior"
#TODO implement this (see also PosteriorLoss elsewhere in the repo)
    end

    # Rebuild the object with the trained networks
    estimator = MAPEstimator(summary_network, inference_network, inference_network_ratio)
    
    # Save the trained estimator
    if !isnothing(savepath)
        _savestate(estimator, savepath; file_name = "MAP_estimator")
    end

    return estimator
end

function architecture(d, a, b; input_channels = 1, kwargs...)
	architecture(d, input_channels; outputactivation = Compress(a, b), kwargs...)
end

function architecture(d, input_channels = 1; outputactivation = softplus, J::Integer = 5, kwargs...)
  d = Int(d)
  input_channels = Int(input_channels)
  estimators = [initialize_estimator(d; input_channels = input_channels, outputactivation = outputactivation, kwargs...) for _ in 1:J]
  return Ensemble(estimators)
end

function initialize_estimator(d::Integer; input_channels, outputactivation, dropout::Bool = true, activation::Function = relu, width::Integer = 128, depth::Integer = 2)

	ψ = Chain(
		Conv((3, 3), input_channels => 16, pad=1, bias=false), 
		BatchNorm(16, relu),   
		ResidualBlock((3, 3), 16 => 16),                               
		ResidualBlock((3, 3), 16 => 32, stride=2),                     
		ResidualBlock((3, 3), 32 => 64, stride=2),                     
		ResidualBlock((3, 3), 64 => 128, stride=2),                    
		GlobalMeanPool(),                                              
		Flux.flatten,                                       
	)
	ϕ = Chain(
		Dense(128, width, activation),
		([Chain(Dense(width, width, activation), (dropout ? [Dropout(0.1)] : [])...) for _ in 2:depth]...)...,
		Dense(width, d),
		outputactivation
	)

	summary_network = DeepSet(ψ, identity)
	inference_network = ϕ
	estimator = MAPEstimator(summary_network, inference_network)

	return estimator
end