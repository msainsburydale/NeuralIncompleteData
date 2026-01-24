using NeuralEstimators
import NeuralEstimators: simulate
using BSON: @load
using CSV
using CUDA
using DataFrames
using Distributions: Normal, InverseGamma, mode, median, shape, scale, mean, logpdf
using Flux
using Tables
using Random: seed!

savepath = joinpath("intermediates", "univariate", "NBE")

# NB can also try a more complicated model, like the Normal-Inverse-Gamma, which has four parameters and sufficient statistics (I think)

# ---- NRE loss function ---

@doc raw"""
	PosteriorLoss(estimator::RatioEstimator)
	(loss::PosteriorLoss)(θ̂, θ)

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ U(0, 1) and σ ~ U(0, 1)
d = 2     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 30    # number of independent replicates in each data set
sample(K) = rand32(d, K)
simulate(θ, m) = [ϑ[1] .+ ϑ[2] .* randn32(n, m) for ϑ in eachcol(θ)]

# Neural network
function network(d; ratio = false, w = 128) 
    ψ = Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu))
    ϕ_in = ratio ? w + d : w
    ϕ_out = ratio ? 1 : d
    ϕ = Chain(Dense(ϕ_in, w, relu), Dense(w, w, relu), Dense(w, ϕ_out))
    DeepSet(ψ, ϕ)
end

# Initialise the estimators
r̂ = RatioEstimator(network(d; ratio = true))
θ̂ = PointEstimator(network(d; ratio = false))

# Train the estimators
r̂ = train(r̂, sample, simulate, m = m)
θ̂  = train(θ̂ , sample, simulate, m = m, loss = PosteriorLoss(r̂), use_gpu = false)

# Compare MAPs
θ = sample(1000)
z = simulate(θ, m)
θ_grid = f32(expandgrid(0:0.01:1, 0:0.01:1))'  # fine gridding of the parameter space
posteriormode(r̂, z; θ_grid = θ_grid)           # posterior mode 
estimate(θ̂, z)
```
"""
struct PosteriorLoss{R, F} 
    ratioestimator::R
    logprior::F
end

# Default constructor: flat prior (log p(θ) = 0)
PosteriorLoss(ratioestimator) = PosteriorLoss(ratioestimator, θ -> zeros(eltype(θ), 1, size(θ, 2)))

# The ratio estimator is frozen (non-trainable)
Flux.trainable(::PosteriorLoss) = ()

function (loss::PosteriorLoss)(θ̂, Z)
    # Compute likelihood ratio (already in log-space)
    # log_likelihood_ratio = loss.ratioestimator(Z, θ̂) 
    log_likelihood_ratio = loss.ratioestimator((Z, θ̂))

    # Compute prior term
    log_prior = loss.logprior(cpu(θ̂)) # move to cpu since many densities only work on CPU

    # Convert container types to device of θ̂
    log_likelihood_ratio = convert(containertype(θ̂), log_likelihood_ratio)
    log_prior = convert(containertype(θ̂), log_prior)

    return -mean(log_likelihood_ratio + log_prior)
end

import NeuralEstimators: _risk
using NeuralEstimators: DataLoader

function _risk(estimator, loss::PosteriorLoss, set::DataLoader, device, optimiser = nothing)
    loss = loss |> device #NB would be more efficient to do this once outside the hotloop, since this involves moving neural networks, not just a function
    sum_loss = 0.0f0
    K = 0
    for (input, output) in set
        input, output = input |> device, output |> device
        k = size(output)[end]
        loss_fn = est -> loss(est(input), input)
        if isnothing(optimiser)
            ls = loss_fn(estimator)
        else
            # NB storing the loss in this way is efficient, but it means that
            # the final training risk that we report for each epoch is slightly inaccurate
            # (since the neural-network parameters are updated after each batch)
            ls, ∇ = Flux.withgradient(loss_fn, estimator)
            Flux.update!(optimiser, estimator, ∇[1])
        end
        # Convert average loss to a sum and add to total
        sum_loss += ls * k
        K += k
    end

    return cpu(sum_loss/K)
end

# ---- Define the model ----

# Zᵢ ~ N(0, σ²), σ² ~ IG(α, β).

function posterior(S, ξ)
    Ω = ξ.Ω
    α = shape(Ω)
    β = scale(Ω)
    m = 10
	sum_sq = S * m
    InverseGamma(α̃(α, m), β̃(β, sum_sq))
end
MAP(S, ξ) = mode.(posterior.(S, Ref(ξ)))
PosteriorMedian(S, ξ) = median.(posterior.(S, Ref(ξ)))
α̃(α, m) = α + m/2
β̃(β, sum_sq) = β + sum_sq/2

Ω = InverseGamma(3, 1)

ξ = (Ω = Ω, μ = 0, parameter_names = ["θ"])

struct Parameters <: ParameterConfigurations
	θ
	μ
end

Parameters(K::Integer, ξ) = Parameters(rand(ξ.Ω, 1, K), ξ.μ)

function summarystatistic(z)
	μ = 0
	sum((z .- μ).^2)/length(z)
end

function simulate(params::Parameters, m::Integer)
	K = size(params, 2)
	θ = vec(params.θ)
	Z = [rand(Normal(params.μ, √θ[k]), 1, m) for k ∈ 1:K]
	S = summarystatistic.(Z)
	S = reduce(hcat, S)
	return S
end


# ---- Neural-network architecture ----

function architecture(d; likelihood = false, ensemble_components::Integer = 10, depth = 3, kwargs...)

	sum_stat_dim = d

	if likelihood 
	  in = sum_stat_dim + d
	  out = 1
	  output_activation = identity
	else
	  in = sum_stat_dim 
	  out = d
	  output_activation = softplus
	end
  
	estimators = map(1:ensemble_components) do _
		network = MLP(in, out; output_activation = output_activation, depth = depth, kwargs...)
		likelihood ? RatioEstimator(network) : PointEstimator(network)	
	end

	return Ensemble(estimators)
end

# ---- Training ----

K = 10_000  # number of training data sets in each epoch
n = 1       # number of observations in each replicate
m = 10      # number of independent replicates in each data set 
d = 1       # number of parameters
use_gpu = false

seed!(1)

# mean-absolute-error loss 
estimator = architecture(d)
estimator = train(estimator, Parameters, simulate; ξ = ξ, K = K, m = m, savepath = joinpath(savepath, "L1"), use_gpu = use_gpu)

# # posterior loss
ratioestimator = architecture(d; likelihood = true)
ratioestimator = train(ratioestimator, Parameters, simulate, savepath = joinpath(savepath, "ratioestimator"), ξ = ξ, K = K, m = m, use_gpu = use_gpu)
logprior(θ) = logpdf.(InverseGamma(3, 1), θ)
posteriorloss = PosteriorLoss(ratioestimator, logprior)
estimator = architecture(d)
estimator = train(estimator, Parameters, simulate, loss = posteriorloss, savepath = joinpath(savepath, "posteriorloss"), ξ = ξ, K = K, m = m, use_gpu = use_gpu) 

# tanh loss
all_k = Float32.([0.9, 0.5, 0.3, 0.2, 0.1, 0.05])
estimator = architecture(d)
pretrain = true
for k ∈ all_k
	@info "training the NBE under the tanh() loss with k=$k"
	if !pretrain
		global estimator = architecture(d) 
	end
	global estimator = train(
		estimator, Parameters, simulate;
		ξ = ξ, K = K, m = m, 
		savepath = joinpath(savepath, "tanhloss_k$k"),
		loss = (ŷ, y) -> tanhloss(ŷ, y, k), 
		use_gpu = use_gpu
	)
end



# ---- Testing ----

estimators = []
estimator_names = String[]

# Load the NBE trained under the L1 loss
estimator = architecture(d)
loadpath  = joinpath(savepath, "L1", "ensemble.bson")
@load loadpath model_state
Flux.loadmodel!(estimator, model_state)
push!(estimators, estimator)
push!(estimator_names, "L1")

# Load the NBE trained under the posterior loss
estimator = architecture(d)
loadpath  = joinpath(savepath, "posteriorloss", "ensemble.bson")
@load loadpath model_state
Flux.loadmodel!(estimator, model_state)
push!(estimators, estimator)
push!(estimator_names, "posteriorloss")

# Load the NBEs trained under the tanh loss
for k ∈ all_k
  local estimator = architecture(d)
  local loadpath  = joinpath(savepath, "tanhloss_k$k", "ensemble.bson")
  @load loadpath model_state
  Flux.loadmodel!(estimator, model_state)
  push!(estimators, estimator)
  push!(estimator_names, "k$k")
end



# Test set, focusing on the sampling distribution for a single parameter configuration
seed!(1)
θ = mean(ξ.Ω)
θ = reshape([Float32(θ)], 1, 1)
θ = Parameters(θ, ξ.μ)
J = 100_000
Z = simulate(θ, m, J) |> permutedims

assessment = assess(
	estimators, θ, Z;
	parameter_names = ξ.parameter_names,
	estimator_names = estimator_names
)

# Analytic estimators 
assessment = merge(
		assessment, 
		assess(MAP, θ, Z; parameter_names = ξ.parameter_names, estimator_names = "MAP", ξ = ξ)
		)
assessment = merge(
		assessment, 
		assess(PosteriorMedian, θ, Z; parameter_names = ξ.parameter_names, estimator_names = "PosteriorMedian", ξ = ξ)
		)

rmse(assessment)

CSV.write(joinpath("intermediates", "univariate", "estimates.csv"), assessment.df)




# ---- Visualize the learned posterior ----

# Load the NBE trained under the posterior loss
ratioestimator = architecture(d; likelihood = true)
loadpath  = joinpath(savepath, "ratioestimator", "ensemble.bson")
@load loadpath model_state
Flux.loadmodel!(ratioestimator, model_state)

# Vizualise the posterior given data sampled conditionally on the prior mean
seed!(1)
θ = mean(ξ.Ω)
θ = reshape([Float32(θ)], 1, 1)
θ = Parameters(θ, ξ.μ)
Z = simulate(θ, m)
θ = θ.θ 

# Compute log densities
θ_grid = f32(0.1:0.001:3)'  # fine gridding of the parameter space
logdensity = ratioestimator((Z, θ_grid)) + logprior.(θ_grid)
postdis = posterior(Z, ξ)
truelogdensity = logpdf(postdis, θ_grid)

# Create DataFrame and save
df = DataFrame(
    theta = vec(θ_grid),  # Convert to vector
    estimated_logdensity = vec(logdensity),
    true_logdensity = vec(truelogdensity)
)
CSV.write(joinpath("intermediates", "univariate", "logdensity.csv"), df)



# using Plots

# # Find indices of maximum values
# estimated_max_idx = argmax(logdensity)
# true_max_idx = argmax(truelogdensity)

# # Extract the corresponding θ values
# θ_max_estimated = θ_grid[estimated_max_idx]
# θ_max_true = θ_grid[true_max_idx]

# # Create the plot
# plot(θ_grid[:], logdensity[:], 
#      linewidth=2, 
#      label="Estimated Log-Ratios",
#      xlabel="θ",
#      ylabel="Log-Ratio",
#      title="Comparison of Log-Ratios",
#      legend=:topright)

# plot!(θ_grid[:], truelogdensity[:], 
#       linewidth=2, 
#       label="True Log-Ratios",
#       linestyle=:dash)

# # Add vertical lines at the maximum points
# vline!([θ_max_estimated], 
#        linewidth=1.5, 
#        linestyle=:dot, 
#        color=:blue,
#        label="Estimated Max at θ = $(round(θ_max_estimated, digits=3))")

# vline!([θ_max_true], 
#        linewidth=1.5, 
#        linestyle=:dot, 
#        color=:red,
#        label="True Max at θ = $(round(θ_max_true, digits=3))")

# # Optional: Add markers at the maximum points
# scatter!([θ_max_estimated], [logdensity[estimated_max_idx]], 
#          markersize=6, 
#          color=:blue,
#          label="")

# scatter!([θ_max_true], [truelogdensity[true_max_idx]], 
#          markersize=6, 
#          color=:red,
#          label="")

# # Display the plot
# display(plot!())

# Print summary information
# println("Estimated maximum:")
# println("  θ = $(θ_max_estimated), log-ratio = $(logdensity[estimated_max_idx])")
# println("True maximum:")
# println("  θ = $(θ_max_true), log-ratio = $(truelogdensity[true_max_idx])")
# println("Difference in θ: $(abs(θ_max_estimated - θ_max_true))")