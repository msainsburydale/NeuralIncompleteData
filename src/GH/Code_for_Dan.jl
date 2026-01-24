using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
end
parsed_args = parse_args(arg_table)
quick = parsed_args["quick"]

m = 150

using NeuralEstimators
using NeuralEstimators: estimate
using BenchmarkTools
using BSON: @load
using DelimitedFiles
using Random: seed!
using StatsBase
using CUDA
using CSV
using ProgressMeter

loss = Flux.mse

include(joinpath(pwd(), "src", "GH", "Simulation.jl")) 
include(joinpath(pwd(), "src", "Architecture.jl"))
int_path = joinpath(pwd(), "intermediates", "GH", "Dan", string(nameof(loss)))
if !isdir(int_path) mkpath(int_path) end

prior_mean = mean.([ξ.Ω...])
prior_upper_bound = maximum.([ξ.Ω...])
prior_lower_bound = minimum.([ξ.Ω...])
d = length(prior_mean)

function initialize_estimator(p::Integer; input_channels, outputactivation, dropout::Bool = false, activation = relu)
    ψ = Chain(
        Conv((3, 3), input_channels => 16, pad = 1, bias = false),
        BatchNorm(16, relu),
        ResidualBlock((3, 3), 16 => 32, stride=2),
        ResidualBlock((3, 3), 32 => 64),
        GlobalMeanPool(),
        Flux.flatten
    )
    ϕ = Chain(
        Dense(64, 128, activation), 
        (dropout ? [Dropout(0.5)] : [])...,
        Dense(128, 128, activation),
        (dropout ? [Dropout(0.5)] : [])..., 
        Dense(128, p), 
        outputactivation
    )

	network = DeepSet(ψ, ϕ)

    return PointEstimator(network)
end

NBE = architecture(d, prior_lower_bound, prior_upper_bound; input_channels = 1)

# ---- Train the NBE ----

# Number of parameter vectors and epochs used during training
K = quick ? 2000 : 25000
epochs = quick ? 10 : 100

sim_time = @elapsed begin
    seed!(1)
    @info "Sampling parameter vectors used for validation..."
    θ_val = Parameters(K ÷ 5, ξ)
    @info "Sampling parameter vectors used for training..."
    θ_train = Parameters(K, ξ)
end
writedlm(joinpath(int_path, "sim_time.csv"), sim_time, ',')

@info "Training NBE..."
NBE = train(NBE, θ_train, θ_val, simulate, m = m, loss = loss, savepath = joinpath(int_path, "NBE"), epochs = epochs, epochs_per_Z_refresh = 3)

# ---- Load the NBE ----

loadpath = joinpath(int_path, "NBE", "ensemble.bson")
@load loadpath model_state
Flux.loadmodel!(NBE, model_state)

# ---- Apply the NBE over test data ----

# Generate the data and estimates on the fly, to avoid memory issues
function estimateonthefly(K; batch_size = 1000)
    # Safety checks
    if K <= 0
        throw(ArgumentError("K must be positive, got $K"))
    end
    if batch_size <= 0
        throw(ArgumentError("batch_size must be positive, got $batch_size"))
    end
    if K % batch_size != 0
        @warn "K ($K) is not evenly divisible by batch_size ($batch_size). The last batch will be smaller."
    end
    
    # Calculate batches - handle case where K is not divisible by batch_size
    batches = ceil(Int, K / batch_size)
    
    # Pre-allocate arrays
    θ_all = Matrix{Float64}(undef, ξ.p, K)
    estimates_all = Matrix{Float64}(undef, ξ.p, K)
    
    # Create progress bar
    progress = Progress(batches, 1, "Computing estimates...")
    
    for batch in 1:batches
        # Calculate the actual batch size for this iteration
        current_batch_size = if batch == batches
            K - (batch - 1) * batch_size  # Last batch might be smaller
        else
            batch_size
        end
        
        # Generate data and estimates
        parameters = Parameters(current_batch_size, ξ)
        Z = simulate(parameters, m)
        θ = parameters.θ
        estimates = estimateinbatches(NBE, Z)

        # Fill the appropriate slice
        start_idx = (batch-1)*batch_size + 1
        end_idx = start_idx + current_batch_size - 1
        
        # Additional safety check for array bounds
        if end_idx > K
            end_idx = K
        end
        
        θ_all[:, start_idx:end_idx] = θ
        estimates_all[:, start_idx:end_idx] = estimates
        
        # Update progress bar
        next!(progress)
    end
    
    return θ_all, estimates_all
end

θ, estimates = estimateonthefly(100_000; batch_size = 1000)
writedlm(joinpath(int_path, "estimates.csv"), estimates, ',')
writedlm(joinpath(int_path, "theta.csv"), θ, ',')
