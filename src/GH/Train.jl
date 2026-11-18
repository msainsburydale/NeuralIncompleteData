# Spatial generalised-hyperbolic (GH) model (Section S4 of the supplement): simulation of the training data, training
# of the NBEs, and the ABC summary statistics of the training data. The trained networks are used by
# src/GH/Experiment.jl.
#
# Usage (from the top level of the repository):
#   julia --project=. --threads=auto src/GH/Train.jl [--quick]

quick = "--quick" in ARGS

using NeuralEstimators
using NeuralEstimators: estimate
using Flux
using CUDA, cuDNN
using BSON: @save
using DelimitedFiles
using LinearAlgebra
using Random: seed!
using StatsBase
using Statistics: mean

include(joinpath(pwd(), "src", "Architecture.jl"))
include(joinpath(pwd(), "src", "GH", "Architecture.jl"))
include(joinpath(pwd(), "src", "GH", "Simulation.jl"))
include(joinpath(pwd(), "src", "GH", "ABC_summaries.jl"))
int_path = joinpath(pwd(), "intermediates", "GH")
abc_path = joinpath(int_path, "ABC")
mkpath(abc_path)

m = 150 # number of replicates in each data set
prior_lower_bound = minimum.([ξ.Ω...])
prior_upper_bound = maximum.([ξ.Ω...])
d = length(prior_lower_bound)

# Number of parameter vectors and epochs used during training
K = quick ? 2000 : 10000
epochs = quick ? 10 : 200

# Simulate training data
sim_time = @elapsed begin
    @info "Generating training data..."
    seed!(1)
    θ_val = Parameters(K, ξ; J = 5)
    θ_train = Parameters(K, ξ; J = 5)
    Z_val = simulate(θ_val, m)
    Z_train = simulate(θ_train, m)
end
writedlm(joinpath(int_path, "sim_time.csv"), sim_time, ',')

savenetwork(estimator, folder) = (model_state = Flux.state(Flux.cpu(estimator)); @save joinpath(int_path, folder, "best_estimator.bson") model_state)

@info "Training EM NBEs..."
neuralMAP = gh_architecture(d, prior_lower_bound, prior_upper_bound; input_channels = 1)
neuralMAP = train(neuralMAP, θ_train, θ_val, Z_train, Z_val; epochs = epochs, savepath = joinpath(int_path, "runs_EM"))
savenetwork(neuralMAP, "runs_EM")

@info "Training Masking NBEs..."
function generatemissing_MCAR(Z)
    K = length(Z)
    n = prod(size(Z[1])[1:2])                    # number of elements in the complete-data vector
    n₁ = StatsBase.sample(Int(ceil(0.5*n)):n, K) # number of elements in the incomplete-data vector
    Z₁ = removedata.(Z, n₁; fixed_pattern = true)
    return encodedata.(Z₁)
end
maskedestimator = gh_architecture(d, prior_lower_bound, prior_upper_bound; input_channels = 2)
maskedestimator = train(maskedestimator, θ_train, θ_val, generatemissing_MCAR(Z_train), generatemissing_MCAR(Z_val); epochs = epochs, savepath = joinpath(int_path, "runs_masking"))
savenetwork(maskedestimator, "runs_masking")

# Complete-data estimates of the training set
writedlm(joinpath(int_path, "estimates_train_complete.csv"), estimate(neuralMAP, Z_train), ',')
writedlm(joinpath(int_path, "theta_train.csv"), θ_train.θ, ',')

# ABC summary statistics of the training (and validation) data
@info "Computing ABC summary statistics..."
t₁ = @elapsed T_val   = summary_statistic(Z_val, D)
t₂ = @elapsed T_train = summary_statistic(Z_train, D)
writedlm(joinpath(abc_path, "sumstats_train.csv"), hcat(T_train, T_val), ',')
writedlm(joinpath(abc_path, "theta_train.csv"), hcat(θ_train.θ, θ_val.θ), ',')
writedlm(joinpath(abc_path, "sumstats_time.csv"), t₁ + t₂, ',')
