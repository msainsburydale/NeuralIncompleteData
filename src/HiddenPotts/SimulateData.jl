# Hidden Potts model (Section 3.3): simulation of the training, validation, test and scenario data with
# HiddenMarkovRandomFields.jl. The NBEs are trained with the training and validation data by src/HiddenPotts/Train.jl,
# src/HiddenPotts/ABC.R uses the training data as its reference table, and src/HiddenPotts/Experiment.jl assesses the
# estimators with the test and scenario data.
#
# Usage (from the top level of the repository):
#   julia --project=. --threads=auto src/HiddenPotts/SimulateData.jl [--quick]

quick = "--quick" in ARGS

using CUDA # simulation on an NVIDIA GPU, if there is one
using DelimitedFiles
using Distributions: Normal, Uniform, truncated
using HDF5
using Random
using Statistics: mean

include(joinpath(pwd(), "src", "HiddenPotts", "Simulation.jl"))
int_path = joinpath("intermediates", "HiddenPotts")
mkpath(int_path)

Random.seed!(1) # the parameters
rng = Xoshiro(1) # the fields
backend = CUDA.functional() ? CUDABackend() : nothing # simulation on the GPU if there is one

# ---- Prior ----

q = 3      # number of labels
n = 64     # the images are n × n
K = quick ? 1000 : 25000 # number of data sets
num_iterations = 200     # MCMC iterations of each field; 100 were too few within about 0.01 of the critical value on the 199 × 219 sea-ice grid

β_max = 1.5
βcrit_q = log(1 + sqrt(q))

# Emission parameters: sorted means drawn about (-1, 0, 1) and truncated to [-1, 1], and standard deviations
function sample_emission_parameters(K, q; m_bounds = (-1.0, 1.0))
    m = range(m_bounds...; length = q) # prior means of μ_k
    spacing = step(m)
    μ = Matrix{Float64}(undef, K, q)
    σ = Matrix{Float64}(undef, K, q)
    for i = 1:K
        μ[i, :] = sort([rand(truncated(Normal(m[k], 0.2 * spacing), m_bounds...)) for k = 1:q])
        σ[i, :] = rand(Uniform(0.03, 1 / q), q)
    end
    return μ, σ
end
emissions(μ, σ, i) = [Normal(μ[i, k], σ[i, k]) for k in axes(μ, 2)]

# ---- Simulation ----

@info "Simulating the data..."
β = rand(Uniform(0.03, β_max), K)
μ, σ = sample_emission_parameters(K, q)
sim_time = @elapsed Z = simulate_fields(n, n, β, [emissions(μ, σ, i) for i = 1:K]; num_iterations = num_iterations, rng = rng, backend = backend)
writedlm(joinpath(int_path, "sim_time.csv"), sim_time, ',')
θ = vcat(β', μ', σ') # (β, μ₁, …, μ_q, σ₁, …, σ_q) × K

# Prior mean and bounds (approximated by the simulated parameters)
prior_mean = vec(mean(θ; dims = 2))
prior_lower_bound = vec(minimum(θ; dims = 2))
prior_upper_bound = vec(maximum(θ; dims = 2))
for (name, v) in (("prior_mean", prior_mean), ("prior_lower_bound", prior_lower_bound), ("prior_upper_bound", prior_upper_bound))
    writedlm(joinpath(int_path, "$name.csv"), v, ',')
end

# Scenarios: β = 0.4, 0.5, …, 1.5, each with its own emission parameters and J data sets; data set (j - 1) * 12 + i has
# the parameters of scenario i
β_scenarios = collect(range(0.4, floor(β_max * 10) / 10; step = 0.1))
μ_s, σ_s = sample_emission_parameters(length(β_scenarios), q)
θ_scenarios = vcat(β_scenarios', μ_s', σ_s')
J = quick ? 5 : 50
scenario = repeat(eachindex(β_scenarios), outer = J)
Z_scenarios = simulate_fields(n, n, β_scenarios[scenario], [emissions(μ_s, σ_s, i) for i in scenario]; num_iterations = num_iterations, rng = rng, backend = backend)

# ---- Training, validation and test sets ----

K1 = ceil(Int, 0.8K)    # training set
K3 = quick ? 15 : 1000  # test set
K2 = K - K1 - K3        # validation set
idx_train, idx_val, idx_test = 1:K1, (K1 + 1):(K1 + K2), (K1 + K2 + 1):K

# Test and scenario data (for Experiment.jl and ABC.R), and the training and validation data (for Train.jl; the
# training data are also the reference table of ABC.R)
writedlm(joinpath(int_path, "theta_test.csv"), θ[:, idx_test], ',')
writedlm(joinpath(int_path, "theta_scenarios.csv"), θ_scenarios, ',')
h5open(joinpath(int_path, "test_data.h5"), "w") do f
    f["Z_test"] = Z[:, :, idx_test]
    f["Z_scenarios"] = Z_scenarios
end
h5open(joinpath(int_path, "training_data.h5"), "w") do f
    f["Z_train"] = Z[:, :, idx_train]
    f["theta_train"] = θ[:, idx_train]
    f["Z_val"] = Z[:, :, idx_val]
    f["theta_val"] = θ[:, idx_val]
end
