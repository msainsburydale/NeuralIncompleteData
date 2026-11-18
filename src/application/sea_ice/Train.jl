# Sea-ice application (Section 4): simulation of the training, validation and test data with HiddenMarkovRandomFields.jl,
# and training of the NBE, for the full domain or the subdomain. The NBE is used by
# src/application/sea_ice/Inference.jl.
#
# Usage (from the top level of the repository, after src/application/sea_ice/Preprocessing.R):
#   julia --project=. --threads=auto src/application/sea_ice/Train.jl [--quick] --domain=full|sub

quick = "--quick" in ARGS
domain_arg = filter(a -> startswith(a, "--domain="), ARGS)
domain = isempty(domain_arg) ? "full" : replace(only(domain_arg), "--domain=" => "")
domain in ("full", "sub") || error("--domain must be 'full' or 'sub'")
println("quick = $quick, domain = $domain")

using NeuralEstimators
using Flux
using CUDA, cuDNN # cuDNN: Flux convolutions on NVIDIA GPUs
using BSON: @save
using CSV
using DelimitedFiles
using Distributions: Beta, Dirac, Uniform
using LinearAlgebra: BLAS
using Random
using RData
using Statistics: mean
Threads.nthreads() > 1 && BLAS.set_num_threads(1) # see src/HiddenPotts/Experiment.jl

include(joinpath(pwd(), "src", "Architecture.jl"))
include(joinpath(pwd(), "src", "HiddenPotts", "Simulation.jl"))
int_path = joinpath("intermediates", "application", "sea_ice", domain)

Random.seed!(1) # the parameters (and the initial weights of the networks)
rng = Xoshiro(1) # the fields
backend = CUDA.functional() ? CUDABackend() : nothing # simulation on the GPU if there is one

# ---- Prior ----

K = quick ? 5000 : 50000 # number of data sets
num_iterations = 200     # MCMC iterations of each field; 100 were too few within about 0.01 of the critical value on the 199 × 219 sea-ice grid
n_beta = 2               # Beta components
q = 2 + n_beta           # labels: two Beta components, open water (0) and full ice (1)

βcrit_q = log(1 + sqrt(q))
β_lower, β_upper = 0.03, 1.15 * βcrit_q
a1_support, a2_support, b1_support, b2_support = (2.0, 5.0), (2.0, 5.0), (2.0, 5.0), (0.1, 0.9)

β = rand(Uniform(β_lower, β_upper), K)
a = Matrix{Float64}(undef, n_beta, K)
b = Matrix{Float64}(undef, n_beta, K)
for i = 1:K
    while true # draw until the means of the Beta components are ordered (identifiability)
        a_prop = [rand(Uniform(a1_support...)), rand(Uniform(a2_support...))]
        b_prop = [rand(Uniform(b1_support...)), rand(Uniform(b2_support...))]
        means = a_prop ./ (a_prop .+ b_prop)
        if all(diff(means) .> 0)
            a[:, i] = a_prop
            b[:, i] = b_prop
            break
        end
    end
end
θ = vcat(β', a, b) # (β, a₁, a₂, b₁, b₂) × K

prior_lower_bound = [β_lower, a1_support[1], a2_support[1], b1_support[1], b2_support[1]]
prior_upper_bound = [β_upper, a1_support[2], a2_support[2], b1_support[2], b2_support[2]]
prior_mean = vec(mean(θ; dims = 2))
for (name, v) in (("prior_mean", prior_mean), ("prior_lower_bound", prior_lower_bound), ("prior_upper_bound", prior_upper_bound))
    writedlm(joinpath(int_path, "$name.csv"), v, ',')
end

# ---- Simulation ----

# The labels in the order of src/application/sea_ice/Inference.jl (the Potts model does not depend on the order)
emissions(i) = [Beta(a[1, i], b[1, i]), Beta(a[2, i], b[2, i]), Dirac(0.0), Dirac(1.0)]
grid_dim = Int.(RData.load(joinpath(int_path, "dim_grid.rds")))
@info "Simulating $K fields of $(grid_dim[1]) × $(grid_dim[2]) pixels..."
sim_time = @elapsed Z = simulate_fields(grid_dim[1], grid_dim[2], β, [emissions(i) for i = 1:K];
                                        num_iterations = num_iterations, rng = rng, backend = backend, batchsize = 2000)
writedlm(joinpath(int_path, "sim_time.csv"), sim_time, ',')
@info "Simulation took $(round(sim_time, digits = 1)) s"

# ---- Training, validation and test sets ----

K1 = ceil(Int, 0.8K)    # training set
K3 = quick ? 50 : 1000  # test set
K2 = K - K1 - K3        # validation set
idx_train, idx_val, idx_test = 1:K1, (K1 + 1):(K1 + K2), (K1 + K2 + 1):K
Z_train, Z_val, Z_test = as_datasets(Z[:, :, idx_train]), as_datasets(Z[:, :, idx_val]), as_datasets(Z[:, :, idx_test])
θ_train, θ_val, θ_test = θ[:, idx_train], θ[:, idx_val], θ[:, idx_test]
Z = nothing

# ---- Training ----

@info "Training the NBE..."
estimator = architecture(size(θ, 1), prior_lower_bound, prior_upper_bound; input_channels = 1, J = 5)
epochs = quick ? 3 : 100 # 100 is the default of train(), as used before
estimator = train(estimator, θ_train, θ_val, Z_train, Z_val; epochs = epochs, savepath = joinpath(int_path, "NBE"))
model_state = Flux.state(Flux.cpu(estimator))
@save joinpath(int_path, "NBE", "best_estimator.bson") model_state

# Assessment with the test data
assessment = assess(estimator, θ_test, Z_test; estimator_name = "NBE", parameter_names = ["β", "a₁", "a₂", "b₁", "b₂"])
CSV.write(joinpath(int_path, "assessment_test.csv"), assessment.estimates)
CSV.write(joinpath(int_path, "rmse_test.csv"), rmse(assessment))
