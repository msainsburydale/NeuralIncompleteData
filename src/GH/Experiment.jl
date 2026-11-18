# Spatial generalised-hyperbolic (GH) model (Section S4 of the supplement): assessment of the estimators with
# incomplete data. The NBEs are trained by src/GH/Train.jl.
#
# Usage (from the top level of the repository):
#   julia --project=. --threads=auto src/GH/Experiment.jl [--quick]

quick = "--quick" in ARGS

# On x86-64, MKL is used for BLAS and LAPACK: OpenBLAS serialises concurrent calls from many Julia threads (which
# then spend most of their time waiting on a lock in the kernel), whereas MKL does not. MKL's sequential threading
# layer is used, since parallelism is over Julia threads: with its default OpenMP layer, every Julia thread that calls
# MKL starts its own team of threads. The layer must be chosen before MKL is loaded.
@static if Sys.ARCH === :x86_64
    ENV["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    using MKL
end
using NeuralEstimators
using NeuralEstimators: estimate
using Flux
using CUDA, cuDNN # cuDNN: Flux convolutions on NVIDIA GPUs
using BenchmarkTools
using BSON: @load
using DelimitedFiles
using LinearAlgebra
using Random: seed!, Xoshiro
using StatsBase
using Statistics: mean
using CSV
using DataFrames

# Parallelism is over data sets (and conditional simulations), so BLAS uses a single thread
BLAS.set_num_threads(1)

include(joinpath(pwd(), "src", "Architecture.jl"))
include(joinpath(pwd(), "src", "GH", "Architecture.jl"))
include(joinpath(pwd(), "src", "GH", "Simulation.jl"))
include(joinpath(pwd(), "src", "GH", "MAP.jl"))
include(joinpath(pwd(), "src", "GH", "ABC_summaries.jl"))
include(joinpath(pwd(), "src", "EM.jl"))
int_path = joinpath(pwd(), "intermediates", "GH")
abc_path = joinpath(int_path, "ABC")
mkpath(abc_path)

m_test = 150 # number of replicates in each data set
prior_mean = mean.([ξ.Ω...])
prior_lower_bound = minimum.([ξ.Ω...])
prior_upper_bound = maximum.([ξ.Ω...])
d = length(prior_mean)
ξ = merge(ξ, (θ₀ = prior_mean,)) # initial value for the MAP optimiser (the EM algorithm also starts from the prior mean)

# ---- Load the NBEs ----

function loadnetwork(input_channels, folder)
    estimator = gh_architecture(d, prior_lower_bound, prior_upper_bound; input_channels = input_channels)
    loadpath = joinpath(int_path, folder, "best_estimator.bson")
    @load loadpath model_state
    Flux.loadmodel!(estimator, model_state)
    return estimator
end
neuralMAP = loadnetwork(1, "runs_EM")
maskedestimator = loadnetwork(2, "runs_masking")

# The networks are moved to the GPU (if there is one) once: batched_em and estimate then use them there without
# copying them, so the timings do not include copying the weights to the GPU
neuralMAP = Flux.gpu_device()(neuralMAP)
maskedestimator = Flux.gpu_device()(maskedestimator)

# ---- EM algorithm ----

# Hyperparameters of the EM approach (the same for all simulation studies)
sim = GHEMSimulator(ξ; nsims = quick ? 3 : 30)
em_kwargs = (niterations = quick ? 3 : 50, burnin = quick ? 1 : 5, tol = 0.01, nconsecutive = quick ? 1 : 3)

# ---- EM convergence ----

seed!(2025)
parameters = Parameters(100, ξ)
Z = simulate(parameters, m_test)
θ = parameters.θ
distances = [norm(θ[:, j] .- prior_mean) for j in 1:size(θ, 2)]
closest = argmin(distances)
θ = θ[:, closest]
Z = Z[closest]
n = prod(size(Z)[1:end-1])
n₁ = Int(ceil(0.8n)) # number of observed pixels in each image
Z₁ = removedata(Z, n₁; fixed_pattern = true)

prior_range = prior_upper_bound - prior_lower_bound
all_θ₀ = [prior_lower_bound + 0.05 * prior_range, prior_upper_bound - 0.05 * prior_range]
df = run_EM(sim, neuralMAP, Z₁, all_θ₀; parameter_names = parameter_names, all_nsims = [1, 10, 30], rng = Xoshiro(1))
CSV.write(joinpath(int_path, "EM_iterates.csv"), df)
CSV.write(joinpath(int_path, "EM_iterates_truth.csv"), DataFrame(parameter = parameter_names, value = θ))

# ---- Test data ----

seed!(1)
θ_scenarios = Parameters(5, ξ)
Z_scenarios = simulate(θ_scenarios, m_test, quick ? 10 : 100)

seed!(1)
θ_test = Parameters(quick ? 50 : 1000, ξ)
Z_test = simulate(θ_test, m_test)

# ABC summary statistics of the complete data
writedlm(joinpath(abc_path, "sumstats_test_complete.csv"), summary_statistic(Z_test, D), ',')
writedlm(joinpath(abc_path, "theta_test.csv"), θ_test.θ, ',')
writedlm(joinpath(abc_path, "sumstats_scenarios_complete.csv"), summary_statistic(Z_scenarios, D), ',')
writedlm(joinpath(abc_path, "theta_scenarios.csv"), θ_scenarios.θ, ',')

# Complete-data estimates
writedlm(joinpath(int_path, "estimates_test_complete.csv"), estimate(neuralMAP, Z_test), ',')
writedlm(joinpath(int_path, "theta_test.csv"), θ_test.θ, ',')

# ---- Missingness mechanisms ----

function remove_complex_missingness(Z)
    n_rows, n_cols = size(Z)[1:2]

    # Midpoints
    row_mid = n_rows ÷ 2
    col_mid = n_cols ÷ 2

    # Quarter-circle radius
    radius = min(row_mid, n_cols - col_mid)

    # Storage for missing indices
    indices_to_remove = Set{Tuple{Int,Int}}()

    # --- Quarter circle ---
    for i in 1:row_mid  # Loop through the top half rows
        for j in col_mid:n_cols  # Loop through the right half columns
            # Calculate the distance from the top-right corner
            dist = sqrt((i - 1)^2 + (j - n_cols)^2)

            # If distance is less than or equal to the radius, mark for removal
            if dist <= radius
                push!(indices_to_remove, (i, j))
            end
        end
    end

    # --- Ellipse-shaped missing region in bottom-left ---
    for i in row_mid+1:n_rows
        for j in 1:col_mid
            # Ellipse centered at bottom-left
            di = (i - n_rows)^2 / (0.3n_rows)^2
            dj = (j - 1)^2 / (0.5n_cols)^2
            if di + dj <= 1.0
                push!(indices_to_remove, (i, j))
            end
        end
    end

    # --- Apply missingness ---
    Z₁ = Array{Union{Missing, eltype(Z)}}(Z)
    for (i, j) in indices_to_remove
        Z₁[i, j, :, :] .= missing
    end

    return Z₁
end

# ---- Assessment with incomplete data ----

runtimes = DataFrame(set = String[], missingness = String[], estimator = String[], time = Float64[], n = Int[])

# Long-format data frame of estimates (the same format as src/GH/ABC.R): estimates[:, i] is the estimate for data set
# i, with true value θ[:, k] for k = (i - 1) mod K + 1
function estimates_dataframe(estimates, θ, name)
    K = size(θ, 2)
    J = size(estimates, 2) ÷ K
    DataFrame(
        m = m_test,
        k = repeat(repeat(1:K, inner = d), outer = J),
        j = repeat(1:J, inner = K * d),
        estimator = name,
        parameter = repeat(parameter_names, outer = K * J),
        estimate = vec(estimates),
        truth = vec(repeat(θ, 1, J))
    )
end

function assessmissing(Z, θ, missingness::String, set::String)

    println("\nAssessing the estimators with $missingness data ($set set, $(length(Z)) data sets)...")
    record(estimator, t) = push!(runtimes, (set, missingness, estimator, t, length(Z)))

    seed!(1)

    # Generate missingness
    if missingness == "MCAR"
        n = prod(size(Z[1])[1:end-1])
        n₁ = Int(ceil(0.8n)) # number of observed pixels in each image
        Z₁ = removedata.(Z, n₁; fixed_pattern = true)
    elseif missingness == "MB"
        Z₁ = remove_complex_missingness.(Z)
    end

    # ABC summary statistics
    writedlm(joinpath(abc_path, "sumstats_$(missingness)_$(set).csv"), summary_statistic(Z₁, D), ',')

    # Save data for plotting
    if set == "scenarios"
        K = size(θ, 2)
        num_rep = length(Z) ÷ K
        colons = ntuple(_ -> (:), ndims(Z₁[1]) - 1)
        z = broadcast(z -> vec(z[colons..., 1]), Z₁) # save only the first replicate of each data set
        z = vcat(z...)
        n = prod(size(Z₁[1])[1:end-1])
        k = repeat(repeat(1:K, outer = num_rep), inner = n)
        j = repeat(repeat(1:num_rep, inner = n), inner = K)
        CSV.write(joinpath(int_path, "Z_$(missingness).csv"), DataFrame(Z = z, k = k, j = j))
    end

    println("  Running the masking NBE...")
    UW = encodedata.(Z₁)
    estimate(maskedestimator, UW[1:1]) # compilation
    t = @elapsed θ̂_masking = estimate(maskedestimator, UW)
    record("masking", t)

    println("  Running the EM NBE...")
    # The EM algorithm starts from the prior mean (as does the MAP optimiser), as in Section 3.1 of the paper
    t = @elapsed em = batched_em(sim, neuralMAP, Z₁, prior_mean; em_kwargs..., rng = Xoshiro(1), verbose = true)
    record("EM", t)
    CSV.write(joinpath(int_path, "EM_diagnostics_$(missingness)_$(set).csv"), DataFrame(k = eachindex(Z₁), niterations = em.niterations, converged = em.converged))
    @info "Covariance matrices regularised so far: $(gh_jitter_count[])"

    println("  Running the MAP estimator...")
    t = @elapsed θ̂_MAP = MAP(Z₁, ξ)
    record("MAP", t)

    df = vcat(
        estimates_dataframe(θ̂_masking, θ, "masking"),
        estimates_dataframe(em.estimates, θ, "EM"),
        estimates_dataframe(θ̂_MAP, θ, "MAP")
    )
    CSV.write(joinpath(int_path, "estimates_$(missingness)_$(set).csv"), df)
    return df
end

for missingness in ["MCAR", "MB"]
    assessmissing(Z_test, θ_test.θ, missingness, "test")
    assessmissing(Z_scenarios, θ_scenarios.θ, missingness, "scenarios")
end
CSV.write(joinpath(int_path, "runtime_total.csv"), runtimes)

# ---- Run times for a single data set ----

println("\nAssessing the run times for a single data set...")

seed!(1)
n = prod(size(Z_test[1])[1:end-1])
Z₁ = removedata(Z_test[1], Int(ceil(0.8n)); fixed_pattern = true)
df = DataFrame(estimator = String[], time = Float64[])

# Masking
UW₁ = encodedata(Z₁)
push!(df, ("masking", @belapsed estimate($maskedestimator, $UW₁)))

# EM (started from the prior mean)
em_single(Z₁) = batched_em(sim, neuralMAP, Z₁, prior_mean; em_kwargs..., rng = Xoshiro(1))
em_result = em_single(Z₁)
push!(df, ("neuralEM", @belapsed em_single($Z₁) samples = 10 evals = 1))
push!(df, ("EM_niterations", size(em_result.iterates, 2) - 1))

# MAP
push!(df, ("MAP", @belapsed MAP($([Z₁]), ξ) samples = 3 evals = 1))

# ABC summary statistics
push!(df, ("ABC", @belapsed summary_statistic($Z₁, D)))

CSV.write(joinpath(int_path, "runtime.csv"), df)
println(df)
