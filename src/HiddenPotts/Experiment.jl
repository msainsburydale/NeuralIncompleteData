using NeuralEstimators
using NeuralEstimators: estimate
using BenchmarkTools
using CSV
using CUDA, cuDNN # cuDNN: Flux convolutions on NVIDIA GPUs
using DataFrames
using Distributions: Normal
using DelimitedFiles
using HDF5
using Random: seed!
using StatsBase
using Statistics: median
using LinearAlgebra
# NNlib parallelises convolutions over images with Julia threads, and the convolution for each image calls BLAS:
# a multithreaded BLAS then oversubscribes the CPU (CNN forward passes ≈10–40× slower on a 10-core machine)
Threads.nthreads() > 1 && BLAS.set_num_threads(1)

include(joinpath(pwd(), "src", "Architecture.jl"))
include(joinpath(pwd(), "src", "HiddenPotts", "Simulation.jl"))
include(joinpath(pwd(), "src", "EM.jl"))
int_path = joinpath("intermediates", "HiddenPotts")

# Load the parameters and data simulated by src/HiddenPotts/SimulateData.jl
θ_test = readdlm(joinpath(int_path, "theta_test.csv"), ',')
θ_scenarios = readdlm(joinpath(int_path, "theta_scenarios.csv"), ',')
Z_test, Z_scenarios = h5open(joinpath(int_path, "test_data.h5")) do f
    as_datasets(read(f, "Z_test")), as_datasets(read(f, "Z_scenarios"))
end

# Number of parameters to estimate
d = size(θ_test, 1)

# Number of states 
q = (d - 1) ÷ 2

# Parameter names 
parameter_names = ["β"] ∪ "μ" .* string.(1:q) ∪ "σ" .* string.(1:q) 

# Prior information 
prior_mean = vec(readdlm(joinpath(int_path, "prior_mean.csv"), ','))
prior_lower_bound = vec(readdlm(joinpath(int_path, "prior_lower_bound.csv"), ','))
prior_upper_bound = vec(readdlm(joinpath(int_path, "prior_upper_bound.csv"), ','))

# ---- EM NBE ----

# The trained ensembles are stored as best_estimator.bson. The networks are moved to the GPU (if there is one) once:
# batched_em and estimate then use them there without copying them, so the timings below do not include copying the
# weights to the GPU.
neuralMAP = architecture(d, prior_lower_bound, prior_upper_bound; input_channels = 1)
neuralMAP = loadestimator(neuralMAP, joinpath(pwd(), int_path, "runs_EM"))
neuralMAP = Flux.gpu_device()(neuralMAP)

# β and the emission distributions from θ = (β, μ₁, …, μ_q, σ₁, …, σ_q)
function emissions(θ)
    d = length(θ)
    q = (d - 1) ÷ 2
    β = θ[1]
    μ = θ[2:(q + 1)]
    σ = θ[(q + 2):end]
    return β, [Normal(μ[k], σ[k]) for k in 1:q]
end

# Conditional simulator for the neural EM algorithm (see batched_em in src/EM.jl): 30 persistent chains per data set,
# initialised and run for 50 iterations in the first EM iteration, then advanced by 5 iterations per EM iteration;
# each iteration is a Swendsen–Wang sweep followed by a chequerboard Gibbs sweep. The chains run on the GPU (if there
# is one), where the missing values are also drawn, so that the completed data never leave the GPU.
backend = CUDA.functional() ? CUDABackend() : nothing
sim = HiddenPottsEMSimulator(emissions; nsims = 30, sweeps_init = 50, sweeps_warm = 5, sw_every = 1, backend = backend)

# ---- Masking NBE ----

maskedestimator = architecture(d, prior_lower_bound, prior_upper_bound; input_channels = 2)
maskedestimator = loadestimator(maskedestimator, joinpath(pwd(), int_path, "runs_masking"))
maskedestimator = Flux.gpu_device()(maskedestimator)

# ---- EM convergence ----

println("\nGenerating EM NBE sequences for visualization...")

distances = [norm(θ_test[:, j] .- prior_mean) for j in 1:size(θ_test, 2)]
closest = argmin(distances)
θ = θ_test[:, closest]
Z₁ = removedata(Z_test[closest], 0.2)

prior_range = prior_upper_bound - prior_lower_bound
all_θ₀ = [prior_lower_bound + 0.05 * prior_range, prior_upper_bound - 0.05 * prior_range]
df = run_EM(sim, neuralMAP, Z₁, all_θ₀; parameter_names = parameter_names)
df_theta = DataFrame(parameter = parameter_names, value = θ)
CSV.write(joinpath(int_path, "EM_iterates.csv"), df)
CSV.write(joinpath(int_path, "EM_iterates_truth.csv"), df_theta)

# ---- Assess the estimators' computational efficiency ----

println("\nAssessing the run-times for a single data set...")

df = DataFrame(estimator = [], time = [])

# Missing data
Z₁ = removedata(Z_test[1], 0.2)

# Masking 
t = @belapsed estimate(maskedestimator, encodedata(Z₁))
append!(df, DataFrame(estimator = "masking", time = t))

# EM, started from the prior mean
t = @belapsed batched_em($sim, $neuralMAP, $Z₁, $prior_mean; burnin = 5, niterations = 50)
append!(df, DataFrame(estimator = "EM", time = t))

CSV.write(joinpath(int_path, "runtime_single_estimate.csv"), df)

# ---- Assess the estimators' statistical efficiency ----

function remove_quarter_circle(Z)
    data = Z[:, :, 1, 1]  # Extract the matrix
    data = Matrix{Union{Missing, eltype(data)}}(data)  # Allow `missing` values in the matrix
    n_rows, n_cols = size(data)  # Get matrix dimensions

    # Calculate row and column midpoints (top-right quadrant)
    row_mid = ceil(Int, n_rows / 2)
    col_mid = ceil(Int, n_cols / 2)

    # Determine the radius of the quarter circle (smallest half-dimension)
    radius = min(row_mid, n_cols - col_mid)

    # Create Cartesian indices for the top-right quadrant
    for i in 1:row_mid  # Loop through the top half rows
        for j in col_mid:n_cols  # Loop through the right half columns
            # Calculate the distance from the top-right corner
            dist = sqrt((i - 1)^2 + (j - n_cols)^2)

            # If distance is less than or equal to the radius, mark for removal
            if dist <= radius
                data[i, j] = missing  
            end
        end
    end

    return data[:, :, :, :]
end

# remove_complex_missingness (the MICB pattern) is in src/HiddenPotts/Simulation.jl

function assessmissing(Z, θ, missingness::String, set::String)

	println("\nEstimating over the $set set...")

	seed!(1)

	# Generate missingness
	if missingness == "MCAR"
		n = prod(size(Z[1])[1:end-1])
		n₁ = Int(ceil(0.8n)) # number of observed pixels in each image
		Z₁ = removedata.(Z, n₁; fixed_pattern = true)
	elseif missingness == "MB"
		# Z₁ = remove_quarter_circle.(Z)
		Z₁ = remove_complex_missingness.(Z)
	end

	# Save data for plotting
	if set == "scenarios"
	  	K = size(θ, 2)
		num_rep = length(Z) ÷ K
		colons = ntuple(_ -> (:), ndims(Z₁[1]) - 1)
		z = broadcast(z -> vec(z[colons..., 1]), Z₁) # save only the first replicate of each parameter configuration
		z = vcat(z...)
		n = prod(size(Z₁[1])[1:end-1])
		k = repeat(repeat(1:K, outer = num_rep), inner = n)
		j = repeat(repeat(1:num_rep, inner = n), inner = K)
		df = DataFrame(Z = z, k = k, j = j)
		CSV.write(joinpath(int_path, "Z_$(missingness).csv"), df)
	end

    println("  Running the masking NBE...")
	assessment = assess(
		maskedestimator, θ, encodedata.(Z₁);
		estimator_name = "masking", 
		parameter_names = parameter_names
	)

	println("  Running the EM NBE...")
	θ₀ = prior_mean # the EM algorithm starts from the prior mean, as in Section 3.1 of the paper
	# All data sets are processed in lock-step: in each EM iteration, the chains of a chunk of data sets are advanced on
	# the GPU (one data set after another; on the CPU, in parallel over threads), and the MAP estimator is applied to the
	# chunk's completed data in a single call
	t = @elapsed em = batched_em(sim, neuralMAP, Z₁, θ₀; burnin = 5)
	println("  EM iterations: median $(median(em.niterations)), maximum $(maximum(em.niterations)); converged: $(count(em.converged)) of $(length(em.converged)); time: simulation $(round(em.timings.simulation, digits = 1)) s, estimation $(round(em.timings.estimation, digits = 1)) s")
	assessment = merge(assessment, assess(θ, em.estimates; estimator_name = "EM", parameter_names = parameter_names, runtime = t))

	CSV.write(joinpath(int_path, "estimates_$(missingness)_$(set).csv"), assessment.estimates)

	return assessment
end

for missingness in ["MCAR", "MB"]
    println("\nAssessing the estimators with $missingness data...")
	assessmissing(Z_test, θ_test, missingness, "test") 
    assessmissing(Z_scenarios, θ_scenarios, missingness, "scenarios")
end