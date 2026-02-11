using NeuralEstimators
using NeuralEstimators: estimate
using BenchmarkTools
using BSON: @load
using CSV
using CUDA
using DataFrames
using Distributions: Normal
using Random: seed!
using RData
using StatsBase
using LinearAlgebra

include(joinpath(pwd(), "src", "Architecture.jl"))
include(joinpath(pwd(), "src", "EM.jl"))
include(joinpath(pwd(), "src", "HiddenPotts", "Simulation.jl")) 
int_path = joinpath("intermediates", "HiddenPotts")

# Load parameters and data 
θ_test  = RData.load(joinpath(int_path, "theta_test.rds"))
θ_scenarios = RData.load(joinpath(int_path, "theta_scenarios.rds"))
Z_test = RData.load(joinpath(int_path, "Z_test.rds"))
Z_scenarios = RData.load(joinpath(int_path, "Z_scenarios.rds"))
Z_scenarios = broadcast.(Float32, Z_scenarios)
Z_test = broadcast.(Float32, Z_test)

# Number of parameters to estimate
d = size(θ_test, 1)

# Number of states 
q = (d - 1) ÷ 2

# Parameter names 
parameter_names = ["β"] ∪ "μ" .* string.(1:q) ∪ "σ" .* string.(1:q) 

# Prior information 
prior_mean = RData.load(joinpath(int_path, "prior_mean.rds"))
prior_lower_bound = RData.load(joinpath(int_path, "prior_lower_bound.rds"))
prior_upper_bound = RData.load(joinpath(int_path, "prior_upper_bound.rds"))

# ---- EM NBE ----

neuralMAP = architecture(d, prior_lower_bound, prior_upper_bound; input_channels = 1)
loadpath  = joinpath(pwd(), int_path, "runs_EM", "ensemble.bson")
@load loadpath model_state
Flux.loadmodel!(neuralMAP, model_state)

function simulateconditional(
    Z₁, θ; 
    nsims::Integer = 1, 
    num_iterations::Integer = 50, 
    )
    
    # θ = cpu(θ)    

    # Argument validation
    @assert nsims > 0 "nsims must be positive"
    @assert num_iterations > 0 "num_iterations must be positive"
    if ndims(Z₁) > 2 
        @assert all(size(Z₁)[3:end] .== 1)
    end 
    Z₁ = Z₁[:, :]

    # Construct the emmisions distributions
  	d = length(θ)
  	q = (d-1)÷2
  	β = θ[1]
  	μ = θ[2:q+1]
  	σ = θ[q+2:end]
    λ = vcat(μ', σ')
    distribution(λ::AbstractVector) = Normal(λ[1], λ[2])
    distributions = distribution.(eachcol(λ))

    # Run multiple chains to get independent samples
    Z = simulatehiddenpotts_parallel(Z₁, β, distributions; num_iterations = num_iterations, num_chains = nsims).Z[:, :, num_iterations:num_iterations, :]

    return Z
end 
θ₀ = prior_mean
neuralem = EM(simulateconditional, neuralMAP, θ₀)

# ---- Masking NBE ----

maskedestimator = architecture(d, prior_lower_bound, prior_upper_bound; input_channels = 2)
loadpath  = joinpath(pwd(), int_path, "runs_masking", "ensemble.bson")
@load loadpath model_state
Flux.loadmodel!(maskedestimator, model_state)

# ---- EM convergence ----

println("\nGenerating EM NBE sequences for visualization...")

distances = [norm(θ_test[:, j] .- prior_mean) for j in 1:size(θ_test, 2)]
closest = argmin(distances)
θ = θ_test[:, closest]
Z₁ = removedata(Z_test[closest], 0.2)

prior_range = prior_upper_bound - prior_lower_bound
all_θ₀ = [prior_lower_bound + 0.05 * prior_range, prior_upper_bound - 0.05 * prior_range]
df = run_EM(neuralem, Z₁, all_θ₀; parameter_names = parameter_names)
df_theta = DataFrame(parameter = parameter_names, value = θ)
CSV.write(joinpath(int_path, "EM_iterates.csv"), df)
CSV.write(joinpath(int_path, "EM_iterates_truth.csv"), df_theta)

# ---- Assess the estimators' computational efficiency ----

println("\nAssessing the run-times for a single data set...")

df = DataFrame(estimator = [], time = [])

# Missing data
Z₁ = removedata(Z_test[1], 0.2)

# Masking 
t = @belapsed θ₀ = estimate(maskedestimator, encodedata(Z₁))
append!(df, DataFrame(estimator = "masking", time = t))

# EM
t += @belapsed neuralem(Z₁; burnin = 5, niterations = 50, nsims = 30)
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
	θ₀ = estimate(maskedestimator, encodedata.(Z₁))
	assessment = merge(assessment, assess(
      Z₁ -> neuralem(Z₁, θ₀; burnin = 5, nsims = 30, use_gpu = false), # TODO implement EM for multiple data sets in a more memory-safe manner (e.g., in a given iteration, all conditional simulations in parallel; then do all estimates)
			θ, Z₁;
			estimator_name = "EM",
			use_gpu = false, 
			parameter_names = parameter_names
		))

	CSV.write(joinpath(int_path, "estimates_$(missingness)_$(set).csv"), assessment.df)

	return assessment
end

for missingness in ["MCAR", "MB"]
    println("\nAssessing the estimators with $missingness data...")
	assessmissing(Z_test, θ_test, missingness, "test") 
    assessmissing(Z_scenarios, θ_scenarios, missingness, "scenarios")
end