using NeuralEstimators
using BenchmarkTools
using BSON: @load
using CSV
using CUDA
using DataFrames
using Random: seed!
using RData
using StatsBase

include(joinpath(pwd(), "src", "Architecture.jl"))
relative_loadpath = joinpath("intermediates", "Potts")
relative_savepath = joinpath(relative_loadpath, "Estimates")
savepath = joinpath(pwd(), relative_savepath)
if !isdir(savepath) mkdir(savepath) end

# Load parameters and data 
θ_test  = RData.load(joinpath(relative_loadpath, "theta_test.rds"))
θ_scenarios = RData.load(joinpath(relative_loadpath, "theta_scenarios.rds"))
Z_test = RData.load(joinpath(relative_loadpath, "Z_test.rds"))
Z_test = broadcast.(Int, Z_test)
Z_scenarios = RData.load(joinpath(relative_loadpath, "Z_scenarios.rds"))
Z_scenarios = broadcast.(Int, Z_scenarios)

p = 1 # number of parameters in the model

# ---- Load the neural MAP estimator used in the EM algorithm ----

neuralMAP = architecture(p)
loadpath  = joinpath(pwd(), relative_loadpath, "runs_EM", "ensemble.bson")
@load loadpath model_state
Flux.loadmodel!(neuralMAP, model_state)

θ₀ = reshape([0.7], 1) # TODO should be able to give theta as a Number or a Vector, add a convenience constructor
simulatepottsquick(args...; kwargs...) = simulatepotts(args...; num_iterations = 500, kwargs...)
neuralem = EM(simulatepottsquick, neuralMAP, θ₀)


# ---- Load the neural MAP estimators using the masking approach ----

maskedestimator = architecture(p, 2)
loadpath  = joinpath(pwd(), relative_loadpath, "runs_masking", "ensemble.bson")
@load loadpath model_state
Flux.loadmodel!(maskedestimator, model_state)

# ---- Assess the estimators ----

function savedata(Z, K, num_rep, missingness)
	colons  = ntuple(_ -> (:), ndims(Z[1]) - 1)
	z = broadcast(z -> vec(z[colons..., 1]), Z) # save only the first replicate of each parameter configuration
	z = vcat(z...)
	d = prod(size(Z[1])[1:end-1])
	k = repeat(repeat(1:K, outer = num_rep), inner = d)
	j = repeat(repeat(1:num_rep, inner = d), inner = K)
	df = DataFrame(Z = z, k = k, j = j)
	CSV.write(joinpath(relative_savepath, "Z_$(missingness).csv"), df)
end

neuralemclosure(z) = neuralem(z; use_gpu = false)

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

function assessmissing(Z, θ, missingness::String, set::String)

	println("\nEstimating over the $set set...")

	d = prod(size(Z[1])[1:end-1])
	n = Int(ceil(0.8d)) # number of observed pixels in each image

	seed!(1)
	if missingness == "MCAR"
		Z = removedata.(Z, n; fixed_pattern = true)
	elseif missingness == "MB"
		Z = remove_quarter_circle.(Z)
	end

	if set == "scenarios"
	  K = size(θ, 2)
	  J = length(Z) ÷ K
		savedata(Z, K, J, missingness) 
	end
	
	parameter_names = ["β"]

  println("  Running the masked neural Bayes estimator...")
	assessment = assess(
		maskedestimator, θ, encodedata.(Z);
		estimator_name = "masking",
		parameter_names = parameter_names, 
		use_gpu = true
	)

   println("  Running the neural EM algorithm...")
   assessment = merge(assessment, assess(
		neuralemclosure, θ, Z;
		estimator_name = "neuralEM",
		parameter_names = parameter_names, 
		use_gpu = true
	))
	
	CSV.write(joinpath(relative_savepath, "estimates_$(missingness)_$(set).csv", assessment.df)) 
  CSV.write(joinpath(relative_savepath, "runtime_$(missingness)_$(set).csv", assessment.runtime)) 

	assessment
end

for missingness in ["MCAR", "MB"]
    println("\nAssessing the estimators with $missingness data...")
    assessmissing(Z_test, θ_test, missingness, "test") 
    assessmissing(Z_scenarios, θ_scenarios, missingness, "scenarios") 
end

# ---- Accurately assess the run-time for a single data set ----

println("\nAssessing the run-times for a single data set...")

# Missing data
Z1 = removedata.(Z_test, 0.1)
df = DataFrame(estimator = [], time = [])

# Masked neural Bayes estimator
t = @belapsed gpu(maskedestimator)(gpu(encodedata(Z1[1])))
append!(df, DataFrame(estimator = "masking", time = t))

# Neural EM
t = @belapsed neuralem(Z1[1])
append!(df, DataFrame(estimator = "neuralEM", time = t))

CSV.write(joinpath(relative_savepath, "runtime_singledataset.csv"), df)
