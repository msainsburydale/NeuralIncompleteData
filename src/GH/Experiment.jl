using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "A flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
end
parsed_args = parse_args(arg_table)
quick = parsed_args["quick"]

m_test = 150

using NeuralEstimators
using NeuralEstimators: estimate
using BenchmarkTools
using BSON: @load
using DelimitedFiles
using Random: seed!
using StatsBase
using CUDA
using CSV
using DataFrames

include(joinpath(pwd(), "src", "GH", "MAP.jl"))
include(joinpath(pwd(), "src", "EM.jl"))
include(joinpath(pwd(), "src", "GH", "Simulation.jl")) 
include(joinpath(pwd(), "src", "GH", "ABC_summaries.jl"))
include(joinpath(pwd(), "src", "Architecture.jl"))
int_path = joinpath(pwd(), "intermediates", "GH")
abc_path = joinpath(int_path, "ABC")
if !isdir(int_path) mkpath(int_path) end
if !isdir(abc_path) mkpath(abc_path) end

prior_mean = mean.([ξ.Ω...])
prior_upper_bound = maximum.([ξ.Ω...])
prior_lower_bound = minimum.([ξ.Ω...])
d = length(prior_mean)

function initialize_estimator(d::Integer; input_channels, outputactivation, dropout::Bool = true, activation = relu)
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
        (dropout ? [Dropout(0.05)] : [])...,
        Dense(128, 128, activation),
        (dropout ? [Dropout(0.05)] : [])..., 
        Dense(128, d), 
        outputactivation
    )

	# estimator = PointEstimator(DeepSet(ψ, ϕ))

	summary_network = DeepSet(ψ, identity)
	inference_network = ϕ
	estimator = MAPEstimator(summary_network, inference_network)

    return estimator
end 

# ---- Train the NBEs ----

# Number of parameter vectors and epochs used during training
K = quick ? 2000 : 10000 #25000
epochs = quick ? 10 : 200

# Simulate training data
sim_time = @elapsed begin
    @info "Generating training data..."
    seed!(1)
    @info "Sampling parameter vectors used for validation..."
    θ_val = Parameters(K, ξ; J = 5)
    @info "Sampling parameter vectors used for training..."
    θ_train = Parameters(K, ξ; J = 5)
    @info "Simulating training data..."
    Z_val = simulate(θ_val, m_test)
    Z_train = simulate(θ_train, m_test)
end
writedlm(joinpath(int_path, "sim_time.csv"), sim_time, ',')

@info "Training EM NBEs..."
neuralMAP = architecture(d, prior_lower_bound, prior_upper_bound; input_channels = 1)
neuralMAP = train(neuralMAP, θ_train, θ_val, Z_train, Z_val; epochs = epochs, savepath = joinpath(int_path, "runs_EM"))

@info "Training Masking NBEs..."
maskedestimator = architecture(d, prior_lower_bound, prior_upper_bound; input_channels = 2)
function generatemissing_MCAR(Z)
	K = length(Z)
	n = prod(size(Z[1])[1:2])                    # number of elements in the complete-data vector 
	n₁ = StatsBase.sample(Int(ceil(0.5*n)):n, K) # number of elements in the incomplete-data vector 
	Z₁ = removedata.(Z, n₁; fixed_pattern = true) 
	UW = encodedata.(Z₁) 
	return UW
end
# simulatemissing_MCAR(parameters, m) = generatemissing_MCAR(simulate(parameters, m))
# maskedestimator = train(maskedestimator, θ_train, θ_val, simulatemissing_MCAR, m = m_test, epochs = epochs, epochs_per_Z_refresh = 3, savepath = joinpath(int_path, "runs_masking"))
UW_train = generatemissing_MCAR(Z_train)
UW_val = generatemissing_MCAR(Z_val)
maskedestimator = train(maskedestimator, θ_train, θ_val, UW_train, UW_val; epochs = epochs, savepath = joinpath(int_path, "runs_masking"))

# ABC summary statistics
@info "Running ABC..."
t₁ = @elapsed T_val   = summary_statistic(Z_val, D)
t₂ = @elapsed T_train = summary_statistic(Z_train, D)
t = t₁ + t₂
T = hcat(T_train, T_val)
θ = hcat(θ_train.θ, θ_val.θ)
writedlm(joinpath(abc_path, "sumstats_train.csv"), T, ',')
writedlm(joinpath(abc_path, "theta_train.csv"), θ, ',')
writedlm(joinpath(abc_path, "sumstats_time.csv"), t, ',')

# ---- Load NBEs ----

neuralMAP = architecture(d, prior_lower_bound, prior_upper_bound; input_channels = 1)
loadpath = joinpath(int_path, "runs_EM", "ensemble.bson")
@load loadpath model_state
Flux.loadmodel!(neuralMAP, model_state)

maskedestimator = architecture(d, prior_lower_bound, prior_upper_bound; input_channels = 2)
loadpath = joinpath(int_path, "runs_masking", "ensemble.bson")
@load loadpath model_state
Flux.loadmodel!(maskedestimator, model_state)

# ---- EM object ----

θ₀ = prior_mean
ξ = merge(ξ, (θ₀ = θ₀,))
neuralem = EM(simulateconditional, neuralMAP, θ₀)

# ---- EM convergence ----

seed!(2025)
parameters = Parameters(100, ξ)
Z = simulate(parameters, m_test)
θ = parameters.θ
distances = [norm(θ[:, j] .- prior_mean) for j in 1:size(θ, 2)]
closest = argmin(distances)
θ = θ[:, closest ]
Z = Z[closest]
n = prod(size(Z)[1:end-1])
n₁ = Int(ceil(0.8n)) # number of observed pixels in each image
Z₁ = removedata(Z, n₁; fixed_pattern = true)

prior_range = prior_upper_bound - prior_lower_bound
all_θ₀ = [prior_lower_bound + 0.05 * prior_range, prior_upper_bound - 0.05 * prior_range]
df = run_EM(neuralem, Z₁, all_θ₀; parameter_names = parameter_names, ξ = ξ)
df_theta = DataFrame(parameter = parameter_names, value = θ)
CSV.write(joinpath(int_path, "EM_iterates.csv"), df)
CSV.write(joinpath(int_path, "EM_iterates_truth.csv"), df_theta)

# ---- Assess the estimators' statistical efficiency ----

# Simulate testing data 
seed!(1)
θ_scenarios = Parameters(5, ξ) 
Z_scenarios = simulate(θ_scenarios, m_test, quick ? 10 : 100)

seed!(1)
θ_test = Parameters(quick ? 50 : 1000, ξ)
Z_test = simulate(θ_test, m_test)

# ABC summary statistics  
T_test = summary_statistic(Z_test, D)
writedlm(joinpath(abc_path, "sumstats_test_complete.csv"), T_test, ',')
writedlm(joinpath(abc_path, "theta_test.csv"), θ_test.θ, ',')
T_scenarios = summary_statistic(Z_scenarios, D)
writedlm(joinpath(abc_path, "sumstats_scenarios_complete.csv"), T_scenarios, ',')
writedlm(joinpath(abc_path, "theta_scenarios.csv"), θ_scenarios.θ, ',')

# Save complete-data estimates
estimates_complete = estimateinbatches(neuralMAP, Z_test)
writedlm(joinpath(int_path, "estimates_test_complete.csv"), estimates_complete, ',')
writedlm(joinpath(int_path, "theta_test.csv"), θ_test.θ, ',')
estimates_complete = estimateinbatches(neuralMAP, Z_train)
writedlm(joinpath(int_path, "estimates_train_complete.csv"), estimates_complete, ',')
writedlm(joinpath(int_path, "theta_train.csv"), θ_train.θ, ',')

function remove_quarter_circle(Z)
    n_rows, n_cols = size(Z)[1:2]  # Get matrix dimensions

    # Calculate row and column midpoints (top-right quadrant)
    row_mid = ceil(Int, n_rows / 2)
    col_mid = ceil(Int, n_cols / 2)

    # Determine the radius of the quarter circle (smallest half-dimension)
    radius = min(row_mid, n_cols - col_mid)

    # Create Cartesian indices for the top-right quadrant
	indices_to_remove = []
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

	# Set the identified indices to missing
	Z₁ = Array{Union{Missing, eltype(Z)}}(Z) # allow missing values in array
    for idx in indices_to_remove
        Z₁[idx[1], idx[2], :, :] .= missing
    end

    return Z₁
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
		Z₁ = remove_complex_missingness.(Z)
	end

    # ABC summary statistics  
    T = summary_statistic(Z₁, D)
    writedlm(joinpath(abc_path, """sumstats_$(missingness)_$(set).csv"""), T, ',')

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
		parameter_names = ξ.parameter_names,
	)

   println("  Running the EM NBE...")
   θ₀ = estimate(maskedestimator, encodedata.(Z₁))
   assessment = merge(assessment, assess(
		(Z₁, ξ) -> neuralem(Z₁, θ₀; burnin = 5, nsims = 10, niterations = 10, use_gpu = false, ξ = ξ),
		θ, Z₁;
        ξ = ξ,
		estimator_name = "EM",
		parameter_names = ξ.parameter_names, 
		use_gpu = false 
	))

	println("  Running the MAP estimator...")
	assessment = merge(assessment, assess(
		MAP, θ, Z₁; 
        ξ = ξ,
		estimator_name = "MAP",
		parameter_names = ξ.parameter_names, 
		use_gpu = false
	))
	
	CSV.write(joinpath(int_path, "estimates_$(missingness)_$(set).csv"), assessment.df)

	return assessment
end

for missingness in ["MCAR", "MB"]
    println("\nAssessing the estimators with $missingness data...")
    assessmissing(Z_test, θ_test, missingness, "test") 
    assessmissing(Z_scenarios, θ_scenarios, missingness, "scenarios")
end

# ---- Assess the estimators' computational efficiency ----

println("\nAssessing the run-times for a single data set...")

# Missing data
Z₁ = removedata(Z_test[1], 0.1)
df = DataFrame(estimator = [], time = [])

# Masking
t = @belapsed gpu(maskedestimator)(gpu(encodedata(Z₁)))
append!(df, DataFrame(estimator = "masking", time = t))

# EM
t = @belapsed θ₀ = cpu(gpu(maskedestimator)(gpu(encodedata(Z₁))))
t += @belapsed neuralem(Z₁, θ₀; ξ = ξ, burnin = 2, niterations = 5)
append!(df, DataFrame(estimator = "neuralEM", time = t))

# MAP
t = @belapsed MAP([Z₁], ξ) 
append!(df, DataFrame(estimator = "MAP", time = t))

# ABC summary statistics 
t = @belapsed summary_statistic(Z₁, D)
append!(df, DataFrame(estimator = "ABC", time = t))

CSV.write(joinpath(int_path, "runtime.csv"), df)