println("Starting estimation stage for sea-ice application...")

using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "Flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
end
quick         = parsed_args["quick"]

using NeuralEstimators
using BSON: @load
using CSV
using DataFrames
using Folds
using RData
using Statistics: mean
using HDF5

include(joinpath(pwd(), "src", "Architecture.jl"))

# Load the data and coerce from 3D array to vector of 4D arrays
sea_ice = RData.load(joinpath("data", "sea_ice", "sea_ice_3Darray.rds"))
w, h, K = size(sea_ice) # width, height, and number of images
sea_ice = [sea_ice[:, :, k] for k in 1:K]
sea_ice = reshape.(sea_ice, w, h, 1, 1)
sea_ice = copy.(sea_ice)

# Convert to Int eltype 
IntMissing(x) = ismissing(x) ? x : Int(x)
sea_ice = broadcast.(IntMissing, sea_ice)

# Load the neural MAP estimator 
p = 1
neuralMAP = architecture(p)
path = joinpath("intermediates", "application", "sea_ice")
loadpath  = joinpath(pwd(), path, "NBE", "ensemble.bson")
@load loadpath model_state
Flux.loadmodel!(neuralMAP, model_state)
neuralMAP = neuralMAP[1] # use a single NBE rather than an ensemble 

# Construct the neural EM object
θ₀ = [0.9] 
simulatepottsquick(args...; kwargs...) = simulatepotts(args...; kwargs..., num_iterations = 100)
neuralem = EM(simulatepotts, neuralMAP, θ₀)
@elapsed neuralem(sea_ice[1])
@elapsed neuralem(sea_ice[1:2])

# Point estimates 
println("Obtaining point estimates...")
tm = @elapsed estimates = neuralem(sea_ice)
tm /= length(sea_ice) # average time for a single estimate  

# Bootstrap uncertainty quantification
println("Performing bootstrap-based uncertainty quantification...")
B = quick ? 20 : 400 # number of bootstrap samples 
Z = Folds.map(eachcol(estimates)) do β
  num_states = 2 # two states (ice, no ice)
  z = simulatepotts(w, h, num_states, β; nsims = B, thin = 10, burn = 1000) 
  z = broadcast(x -> x .-= 1, z) # decrease state labels by 1 (for consistency with format of simulated data used during training, which starts the labels at 0)
  z = reshape.(z, w, h, 1, 1)    # reshape to 4-dimensional array, as required by CNN architecture
  z
end
# Ignoring missingness, which may be ok if missingness proportion is small
# bs_estimates_completedata = estimateinbatches.(Ref(neuralMAP), Z)
# bs_estimates_completedata = reduce(vcat, bs_estimates_completedata)

# Account for missingness in bootstrap uncertainty quantification
Z1 = deepcopy(Z)
# Promote eltype of Z1 from Int64 to Union{T, Missing} to allow missing values 
T = eltype(Z1[1][1])
Z1 = broadcast.(convert, Ref(Array{Union{T, Missing}}), Z1)
for i in 1:length(Z1)
    for j in 1:length(Z1[i])
        Z1[i][j] = map((z, s) -> ismissing(s) ? missing : z, Z1[i][j], sea_ice[i])
    end
end
# We have many estimates to obtain, so initialise the algorithm with the 
# observed data estimates and then do a short run of the EM algorithm
bs_estimates = neuralem.(Z1, estimates, niterations = 2) 
bs_estimates = reduce(vcat, bs_estimates)

# Conditional simulation 
sea_ice_complete = Folds.map(1:K) do k 
  Z = sea_ice[k][:, :, 1, 1]
  β = estimates[k]
  z = simulatepotts(Z, β; nsims = B, thin = 5, burn = 200)
  z
end

# Sea ice extent 
sie = broadcast.(sum, sea_ice_complete)
sie = permutedims(reduce(hcat, sie))

# Inference on single missing pixels 
idx = 17 # focus on 1995
z = copy(sea_ice[idx][:, :, 1, 1])
β = estimates[idx]
z_sims = simulatepotts(z, β; nsims = B, thin = 10, burn = 500)
z_sims = stackarrays(z_sims; merge = false)
# Remove "sims1995.h5" if it already exists, then save the new one
file_path = joinpath(path, "sims1995.h5")
if isfile(file_path)
    rm(file_path)
end
h5write(file_path, "dataset", z_sims)

# Save results 
CSV.write(joinpath(path, "estimates.csv"), DataFrame(beta = vec(estimates), sie = mean.(eachrow(sie))))
CSV.write(joinpath(path, "estimation_time.csv"), Tables.table([tm]), writeheader=false)
CSV.write(joinpath(path, "bs_estimates.csv"), Tables.table(bs_estimates), writeheader=false)
CSV.write(joinpath(path, "sie.csv"), Tables.table(sie), writeheader=false)

