using NeuralEstimators
using CSV
using DataFrames
using RData
using HDF5

# Load the neural-network architecture
include(joinpath(pwd(), "src", "application", "sea_ice", "Architecture.jl"))

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
loadpath  = joinpath(pwd(), path, "NBE")
Flux.loadparams!(neuralMAP, loadbestweights(loadpath))

# Construct the neural EM object
θ₀ = reshape([0.9], 1) # TODO should be able to give theta as a Number or a Vector, add a convenience constructor
simulatepottsquick(args...; kwargs...) = simulatepotts(args...; num_iterations = 100, kwargs...) #TODO more iterations?
neuralem = EM(simulatepottsquick, neuralMAP, θ₀)
@elapsed neuralem(sea_ice[1])
@elapsed neuralem(sea_ice[1:2])

# Parameter point estimates 
tm = @elapsed estimates = neuralem(sea_ice)

# Bootstrap uncertainty quantification
#TODO Is this right? Shouldn't we also use missing data? I suppose it's not a big deal if the missingness proportion is relatively small
B = 400 # number of bootstrap samples 
Z = Folds.map(eachcol(estimates)) do β
  z = simulatepotts(w, h, 2, β; nsims = B, thin = 5, burn = 2000)
  z = broadcast(x -> x .-= 1, z) # decrease state labels by 1 (for consistency with expected input)
  z = reshape.(z, w, h, 1, 1)    # reshape to 4-dimensional array, as required by CNN architecture
  z
end
bs_estimates = estimateinbatches.(Ref(neuralMAP), Z)
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
h5write(joinpath(path, "sims1995.h5"), "dataset", z_sims)

# Save results 
CSV.write(joinpath(path, "estimates.csv"), DataFrame(beta = vec(estimates), sie = mean.(eachrow(sie))))
CSV.write(joinpath(path, "estimation_time.csv"), Tables.table([tm]), writeheader=false)
CSV.write(joinpath(path, "bs_estimates.csv"), Tables.table(bs_estimates), writeheader=false)
CSV.write(joinpath(path, "sie.csv"), Tables.table(sie), writeheader=false)

