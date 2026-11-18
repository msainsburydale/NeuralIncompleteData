using ArgParse
arg_table = ArgParseSettings()
@add_arg_table arg_table begin
	"--quick"
		help = "Flag controlling whether or not a computationally inexpensive run should be done."
		action = :store_true
  "--domain"
		help = "The domain over which to perform inference ('full' or 'sub')"
		arg_type = String
		required = true
end
parsed_args = parse_args(arg_table)
quick = parsed_args["quick"]
domain = parsed_args["domain"]

println("Starting estimation stage for sea-ice application over the $(domain) domain...")

using NeuralEstimators
using NeuralEstimators: estimate
using CUDA, cuDNN # cuDNN: Flux convolutions on NVIDIA GPUs
using CSV
using DataFrames
using Distributions: Beta, Dirac
using Folds
using DelimitedFiles
using RData
using Statistics: mean
using HDF5
using Plots
using LinearAlgebra: BLAS
# NNlib parallelises convolutions over images with Julia threads, and the convolution for each image calls BLAS:
# a multithreaded BLAS then oversubscribes the CPU (CNN forward passes ≈10–40× slower on a 10-core machine)
Threads.nthreads() > 1 && BLAS.set_num_threads(1)

include(joinpath(pwd(), "src", "Architecture.jl"))
include(joinpath(pwd(), "src", "HiddenPotts", "Simulation.jl"))
include(joinpath(pwd(), "src", "EM.jl"))
int_path = joinpath("intermediates", "application", "sea_ice", domain)

# Load the data and coerce from 3D array to vector of 4D arrays
sea_ice = RData.load(joinpath(int_path, "sea_ice_3Darray.rds"));
w, h, K = size(sea_ice) # width, height, and number of images
sea_ice = convert(Array{Union{Missing, Float32}, 3}, sea_ice)
sea_ice = [copy(reshape(sea_ice[:, :, k], w, h)) for k in 1:K]
# Sanity check: sea_ice[15] |> heatmap
# Sanity check: mean.(broadcast.(ismissing, sea_ice))
# Sanity check: mean.(broadcast.(ismissing, sea_ice)) |> histogram
# Sanity check: all(any.(broadcast.(ismissing, sea_ice)))
# Sanity check: mean(any.(broadcast.(ismissing, sea_ice)))

# Prior information 
prior_mean = vec(readdlm(joinpath(int_path, "prior_mean.csv"), ',')) # written by src/application/sea_ice/Train.jl
prior_lower_bound = vec(readdlm(joinpath(int_path, "prior_lower_bound.csv"), ','))
prior_upper_bound = vec(readdlm(joinpath(int_path, "prior_upper_bound.csv"), ','))
d = length(prior_mean)

# Load the NBE (stored as best_estimator.bson), and move it to the GPU (if there is one) once: batched_em and estimate
# then use it there without copying it
neuralMAP = architecture(d, prior_lower_bound, prior_upper_bound)
neuralMAP = loadestimator(neuralMAP, joinpath(pwd(), int_path, "NBE"))
neuralMAP = Flux.gpu_device()(neuralMAP)

# Function to construct emissions distributions, including dirac functions on 0 and 1
betadistribution(λ::AbstractVector) = Beta(λ[1], λ[2])
function emissiondistributions(θ)
  d = length(θ)
  q = (d-1)÷2
  a = θ[2:q+1]
  b = θ[q+2:end]
  λ = vcat(a', b')
  distributions = betadistribution.(eachcol(λ))
  distributions = [distributions..., Dirac(0.0), Dirac(1.0)]
  return distributions
end

# Conditional simulation
function simulateconditional(
    Z₁, θ; 
    nsims::Integer = 1, 
    num_iterations::Integer = 100, 
    )
    
    # Argument validation
    @assert nsims > 0 "nsims must be positive"
    @assert num_iterations > 0 "num_iterations must be positive"
    if ndims(Z₁) > 2 
        @assert all(size(Z₁)[3:end] .== 1)
    end 
    Z₁ = Z₁[:, :]

    # Complete-data case
    if !any(ismissing.(Z₁))
      z = convert(Matrix{nonmissingtype(eltype(Z₁))}, Z₁)
      return repeat(reshape(z, size(z)..., 1, 1), 1, 1, 1, nsims)
    end

    # Construct the emissions distributions
    distributions = emissiondistributions(θ)
  	β = θ[1]

    # Run nsims independent chains: array of dimension w × h × 1 × nsims
    return simulatehiddenpotts(Z₁, β, distributions; nchains = nsims, num_iterations = num_iterations).Z
end

# Conditional simulator for the neural EM algorithm (see batched_em in src/EM.jl): one persistent chain per data set,
# initialised and run for 100 iterations in the first EM iteration, then advanced by 10 iterations per EM iteration.
# The chains run on the GPU (if there is one), where the missing values are also drawn, so that the completed data
# never leave the GPU.
backend = CUDA.functional() ? CUDABackend() : nothing
sim = HiddenPottsEMSimulator(θ -> (θ[1], emissiondistributions(θ)); nsims = 1, sweeps_init = 100, sweeps_warm = 10, sw_every = 1, backend = backend)

θ₀ = prior_mean 
θ₀ = [1.0, 5.0, 2.0, 3.0, 0.5]
burnin = 3

# Quick test and timing for a single estimate and for compilation
Z₁ = sea_ice[17]
@elapsed batched_em(sim, neuralMAP, Z₁, θ₀; burnin = burnin)

# Point estimates
println("Obtaining point estimates...")
tm = @elapsed estimates = batched_em(sim, neuralMAP, sea_ice, θ₀; burnin = burnin).estimates

# Bootstrap data sets
println("Performing bootstrap uncertainty quantification...")
B = quick ? 10 : 100 # number of bootstrap samples 
Z_bs = map(eachcol(estimates)) do θ
  # Simulate marginally
  β = θ[1]
  distributions = emissiondistributions(θ)
  z = simulatehiddenpotts(w, h, β, distributions; nchains = B, num_iterations = 100).Z

  # Convert to vector of matrices
  z = convert(Array{Float32}, z)
  z = [z[:, :, 1, b] for b in 1:B]
end

# Boostrap estimates: Ignoring missingness (okay if missingness proportion is small)
Z = reduce(vcat, Z_bs)
Z = cat.(Z; dims = 4)
bs_time_complete = @elapsed bs_estimates_complete = estimate(neuralMAP, Z)

 # Boostrap estimates: Account for missingness in bootstrap uncertainty quantification
Z1_bs = deepcopy(Z_bs)
# Promote eltype of Z1 from T to Union{T, Missing} to allow missing values 
T = eltype(Z1_bs[1][1])
Z1_bs = broadcast.(convert, Ref(Array{Union{T, Missing}}), Z1_bs)
for k in 1:K
    for b in 1:B
        Z1_bs[k][b] = map((z, s) -> ismissing(s) ? missing : z, Z1_bs[k][b], sea_ice[k])
    end
end
# Sanity check: Z1_bs[15][1] |> heatmap 
# Sanity check: sea_ice[15] |> heatmap 
# Initialise the algorithm with the observed-data estimates and then run the EM algorithm. The K × B bootstrap data
# sets are processed in groups of 500: on the GPU, the chains of a data set over the full domain take about 5 MB (with
# the Swendsen–Wang scratch memory and the likelihood weights), so all 4500 at once would not fit
z = reduce(vcat, Z1_bs)
θ₀ = repeat(estimates, inner = (1, B))
bs_groups = Iterators.partition(eachindex(z), 500)
bs_time_missing = @elapsed bs_estimates_missing = reduce(hcat, [batched_em(sim, neuralMAP, z[g], θ₀[:, g]; burnin = burnin).estimates for g in bs_groups])

# Conditional simulation 
sea_ice_complete = Folds.map(1:K) do k 
  Z₁ = sea_ice[k]

  # Complete-data case
  if !any(ismissing.(Z₁))
    z = convert(Matrix{nonmissingtype(eltype(Z₁))}, Z₁)
    return repeat([z], B)
  end

  θ = estimates[:, k]
  z = simulateconditional(Z₁, θ; nsims = B)
  z = [z[:, :, 1, b] for b in 1:B]
  z
end
# Sanity check: sea_ice[15] |> heatmap 
# Sanity check: sea_ice_complete[15][1] |> heatmap 
# Sanity check: sea_ice_complete[15][2] |> heatmap 

# Sea-ice extent 
sie = broadcast.(sum, sea_ice_complete)
sie = permutedims(reduce(hcat, sie))

# Conditional simulations, storing both Y and Z
println("Computing predictions at missing pixels...")
all_year = [1979, 1990, 1993, 1995, 1999, 2023]
# all_year = 1979:2023
for year in all_year
  idx = year - 1978
  dat = sea_ice[idx]
  θ = estimates[:, idx]
  β = θ[1]
  distributions = emissiondistributions(θ)
  y_sims, z_sims = simulatehiddenpotts(dat, β, distributions; nchains = max(B, 100), num_iterations = 1000)
  z_sims = z_sims[:, :, 1, :]
  y_sims = y_sims[:, :, 1, :]
  file_path = joinpath(int_path, "conditional_sims_$(year).h5")
  isfile(file_path) && rm(file_path) # remove file if it already exists
  h5write(file_path, "Z", z_sims)
  h5write(file_path, "Y", y_sims)
  h5write(file_path, "year", year)
  h5write(file_path, "idx", idx)
end

# Save results 
CSV.write(joinpath(int_path, "estimates.csv"), Tables.table(estimates), writeheader=false)
inference_time = DataFrame(
    estimation_time = tm,
    bootstrap_time_complete = bs_time_complete,
    bootstrap_time_missing = bs_time_missing
)
CSV.write(joinpath(int_path, "inference_time.csv"), inference_time)
CSV.write(joinpath(int_path, "bs_estimates_complete.csv"), Tables.table(bs_estimates_complete), writeheader=false)
CSV.write(joinpath(int_path, "bs_estimates_missing.csv"), Tables.table(bs_estimates_missing), writeheader=false)
CSV.write(joinpath(int_path, "sie.csv"), Tables.table(sie), writeheader=false)

