using NeuralEstimators
import NeuralEstimators: simulate
using BSON: @load
using CSV
using CUDA
using DataFrames
using Distributions: Normal, InverseGamma, mode, median, shape, scale, mean
using Flux
using Tables
using Random: seed!


# ---- Define the model ----

# Zᵢ ~ N(0, σ²), σ² ~ IG(α, β).

MAP(Z, ξ) = mode.(posterior.(Z, Ref(ξ)))'
PosteriorMedian(Z, ξ) = median.(posterior.(Z, Ref(ξ)))'

function posterior(Z, ξ)
    μ = ξ.μ
    Ω = ξ.Ω
    α = shape(Ω)
    β = scale(Ω)
    m = length(Z)
    InverseGamma(α̃(α, m), β̃(β, Z, μ))
end

α̃(α, m) = α + m/2
β̃(β, Z, μ) = β +sum((Z .- μ).^2)/2

Ω = InverseGamma(3, 1)

ξ = (
	Ω = Ω,
	μ = 0,
	parameter_names = ["θ"]
)

struct Parameters <: ParameterConfigurations
	θ
	μ
end

Parameters(K::Integer, ξ) = Parameters(rand(ξ.Ω, 1, K), ξ.μ)

function simulate(params::Parameters, m::R) where {R <: AbstractRange{I}} where I <: Integer
	K = size(params, 2)
	m̃ = rand(m, K)
	θ = vec(params.θ)

	return [rand(Normal(params.μ, √θ[k]), 1, m̃[k]) for k ∈ eachindex(m̃)]
end
simulate(parameters::Parameters, m::Integer) = simulate(parameters, range(m, m))


# ---- Neural-network architecture ----

function architecture(n, p; q = 128, w = 128, activation = relu, likelihood = false)

	if likelihood 
	  ϕ_in = q + p
	  ϕ_out = 1
	else
	  ϕ_in = q
	  ϕ_out = p
	end
  
	estimators = map(1:5) do _
		ψ = Chain(
			Dense(n, w, activation),
			Dense(w, w, activation), 
			Dense(w, q, activation)
		)
		ϕ = Chain(
			Dense(ϕ_in, w, activation),
			Dense(w, w, activation),
			Dense(w, ϕ_out),
		)
		deepset = DeepSet(ψ, ϕ)
		likelihood ? RatioEstimator(deepset) : PointEstimator(deepset)	
	end

	return Ensemble(estimators)
end


# ---- Training ----

K = 10_000  # number of training data sets in each epoch
n = 1       # number of observations in each replicate
m = 10      # number of independent replicates in each data set 
p = 1       # number of parameters
seed!(1)

estimator = architecture(n, p)
savepath = joinpath("intermediates", "univariate", "NBE")

# mean-absolute-error loss 
L1_path = joinpath(pwd(), savepath, "L1")
estimator = train(
	  estimator, Parameters, simulate;
	  ξ = ξ, K = K, m = m,
	  savepath = L1_path
)

# tanh() loss
all_k = Float32.([0.9, 0.7, 0.5, 0.3, 0.1, 0.05])
for k ∈ all_k
	@info "training the NBE under the tanh() loss with k=$k"
	train(
		estimator, Parameters, simulate;
		ξ = ξ, K = K, m = m, 
		savepath = joinpath(savepath, "tanhloss_k$k"),
		loss = (ŷ, y) -> tanhloss(ŷ, y, k)
	)
end

# ---- Testing ----

seed!(1)
θ = mean(ξ.Ω)
θ = reshape([Float32(θ)], 1, 1)
θ = Parameters(θ, ξ.μ)
J = 100_000
Z = simulate(θ, m, J)

estimators = []
estimator_names = String[]

# Load the NBE trained under the L1 loss
estimator = architecture(n, p)
loadpath  = joinpath("intermediates", "univariate", "NBE", "L1", "ensemble.bson")
@load loadpath model_state
Flux.loadmodel!(estimator, model_state)
push!(estimators, estimator)
push!(estimator_names, "L1")

# Load the NBEs trained under the tanh loss
relative_loadpath = joinpath("intermediates", "univariate")
for k ∈ all_k
  estimator = architecture(n, p)
  loadpath  = joinpath("intermediates", "univariate", "NBE", "tanhloss_k$k", "ensemble.bson")
  @load loadpath model_state
  Flux.loadmodel!(estimator, model_state)
  push!(estimators, estimator)
  push!(estimator_names, "k$k")
end

assessment = assess(
	estimators, θ, Z;
	parameter_names = ξ.parameter_names,
	estimator_names = estimator_names
)

# Analytic estimators 
assessment = merge(
		assessment, 
		assess(MAP, θ, Z; parameter_names = ξ.parameter_names, estimator_names = "MAP", ξ = ξ)
		)

assessment = merge(
		assessment, 
		assess(PosteriorMedian, θ, Z; parameter_names = ξ.parameter_names, estimator_names = "PosteriorMedian", ξ = ξ)
		)

relative_savepath = joinpath("intermediates", "univariate", "Estimates")
savepath = joinpath(pwd(), relative_savepath)
if !isdir(savepath) mkdir(savepath) end
CSV.write(joinpath(relative_savepath, "estimates.csv"), assessment.df)
CSV.write(joinpath(relative_savepath, "runtime.csv"), assessment.runtime)

