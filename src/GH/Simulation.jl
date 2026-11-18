using NeuralEstimators
using LinearAlgebra
using Distributions
using Distances: pairwise, Euclidean
using Folds
using Statistics
using Test
using Random
using Random: seed!
using Bessels: besselk, gamma
import SpecialFunctions

@isdefined(_tforeach) || include(joinpath(@__DIR__, "..", "Threading.jl")) # _tforeach

"""
	_besselk(ν, x)
Modified Bessel function of the second kind, ``K_ν(x)``, for the small orders used in this model (the Matérn
smoothness ν and the shape λ of the generalised inverse Gaussian mixing distribution).

`Bessels.besselk` (v0.2.8) is inaccurate for small `x` when `ν` is within about 1e-12 of an integer: for example, at
`x = 0.654` it is 1% off at a distance of 1e-14 and 30% too small at `ν = prevfloat(1.0)`, and it returns `NaN` for
orders within about 1e-323 of zero. In the GH likelihood, log K_λ(ω) enters once per replicate, so the error created
spurious likelihood maxima at λ → ±1, the bounds of the prior, which the scaled-logit parameterisation of the MAP
optimiser approaches arbitrarily closely. Within 1e-6 of an integer, `SpecialFunctions.besselk` (AMOS, relative error
below 1e-14 there) is therefore used, and within 1e-12 the order is rounded to the integer (AMOS also fails for
subnormal orders; K_ν(x) is smooth in ν, so the rounding changes the value by less than about 1e-11 relative).
Elsewhere, `Bessels.besselk` is used (it is faster, and it was used to simulate the data, which are hence unchanged).
"""
function _besselk(ν, x)
	n = round(ν)
	abs(ν - n) < 1e-12 && return SpecialFunctions.besselk(n, x)
	abs(ν - n) < 1e-6 && return SpecialFunctions.besselk(ν, x)
	return besselk(ν, x)
end

# Prior
Ω = (
	γ = Uniform(-0.3, 0.3),
	ω = Uniform(0.1, 1),
	λ = Uniform(-1, 1),
	ρ = Uniform(0.05, 0.35),
	ν = Uniform(0.5, 2.0)
)
parameter_names = String.(collect(keys(Ω)))

# Spatial domain
pts = range(0.0, 1.0, 16)
S = expandgrid(pts, pts)
D = pairwise(Euclidean(), S, S, dims = 1)

# ---- Matérn covariance function ----

# (These functions were provided by NeuralEstimators.jl before v0.4.)

@doc raw"""
    matern(h, ρ, ν, σ² = 1)
Given distance ``h``, computes the Matérn covariance function
``C(h) = \sigma^2 \frac{2^{1 - \nu}}{\Gamma(\nu)} (h/\rho)^\nu K_\nu(h/\rho)``.
"""
function matern(h, ρ, ν, σ² = one(typeof(h)))
	@assert h >= 0 "h should be non-negative"
	@assert ρ > 0 "ρ should be positive"
	@assert ν > 0 "ν should be positive"
	if h == 0
		return σ²
	else
		d = h / ρ
		return σ² * ((2^(1 - ν)) / gamma(ν)) * d^ν * _besselk(ν, d)
	end
end

"""
	maternchols(D, ρ, ν)
Lower Cholesky factors of the Matérn covariance matrices with distance matrix `D` and parameters `ρ[k]`, `ν[k]`,
stacked into a three-dimensional array.
"""
function maternchols(D, ρ, ν)
	L = map(eachindex(ρ)) do k
		Matrix(cholesky(Symmetric(matern_covariance(D, ρ[k], ν[k]))).L)
	end
	return stackarrays(L, merge = false)
end

# The distance matrix of a regular grid has few distinct values, and the Matérn covariance function is expensive
# (it involves a Bessel function), so it is evaluated once per distinct distance. `distancegroups(D)` gives the
# distinct distances and, for each element of D, the index of its distance.
function distancegroups(D)
	keys = round.(D; digits = 10)
	u = unique(keys)
	idx = Dict(k => i for (i, k) in enumerate(u))
	return (h = [D[findfirst(==(k), keys)] for k in u], index = [idx[k] for k in keys])
end

"""
	matern_covariance(D, ρ, ν; groups = distancegroups(D))
The Matérn covariance matrix (unit variance) for the distance matrix `D`.
"""
function matern_covariance(D, ρ, ν; groups = distancegroups(D))
	c = [matern(h, ρ, ν) for h in groups.h]
	return c[groups.index]
end
matern_covariance(ξ::NamedTuple, ρ, ν) = matern_covariance(ξ.D, ρ, ν; groups = ξ.Dgroups)

# Tuple containing key information
ξ = (
	Ω = Ω, pts = pts, S = S, D = D, p = length(Ω), d = size(S, 1),
	parameter_names = parameter_names,
	ρ_idx = findfirst(parameter_names .== "ρ"),
	ν_idx = findfirst(parameter_names .== "ν"),
	Dgroups = distancegroups(D)
)


# ---- Parameter configurations ----

# Named tuple without the fields in `ks` (NeuralEstimators.drop() before v0.4)
_drop(nt::NamedTuple, ks) = (; (k => nt[k] for k in keys(nt) if k ∉ ks)...)

struct Parameters{T, V, I} <: ParameterConfigurations
	θ::Matrix{T}
	chols::V
	chol_pointer::Vector{I} # chol_pointer[i] gives the Cholesky factor associated with θ[:, i]
end

function Parameters(K::Integer, ξ; J::Integer = 1)

	# All parameters not associated with the Gaussian process
	θ = [rand(ϑ, K * J) for ϑ in _drop(ξ.Ω, (:ρ, :ν))]

	# Determine if we are estimating ν
	estimate_ν = "ν" ∈ ξ.parameter_names

	# Covariance parameters
	ρ = rand(ξ.Ω.ρ, K)
	ν = estimate_ν ? rand(ξ.Ω.ν, K) : fill(ξ.ν, K)
	chols = stackarrays([Matrix(cholesky(Symmetric(matern_covariance(ξ, ρ[k], ν[k]))).L) for k in eachindex(ρ)], merge = false)
	ρ = repeat(ρ, inner = J)
	ν = repeat(ν, inner = J)
	chol_pointer = repeat(1:K, inner = J)

	# Now insert ρ (and possibly ν) into θ.
	θ₁ = θ[1:(ξ.ρ_idx-1)]
	θ₂ = θ[ξ.ρ_idx:end] # Note that ρ and ν are not in θ, so we don't need to skip any indices.
	if estimate_ν
		@assert ξ.ν_idx == ξ.ρ_idx + 1 "We assume that ρ and ν are stored continguously in θ, that is, that ξ.ν_idx = ξ.ρ_idx + 1"
		θ = [θ₁..., ρ, ν, θ₂...]
	else
		θ = [θ₁..., ρ, θ₂...]
	end

	# Concatenate into a matrix
	θ = hcat(θ...)'
	θ = Float32.(θ)

	Parameters(θ, chols, chol_pointer)
end


# ---- Marginal simulation ----

function simulate(parameters::Parameters, m::R) where {R <: AbstractRange{I}} where I <: Integer

	K = size(parameters, 2)
	m̃ = rand(m, K)

	# Extract the parameters from θ, and the Cholesky factors
	θ = parameters.θ
	T = eltype(θ)
	γ = θ[findfirst(parameter_names .== "γ"), :]
	λ = θ[findfirst(parameter_names .== "λ"), :]
	ω = θ[findfirst(parameter_names .== "ω"), :]
	chols = parameters.chols
	chol_pointer = parameters.chol_pointer

	# infer the number of observations
	d = size(chols, 1)
	@assert √(d) == isqrt(d) "This code assume that the spatial domain is a square"

	# One random-number generator per data set, seeded from the global one, so that the data are reproducible
	# regardless of the number of threads
	seeds = rand(UInt64, K)

	Z = Folds.map(1:K) do k
		L = view(chols, :, :, chol_pointer[k])
		z = simulateGH(L, m̃[k]; μ = zero(T), γ = γ[k], λ = λ[k], ω = ω[k], rng = Xoshiro(seeds[k]))
		z = Float32.(z)
		z = reshape(z, isqrt(d), isqrt(d), 1, :)
		z
	end

	return Z
end
simulate(parameters::Parameters, m::Integer) = simulate(parameters, range(m, m))
simulate(parameters::Parameters) = stackarrays(simulate(parameters, 1))
# J independent data sets for each parameter vector: data set (j - 1)K + k corresponds to parameter vector k
simulate(parameters::Parameters, m, J::Integer) = vcat([simulate(parameters, m) for _ in 1:J]...)

# ---- Conditional simulation ----

"""
Generic function for defining conditional simulation, used in the EM algorithm.

The user must provide a method:

	simulateconditional(Z₁::M, θ; nsims::Integer = 1, kwargs...) where {M <: AbstractMatrix{Union{Missing, T}}} where T

The two positional arguments are:

- `Z₁`: the observed, incomplete data. The last dimension of `Z₁` contains the replicates; the other dimensions store the response variable.
- `θ`: vector of parameters used for conditional simulation.

The argument `nsims` controls the number of simulations.
"""
function simulateconditional end

function simulateconditional(Z::A, θ; nsims::Integer = 1, kwargs...) where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}

	# Save the original dimensions
	dims = size(Z)

	# Convert to matrix and pass to the matrix method
	Z = simulateconditional(Flux.flatten(Z), θ; nsims = nsims, kwargs...)

	# Convert Z to the correct dimensions
	Z = reshape(Z, dims[1:end-1]..., :)

	return Z
end

function simulateconditional(Z::V, θ; nsims::Integer, kwargs...) where {V <: AbstractVector{Union{Missing, T}}} where T

	## Convert to matrix and pass to the matrix method
	Z = reshape(Z, (length(Z), 1))
	Z = simulateconditional(Z, θ; nsims = nsims, kwargs...)

	return Z
end


"""
Replicates are stored in the columns of `Z`
"""
function simulateconditional(Z::M, θ; nsims::Integer, ξ) where {M <: AbstractMatrix{Union{Missing, T}}} where T

	θ = Float64.(θ)
	
	# Extract objects from ξ
	D = ξ.D # distance matrix 
	d = size(D, 1)

	# Extract the parameters from θ using information in ξ
	parameter_names = ξ.parameter_names
	γ = θ[findfirst(parameter_names .== "γ")]; γ = repeat([γ], d)
	μ = repeat([0], length(γ))
	η = one(eltype(θ)) 
	ω = θ[findfirst(parameter_names .== "ω")]
	λ = θ[findfirst(parameter_names .== "λ")]
	ρ = θ[findfirst(parameter_names .== "ρ")]
	ν = θ[findfirst(parameter_names .== "ν")]
	
	# Compute covariance matrix
	Σ = Symmetric(matern_covariance(D, ρ, ν; groups = haskey(ξ, :Dgroups) ? ξ.Dgroups : distancegroups(D)))

	# Remove any columns that contain only missing elements
	Z = Z[:, vec(mapslices(z -> !all(ismissing.(z)), Z, dims = 1))]

	# Conditional simulation
	simulateconditionalGH(Z, Σ; γ = γ, μ = μ, λ = λ, ω = ω, η = η, H = nsims)
end


# ---- Generalised-hyperbolic simulation ----

# Marginal simulation is easy, we just need to simulate from a Gaussian process
# and from a generalised inverse Gaussian (GIG) distribution.

function simulateGH(L::AbstractArray{T, 2}; μ, γ, λ, ω, η = 1, rng::AbstractRNG = Random.default_rng()) where T <: Number
	W = L * randn(rng, size(L, 1))
	M = simulategig(λ = λ, ω = ω, η = η, rng = rng)
	Z = μ .+ γ * M .+ sqrt(M) * W
	return Z
end

function simulateGH(L::AbstractArray{T, 2}, m::Integer; args...) where T <: Number
	n = size(L, 1)
	Z = similar(L, n, m)
	for h ∈ 1:m
		Z[:, h] = simulateGH(L; args...)
	end
	return Z
end

"""
	simulateconditionalGH(Z::Matrix{Union{Missing, T}, Σ::Matrix; γ, μ, λ, ω, η, H::Integer) where T
Conditional simulation from the generalised hyperbolic distribution, described
below.

The data Z should be a d×n matrix of incompletely observed replicates.
The return type is a d×n`H` matrix obtained by conditionally simulating `H`
times for each replicate (note that the order of replicates is not preserved).

# GH distribution

The d-dimensional random variable 𝐙 is called a normal mean-variance mixture
(NMVM) if it can be represented as,

```math
𝐙 = 𝛍 + M𝛄 + M𝐕
```

where 𝛍 ∈ ℝᵈ and 𝛄 ∈ ℝᵈ, and where M is a positive mixing random variable that
is independent of 𝐕 ∼ Gau(𝟎, 𝚺) for covariance matrix 𝚺. This flexible family of
distributions is closed under conditioning. With 𝐙 = (𝐙₁', 𝐙₂')' and with 𝛍, 𝛄,
and 𝚺 partitioned accordingly, 𝐙₂ | 𝐙₁ is also a NMVM, with mixing variable
M | 𝐙₁ and parameters (Jamalizadeh and Balakrishnan, 2019, Thm. 1),

```math
𝛍₂∣₁  = 𝛍₂ + 𝚺₂₁𝚺₁₁⁻¹(𝐙₁−𝛍₁),
𝛄₂|₁  = 𝛄₂ - 𝚺₂₁𝚺₁₁⁻¹𝛄₁,
𝚺₂₂∣₁ = 𝚺₂₂ - 𝚺₂₁𝚺₁₁⁻¹𝚺₁₂.
```

The generalised hyperbolic (GH) distribution (Barndorff-Nielsen, 1977) is
obtained when M follows a generalised inverse Gaussian (GIG) distribution.
If M ∼ GIG(ω, η, λ) with density function,

```math
f(m; λ, η, ω) ∝ m^{λ-1}e^{-ω(η/m + m/η)/2}, 	m > 0,
```

then M | Z₁ also follows a GIG distribution with parameters
(Jamalizadeh and Balakrishnan, 2019, Cor. 2),

```math
ω₂∣₁ = [(q(𝐙₁; 𝛍₁, 𝚺₁₁) + ωη)(ω/η + 𝛄₁'𝚺₁₁⁻¹𝛄₁)]^{1/2},
η₂∣₁ = [(q(𝐙₁; 𝛍₁, 𝚺₁₁) + ωη)/(ω/η + 𝛄₁'𝚺₁₁⁻¹𝛄₁)]^{1/2},
λ₂∣₁ = λ − dim(Z₁)/2,
```

where ``q(𝐙₁; 𝛍₁, 𝚺₁₁) = (𝐙₁−𝛍₁)'𝚺₁₁⁻¹(𝐙₁−𝛍₁)``.

# Examples
```
d = 5
n = 1000
Z = rand(d, n)
Z = removedata(Z, 0.5)

μ = randn(d)
γ = randn(d)
Σ = randn(d, d); Σ = Symmetric(0.5 * (Σ + Σ')); Σ = Σ +  d * I(d)
λ = randn(1)[1]
ω = rand(1)[1]
η = rand(1)[1]

simulateconditionalGH(Z, Σ; γ = γ, λ = λ, ω = ω, η = η, μ = μ, H = 10)
```
"""
function simulateconditionalGH(Z::M₁, Σ::M₂; γ, μ, λ, ω, η, H::Integer) where {M₁ <: AbstractMatrix{Union{Missing, T}},  M₂ <: AbstractMatrix} where T

  	# Dimension of the response and number of replicates
  	d  = size(Z, 1)
  	n  = size(Z, 2)

  	# Get the indices of the observed component for each replicate.
	idx = [findall(x -> !ismissing(x), vec(Z[:, i])) for i ∈ 1:n]

	# Handle replicates without any observations, and completely observed replicates
	unique_idx = unique(idx)
	@assert !any(length.(unique_idx) == 0) "dim(Z₁) = 0: cannot do conditional simulation without data to condition on" # NB could just simulate from the marginal in this case
	deleteat!(unique_idx, findall(Ref(1:d) .== unique_idx))

	# Consider groups of replicates with the same missingness pattern for computational efficiency
	W = map(unique_idx) do I₁

    	# Missing elements
    	I₂ = (1:d)[1:d .∉ Ref(I₁)]

    	# Extract replicates with the missingness pattern I₁
    	z =  Z[:, findall(Ref(I₁) .== idx)]
		ñ = size(z, 2)

		# Extract parameter subvectors and submatrices
		μ₁ = μ[I₁]
		μ₂ = μ[I₂]
		γ₁ = γ[I₁]
		γ₂ = γ[I₂]
		Σ₁₁ = Σ[I₁, I₁]
		Σ₂₂ = Σ[I₂, I₂]
		Σ₂₁ = Σ[I₂, I₁]
		Σ₁₂ = Σ[I₁, I₂]

		# Compute conditional parameters that do not depend on Z₁
		L₁₁ = try
			cholesky(Symmetric(Σ₁₁)).L
		catch
			@warn "Failed to compute the Cholesky factor of Σ₁₁: ignoring this batch of $ñ out of $n total replicates"
			return missing
		end
		u = L₁₁ \ Σ₁₂   # L₁₁⁻¹Σ₁₂
		w = L₁₁ \ γ₁    # L₁₁⁻¹γ₁
		quadform_γγ = w'w
		λstar   = λ - length(I₁)/2
		γ₂star  = γ₂ - u'w
		Σ₂₂star = Σ₂₂ - u'u
		L₂₂star = try
			cholesky(Symmetric(Σ₂₂star)).L
		catch
			@warn "Failed to compute the Cholesky factor of Σ₂₂∣₁: ignoring this batch of $ñ out of $n total replicates"
			return missing
		end

		# Conditional simulation for each Z₁
		W = map(eachcol(z)) do Z₁

			# Extract the observed data and drop Missing from the eltype
			Z₁ = Z₁[I₁]
			Z₁ = [Z₁...]

			# Compute conditional parameters that depend on Z₁
			v = L₁₁ \ (Z₁ - μ₁) # L₁₁⁻¹(Z₁ - μ₁)
			μ₂star  = μ₂ + u'v
			quadform_zz = v'v
			ωstar = try
				sqrt( (ω*η + quadform_zz) * (ω/η + quadform_γγ) )
			catch
				@warn "Computation of ω₂∣₁ failed: ignoring this batch of $ñ out of $n total replicates"
				return missing
			end
			ηstar = try
				sqrt( (ω*η + quadform_zz) / (ω/η + quadform_γγ) )
			catch
				@warn "Computation of η₂∣₁ failed: ignoring this batch of $ñ out of $n total replicates"
				return missing
			end

			# Simulate from the conditional distribution Z₂ ∣ Z₁, θ
			Z₂ = simulateGH(L₂₂star, H; μ = μ₂star, γ = γ₂star, λ = λstar, ω = ωstar, η = ηstar)

			# Combine the observed and missing data to form the complete data
			W = map(1:H) do h
				w = Vector{T}(undef, d)
				w[I₁] = Z₁
				w[I₂] = Z₂[:, h]
				w
			end
			hcat(W...) # d × H Matrix{T}
		end
		W = skipmissing(W)
		W = hcat(W...) # d × (nᵢH) Matrix{T}, where nᵢ is the number of replicates in the current batch
		if all(ismissing.(W)) W =  Matrix{T}(undef, d, 0) end # when all simulations fails
		W
	end
	W = skipmissing(W)
	W = hcat(W...) # d × (ñH) Matrix{T}, where ñ is the total number of replicates excluding those replicates that are fully observed
  if all(ismissing.(W)) W =  Matrix{T}(undef, d, 0) end # when all simulations fails

	# Now add the completely observed replicates (if there are any)
	if any(idx .== Ref(1:d))
		z =  Z[:, findall(idx .== Ref(1:d))] # extract completely observed replicates
		w = repeat(z, inner = (1, H))        # repeat each replicate H times
		w = convert(Matrix{T}, w)            # drop Missing from the eltype
		W = hcat(W, w) # d × (nH) Matrix{T}
	end

	return W # d × (nH) Matrix{T}
end

function simulateconditionalGH(Z::V, Σ::M₂; args...) where {V <: AbstractVector{Union{Missing, T}},  M₂ <: AbstractMatrix} where T
	Z = reshape(Z, :, 1)
	simulateconditionalGH(Z, Σ; args...)
end


# ---- Generalised Inverse Gaussian (GIG) ----

# This code was adapted from the following Distributions.jl pull request:
# https://github.com/JuliaStats/Distributions.jl/pull/1300/files

using Distributions
import Distributions: convert
import Base: rand

"""
	simulategig(n = 1; ω, η, λ)
Simulates `n` realisations of a generalized inverse Gaussian (GIG) random
variable with density,

```math
f(x; λ, η, ω)  ∝ x^{λ-1}e^{-ω(η/x + x/η)/2}, x > 0,
```

for concentration parameter ω > 0, scale parameter η > 0, and shape
parameter λ ∈ ℝ. Note that this is equivalent to the parameterisation,

```math
f(x; λ, χ, ψ)  ∝ x^{λ-1}e^{-(χ/x + ψx)/2}
```

with χ = ωη and ψ = ω/η. Note that the inverse parameterisation is η = √{χ/ψ} and ω = √{χψ}.

The sampling procedure is based on [Hörmann & Leydold (2014)](https://doi.org/10.1007/s11222-013-9387-3).
"""
function simulategig(n::Integer; λ, ω, η, rng::AbstractRNG = Random.default_rng())
	ψ = ω/η
	χ = ω * η
	rand(rng, GeneralisedInverseGaussian(ψ, χ, λ), n)
end
function simulategig(; λ, ω, η, rng::AbstractRNG = Random.default_rng())
	ψ = ω/η
	χ = ω * η
	rand(rng, GeneralisedInverseGaussian(ψ, χ, λ))
end

struct GeneralisedInverseGaussian{T<:Real} <: ContinuousUnivariateDistribution
    ψ::T
    χ::T
    p::T #NB p ⟺ λ
    GeneralisedInverseGaussian{T}(ψ::T, χ::T, p::T) where T = new{T}(ψ,χ,p)
end

function GeneralisedInverseGaussian(ψ::T,χ::T,p::T) where {T<:Real}
    @assert ψ > zero(ψ) && χ > zero(χ)
    return GeneralisedInverseGaussian{T}(ψ,χ,p)
end

GeneralisedInverseGaussian(ψ::Real, χ::Real, p::Real) = GeneralisedInverseGaussian(promote(ψ,χ,p)...)
GeneralisedInverseGaussian(ψ::Integer, χ::Integer, p::Integer) = GeneralisedInverseGaussian(float(ψ), float(χ), float(p))
function convert(::Type{GeneralisedInverseGaussian{T}}, ψ::Real, χ::Real, p::Real) where T<:Real
    GeneralisedInverseGaussian(T(ψ),T(χ),T(p))
end
function convert(::Type{GeneralisedInverseGaussian{T}}, d::GeneralisedInverseGaussian{S}) where {T <: Real, S<: Real}
    GeneralisedInverseGaussian(T(d.ψ), T(d.χ), T(d.p))
end
params(d::GeneralisedInverseGaussian) = (d.ψ, d.χ, d.p)
@inline partype(d::GeneralisedInverseGaussian{T}) where {T<:Real} = T

#### Sampling

# NB Added iteration limit in the while loops to prevent hanging when the
# parameters are extreme

rand(rng::AbstractRNG, d::GeneralisedInverseGaussian, n::Int64) = [rand(rng, d) for _ in 1:n]
rand(d::GeneralisedInverseGaussian, n::Int64) = rand(Random.default_rng(), d, n)
rand(d::GeneralisedInverseGaussian) = rand(Random.default_rng(), d)

function rand(rng::AbstractRNG, d::GeneralisedInverseGaussian)
    (ψ,χ,p) = params(d)
    ω = sqrt(ψ/χ) # NB This is not the same ω as described above
    η = sqrt(ψ*χ) # NB This is not the same η as described above
    λ = abs(p)
    if η > 1 || λ > 1
        x = sample_unif_mode_shift(rng,λ,η)
    else
        η_bound = min(1/2, (2/3)*sqrt(1 - p))
        if η < 1 && η >= η_bound
            x = sample_unif_no_mode_shift(rng,λ,η)
        elseif η < η_bound && η > 0
            x = concave_sample(rng,λ,η)
        else
            throw(ArgumentError("None of the required conditions on the parameters are satisfied"))
        end
    end
    if p >= 0
        return x/ω
    else
        return 1 / (ω*x)
    end
end

#Sample the 2-parameter GIG distribution with parameters p and η using the Rejection method
#as described by Hörmann & Leydold (2014).
function concave_sample(rng::AbstractRNG, p::Real, η::Real; max_iteration::Integer = 10000, verbose::Bool = false)

    ϵ = typeof(η)(0.000001) # small positive constant to prevent log(<0)

    m = η/((1 - p) + sqrt((1 - p)^2 + η^2))
    x_naut = η/(1 - p)
    x_star = max(x_naut,2/η)
    k1 = g(m,p,η)
    A1 = k1 * x_naut
    k2 = 0
    A2 = 0
    if x_naut < 2/η
        k2 = exp(-η)
        if p > 0
            A2 = k2 * ((2/η)^p - x_naut^p) / p
        else
            y = 2/η^2
            if y < 0 y += ϵ end
            A2 = k2 * log(y)
        end
    end
    k3 = x_star^(p-1)
    A3 = 2 * k3 * exp(-x_star * η / 2) / η
    A = A1 + A2 + A3

    iteration = 1
    while true

        u = rand(rng)
        v = rand(rng)*A
        h = Inf
        if v <= A1
            x = x_naut * v / A1
            h = k1
        elseif v <= A1 + A2
            v = v - A1
            if p > 0
                x = (x_naut^p + (v*p/k2))^(1/p)
            else
                x = η*exp(v * exp(η))
            end
            h = k2 * x^(p-1)
        else
            v = v - (A1 + A2)
            y = exp(-x_star * η / 2) - (v * η) / (2 * k3)
            if y < 0 y += ϵ end
            x = -2 * log(y) / η
            h = k3 * exp(-x * η / 2)
        end
        if (u * h) <= g(x,p,η) || iteration == max_iteration
            iteration == max_iteration && verbose && @warn "GIG sampling reached max iteration: check results carefully"
            return x
        end
        iteration += 1
    end
end

#Sample the 2-parameter GIG distribution with parameters p and η using the
#Ratio-of-Uniforms without mode shift as described by Hörmann & Leydold (2014).
function sample_unif_no_mode_shift(rng::AbstractRNG, p::Real,η::Real; max_iteration::Integer = 10000, verbose::Bool = false)
    m = η/((1 - p) + sqrt((1 - p)^2 + η^2))
    x⁺= ((1 + p) + sqrt((1 + p)^2 + η^2))/η
    v⁺= sqrt(g(m,p,η))
    u⁺= x⁺ * sqrt(g(x⁺,p,η))

    iteration = 1
    while true
        u = rand(rng)*u⁺
        v = rand(rng)*v⁺
        x = u/v
        if v^2 <= g(x,p,η) || iteration == max_iteration
            iteration == max_iteration && verbose && @warn "GIG sampling reached max iteration: check results carefully"
            return x
        end
        iteration += 1
    end
end

#Sample the 2-parameter GIG distribution with parameters p and η using the
#Ratio-of-Uniforms with mode shift as described by Hörmann & Leydold (2014) (originally from Dagpunar (1989)).
function sample_unif_mode_shift(rng::AbstractRNG, p::Real, η::Real; max_iteration::Integer = 10000, verbose::Bool = false)
    m = (sqrt((p - 1)^2 + η^2) + (p-1)) / η
    a = -(2*(p+1)/η) - m
    b = (2*(p-1)/η) * m - 1
    p2 = b - (a^2)/3
    q = (2*a^3)/27 - (a*b/3) + m
    ϕ = acos(-(q/2) * sqrt(-27 / (p2^3)))
    x⁻ = sqrt((-4/3)*p2)*cos(ϕ/3 + (4/3)*π) - (a / 3)
    x⁺ = sqrt((-4/3)*p2)*cos(ϕ/3) - (a/3)
    v⁺ = sqrt(g(m,p,η))
    u⁻ = (x⁻ - m)*sqrt(g(x⁻,p,η))
    u⁺ = (x⁺ - m)*sqrt(g(x⁺,p,η))

    iteration = 1
    while true
        u = rand(rng)*(u⁺ - u⁻) + u⁻
        v = rand(rng)*v⁺
        x = (u / v) + m
        if (x > 0 && v^2 <= g(x,p,η)) || iteration == max_iteration
            iteration == max_iteration && verbose && @warn "GIG sampling reached max iteration: check results carefully"
            return x
        end
        iteration += 1
    end
end

function g(x::Real,p::Real,η::Real)
    x^(p-1)*exp(-(η/2)*(x + (1/x)))
end

# ---- Conditional simulator for batched_em (src/EM.jl) ----

"""
	GHEMSimulator(ξ; nsims = 30)
Conditional simulator of the spatial GH model for `batched_em`. A data set is an n₁ × n₂ × 1 × m array of m
replicates (spatial fields). Each EM iteration draws `nsims` completions of every replicate, which are returned as
the n₁ × n₂ × 1 × (m⋅nsims) array of the completed data. Conditional simulation is exact (see
`simulateconditionalGH`), so each "chain" is an independent draw; the draws are made in parallel over draws and data
sets. Replicates are grouped by missingness pattern, and the quantities that depend only on the pattern (Cholesky
factors) are computed once per pattern and EM iteration, with matrix solves over all replicates of a pattern.
"""
struct GHEMSimulator{X}
	ξ::X
	nsims::Int
end
GHEMSimulator(ξ; nsims::Integer = 30) = GHEMSimulator(ξ, Int(nsims))
withnsims(sim::GHEMSimulator, nsims::Integer) = GHEMSimulator(sim.ξ, nsims)

# Replicates that share a missingness pattern
struct GHPattern
	I₁::Vector{Int}        # observed elements
	I₂::Vector{Int}        # missing elements
	reps::Vector{Int}      # replicates with this pattern
	Z₁::Matrix{Float64}    # their observed data, |I₁| × |reps|
end

# Parameters of the conditional distribution of the missing data given the observed data, for one pattern
struct GHPatternParams
	L₂₂::LowerTriangular{Float64, Matrix{Float64}}  # Cholesky factor of Σ₂₂∣₁
	μ₂::Matrix{Float64}                              # μ₂∣₁ for each replicate, |I₂| × |reps|
	γ₂::Vector{Float64}                              # γ₂∣₁
	λ::Float64                                       # λ₂∣₁
	ω::Vector{Float64}                               # ω₂∣₁ for each replicate
	η::Vector{Float64}                               # η₂∣₁ for each replicate
end

struct GHParams
	patterns::Vector{GHPatternParams}
	draws::Vector{Vector{Matrix{Float64}}}  # draws[c][p]: draw c of the missing data of pattern p, |I₂| × |reps|
end

mutable struct GHEMState
	zfill::Array{Float32, 4}   # the data with missing values set to zero
	complete::Bool
	patterns::Vector{GHPattern}
	params::Union{Nothing, GHParams}
	rngs::Vector{Xoshiro}      # one per draw, so that draws can be made in parallel
end

outputbuffer(state::GHEMState, nsims::Integer) = Array{Float32}(undef, size(state.zfill)[1:3]..., size(state.zfill, 4) * nsims)
completeinput(state::GHEMState) = state.zfill

function init_state(sim::GHEMSimulator, Z₁::AbstractArray, θ₀; rng::AbstractRNG = Random.default_rng())
	@assert ndims(Z₁) == 4 && size(Z₁, 3) == 1 "each data set must be an n₁ × n₂ × 1 × m array"
	zfill = Float32[ismissing(z) ? 0f0 : z for z in Z₁]
	d = size(Z₁, 1) * size(Z₁, 2)
	Z = reshape(Z₁, d, :)
	idx = [findall(!ismissing, view(Z, :, r)) for r in axes(Z, 2)]
	@assert all(!isempty, idx) "cannot do conditional simulation for a replicate without observed data"
	patterns = GHPattern[]
	for I₁ in unique(idx)
		length(I₁) == d && continue # completely observed replicates need no simulation
		reps = findall(==(I₁), idx)
		I₂ = setdiff(1:d, I₁)
		push!(patterns, GHPattern(I₁, I₂, reps, Float64[Z[i, r] for i in I₁, r in reps]))
	end
	rngs = [Xoshiro(rand(rng, UInt64)) for _ = 1:sim.nsims]
	return GHEMState(zfill, isempty(patterns), patterns, nothing, rngs)
end

# Cholesky factor of a covariance matrix; if it is numerically singular, a small multiple of the identity is added
# (the previous implementation instead discarded the affected replicates)
const gh_jitter_count = Threads.Atomic{Int}(0)
function _cholesky_jitter(Σ)
	C = cholesky(Symmetric(Σ); check = false)
	issuccess(C) && return C.L
	ϵ = 1e-10 * tr(Σ) / size(Σ, 1)
	while true
		C = cholesky(Symmetric(Σ + ϵ * I); check = false)
		if issuccess(C)
			Threads.atomic_add!(gh_jitter_count, 1)
			return C.L
		end
		ϵ *= 10
	end
end

function _ghparams(ξ, θ)
	θ = Float64.(θ)
	names = ξ.parameter_names
	param(name) = θ[findfirst(==(name), names)]
	return (γ = param("γ"), ω = param("ω"), λ = param("λ"), ρ = param("ρ"), ν = param("ν"), η = 1.0)
end

function setparams!(state::GHEMState, sim::GHEMSimulator, θ)
	(; γ, ω, λ, ρ, ν, η) = _ghparams(sim.ξ, θ)
	Σ = matern_covariance(sim.ξ, ρ, ν)
	patterns = map(state.patterns) do pat
		I₁, I₂ = pat.I₁, pat.I₂
		L₁₁ = _cholesky_jitter(Σ[I₁, I₁])
		u = L₁₁ \ Σ[I₁, I₂]            # L₁₁⁻¹Σ₁₂
		w = L₁₁ \ fill(γ, length(I₁))  # L₁₁⁻¹γ₁
		V = L₁₁ \ pat.Z₁               # L₁₁⁻¹(Z₁ − μ₁), with μ = 0
		qγγ = dot(w, w)
		qzz = vec(sum(abs2, V; dims = 1))
		GHPatternParams(
			_cholesky_jitter(Σ[I₂, I₂] - u'u),
			u'V,
			γ .- u'w,
			λ - length(I₁) / 2,
			sqrt.((ω * η .+ qzz) .* (ω / η + qγγ)),
			sqrt.((ω * η .+ qzz) ./ (ω / η + qγγ))
		)
	end
	draws = [[Matrix{Float64}(undef, length(p.I₂), length(p.reps)) for p in state.patterns] for _ = 1:sim.nsims]
	state.params = GHParams(patterns, draws)
	return state
end

function update_chain!(state::GHEMState, sim::GHEMSimulator, c::Integer, ws)
	p = state.params::GHParams
	rng = state.rngs[c]
	for (k, pp) in enumerate(p.patterns)
		X = p.draws[c][k]
		randn!(rng, X)
		lmul!(pp.L₂₂, X) # Gaussian component, Σ₂₂∣₁^{1/2} W
		for r in axes(X, 2)
			M = simulategig(λ = pp.λ, ω = pp.ω[r], η = pp.η[r], rng = rng)
			sM = sqrt(M)
			@inbounds for i in axes(X, 1)
				X[i, r] = pp.μ₂[i, r] + pp.γ₂[i] * M + sM * X[i, r]
			end
		end
	end
	return nothing
end

function drawZ!(zbuf::AbstractArray{<:Any, 4}, state::GHEMState)
	p = state.params::GHParams
	m = size(state.zfill, 4)
	d = size(state.zfill, 1) * size(state.zfill, 2)
	for c in eachindex(p.draws)
		block = view(zbuf, :, :, :, ((c - 1) * m + 1):(c * m))
		copyto!(block, state.zfill)
		B = reshape(block, d, m)
		for (k, pat) in enumerate(state.patterns)
			X = p.draws[c][k]
			for (ir, r) in enumerate(pat.reps), (ii, i) in enumerate(pat.I₂)
				B[i, r] = X[ii, ir]
			end
		end
	end
	return zbuf
end
