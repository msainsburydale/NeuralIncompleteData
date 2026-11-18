using LinearAlgebra
using NeuralEstimators
using Folds
using Flux: flatten
using Bessels: besselk
using LBFGSB: lbfgsb

function MAP(Z::V, ξ) where {T, N, A <: AbstractArray{T, N}, V <: AbstractVector{A}}

	# Convert to Float64 to avoid rounding errors
	Z  = broadcast.(x -> !ismissing(x) ? Float64(x) : identity(x), Z)
	θ₀ = Float64.(ξ.θ₀)

	# Repeat θ₀ to match the length of Z
	K = size(θ₀, 2)
	m = length(Z)
	if m != K
		if (m ÷ K) == (m / K)
			θ₀ = repeat(θ₀, outer = (1, m ÷ K))
		else
			error("The length of the data vector, m = $m, and the number of parameter configurations, K = $K, do not match; further, m is not a multiple of K, so we cannot replicate θ to match Z.")
		end
	end

	# Optimise
	θ = Folds.map(eachindex(Z)) do k
		 MAP(Z[k], θ₀[:, k], ξ)
	end

	# Convert to matrix
	θ = hcat(θ...)

	return θ
end

"""
	MAP(Z::AbstractMatrix, θ₀, ξ)
MAP estimate of θ from the incomplete data `Z` (replicates in the columns, missing values as `missing`). The prior is
uniform, so the estimate maximises the likelihood over the prior support. It is computed with L-BFGS-B on the support
rescaled to [0, 1]ᵈ, starting from `θ₀` (the prior mean), with a central finite-difference gradient (step 1e-5,
truncated at the bounds), as for the GP model (`gp_MAP` in src/GP/Simulation.jl).

NB Nelder–Mead on the scaled-logit scale, used previously, stalled near the bounds of the support, where the logistic
function is flat: from the prior mean, it stopped short of the maximum (by up to 125 log-likelihood units, at points
that are not local maxima) in 37 of the 1000 MCAR and 56 of the 1000 MICB test data sets. L-BFGS-B reaches the
highest likelihood found from any starting value in all of them.
"""
function MAP(Z::M, θ₀::V, ξ) where {T, R, V <: AbstractVector{T}, M <: AbstractMatrix{R}}
	Ω = [ξ.Ω...]
	a, b = minimum.(Ω), maximum.(Ω)
	d = length(Ω)
	data = GHLikelihoodData(Z) # the replicates are grouped by missingness pattern once, here
	θ(u) = a .+ (b .- a) .* clamp.(u, 0, 1)
	# Non-finite values (e.g., Σ₁₁ numerically singular at a corner of the support) are replaced by a large value, so
	# that the line search backs off instead of stopping
	f(u) = (v = ghnll(θ(u), data, ξ); isfinite(v) ? v : 1e10)
	g!(g, u) = _fdgradient_box!(g, f, copy(u), 1e-5)
	_, u = lbfgsb(f, g!, (Float64.(θ₀) .- a) ./ (b .- a); lb = zeros(d), ub = ones(d), m = 5, factr = 1e5, pgtol = 0.0, maxiter = 500)
	return θ(u)
end

# Central finite-difference gradient on [0, 1]ᵈ, with the step truncated at the bounds
function _fdgradient_box!(g, f, u, h)
	for i in eachindex(u)
		ui = u[i]
		up = min(ui + h, one(ui)); u[i] = up; fp = f(u)
		um = max(ui - h, zero(ui)); u[i] = um; fm = f(u)
		u[i] = ui
		g[i] = (fp - fm) / (up - um)
	end
	return g
end

function MAP(Z::A, θ₀::V, ξ) where {T, R, V <: AbstractVector{T}, A <: AbstractArray{R}}
	MAP(flatten(Z), θ₀, ξ)
end

# ---- Negative log-likelihood ----

# The incomplete data grouped by missingness pattern: for each pattern, the observed elements I₁ and the observed
# data of the replicates with that pattern (|I₁| × number of replicates)
struct GHLikelihoodData
	groups::Vector{Tuple{Vector{Int}, Matrix{Float64}}}
end
function GHLikelihoodData(Z::AbstractMatrix)
	idx = [findall(!ismissing, view(Z, :, r)) for r in axes(Z, 2)]
	groups = [(I₁, Float64[Z[i, r] for i in I₁, r in findall(==(I₁), idx)]) for I₁ in unique(idx) if !isempty(I₁)]
	return GHLikelihoodData(groups)
end

"""
	ghnll(θ, data::GHLikelihoodData, ξ)
	nll_fast(θ, data::GHLikelihoodData, ξ)
Negative log-likelihood of the spatial GH model, with θ on the original scale (`ghnll`) or on the unconstrained
(scaled-logit) scale (`nll_fast`). This computes the same quantity as `nll` below, but evaluates the Matérn covariance
function once per distinct distance, factorises Σ₁₁ once per missingness pattern, uses matrix solves over all
replicates of a pattern, and computes the terms that do not depend on the data once per pattern. As in `nll`,
replicates for which the density cannot be evaluated are ignored. If Σ₁₁ is numerically singular, the negative
log-likelihood is infinite.
"""
nll_fast(θ, data::GHLikelihoodData, ξ) = ghnll(scaledlogistic.(θ, [ξ.Ω...]), data, ξ)
function ghnll(θ, data::GHLikelihoodData, ξ)
	(; γ, ω, λ, ρ, ν, η) = _ghparams(ξ, θ)
	Σ = matern_covariance(ξ, ρ, ν)
	logKλω = log(_besselk(λ, ω)) # _besselk (src/GH/Simulation.jl): accurate when λ is near an integer
	ℓ = 0.0
	for (I₁, Z₁) in data.groups
		C = cholesky(Symmetric(Σ[I₁, I₁]); check = false)
		issuccess(C) || return Inf
		L = C.L
		d = length(I₁)
		X = L \ Z₁                 # L⁻¹(z − μ) for each replicate, with μ = 0
		y = L \ fill(γ, d)         # L⁻¹γ
		qγγ = dot(y, y)
		const_term = (d/2 - λ) * log(ω/η + qγγ) - ((d/2) * log(2π) + logdet(L) + λ * log(η) + logKλω)
		for r in axes(X, 2)
			x = view(X, :, r)
			qzz = dot(x, x)
			qzγ = dot(x, y)
			sqrt_term = sqrt((ω*η + qzz) * (ω/η + qγγ))
			bessel_term = besselk(λ - d/2, sqrt_term)
			bessel_term < 0 && continue # (log(bessel_term) fails, and nll ignores the replicate)
			ℓ += log(bessel_term) + qzγ - (d/2 - λ) * log(sqrt_term) + const_term
		end
	end
	return -ℓ
end

# Reference implementation (the original), used in tests
function nll(θ, Z, ξ)

  D = ξ.D
  Ω = ξ.Ω

  # Constrain θ to be within the prior support and valid
  Ω = [Ω...]
  θ = copy(θ)
  θ = scaledlogistic.(θ, Ω)

  # Extract parameters
	d  = size(Z, 1)
	γ  = θ[findfirst(parameter_names .== "γ")]; γ = repeat([γ], d)
	η  = one(eltype(θ)) 
	ω = θ[findfirst(parameter_names .== "ω")]
	λ = θ[findfirst(parameter_names .== "λ")]
	ρ = θ[findfirst(parameter_names .== "ρ")]
	ν = θ[findfirst(parameter_names .== "ν")]
	
	# Covariance matrix 
  Σ = matern.(UpperTriangular(D), ρ, ν)
	Σ = Symmetric(Σ)

	# Compute the log-likelihood
	ℓ = GHdensity(Z, Σ, γ = γ, λ = λ, ω = ω, η = η)

	return -ℓ
end


# ---- Generalised-hyperbolic density ----

function GHdensity(Z::A, Σ::M; args...) where {A <: Union{AbstractVector{T}, AbstractMatrix{T}}, M <: AbstractMatrix} where {T}
	L = cholesky(Symmetric(Σ)).L
	GHdensity(Z, L; args...)
end

function GHdensity(Z::M, L::LT; γ, λ, ω, η, μ = repeat([zero(T)], length(γ)), logdensity::Bool = true) where {M <: AbstractMatrix{Union{Missing, T}}, LT <: LowerTriangular} where {T}

	# Get the indices of the observed component for each replicate.
	m   = size(Z, 2)
	idx = [findall(x -> !ismissing(x), vec(Z[:, i])) for i ∈ 1:m]

	# This code caters for complete and incomplete data. Here, we drop any
	# missing observations from Z, and in the process convert the eltype of Z
	# from Union{Missing, x} to x, where x is some basic type (e.g., Float64).
	Z = [[Z[idx[i], i]...] for i ∈ 1:m]

	# Compute the density over groups of replicates with the same missingness pattern
	ℓ = map(unique(idx)) do I₁
		 x = findall(Ref(I₁) .== idx) # find all replicates with the same missingness pattern
		 z = hcat(Z[x]...)
		 μstar, γstar, Σstar, λstar, ωstar, ηstar = GHmarginalparameters(I₁, μ, γ, L, λ, ω, η)
		 Lstar = cholesky(Σstar).L
		 GHdensity(z, Lstar; μ = μstar, γ = γstar, λ = λstar, ω = ωstar, η = ηstar, logdensity = logdensity)
	end
	ℓ  = skipmissing(ℓ)
	return logdensity ? sum(ℓ) : prod(ℓ)
end

# function GHdensity(Z::A, L::LT; logdensity::Bool = true, args...) where {A <: AbstractArray{T, N}, LT <: LowerTriangular} where {T, N}
function GHdensity(Z::A, L::LT; logdensity::Bool = true, args...) where {A <: Union{AbstractVector{T}, AbstractMatrix{T}}, LT <: LowerTriangular} where {T}
	ℓ  = [GHdensity(z, L; logdensity = logdensity, args...) for z in eachcol(Z)]
	ℓ  = skipmissing(ℓ)
	return logdensity ? sum(ℓ) : prod(ℓ)
end

function GHdensity(z::V, L::LT; γ, λ, ω, η, μ = repeat([zero(T)], length(γ)), logdensity::Bool = true) where {V <: AbstractVector{T}, LT <: LowerTriangular} where T

	d = length(z)

	# Quadratic forms: uses the result u'Σv = x'y where x = L⁻¹u and y = L⁻¹v.
	x = L \ (z - μ)  # solution to Lx = z-μ
	y = L \ γ        # solution to Ly = γ
	quadform_zz = dot(x, x)
	quadform_γγ = dot(y, y)
	quadform_zγ = dot(x, y)
	sqrt_term = sqrt((ω*η + quadform_zz) * (ω/η + quadform_γγ))
	bessel_term = besselk(λ - d/2, sqrt_term)
	try
		ℓ = log(bessel_term) + quadform_zγ - (d/2 - λ) * log(sqrt_term)
		ℓ += (d/2-λ) * log(ω/η + quadform_γγ) - ( (d/2)*log(2π) + logdet(L) + λ * log(η) + log(_besselk(λ, ω)) )
		return logdensity ? ℓ : exp(ℓ)
	catch
		# Try to identify why the computation failed
		if sqrt_term > 10^6
			@warn "sqrt_term blew up: ignoring this replicate"
		elseif bessel_term < 0
			@warn "bessel_term < 0: ignoring this replicate"
		else
		  @warn "ignoring replicate for unknown reason"
		end
		return missing
	end
end

# I₁ are the indices of the observed elements
function GHmarginalparameters(I₁, μ, γ, L::LowerTriangular, λ, ω, η)

  r = length(I₁)
	@assert r > 0 "Need to have at least one observed element: that is, need `length(I₁) > 0`)"

	d = length(μ)
	I₂ = (1:d)[1:d .∉ Ref(I₁)]
	@assert d == length(γ) == length(μ)
	@assert size(L) == (d, d)

	Σ = L * L'

	if r == d
		return μ, γ, Σ, λ, ω, η
	else
		μ₁  = μ[I₁]
		γ₁  = γ[I₁]
		Σ₁₁ = Σ[I₁, I₁]
		return μ₁, γ₁, Σ₁₁, λ, ω, η
	end
end
