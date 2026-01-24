using LinearAlgebra
using NeuralEstimators
using Optim
using Folds
using Flux: flatten
using Bessels: besselk

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

function MAP(Z::M, θ₀::V, ξ) where {T, R, V <: AbstractVector{T}, M <: AbstractMatrix{R}}

    # Since we transform the parameters during optimisation to force the estimates
    # to be within the prior support, here we compute the inverse-transformed true
    # values, which will then be passed into the optimiser.
	Ω = [ξ.Ω...]
	θ₀ = scaledlogit.(θ₀, Ω)

	# Closure that will be minimised
	loss(θ) = nll(θ, Z, ξ)

	# Estimate the parameters
	θ = optimize(loss, θ₀, NelderMead()) |> Optim.minimizer

	# Convert estimates to the original scale
  θ = scaledlogistic.(θ, Ω)

	return θ
end

function MAP(Z::A, θ₀::V, ξ) where {T, R, V <: AbstractVector{T}, A <: AbstractArray{R}}
	MAP(flatten(Z), θ₀, ξ)
end

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
		ℓ += (d/2-λ) * log(ω/η + quadform_γγ) - ( (d/2)*log(2π) + logdet(L) + λ * log(η) + log(besselk(λ, ω)) )
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
