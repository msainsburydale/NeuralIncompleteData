using LinearAlgebra
using NeuralEstimators
using Optim
using Folds
using Flux: flatten
using RecursiveArrayTools
using Bessels: besselk

function MAP(Z::V, ξ) where {T, N, A <: AbstractArray{T, N}, V <: AbstractVector{A}}

	# Compress the data from an n-dimensional array to a matrix
	Z = flatten.(Z)

	# Convert to Float64 to avoid rounding errors
	Z  = broadcast.(x -> !ismissing(x) ? Float64(x) : identity(x), Z)
	θ₀ = Float64.(ξ.θ₀)

	# If Z is replicated, try to replicate θ₀ accordingly
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
	θ₀[ξ.nonΣidx] = scaledlogit.(θ₀[ξ.nonΣidx], Ω[ξ.nonΣidx])

	# Closure that will be minimised
	loss(θ) = nll(θ, Z, ξ)

	# Estimate the parameters
	θ = optimize(loss, θ₀, NelderMead()) |> Optim.minimizer

	# Convert estimates to the original scale
	# Since we parameterised Σ directly in terms of its Cholesky factor,
	# we need to now construct Σ and extract the implied parameters.
	θ[ξ.nonΣidx] = scaledlogistic.(θ[ξ.nonΣidx], Ω[ξ.nonΣidx])
	vectocholesky = ξ.correlations_only ? vectocorrelationcholesky : vectocovariancecholesky
	L = vectocholesky(θ[ξ.Σidx])
	Σ = L * L'
	θ[ξ.Σidx] = Σ[tril(trues(ξ.d, ξ.d), ξ.correlations_only ? -1 : 0)]

	return θ
end

function nll(θ, Z, ξ)

    # Constrain θ to be within the prior support and valid
    Ω = [ξ.Ω...]
    θ = copy(θ)
    θ[ξ.nonΣidx] = scaledlogistic.(θ[ξ.nonΣidx], Ω[ξ.nonΣidx])
    vectocholesky = ξ.correlations_only ? vectocorrelationcholesky : vectocovariancecholesky
	L = vectocholesky(θ[ξ.Σidx])

	d  = size(Z, 1)
	γ  = θ[findfirst(parameter_names .== "γ")]; γ = repeat([γ], d)
	ω  = θ[findfirst(parameter_names .== "ω")]
	η  = one(eltype(θ)) 
	λ  = θ[findfirst(parameter_names .== "λ")]

	# Compute the log-likelihood
	ℓ = GHdensity(Z, L, γ = γ, λ = λ, ω = ω, η = η)

	return -ℓ
end

# Based on the approach described [here](https://stats.stackexchange.com/a/404453).
# The mappping of `v` to the Cholesky factor is a well behaved diffeomorphism,
# so it can be inverted; however, this inversion is not implemented.
"""
	vectocorrelationcholesky(v::Vector)
Transforms a vector `v` ∈ ℝᵈ, where d is a triangular number, into the Cholesky
factor 𝐋 of a correlation matrix 𝐑 = 𝐋𝐋'.
"""
function vectocorrelationcholesky(v)
	L = vectotril(v; strict = true)
	L = unitdiagonal(L)
	x = rowwisenorm(L)
	L = L ./ x
	L = LowerTriangular(L)
	return L
end
unitdiagonal(A) = A - Diagonal(A[diagind(A)]) + I

# # Alternative vectocorrelationcholesky that Stan uses (I prefer the simpler version above)
# function vectocorrelationcholesky(v)
# 	v = cpu(v)
# 	z = tanh.(vectotril(v; strict=true))
# 	n = length(v)
# 	d = (-1 + isqrt(1 + 8n)) ÷ 2 + 1
# 	L = [ correlationcholeskyterm(i, j, z)  for i ∈ 1:d, j ∈ 1:d ]
# end
# function correlationcholeskyterm(i, j, z)
# 	T = eltype(z)
# 	if i < j
# 		zero(T)
# 	elseif 1 == i == j
# 		one(T)
# 	elseif 1 == j < i
# 		z[i, j]
# 	elseif 1 < j == i
# 		prod(sqrt.(one(T) .- z[i, 1:j-i].^2))
# 	else
# 		z[i, j] * prod(sqrt.(one(T) .- z[i, 1:j-i].^2))
# 	end
# end

# Based on Williams (1996) "Using Neural Networks to Model Conditional Multivariate Densities"
"""
	vectocovariancecholesky(v::Vector)
Transforms a vector `v` ∈ ℝᵈ, where d is a triangular number, into the Cholesky
factor 𝐋 of a covariance matrix 𝚺 = 𝐋𝐋'.
"""
function vectocovariancecholesky(v)
	L = vectotril(v)
	diag = L[diagind(L)]
	L = L - Diagonal(diag) + Diagonal(softplus.(diag))
	L = LowerTriangular(L)
	return L
end


