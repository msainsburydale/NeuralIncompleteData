using Distributions
using Flux
using NeuralEstimators
import Flux: Bilinear

function (b::Bilinear)(Z::A) where A <: AbstractArray{T, 3} where T
	@assert size(Z, 2) == 2
	x = Z[:, 1, :]
	y = Z[:, 2, :]
	b(x, y)
end

function architecture(ξ; input_channels::Integer = 1)

    d = ξ.d  # dimension of each replicate
    p = ξ.p  # total number of parameters
	  correlations_only = ξ.correlations_only # correlation or covariance matrix?

    first_layer = input_channels == 1 ? Chain(Flux.flatten, Dense(d, 128, relu)) : Bilinear((d, d) => 128, relu)

    # Construct the final layer to ensure valid parameter estimates. Usually 
    # this is relatively simple, but its slightly more complicated when we have
    # a correlation/covariance matrix to estimate, since then there are joint
    # conditions on the parameters (namely, the paramter estimates must combine
    # to yield a postive definite matrix). 

  	# Non-Σ parameters
  	Ω₁ = drop(ξ.Ω, :Σ)
  	p₁ = length(Ω₁)
  	a = [minimum.(values(Ω₁))...]
  	b = [maximum.(values(Ω₁))...]
  	l₁ = Compress(a, b)
  	# Σ parameters
  	l₂ = correlations_only ? CorrelationMatrix(ξ.d) : CovarianceMatrix(ξ.d)
  	p₂ = p - p₁
  	
  	final_layer = Parallel(
  	  vcat,
  	  Chain(Dense(128, p₁, identity), l₁),
  	  #Chain(Dense(128, p₂, identity), x -> l₂(x))
  	  Dense(128, p₂, identity) 
  	)

    ψ = Chain(
  		first_layer,
  		Dense(128, 256, relu),
  		Dense(256, 512, relu)
		)

    ϕ = Chain(
  		Dense(512, 256, relu),
  		Dense(256, 128, relu),
  		final_layer
		)

	return DeepSet(ψ, ϕ)
end
