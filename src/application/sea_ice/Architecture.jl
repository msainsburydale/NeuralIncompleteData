using Flux
using Distributions
using NeuralEstimators

function architecture(ξ; input_channels = 1)

	Ω = ξ.Ω

	p = length(Ω)
	a = [minimum.(values(Ω))...]
	b = [maximum.(values(Ω))...]

	architecture(p; input_channels = input_channels, outputactivation = Compress(a, b))
end

# TODO use ResNet architecture

function architecture(p::Integer; input_channels = 1, outputactivation = identity)

 stride = 1

 ψ = Chain(
 		Conv((11,11), input_channels => 16, relu; stride = stride),
		MaxPool((2, 2)),
		Conv((9, 9), 16 => 16, relu; stride = stride),
		MaxPool((2, 2)),
		Conv((7, 7), 16 => 32, relu; stride = stride),
		MaxPool((2, 2)),
		Conv((5, 5),  32 => 64,  relu; stride = stride),
		MaxPool((2, 2)),
		Conv((3, 3),  64 => 64, relu; stride = stride),
		MaxPool((2, 2)),
		Flux.flatten
		)
	ϕ = Chain(
		Dense(384, 500, relu),
		Dense(500, p)
	)

	return DeepSet(ψ, ϕ)
end

# Compute the dimensions at each layer:
#p = 1
#ψ = architecture(p).ψ
#ϕ = architecture(p).ϕ
#z = rand32(199, 219, 1, 1)
#ψ = ψ |> gpu
#z = z |> gpu
#ψ(z)
#[size(ψ[1:i](z)) for i in eachindex(ψ)]
#nparams.(ψ)
#nparams.(ϕ)
#sum(nparams.(ψ)) + sum(nparams.(ϕ)[1:end-1])




