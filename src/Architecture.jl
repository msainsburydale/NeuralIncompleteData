using Flux
using NeuralEstimators

function architecture(ξ; input_channels = 1)

	Ω = ξ.Ω

	p = length(Ω)
	a = [minimum.(values(Ω))...]
	b = [maximum.(values(Ω))...]

	architecture(p; input_channels = input_channels, outputactivation = Compress(a, b))
end

function architecture(p, input_channels = 1; outputactivation = softplus)

  p = Int(p)
  input_channels = Int(input_channels)

	estimators = map(1:5) do _
		ψ = Chain(
			Conv((3, 3), input_channels=>16, pad=1, bias=false), 
			BatchNorm(16, relu),   
			ResidualBlock((3, 3), 16 => 16),                               
			ResidualBlock((3, 3), 16 => 32, stride=2),                     
			ResidualBlock((3, 3), 32 => 64, stride=2),                     
			ResidualBlock((3, 3), 64 => 128, stride=2),                    
			GlobalMeanPool(),                                              
			Flux.flatten,
			Dense(128, 128)                                                
		)
		ϕ = Chain(
			Dense(128, 512, relu),
			Dense(512, p),
			outputactivation
		)
		PointEstimator(DeepSet(ψ, ϕ))
	end

  return Ensemble(estimators)
end

# Compute the dimensions at each layer:
#p = 1
#nn = Chain(
#			Conv((3, 3), 1=>16, pad=1, bias=false), 
#			BatchNorm(16, relu),   
#			ResidualBlock((3, 3), 16 => 16),                               
#			ResidualBlock((3, 3), 16 => 32, stride=2),                     
#			ResidualBlock((3, 3), 32 => 64, stride=2),                     
#			ResidualBlock((3, 3), 64 => 128, stride=2),                    
#			GlobalMeanPool(),                                              
#			Flux.flatten,
#			Dense(128, 128),                                         
#			Dense(128, 512, relu),
#			Dense(512, p)
#		)
#z  = rand32(32, 17, 1, 1)
#[size(nn[1:i](z)) for i in eachindex(nn)]
#[nparams(nn[i]) for i in eachindex(nn)]



