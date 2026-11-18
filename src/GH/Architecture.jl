# Architecture of the NBEs used in the GH study (Section S4 of the supplement): a smaller version of the architecture
# in src/Architecture.jl (which must be included first), suited to the 16 × 16 grid. Each data set consists of
# independent replicates (spatial fields), which are combined by the DeepSets summary network.

function gh_initialize_estimator(d::Integer; input_channels, outputactivation, dropout::Bool = true, activation = relu, residualblock = _residualblock)
    ψ = Chain(
        Conv((3, 3), input_channels => 16, pad = 1, bias = false),
        BatchNorm(16, relu),
        residualblock((3, 3), 16 => 32, stride = 2),
        residualblock((3, 3), 32 => 64),
        GlobalMeanPool(),
        Flux.flatten
    )
    ϕ = Chain(
        Dense(64, 128, activation),
        (dropout ? [Dropout(0.05)] : [])...,
        Dense(128, 128, activation),
        (dropout ? [Dropout(0.05)] : [])...,
        Dense(128, d),
        outputactivation
    )
    return MAPEstimator(DeepSet(ψ, identity), ϕ)
end

# Ensemble of J estimators whose outputs are compressed to the prior support [a, b]
function gh_architecture(d, a, b; input_channels = 1, J::Integer = 5, kwargs...)
    Ensemble([gh_initialize_estimator(Int(d); input_channels = Int(input_channels), outputactivation = Compress(a, b), kwargs...) for _ in 1:J])
end
