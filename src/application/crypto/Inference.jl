using NeuralEstimators
using BenchmarkTools
using BSON: @load
using CSV
using DataFrames
using Random: seed!
using Folds
using StatsBase: ecdf

B = 400 # number of bootstrap samples

include(joinpath(pwd(), "src", "application", "crypto", "Model.jl"))
include(joinpath(pwd(), "src", "application", "crypto", "MAP.jl"))
include(joinpath(pwd(), "src", "application", "crypto", "Architecture.jl"))
relative_loadpath = joinpath("intermediates", "application", "crypto")
savepath = joinpath(pwd(), "img", "application", "crypto")
if !isdir(savepath) mkdir(savepath) end

"""
	addsingleton(x; dim)

# Examples
```
x = rand(4, 4, 10)
addsingleton(x; dim = 3)
```
"""
addsingleton(x; dim) = reshape(x, size(x)[1:dim-1]..., 1, size(x)[dim:end]...)


# threshold levels t
all_t = range(0, 0.1, step = 0.001)

# Joint lower- and upper-tail probabilities
lowertailprob(x, y, t) = sum(x .< t .&& y .< t) / length(x)
uppertailprob(x, y, t) = sum(x .> t .&& y .> t) / length(x)

function tailprob(x, y)
    lower = [lowertailprob(x, y, t) for t ∈ all_t]
    upper = [uppertailprob(x, y, t) for t ∈ sort(1 .- all_t)]
    return lower, upper
end

function integraltransform(v::V) where V <: AbstractVector{Union{Missing, T}} where T
    x = filter(!ismissing, v)
    x = convert(Array{T}, x)
    f = ecdf(x)
    broadcast(y -> ismissing(y) ? missing : f(y), v)
end

function integraltransform(v::V) where V <: AbstractVector{T} where T
    f = ecdf(v)
    f(v)
end

function probabilityestimates(θ̂, path)

    u = simulateu(θ̂)

    df = map([(1, 2), (1, 3), (2, 3)]) do (i, j)

        x = u[i]
        y = u[j]

        # point estimates
        tail_probs = tailprob(x, y)
        lower = tail_probs[1]
        upper = tail_probs[2]
        tail  = repeat(["lower", "upper"], inner = length(all_t))
        t     = vcat(all_t, sort(1 .- all_t))
        df    = DataFrame(hcat(t, tail, vcat(lower, upper)), ["t", "tail", "estimate"])

        pair = "$i-$j"
        df[:, :pair] .= pair

        df
    end
    df = vcat(df...)
    CSV.write(joinpath(path, "probability_estimates.csv"), df)
    df
end

function simulateu(θ̂)

    Σ = Symmetric(vectotriu(θ̂[ξ.Σidx], strict=true)); Σ[diagind(Σ)] .= one(eltype(Σ))
    L = cholesky(Σ).L
    L = reshape(L, size(L)..., 1)
    L = convert(Array, L)
    θ̂_parameters = Parameters(θ̂, L, [1], ξ.parameter_names)
    Z  = simulate(θ̂_parameters, 100_000)[1]

    u = map(1:d) do i
        v = Z[i, 1, :]
        integraltransform(v)
    end

    return u
end

function probabilityci(θ̃, path)

    df = map(1:size(θ̃, 2)) do b

        θ̂ = θ̃[:, b]
        θ̂ = reshape(θ̂, :, 1)
        u = simulateu(θ̂)

        df = map([(1, 2), (1, 3), (2, 3)]) do (i, j)

            x = u[i]
            y = u[j]

            # point estimates
            tail_probs = tailprob(x, y)
            lower = tail_probs[1]
            upper = tail_probs[2]
            tail  = repeat(["lower", "upper"], inner = length(all_t))
            t     = vcat(all_t, sort(1 .- all_t))
            df    = DataFrame(hcat(t, tail, vcat(lower, upper)), ["t", "tail", "estimate"])

            pair = "$i-$j"
            df[:, :pair] .= pair
            df[:, :b] .= b

            df
        end
        vcat(df...)
    end
    df = vcat(df...)
    CSV.write(joinpath(path, "probability_bootstrap_estimates.csv"), df)
    df = groupby(df, [:t, :tail, :pair])
    df = combine(df,
                 :estimate => (x -> quantile(x, 0.025)) => :lower,
                 :estimate => (x -> quantile(x, 0.975)) => :upper
                 )
    CSV.write(joinpath(path, "probability_pointwise_intervals.csv"), df)
    df
end


# ---- Load the data ----

formatdata(x) = x == "NA" ? missing : parse(Float32, x)

df = CSV.read(joinpath("data", "crypto", "standardised_data.csv"), DataFrame)

z₁ = Float32.(df[:, :Bitcoin]); z₁ = convert(Vector{Union{Missing, Float32}}, z₁)
z₂ = formatdata.(df[:, :Ethereum])
z₃ = formatdata.(df[:, :Avalanche])

z = permutedims(hcat(z₁, z₂, z₃))
z = copy(addsingleton(z, dim = 2))

m = size(z)[end] # number if iid replicates

# ---- Load the neural MAP estimator used for the EM algorithm ----

neuralMAP = architecture(ξ)
loadpath  = joinpath(pwd(), relative_loadpath, "runs_EM", "best_network.bson")
@load loadpath model_state
Flux.loadmodel!(neuralMAP, model_state)

# Value of H we will use during the EM algorithm
H=1

# ---- Load the neural MAP estimator using the encoding approach ----

estimatorencoding = architecture(ξ; input_channels = 2)
loadpath  = joinpath(pwd(), relative_loadpath, "runs_encoding", "best_network.bson")
@load loadpath model_state
Flux.loadmodel!(estimatorencoding, model_state)

# ---- Inference ----

function bivariateresdepcoefficient(λ, ω, η, γ, ρ)

    # Construct the correlation matrix from ρ
    Σ = Symmetric(vectotriu([one(ρ), ρ, one(ρ)]))

    # γ is a scalar but we need a vector
    γ = repeat([γ], 2)

    # Compute the coefficient
    ψ   = ω / η
    β   = (γ .+ sqrt.(ψ .+ γ.^2)) / ψ
    Σ⁻¹ = inv(Σ)
    ϕ   = (sqrt((ψ + γ' * Σ⁻¹ * γ) * β' * Σ⁻¹ * β) - β'*Σ⁻¹*γ)^-1

    return ϕ
end

function bivariateresdepcoefficient(θ, θ_names)

    γ = θ[findfirst(θ_names .== "γ")]
    η = one(eltype(θ)) # η = θ[findfirst(parameter_names .== "η")]
    ω = θ[findfirst(θ_names .== "ω")]
    λ = θ[findfirst(θ_names .== "λ")]

    # Vector of the correlation parameters
    ρ = θ[ξ.Σidx]

    # Compute the coefficient for each correlation parameter
    ϕ = bivariateresdepcoefficient.(λ, ω, η, γ, ρ)

    return ϕ
end

function saveci(ci, path::String, name::String)
    data = hcat(names(ci)[1], Array(ci[:, "lower"]), Array(ci[:, "upper"]))
    df   = DataFrame(data, ["parameters", "lower", "upper"])
    CSV.write(joinpath(path, "$(name)_pointwise_intervals.csv"), df)
end

θ_names = ξ.parameter_names
ϕ_names = replace.(θ_names[ξ.Σidx], "Σ" => "ϕ")

#  ---- Encoding approach

println("Starting inference on crypto data using masking approach...")

path = joinpath(savepath, "encoding")
if !isdir(path) mkdir(path) end

seed!(1)
zw = encodedata(z)
θ̂  = estimatorencoding(zw)                                           # point estimate
t = @belapsed estimateinbatches(estimatorencoding, zw)               # timing (uses GPU if available)
CSV.write(joinpath(path, "time_per_estimate.csv"), Tables.table([t]), header = false)
θ̃  = bootstrap(estimatorencoding, zw, B = B)                         # bootstrap sample
ci = interval(θ̃, parameter_names = θ_names, probs = [0.025, 0.975])  # credible interval
CSV.write(joinpath(path, "theta_estimates.csv"), DataFrame(θ̂', θ_names))
CSV.write(joinpath(path, "theta_bootstrap_estimates.csv"), DataFrame(θ̃', θ_names))
saveci(ci, path, "theta")

# residual dependence coefficient
ϕ̂  = bivariateresdepcoefficient(θ̂, θ_names)                               # point estimate
ϕ̃  = mapslices(θ -> bivariateresdepcoefficient(θ, θ_names), θ̃ , dims = 1) # bootstrap sample
ci = interval(ϕ̃ , parameter_names = ϕ_names, probs = [0.025, 0.975])      # credible interval
CSV.write(joinpath(path, "phi_estimates.csv"), DataFrame(permutedims(ϕ̂), ϕ_names))
CSV.write(joinpath(path, "phi_bootstrap_estimates.csv"), DataFrame(permutedims(ϕ̃), ϕ_names))
saveci(ci, path, "phi")

probabilityestimates(θ̂, path)
probabilityci(θ̃, path)


# ---- Neural EM algorithm

println("Starting inference on crypto data using EM approach...")

path = joinpath(savepath, "neuralEM")
if !isdir(path) mkdir(path) end

seed!(1)

# Initial value: use the prior mean
θ₀_nonΣ = mean.([ξ.Ω...][ξ.nonΣidx])
θ₀_Σ    = repeat([0.5], length(ξ.Σidx))
θ₀      = vcat(θ₀_nonΣ, θ₀_Σ)
neuralem = EM(simulateconditional, neuralMAP, θ₀)

# Point estimate
θ̂ = neuralem(z; ξ = ξ, ϵ = 0.1, use_ξ_in_simulateconditional = true)

# Timing for single estimate
t = @belapsed  neuralem(z; ξ = ξ, ϵ = 0.1, use_ξ_in_simulateconditional = true)
CSV.write(joinpath(path, "time_per_estimate.csv"), Tables.table([t]), header = false)

# Bootstrap
θ̃ = Folds.map(1:B) do _
    z̃ = subsetdata(z, rand(1:m, m))
    neuralem(z̃; ξ = ξ, ϵ = 0.1, nsims = H, use_ξ_in_simulateconditional = true)
end
θ̃ = hcat(θ̃...)
ci = interval(θ̃, parameter_names = θ_names, probs = [0.025, 0.975])
CSV.write(joinpath(path, "theta_estimates.csv"), DataFrame(θ̂', θ_names))
CSV.write(joinpath(path, "theta_bootstrap_estimates.csv"), DataFrame(θ̃', θ_names))
saveci(ci, path, "theta")

# residual dependence coefficient
ϕ̂  = bivariateresdepcoefficient(θ̂, θ_names)                               # point estimate
ϕ̃  = mapslices(θ -> bivariateresdepcoefficient(θ, θ_names), θ̃ , dims = 1) # bootstrap sample
ci = interval(ϕ̃ , parameter_names = ϕ_names, probs = [0.025, 0.975])      # credible interval
CSV.write(joinpath(path, "phi_estimates.csv"), DataFrame(permutedims(ϕ̂), ϕ_names))
CSV.write(joinpath(path, "phi_bootstrap_estimates.csv"), DataFrame(permutedims(ϕ̃), ϕ_names))
saveci(ci, path, "phi")

probabilityestimates(θ̂, path)
probabilityci(θ̃, path)

# ---- MAP estimator

println("Starting inference on crypto data using analytic MAP estimation...")

path = joinpath(savepath, "MAP")
if !isdir(path) mkdir(path) end

seed!(1)
ξ = merge(ξ, (θ₀ = θ₀,))
θ̂ = MAP([z], ξ)
t = @belapsed  MAP([z], ξ)
CSV.write(joinpath(path, "time_per_estimate.csv"), Tables.table([t]), header = false)
θ̃ = Folds.map(1:B) do _
    z̃ = subsetdata(z, rand(1:m, m))
    MAP([z̃], ξ)
end
θ̃ = hcat(θ̃...)
ci = interval(θ̃, parameter_names = θ_names, probs = [0.025, 0.975])
CSV.write(joinpath(path, "theta_estimates.csv"), DataFrame(θ̂', θ_names))
CSV.write(joinpath(path, "theta_bootstrap_estimates.csv"), DataFrame(θ̃', θ_names))
saveci(ci, path, "theta")

# residual dependence coefficient
ϕ̂  = bivariateresdepcoefficient(θ̂, θ_names)                               # point estimate
ϕ̃  = mapslices(θ -> bivariateresdepcoefficient(θ, θ_names), θ̃ , dims = 1) # bootstrap sample
ci = interval(ϕ̃ , parameter_names = ϕ_names, probs = [0.025, 0.975])      # credible interval
CSV.write(joinpath(path, "phi_estimates.csv"), DataFrame(permutedims(ϕ̂), ϕ_names))
CSV.write(joinpath(path, "phi_bootstrap_estimates.csv"), DataFrame(permutedims(ϕ̃), ϕ_names))
saveci(ci, path, "phi")

probabilityestimates(θ̂, path)
probabilityci(θ̃, path)


# ---- Empirical estimates

path = joinpath(savepath, "empirical")
if !isdir(path) mkdir(path) end

u = map(1:d) do i
    v = z[i, 1, :]
    integraltransform(v)
end

df = map([(1, 2), (1, 3), (2, 3)]) do (i, j)

    x = u[i]
    y = u[j]

    rmidx = ismissing.(x) .|| ismissing.(y)
    x = x[.!rmidx]
    y = y[.!rmidx]

    # point estimates
    tail_probs = tailprob(x, y)
    lower = tail_probs[1]
    upper = tail_probs[2]
    tail  = repeat(["lower", "upper"], inner = length(all_t))
    t     = vcat(all_t, sort(1 .- all_t))
    df_est = DataFrame(hcat(t, tail, vcat(lower, upper)), ["t", "tail", "estimate"])

    # non-parametric bootstrap-based pointwise credible intervals
    tail_probs = map(1:B) do _
        n = length(x)
        idx = rand(1:n, n)
        x̃ = x[idx]
        ỹ = y[idx]
        tailprob(x̃, ỹ)
    end
    lower = broadcast(t -> t[1], tail_probs); lower = hcat(lower...)
    upper = broadcast(t -> t[2], tail_probs); upper = hcat(upper...)
    ci_lower = interval(lower, probs = [0.025, 0.975]); ci_lower = vec(ci_lower)
    ci_upper = interval(upper, probs = [0.025, 0.975]); ci_upper = vec(ci_upper)

    ci = vcat(ci_lower, ci_upper)
    t  = vcat(repeat(all_t, outer = 2), repeat(sort(1 .- all_t), outer = 2))
    bound = repeat(repeat(["lower", "upper"], inner = length(all_t)), outer = 2)
    df_ci = DataFrame(hcat(t, repeat(tail, inner = 2), bound, ci), ["t", "tail", "bound", "estimate"])

    pair = "$i-$j"
    df_est[:, :pair] .= pair
    df_ci[:, :pair] .= pair

    df_est, df_ci
end
df_est = broadcast(t -> t[1], df); df_est = vcat(df_est...)
df_ci  = broadcast(t -> t[2], df); df_ci = vcat(df_ci...)

CSV.write(joinpath(path, "probability_estimates.csv"), df_est)
CSV.write(joinpath(path, "probability_pointwise_intervals.csv"), df_ci)
