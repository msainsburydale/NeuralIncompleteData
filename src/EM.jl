import NeuralEstimators: EM
using DataFrames
using Random: AbstractRNG, Xoshiro, default_rng
using Statistics: mean

@isdefined(_tforeach) || include(joinpath(@__DIR__, "Threading.jl")) # _tforeach

# ---- Batched neural EM with persistent conditional-simulation chains ----

# Default hooks of the simulator interface (see batched_em)
emworkspace(sim) = nothing
ondevice(sim) = false
outputbuffer(state, nsims::Integer) = Array{eltype(state.zfill)}(undef, size(state.zfill)..., 1, nsims)
completeinput(state) = reshape(state.zfill, size(state.zfill)..., 1, 1)

"""
	batched_em(sim, MAP, Z::AbstractArray, θ₀::AbstractVector; kwargs...)
	batched_em(sim, MAP, Z::AbstractVector{<:AbstractArray}, θ₀; kwargs...)
Neural EM algorithm ([Sainsbury-Dale et al., 2025](https://arxiv.org/abs/2501.04330)) for one data set `Z` with
initial estimate `θ₀`, or for many data sets at once (with `θ₀` a vector, used for every data set, or a matrix with
one column per data set). It differs from `NeuralEstimators.EM` in two respects:
- **Persistent chains.** The conditional simulator `sim` (e.g., a `HiddenPottsEMSimulator`; see
  `src/HiddenPotts/Simulation.jl`) keeps its MCMC chains from one EM iteration to the next: the chains are initialised
  and burnt in once, and later iterations continue them under the current estimate (persistent-chain Monte Carlo EM).
- **Lock-step batching.** In each EM iteration, the data sets are processed in chunks of `chunksize`: the conditional
  simulations of a chunk are run in parallel over threads (over data sets and chains, with no nested parallelism),
  and the `MAP` estimator is then applied to the whole chunk on `device` (in a single call, unless the chunk has more
  than `maxbatch` images).
- **Simulation on a device.** A simulator whose chains run on a device, such as a `HiddenPottsEMSimulator` with a GPU
  backend, updates the chains of a data set together, one data set after another, and draws the completed data into
  an array on the device. These arrays are passed to `MAP` as they are, so the completed data never leave the device
  (`MAP` should then be on that device too, e.g. moved there once with `Flux.gpu_device()`).

The convergence assessment is identical to that of `NeuralEstimators.EM`: with `θ̄ₗ` the average of the estimates
from iteration `burnin` to iteration `l` (counting the initial value as iteration 0, so that, as in
`NeuralEstimators.EM`, the average includes iterate `burnin`), a data set has converged once
`maxᵢ |θ̄ₗ₊₁,ᵢ - θ̄ₗ,ᵢ| / (|θ̄ₗ,ᵢ| + ϵ) < tol` for `nconsecutive` consecutive iterations, after which it leaves the batch.

The simulator must support `init_state(sim, Z, θ₀; rng)`, `setparams!(state, sim, θ)`,
`update_chain!(state, sim, c, workspace)`, and `drawZ!(buffer, state)`, and have field `nsims`; the state must have
fields `zfill` (the data with missing values filled in), `complete`, and `params`. Simulators whose completed data
do not have the default shape `(size(zfill)..., 1, nsims)` (e.g., data sets with replicates) also define
`outputbuffer(state, nsims)` and `completeinput(state)`, and simulators that need a per-task workspace in
`update_chain!` define `emworkspace(sim)`. Simulators that draw exactly (rather than by MCMC) simply ignore the
chain history: `update_chain!` then draws the `c`th completion of the data. Simulators whose chains run on a device
define `ondevice(sim) = true`, `update_chains!(state, sim)`, which updates all the chains of a data set, and
`drawZ!(state)`, which returns the completed data on the device; they need neither `update_chain!` nor
`drawZ!(buffer, state)`.

# Keyword arguments
- `niterations = 50`, `burnin = 1`, `nconsecutive = 3`, `tol = 0.01`: as for `NeuralEstimators.EM`.
- `chunksize = 32`: number of data sets processed together in each EM iteration.
- `maxbatch = 65535`: maximum number of images (replicates × completions, summed over data sets) passed to `MAP` in a
  single call; a chunk exceeding it is split over several calls. cuDNN's batch normalisation does not support a batch
  dimension larger than 65535 (e.g., 32 GH data sets × 150 replicates × 30 completions = 144000 images).
- `device = nothing`, `use_gpu = true`: device used for `MAP` (as for `NeuralEstimators.EM`).
- `rng = Random.default_rng()`: random-number generator from which the chains are seeded.
- `verbose = false`: whether to print progress.

# Returns
For a single data set, a named tuple `(estimate, burnin, iterates)` as for `NeuralEstimators.EM`. For multiple data
sets, a named tuple with fields `estimates` (a matrix with one column per data set), `iterates` (one matrix of
iterates per data set), `niterations` (the number of EM iterations used for each data set), `converged`, and
`timings` (seconds spent in conditional simulation and in MAP estimation).
"""
# Applies the estimator to the data sets Zs, split into consecutive groups with at most `maxbatch` images (the last
# dimension of each data set, summed over the group) per call; returns one column per data set.
function _estimate_split(MAPd, Zs, dev, cpudev; maxbatch::Integer = 65535)
    groups = UnitRange{Int}[]
    start, count = 1, 0
    for i in 1:length(Zs)
        m = size(Zs[i], ndims(Zs[i]))
        if count + m > maxbatch && i > start
            push!(groups, start:(i - 1))
            start, count = i, 0
        end
        count += m
    end
    push!(groups, start:length(Zs))
    reduce(hcat, [cpudev(MAPd(dev(collect(Zs[g])))) for g in groups])
end

function batched_em(
    sim, MAP, Zs::AbstractVector{<:AbstractArray}, θ₀;
    niterations::Integer = 50,
    burnin::Integer = 1,
    nconsecutive::Integer = 3,
    tol = 0.01,
    chunksize::Integer = 32,
    maxbatch::Integer = 65535,
    device = nothing,
    use_gpu::Bool = true,
    rng::AbstractRNG = default_rng(),
    verbose::Bool = false
)
    @assert 0 <= burnin < niterations
    @assert chunksize > 0 && maxbatch > 0
    K = length(Zs)
    θ₀ = θ₀ isa Number ? fill(θ₀, 1, K) : θ₀ isa AbstractVector ? repeat(θ₀, 1, K) : θ₀
    @assert size(θ₀, 2) == K "θ₀ must have one column per data set"
    θ₀ = Array(θ₀)
    d = size(θ₀, 1)

    dev = NeuralEstimators._resolvedevice(; use_gpu = use_gpu, device = device, verbose = verbose)
    cpudev = NeuralEstimators.cpu_device()
    MAPd = dev(MAP)

    nsims = sim.nsims
    seeds = [rand(rng, UInt64) for _ = 1:K]
    states = Vector{Any}(undef, K)
    _tforeach(K) do k, _ # set-up for conditional simulation, in parallel over data sets
        states[k] = init_state(sim, Zs[k], θ₀[:, k]; rng = Xoshiro(seeds[k]))
    end
    states = [s for s in states]
    θ_all = [reshape(θ₀[:, k], :, 1) for k = 1:K]
    barθ = Vector{Any}(nothing, K)
    counter = zeros(Int, K)
    done = falses(K)
    converged = falses(K)
    nits = zeros(Int, K)
    estimates = zeros(promote_type(eltype(θ₀), Float32), d, K)

    # Complete data sets need no conditional simulation
    for k = 1:K
        states[k].complete || continue
        estimates[:, k] = vec(cpudev(MAPd(dev(completeinput(states[k])))))
        done[k] = converged[k] = true
    end

    # Reusable buffers for the completed data of a chunk (keyed by image size)
    buffers = Dict{Any, Vector}()
    getbuffers(state, n) = begin
        z = state.zfill
        key = (eltype(z), size(z))
        bufs = get!(() -> [outputbuffer(state, nsims) for _ = 1:chunksize], buffers, key)
        view(bufs, 1:n)
    end

    t_sim = 0.0
    t_est = 0.0
    active = [k for k = 1:K if !done[k]]
    for l = 1:niterations
        isempty(active) && break
        for chunk in Iterators.partition(active, chunksize)
            chunk = collect(chunk)
            n = length(chunk)

            # Conditional simulation: parallel over (data set, chain), or on a device
            t_sim += @elapsed begin
                _tforeach(n) do i, _
                    k = chunk[i]
                    setparams!(states[k], sim, θ_all[k][:, end])
                end
                if ondevice(sim)
                    # The chains of each data set are updated together on the device, one data set after another (on
                    # this task), and the completed data stay there
                    Zchunk = map(chunk) do k
                        update_chains!(states[k], sim)
                        drawZ!(states[k])
                    end
                else
                    _tforeach(n * nsims; workspace = () -> emworkspace(sim)) do i, ws
                        k = chunk[(i - 1) ÷ nsims + 1]
                        update_chain!(states[k], sim, (i - 1) % nsims + 1, ws)
                    end
                    Zchunk = getbuffers(states[chunk[1]], n)
                    if all(size(states[k].zfill) == size(states[chunk[1]].zfill) for k in chunk)
                        _tforeach(n) do i, _
                            drawZ!(Zchunk[i], states[chunk[i]])
                        end
                    else
                        Zchunk = [drawZ!(outputbuffer(states[k], nsims), states[k]) for k in chunk]
                    end
                end
                for k in chunk
                    states[k].params = nothing # free the parameter-dependent memory until the next iteration
                end
            end

            # MAP estimation: batched calls for the chunk
            t_est += @elapsed θ̂ = _estimate_split(MAPd, Zchunk, dev, cpudev; maxbatch = maxbatch)

            # Convergence assessment (identical to NeuralEstimators.EM)
            for (i, k) in enumerate(chunk)
                θ_all[k] = hcat(θ_all[k], θ̂[:, i])
                nits[k] = l
                if l > burnin
                    b = mean(θ_all[k][:, (burnin + 1):end]; dims = 2)
                    if l > burnin + 1 && maximum(abs.(b .- barθ[k]) ./ (abs.(barθ[k]) .+ eps())) < tol
                        counter[k] += 1
                        if counter[k] == nconsecutive
                            converged[k] = done[k] = true
                        end
                    else
                        counter[k] = 0
                    end
                    barθ[k] = b
                end
            end
        end
        active = filter(k -> !done[k], active)
        verbose && @info "EM iteration $l: $(length(active)) of $K data sets still active (simulation $(round(t_sim, digits = 2)) s, estimation $(round(t_est, digits = 2)) s)"
    end
    verbose && !all(converged) && @warn "The EM algorithm failed to converge for $(count(!, converged)) of $K data sets"

    for k = 1:K
        isnothing(barθ[k]) || (estimates[:, k] = vec(barθ[k]))
    end

    return (estimates = estimates, iterates = θ_all, niterations = nits, converged = converged, timings = (simulation = t_sim, estimation = t_est))
end

function batched_em(sim, MAP, Z::AbstractArray{Union{Missing, T}}, θ₀::AbstractVector; burnin::Integer = 1, kwargs...) where {T}
    r = batched_em(sim, MAP, [Z], reshape(θ₀, :, 1); burnin = burnin, kwargs...)
    return (estimate = r.estimates[:, 1], burnin = burnin, iterates = r.iterates[1])
end

# ---- Helpers for visualising EM iterates ----

function _iterates_df(estimates, burnin, parameter_names, i, nsims)
    nparams, niters = size(estimates)

    # compute post–burn-in means for each parameter
    averaged_estimates = similar(estimates)
    averaged_estimates .= NaN
    if niters > burnin
        for p in 1:nparams
            for t in (burnin+1):niters
                averaged_estimates[p, t] = mean(estimates[p, (burnin+1):t])
            end
        end
    end

    if isnothing(parameter_names)
        parameter_names = ["θ$i" for i ∈ 1:nparams]
    end

    # flatten to long DataFrame
    return DataFrame(
        iteration = repeat(1:niters, inner = nparams),
        parameter = repeat(parameter_names, outer = niters),
        estimate = vec(estimates),
        averaged_estimate = vec(averaged_estimates),
        theta_0 = fill(i, nparams * niters),
        nsims = fill(nsims, nparams * niters),
    )
end

# Helper function for visualizing EM iterates (NeuralEstimators.EM)
function run_EM(em::EM, Z1, all_θ₀; parameter_names = nothing, all_nsims = [1, 10, 30], burnin::Integer = 10, kwargs...)
    if isa(parameter_names, String)
        parameter_names = [parameter_names]
    end
    dfs = []
    for (i, θ₀) in enumerate(all_θ₀)
        for nsims in all_nsims
            res = em(Z1, θ₀; nsims = nsims, burnin = burnin, tol = 1e-7, niterations = 100, kwargs...)
            push!(dfs, _iterates_df(res.iterates, burnin, parameter_names, i, nsims))
        end
    end
    return vcat(dfs...)
end

# Helper function for visualizing EM iterates (batched_em with a conditional simulator such as HiddenPottsEMSimulator)
function run_EM(sim, MAP, Z1, all_θ₀; parameter_names = nothing, all_nsims = [1, 10, 30], burnin::Integer = 10, kwargs...)
    if isa(parameter_names, String)
        parameter_names = [parameter_names]
    end
    dfs = []
    for (i, θ₀) in enumerate(all_θ₀)
        for nsims in all_nsims
            res = batched_em(withnsims(sim, nsims), MAP, Z1, θ₀; burnin = burnin, tol = 1e-7, niterations = 100, kwargs...)
            push!(dfs, _iterates_df(res.iterates, burnin, parameter_names, i, nsims))
        end
    end
    return vcat(dfs...)
end
