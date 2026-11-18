# Conditional simulator of the hidden Potts model for the neural EM algorithm (batched_em in src/EM.jl). The sampler is
# that of HiddenMarkovRandomFields.jl (https://github.com/msainsburydale/HiddenMarkovRandomFields.jl): each MCMC
# iteration is a Swendsen–Wang sweep that accounts for the data, followed by a chequerboard heat-bath Gibbs sweep, on
# CPU threads or, with a KernelAbstractions backend such as CUDABackend(), on a GPU. This file uses the package's
# persistent-chain interface (PottsLattice, PottsParams, PottsChain, PottsChains, run!, drawZ!), so that the chains
# continue from one EM iteration to the next.

using Random: AbstractRNG, Xoshiro, default_rng
using KernelAbstractions: allocate
using HiddenMarkovRandomFields: simulatehiddenpotts, PottsLattice, PottsParams, PottsChain, PottsChains
# The package's run! and drawZ! are called with qualified names: Main.drawZ! is part of the simulator interface of
# batched_em. (NeuralEstimators also exports a simulatepotts, so that name is not imported either.)
import HiddenMarkovRandomFields as HMRF

"""
	HiddenPottsEMSimulator(emissions; nsims = 30, sweeps_init = 50, sweeps_warm = 5, sw_every = 1, backend = nothing)
Conditional simulator of the hidden Potts model for the neural EM algorithm (see `batched_em` in `src/EM.jl`).
`emissions(θ)` must return `(β, distributions)`.

The chains persist across EM iterations: in the first iteration they are initialised and run for `sweeps_init`
iterations, and in later iterations they continue from their previous state for `sweeps_warm` iterations under the
current parameters (persistent-chain Monte Carlo EM). Each iteration is a Swendsen–Wang sweep (every `sw_every`
iterations; 0: none) followed by a chequerboard Gibbs sweep.

With `backend = nothing`, each data set has `nsims` chains on the CPU, which `batched_em` updates in parallel over
threads (over data sets and chains). With a KernelAbstractions backend, such as `CUDABackend()`, the `nsims` chains of
each data set run together on that device, and the missing values are drawn into an array on the device, which is
passed to the network as it is. Within the EM algorithm, the only transfers between the CPU and the device are then
the parameter estimates (to the CPU) and, for each new estimate, the likelihood weights of the observed data (to the
device), which `PottsParams` computes on the CPU.
"""
struct HiddenPottsEMSimulator{F, B}
    emissions::F
    nsims::Int
    sweeps_init::Int
    sweeps_warm::Int
    sw_every::Int
    backend::B
end
function HiddenPottsEMSimulator(emissions; nsims::Integer = 30, sweeps_init::Integer = 50, sweeps_warm::Integer = 5, sw_every::Integer = 1, backend = nothing)
    @assert nsims > 0 && sweeps_init >= 0 && sweeps_warm >= 0 && sw_every >= 0
    return HiddenPottsEMSimulator(emissions, Int(nsims), Int(sweeps_init), Int(sweeps_warm), Int(sw_every), backend)
end

# Same simulator with a different number of chains
withnsims(sim::HiddenPottsEMSimulator, nsims::Integer) = HiddenPottsEMSimulator(sim.emissions; nsims = nsims, sweeps_init = sim.sweeps_init, sweeps_warm = sim.sweeps_warm, sw_every = sim.sw_every, backend = sim.backend)

# Whether the chains run on a device, which batched_em then updates one data set at a time (see src/EM.jl)
ondevice(sim::HiddenPottsEMSimulator) = !isnothing(sim.backend)

# State of the chains of one data set on the CPU
mutable struct HiddenPottsEMState{T}
    lattice::PottsLattice
    chains::Vector{PottsChain}
    zfill::Matrix{T}                    # the data, with zeros at the missing pixels
    complete::Bool
    params::Union{Nothing, PottsParams} # the parameter-dependent quantities of the current EM iterate
    nupdates::Int                       # number of EM iterations so far (set by setparams!)
end

# State of the chains of one data set on a device. The chains, and the buffer into which the missing values are drawn,
# are created at the first update, on the task that runs the EM iterations, so that all the work on the device is done
# by one task (and so on one CUDA stream); init_state, which batched_em runs in parallel over data sets, only draws
# their seed.
mutable struct HiddenPottsDeviceEMState{T, B}
    lattice::PottsLattice
    backend::B
    seed::UInt64
    chains::Union{Nothing, PottsChains}
    zbuf::Any                           # nrow × ncol × nsims array on the device: the observed values, and the drawn missing values
    zfill::Matrix{T}
    complete::Bool
    params::Union{Nothing, PottsParams}
    nupdates::Int
end

const _HiddenPottsState = Union{HiddenPottsEMState, HiddenPottsDeviceEMState}

# View a (possibly 4D, with singleton trailing dimensions) incomplete data set as a matrix
function _asmatrix(Z::AbstractArray)
    ndims(Z) == 2 && return Z
    @assert all(size(Z)[3:end] .== 1) "each data set must contain a single replicate"
    return reshape(Z, size(Z, 1), size(Z, 2))
end

_scalarβ(β) = (@assert length(β) == 1 "β must be a scalar"; β[1])

"""
	init_state(sim::HiddenPottsEMSimulator, Z₁, θ₀; rng = Random.default_rng())
Persistent EM state for the incomplete data set `Z₁`, whose chains are seeded from `rng`. The chains are initialised at
their first update, under the parameters of the first EM iteration.
"""
function init_state(sim::HiddenPottsEMSimulator, Z₁::AbstractArray{Union{Missing, T}}, θ₀; rng::AbstractRNG = default_rng()) where {T}
    Zm = _asmatrix(Z₁)
    _, distributions = sim.emissions(θ₀)
    lat = PottsLattice(Zm, distributions) # observed values equal to the atom of a Dirac emission distribution are fixed
    zfill = map(z -> ismissing(z) ? zero(T) : z, Zm)
    complete = !any(ismissing, Zm)
    if ondevice(sim)
        return HiddenPottsDeviceEMState{T, typeof(sim.backend)}(lat, sim.backend, rand(rng, UInt64), nothing, nothing, zfill, complete, nothing, 0)
    end
    chains = [PottsChain(lat, Xoshiro(rand(rng, UInt64))) for _ = 1:sim.nsims]
    return HiddenPottsEMState{T}(lat, chains, zfill, complete, nothing, 0)
end

# Parameter-dependent quantities for the current EM iterate (the likelihood weights of the observed data, computed on
# the CPU), once per EM iteration
function setparams!(state::_HiddenPottsState, sim::HiddenPottsEMSimulator, θ)
    β, distributions = sim.emissions(θ)
    state.params = PottsParams(state.lattice, _scalarβ(β), distributions)
    state.nupdates += 1
    return state
end

# Iterations to run in the current EM iteration: the first initialises the chains and runs sweeps_init iterations
_nsweeps(sim::HiddenPottsEMSimulator, state) = state.nupdates <= 1 ? sim.sweeps_init : sim.sweeps_warm

# ---- Chains on the CPU ----

# Advance chain c (the first call initialises it)
function update_chain!(state::HiddenPottsEMState, sim::HiddenPottsEMSimulator, c::Integer, _)
    HMRF.run!(state.chains[c], state.lattice, state.params::PottsParams, _nsweeps(sim, state); sw_every = sim.sw_every)
    return state
end

function update_chains!(state::HiddenPottsEMState, sim::HiddenPottsEMSimulator)
    for c in eachindex(state.chains)
        update_chain!(state, sim, c, nothing)
    end
    return state
end

"""
	drawZ!(zbuf, state::HiddenPottsEMState)
Completes the data with every chain of `state`: `zbuf` is an `nrow`×`ncol`×1×`nsims` array.
"""
function drawZ!(zbuf::AbstractArray{<:Any, 4}, state::HiddenPottsEMState)
    par = state.params::PottsParams
    for c in eachindex(state.chains)
        Zv = view(zbuf, :, :, 1, c)
        copyto!(Zv, state.zfill)
        HMRF.drawZ!(Zv, state.chains[c], state.lattice, par)
    end
    return zbuf
end

# ---- Chains on a device ----

# Advance all the chains of the data set together. The first call creates the chains and the buffer of completed data,
# into which the observed values are copied once (the chains are then initialised by run!).
function update_chains!(state::HiddenPottsDeviceEMState, sim::HiddenPottsEMSimulator)
    if isnothing(state.chains)
        state.chains = PottsChains(state.lattice, sim.nsims, state.backend; rng = Xoshiro(state.seed))
        zbuf = allocate(state.backend, eltype(state.zfill), (size(state.zfill)..., sim.nsims))
        copyto!(zbuf, repeat(state.zfill, 1, 1, sim.nsims))
        state.zbuf = zbuf
    end
    HMRF.run!(state.chains, state.lattice, state.params::PottsParams, _nsweeps(sim, state); sw_every = sim.sw_every)
    return state
end

"""
	drawZ!(state::HiddenPottsDeviceEMState)
Draws the missing values of every chain of `state` on the device, and returns the completed data there, as an
`nrow`×`ncol`×1×`nsims` array that shares the memory of the state's buffer (and so is overwritten by the next call).
"""
function drawZ!(state::HiddenPottsDeviceEMState)
    zbuf = state.zbuf
    HMRF.drawZ!(zbuf, state.chains, state.lattice, state.params::PottsParams)
    return reshape(zbuf, size(zbuf, 1), size(zbuf, 2), 1, size(zbuf, 3))
end

# ---- Unconditional simulation and missingness patterns (shared by the hidden Potts and sea-ice scripts) ----

"""
	simulate_fields(nrow, ncol, β, distributions; num_iterations = 200, rng = Random.default_rng(), backend = nothing, batchsize = 5000)
Unconditional simulation of `length(β)` fields of the hidden Potts model on an `nrow`×`ncol` grid, field k with inverse
temperature `β[k]` and emission distributions `distributions[k]` (e.g., the training data of a neural estimator).
Returns the data as an `nrow`×`ncol`×`length(β)` `Float32` array on the CPU.

Each field is the state of its own chain after `num_iterations` MCMC iterations (a Swendsen–Wang sweep and a
chequerboard Gibbs sweep each). With a KernelAbstractions `backend`, such as `CUDABackend()`, the chains run on that
device, in batches of `batchsize` fields, which bounds the memory they take there; otherwise they run on CPU threads.
The fields are reproducible given `rng` (the CPU and a device give different fields from the same `rng`).
"""
function simulate_fields(nrow::Integer, ncol::Integer, β::AbstractVector, distributions::AbstractVector;
                         num_iterations::Integer = 200, rng::AbstractRNG = default_rng(), backend = nothing, batchsize::Integer = 5000)
    K = length(β)
    length(distributions) == K || throw(ArgumentError("give one vector of emission distributions per field"))
    Z = Array{Float32}(undef, nrow, ncol, K)
    for batch in Iterators.partition(1:K, batchsize)
        _, Zb = simulatehiddenpotts(nrow, ncol, β[batch], distributions[batch]; nchains = length(batch),
                                    num_iterations = num_iterations, rng = rng, backend = backend)
        Z[:, :, batch] .= Array(Zb)[:, :, 1, :]
    end
    return Z
end

# The fields of an nrow × ncol × K array as K data sets of size nrow × ncol × 1 × 1 (one replicate each)
as_datasets(Z::AbstractArray{T, 3}) where {T} = [reshape(Z[:, :, k], size(Z, 1), size(Z, 2), 1, 1) for k in axes(Z, 3)]

# Missing-data pattern of the MICB scenario: a quarter circle in the top-right corner and an ellipse in the bottom-left
function remove_complex_missingness(Z)
    n_rows, n_cols = size(Z)[1:2]

    # Midpoints
    row_mid = n_rows ÷ 2
    col_mid = n_cols ÷ 2

    # Quarter-circle radius
    radius = min(row_mid, n_cols - col_mid)

    # Storage for missing indices
    indices_to_remove = Set{Tuple{Int,Int}}()

    # --- Quarter circle ---
    for i in 1:row_mid  # Loop through the top half rows
        for j in col_mid:n_cols  # Loop through the right half columns
            # Calculate the distance from the top-right corner
            dist = sqrt((i - 1)^2 + (j - n_cols)^2)

            # If distance is less than or equal to the radius, mark for removal
            if dist <= radius
                push!(indices_to_remove, (i, j))  
            end
        end
    end

    # --- Ellipse-shaped missing region in bottom-left ---
    for i in row_mid+1:n_rows
        for j in 1:col_mid
            # Ellipse centered at bottom-left
            di = (i - n_rows)^2 / (0.3n_rows)^2
            dj = (j - 1)^2 / (0.5n_cols)^2
            if di + dj <= 1.0
                push!(indices_to_remove, (i, j))
            end
        end
    end

    # --- Apply missingness ---
    Z₁ = Array{Union{Missing, eltype(Z)}}(Z)
    for (i, j) in indices_to_remove
        Z₁[i, j, :, :] .= missing
    end

    return Z₁
end
