# Gaussian process model of Section 3.2 in Julia: prior, marginal simulation, conditional simulation and MAP
# estimation with the Vecchia approximation of Vecchia.jl. This mirrors src/GP/Simulation.R (which uses GpGp):
#   - the covariance function is GpGp's "matern_isotropic" with unit variance, smoothness ν = 1 and nugget τ²;
#   - locations are ordered by GpGp's (approximate, randomised) maxmin algorithm, and each location conditions on its
#     30 nearest previous locations;
#   - for conditional simulation, the observed locations are ordered first and the unobserved locations last.
# Given the same orderings and neighbour sets, the inverse Cholesky factor and log-likelihood computed here equal those
# of GpGp (to about 1e-10 and 1e-6).

using Vecchia
using Vecchia: SingletonVecchiaApproximation, ZeroMean, cnllbuf, prepare_conditional!
using StaticArrays: SVector
using Bessels: besselk1
using NearestNeighbors: KDTree, knn
using SparseArrays
using LinearAlgebra
using Random
using Statistics: std
using LBFGSB: lbfgsb

@isdefined(_tforeach) || include(joinpath(@__DIR__, "..", "Threading.jl")) # _tforeach

# ---- Prior ----

const ρ_limits = (0.03, 0.35)   # range parameter
const τ_limits = (0.01, 1.0)    # fine-scale (nugget) standard deviation
const gp_prior_mean = [sum(ρ_limits) / 2, sum(τ_limits) / 2]

gp_sampler(K::Integer; rng::AbstractRNG = Random.default_rng()) = vcat(
    (ρ_limits[1] .+ (ρ_limits[2] - ρ_limits[1]) .* rand(rng, 1, K)),
    (τ_limits[1] .+ (τ_limits[2] - τ_limits[1]) .* rand(rng, 1, K))
)

# ---- Covariance function ----

"""
	matern1(x, y, θ)
GpGp's "matern_isotropic" covariance function with covariance parameters (1, ρ, 1, τ²), that is, unit variance,
range ρ, smoothness ν = 1, and nugget variance τ²: C(h) = (h/ρ) K₁(h/ρ) for h > 0 and C(0) = 1 + τ².
"""
function matern1(x, y, θ)
    ρ, τ = θ[1], θ[2]
    h = sqrt(sum(abs2, x - y))
    iszero(h) && return one(ρ) + τ^2
    t = h / ρ
    return t * besselk1(t)
end

# Locations of the N × N grid, in the order of the elements of an N × N matrix (as expand.grid() in R)
gridlocations(N::Integer) = vec([SVector(x, y) for x in range(1, 0, length = N), y in range(0, 1, length = N)])

# ---- Orderings and neighbour sets ----

# Small random perturbation of the locations, used by GpGp to break ties at random
function _jitter(locs::Vector{SVector{2, Float64}}, rng::AbstractRNG)
    ee = minimum(std(getindex.(locs, i)) for i in 1:2)
    return [s + ee * 1e-4 * SVector(randn(rng), randn(rng)) for s in locs]
end

"""
	maxmin_order(locs; rng)
Approximate maxmin ordering of the locations: a translation of GpGp::order_maxmin() (for Euclidean locations).
"""
function maxmin_order(locs::Vector{SVector{2, Float64}}; rng::AbstractRNG = Random.default_rng())
    n = length(locs)
    n == 1 && return [1]
    locs = _jitter(locs, rng)
    m = round(Int, sqrt(n))
    idxs, _ = knn(KDTree(locs), locs, m + 1, true) # sorted by distance; the first neighbour is the point itself
    NNall = [i[2:end] for i in idxs]
    index_in_position = zeros(Int, 2n) # 0 denotes an empty position (NA in GpGp)
    index_in_position[1:n] = randperm(rng, n)
    position_of_index = Vector{Int}(undef, n)
    position_of_index[index_in_position[1:n]] = 1:n
    curlen = n
    nmoved = 0
    for j = 2:(2n)
        index = index_in_position[j]
        index == 0 && continue # (as in GpGp, where the minimum below is then over no neighbours)
        nneigh = round(Int, min(m, n / (j - nmoved + 1)))
        neighbors = view(NNall[index], 1:nneigh)
        if minimum(position_of_index[neighbors]) < j
            nmoved += 1
            curlen += 1
            position_of_index[index] = curlen
            curlen > length(index_in_position) && push!(index_in_position, 0) # (R extends the vector implicitly)
            index_in_position[curlen] = index
            index_in_position[j] = 0
        end
    end
    return filter(!=(0), index_in_position)
end

"""
	ordered_neighbours(locs, m; rng)
For each location j of the ordered locations `locs`, the indices of its (at most) `m` nearest neighbours among
locations 1,…,j−1, nearest first (as columns 2,…,m+1 of GpGp::find_ordered_nn(), with ties broken at random).
"""
function ordered_neighbours(locs::Vector{SVector{2, Float64}}, m::Integer; rng::AbstractRNG = Random.default_rng())
    locs = _jitter(locs, rng)
    n = length(locs)
    condix = Vector{Vector{Int}}(undef, n)
    d = Vector{Float64}(undef, n)
    for j = 1:n
        k = min(m, j - 1)
        for i = 1:(j - 1)
            d[i] = sum(abs2, locs[i] - locs[j])
        end
        condix[j] = k == 0 ? Int[] : partialsortperm(view(d, 1:(j - 1)), 1:k)
    end
    return condix
end

# Vecchia approximation for the ordered locations `pts` with conditioning sets `condix` (no permutation is applied by
# Vecchia.jl). `chunks` splits the locations into blocks processed in parallel by Vecchia.nll().
function vecchia_approximation(pts, condix, data = nothing; chunks = [collect(eachindex(pts))])
    data = isnothing(data) ? nothing : reshape(Float64.(data), :, 1)
    SingletonVecchiaApproximation(ZeroMean(), matern1, data, pts, condix, collect(eachindex(pts)), chunks)
end

# ---- Marginal simulation ----

# Upper-triangular sparse factor U with Σ⁻¹ ≈ UU' (Vecchia.jl's "reverse Cholesky" factor), for the ordering of V
rcholU(V, θ) = rchol(V, Float64[θ[1], θ[2]]).U

"""
	simulate_gp(θ, N = 64; num_neighbours = 30, rng)
Simulates a Gaussian process on the N × N grid with the Vecchia approximation (as GpGp::fast_Gp_sim()), returned as an
N × N × 1 × 1 array.
"""
function simulate_gp(θ, N::Integer = 64; num_neighbours::Integer = 30, rng::AbstractRNG = Random.default_rng())
    locs = gridlocations(N)
    ord = maxmin_order(locs; rng = rng)
    pts = locs[ord]
    U = rcholU(vecchia_approximation(pts, ordered_neighbours(pts, num_neighbours; rng = rng)), θ)
    z = U' \ randn(rng, length(pts)) # Cov(z) = (UU')⁻¹ ≈ Σ
    Z = Array{Float64}(undef, N, N, 1, 1)
    Z[ord] = z
    return Z
end

# ---- Conditional simulation ----

"""
	GPConditionalSetup(y; num_neighbours = 30, rng)
Set-up for conditional simulation given the incomplete data `y` (done once per data set, regardless of θ): the
observed locations (maxmin ordering) followed by the unobserved locations (maxmin ordering), with their neighbour sets.
"""
struct GPConditionalSetup
    V::SingletonVecchiaApproximation   # joint Vecchia approximation (observed locations first)
    n1::Int                             # number of observed locations
    n2::Int                             # number of unobserved locations
    y1::Vector{Float64}                 # observed data, in the ordering of V
    pred_indices::Vector{Int}           # linear index (in y) of each unobserved location, in the ordering of V
    # Sparsity pattern of U[:, (n1+1):end], split into the rows of the observed (U12) and unobserved (U22) locations
    U12_colptr::Vector{Int}; U12_rowval::Vector{Int}
    U22_colptr::Vector{Int}; U22_rowval::Vector{Int}
end

function GPConditionalSetup(y::AbstractMatrix; num_neighbours::Integer = 30, rng::AbstractRNG = Random.default_rng())
    N = size(y, 1)
    locs = gridlocations(N)
    obs = findall(!ismissing, vec(y))
    mis = findall(ismissing, vec(y))
    ord1 = maxmin_order(locs[obs]; rng = rng)
    ord2 = maxmin_order(locs[mis]; rng = rng)
    pts = vcat(locs[obs][ord1], locs[mis][ord2])
    condix = ordered_neighbours(pts, num_neighbours; rng = rng)
    y1 = Float64[y[i] for i in obs[ord1]]
    return GPConditionalSetup(pts, condix, y1, mis[ord2])
end

# Set-up from given ordered locations `pts` (observed first), conditioning sets `condix`, observed data `y1` (in the
# same order) and the linear indices `pred_indices` of the unobserved locations
function GPConditionalSetup(pts, condix, y1, pred_indices)
    n1 = length(y1)
    n2 = length(pts) - n1

    # Column j of U (j > n1) has nonzeros in rows condix[j] and j
    U12_colptr = [1]; U12_rowval = Int[]
    U22_colptr = [1]; U22_rowval = Int[]
    for j = (n1 + 1):(n1 + n2)
        rows = sort(condix[j])
        append!(U12_rowval, filter(<=(n1), rows))
        append!(U22_rowval, [filter(>(n1), rows) .- n1; j - n1])
        push!(U12_colptr, length(U12_rowval) + 1)
        push!(U22_colptr, length(U22_rowval) + 1)
    end

    V = vecchia_approximation(pts, condix)
    return GPConditionalSetup(V, n1, n2, Float64.(y1), collect(Int, pred_indices), U12_colptr, U12_rowval, U22_colptr, U22_rowval)
end

"""
	conditional_factors(setup, θ)
The blocks U12 (n1 × n2) and U22 (n2 × n2, upper triangular) of the columns of U (Σ⁻¹ ≈ UU') that belong to the
unobserved locations. Only these columns are computed (cf. GpGp's vecchia_Linv(..., start_ind)). With this
factorisation, Z₂ | Z₁ = z₁ ~ N(−U22⁻ᵀU12ᵀz₁, U22⁻ᵀU22⁻¹).
"""
function conditional_factors(s::GPConditionalSetup, θ)
    V = s.V
    params = Float64[θ[1], θ[2]]
    buf = cnllbuf(V, params)
    U12 = SparseMatrixCSC(s.n1, s.n2, s.U12_colptr, s.U12_rowval, Vector{Float64}(undef, length(s.U12_rowval)))
    U22 = SparseMatrixCSC(s.n2, s.n2, s.U22_colptr, s.U22_rowval, Vector{Float64}(undef, length(s.U22_rowval)))
    for jj = 1:s.n2
        j = s.n1 + jj
        idxs = V.condix[j]
        cvar = prepare_conditional!(buf, V, j, params)  # conditional variance; kriging weights in buf.buf_kwts
        isqrt_cvar = inv(sqrt(cvar))
        # Vecchia.jl's rchol(): U[idxs[k], j] = −kwts[k]/√cvar and U[j, j] = 1/√cvar
        for (k, i) in enumerate(idxs)
            w = -buf.buf_kwts[k] * isqrt_cvar
            if i <= s.n1
                r = searchsortedfirst(view(s.U12_rowval, s.U12_colptr[jj]:(s.U12_colptr[jj + 1] - 1)), i)
                U12.nzval[s.U12_colptr[jj] + r - 1] = w
            else
                r = searchsortedfirst(view(s.U22_rowval, s.U22_colptr[jj]:(s.U22_colptr[jj + 1] - 1)), i - s.n1)
                U22.nzval[s.U22_colptr[jj] + r - 1] = w
            end
        end
        U22.nzval[s.U22_colptr[jj + 1] - 1] = isqrt_cvar # the diagonal is the last element of the column
    end
    return U12, UpperTriangular(U22)
end

"""
	simulateconditional(setup, θ, nsims; rng)
`nsims` conditional simulations of the unobserved data (an n2 × nsims matrix, in the ordering of `setup`).
"""
function simulateconditional(s::GPConditionalSetup, θ, nsims::Integer; rng::AbstractRNG = Random.default_rng())
    U12, U22 = conditional_factors(s, θ)
    rhs = randn(rng, s.n2, nsims) .- (U12' * s.y1)
    return U22' \ rhs
end

# ---- Conditional simulator for batched_em (src/EM.jl) ----

"""
	GPEMSimulator(; nsims = 30, num_neighbours = 30)
Conditional simulator of the Gaussian process model for `batched_em`. Conditional simulation is exact (given the
Vecchia approximation), so each "chain" is simply an independent draw, made in parallel over draws and data sets.
"""
struct GPEMSimulator
    nsims::Int
    num_neighbours::Int
end
GPEMSimulator(; nsims::Integer = 30, num_neighbours::Integer = 30) = GPEMSimulator(nsims, num_neighbours)
withnsims(sim::GPEMSimulator, nsims::Integer) = GPEMSimulator(nsims, sim.num_neighbours)

struct GPParams
    U22::UpperTriangular{Float64, SparseMatrixCSC{Float64, Int}}
    b::Vector{Float64}       # −U12ᵀz₁
    draws::Matrix{Float64}   # n2 × nsims
end

mutable struct GPEMState
    setup::Union{Nothing, GPConditionalSetup}
    zfill::Matrix{Float32}   # data with the missing values set to zero
    complete::Bool
    params::Union{Nothing, GPParams}
    rngs::Vector{Xoshiro}    # one per draw, so that draws can be made in parallel
end

function _asmatrix(Z::AbstractArray)
    ndims(Z) == 2 && return Z
    @assert all(size(Z)[3:end] .== 1) "each data set must contain a single replicate"
    return reshape(Z, size(Z, 1), size(Z, 2))
end

function init_state(sim::GPEMSimulator, Z₁::AbstractArray, θ₀; rng::AbstractRNG = Random.default_rng())
    y = _asmatrix(Z₁)
    zfill = Float32[ismissing(z) ? 0f0 : z for z in y]
    complete = !any(ismissing, y)
    setup = complete ? nothing : GPConditionalSetup(y; num_neighbours = sim.num_neighbours, rng = rng)
    rngs = [Xoshiro(rand(rng, UInt64)) for _ = 1:sim.nsims]
    return GPEMState(setup, zfill, complete, nothing, rngs)
end

function setparams!(state::GPEMState, sim::GPEMSimulator, θ)
    s = state.setup
    U12, U22 = conditional_factors(s, θ)
    state.params = GPParams(U22, -(U12' * s.y1), Matrix{Float64}(undef, s.n2, sim.nsims))
    return state
end

function update_chain!(state::GPEMState, sim::GPEMSimulator, c::Integer, ws)
    p = state.params::GPParams
    x = view(p.draws, :, c)
    randn!(state.rngs[c], x)
    x .+= p.b
    ldiv!(p.U22', x)
    return nothing
end

function drawZ!(zbuf::AbstractArray{<:Any, 4}, state::GPEMState)
    p = state.params::GPParams
    idx = state.setup.pred_indices
    for c in axes(zbuf, 4)
        Zv = view(zbuf, :, :, 1, c)
        copyto!(Zv, state.zfill)
        for (i, k) in enumerate(idx)
            Zv[k] = p.draws[i, c]
        end
    end
    return zbuf
end

# ---- MAP estimation ----

# Vecchia approximation of the observed data (maxmin ordering, nearest previous neighbours)
function vecchia_observed(y::AbstractArray; num_neighbours::Integer = 30, rng::AbstractRNG = Random.default_rng(), chunks = nothing)
    y = _asmatrix(y)
    locs = gridlocations(size(y, 1))
    obs = findall(!ismissing, vec(y))
    ord = maxmin_order(locs[obs]; rng = rng)
    pts = locs[obs][ord]
    condix = ordered_neighbours(pts, num_neighbours; rng = rng)
    chunks = isnothing(chunks) ? [collect(eachindex(pts))] : chunks
    vecchia_approximation(pts, condix, Float64[y[i] for i in obs[ord]]; chunks = chunks)
end

# Negative log-likelihood (without the constant n log(2π)/2) as a function of (log ρ, log τ)
gp_nll(V, logθ) = Vecchia.nll(V, exp.(logθ))

# Central finite-difference gradient with step 1e-3, truncated at the bounds, as used by R's optim() with
# method = "L-BFGS-B" (ndeps = 1e-3)
function _fdgradient!(g, f, x, lb, ub; h = 1e-3)
    for i in eachindex(x)
        xi = x[i]
        xp = min(xi + h, ub[i]); x[i] = xp; fp = f(x)
        xm = max(xi - h, lb[i]); x[i] = xm; fm = f(x)
        x[i] = xi
        g[i] = (fp - fm) / (xp - xm)
    end
    return g
end

"""
	gp_MAP(y; threaded = false, rng)
MAP estimate of θ = (ρ, τ) from the incomplete data `y`: the prior is uniform, so this maximises the Vecchia
likelihood over the prior support, with L-BFGS-B on the log scale from the prior mean (as in src/GP/Simulation.R).
If `threaded`, the likelihood is evaluated in parallel over locations.
"""
function gp_MAP(y; threaded::Bool = false, rng::AbstractRNG = Random.default_rng())
    V = vecchia_observed(y; rng = rng)
    if threaded
        n = length(V.pts)
        V = vecchia_approximation(V.pts, V.condix, vec(V.data); chunks = [collect(c) for c in Iterators.partition(1:n, cld(n, Threads.nthreads()))])
    end
    return gp_MAP(V)
end

# MAP estimate given the Vecchia approximation V of the observed data
function gp_MAP(V::SingletonVecchiaApproximation)
    f = logθ -> gp_nll(V, logθ)
    lb = log.([ρ_limits[1], τ_limits[1]])
    ub = log.([ρ_limits[2], τ_limits[2]])
    g! = (g, logθ) -> _fdgradient!(g, f, copy(logθ), lb, ub)
    _, logθ = lbfgsb(f, g!, log.(gp_prior_mean); lb = lb, ub = ub, m = 5, factr = 1e7, pgtol = 0.0, maxiter = 100)
    return exp.(logθ)
end

# MAP estimates for many data sets, in parallel over data sets
function gp_MAP(Z::AbstractVector{<:AbstractArray}; rng::AbstractRNG = Random.default_rng())
    seeds = rand(rng, UInt64, length(Z))
    θ̂ = Matrix{Float64}(undef, 2, length(Z))
    _tforeach(length(Z)) do k, _
        θ̂[:, k] = gp_MAP(Z[k]; rng = Xoshiro(seeds[k]))
    end
    return θ̂
end
