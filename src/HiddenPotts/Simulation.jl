# NB If I add this to NeuralEstimators.jl, I think it would make sense to only include simulatehiddenpotts() for the incomplete data case... marginal simulation of complete fields can be done using other more efficient libraries (using the Swensen and Wang algorithm), and it is only the hidden version of the Potts model that is of interest for neural inference (the regular version has sufficient statistics).

using Random, Base.Threads
using Distributions: logpdf, mean, Categorical, Dirac

"""
	simulatepotts(grid::Matrix{Int}, β)
	simulatepotts(grid::Matrix{Union{Int, Nothing}}, β)
	simulatepotts(nrows::Int, ncols::Int, num_states::Int, β)
Gibbs sampling from a spatial Potts model with parameter `β`>0 (see, e.g., [Sainsbury-Dale et al., 2025, Sec. 3.3](https://arxiv.org/abs/2501.04330), and the references therein).

Approximately independent simulations can be obtained by setting 
`nsims` > 1 or `num_iterations > burn`. The degree to which the 
resulting simulations can be considered independent depends on the 
thinning factor (`thin`) and the burn-in (`burn`).

# Keyword arguments
- `nsims = 1`: number of approximately independent replicates. 
- `num_iterations = 2000`: number of MCMC iterations.
- `burn = num_iterations`: burn-in.
- `thin = 10`: thinning factor.

# Examples
```
using NeuralEstimators 

## Marginal simulation 
β = 0.8
simulatepotts(10, 10, 3, β)

## Marginal simulation: approximately independent samples 
simulatepotts(10, 10, 3, β; nsims = 100, thin = 10)

## Conditional simulation 
β = 0.8
complete_grid   = simulatepotts(100, 100, 3, β)      # simulate marginally 
incomplete_grid = removedata(complete_grid, 0.1)     # randomly remove 10% of the pixels 
imputed_grid    = simulatepotts(incomplete_grid, β)  # conditionally simulate over missing pixels

## Multiple conditional simulations 
imputed_grids   = simulatepotts(incomplete_grid, β; num_iterations = 2000, burn = 1000, thin = 10)

## Recreate Fig. 8.8 of Marin & Robert (2007) “Bayesian Core”
using Plots 
grids = [simulatepotts(100, 100, 2, β) for β ∈ 0.3:0.1:1.2]
heatmaps = heatmap.(grids, legend = false, aspect_ratio = 1)
Plots.plot(heatmaps...)
```
"""
function simulatepotts(
    grid::AbstractMatrix{I},
    β;
    num_iterations::Integer = 1000,
    mask = nothing,
    rng::AbstractRNG = Random.default_rng()
) where {I<:Integer}

    β = β[1]  # unwrap if β was passed as a container

    nrows, ncols = size(grid)
    states = unique(skipmissing(grid))
    num_states = length(states)

    # Map states to 1:num_states for speed
    state_to_idx = Dict(s => i for (i, s) in enumerate(states))
    idx_to_state = collect(states)

    Y = map(s -> state_to_idx[s], grid)  # convert grid to index form

    # Precompute chequerboard patterns
    chequerboard1 = [(i+j) % 2 == 0 for i in 1:nrows, j in 1:ncols]
    chequerboard2 = .!chequerboard1
    
    if !isnothing(mask)
        @assert size(grid) == size(mask)
        chequerboard1 = chequerboard1 .&& mask
        chequerboard2 = chequerboard2 .&& mask
    end
    
    chequerboards = if sum(chequerboard1) == 0
        (chequerboard2,)
    elseif sum(chequerboard2) == 0
        (chequerboard1,)
    else 
        (chequerboard1, chequerboard2)
    end

    # Precompute neighbor offsets as CartesianIndex
    neighbor_offsets = CartesianIndex.([(0,1), (1,0), (0,-1), (-1,0)])

    # Preallocated buffers
    neighbour_counts = zeros(Int, num_states)
    probs = zeros(Float64, num_states)
    cum_probs = zeros(Float64, num_states)

    @inbounds for _ in 1:num_iterations
        for chequerboard in chequerboards
            for ci in findall(chequerboard)

                # Reset and count neighbors
                fill!(neighbour_counts, 0)
                for offset in neighbor_offsets
                    ni = ci + offset
                    checkbounds(Bool, Y, ni) || continue
                    neighbour_counts[Y[ni]] += 1
                end
                
                # If all counts are zero → uniform draw
                if maximum(neighbour_counts) == 0
                    Y[ci] = rand(rng, 1:num_states)
                    continue
                end

                # Calculate probabilities (log-sum-exp trick)
                for s in 1:num_states
                    probs[s] = β * neighbour_counts[s]
                end
                max_log_prob = maximum(probs)
                @. probs = exp(probs - max_log_prob)
                ssum = sum(probs)
                @. probs = probs / ssum

                # Sample new state
                u = rand(rng)
                cumsum!(cum_probs, probs)
                new_state_idx = searchsortedfirst(cum_probs, u)
                Y[ci] = new_state_idx
            end
        end
    end

    # Map back to original states
    return map(i -> idx_to_state[i], Y)
end

function simulatepotts(nrows::Integer, ncols::Integer, num_states::Integer, β; rng::AbstractRNG = Random.default_rng(), kwargs...)
    @assert length(β) == 1
    β = β[1]
    grid = initializepotts(nrows, ncols, num_states, β; rng = rng)
    simulatepotts(grid, β; rng = rng, kwargs...)
end

function simulatepotts(grid::AbstractMatrix{Union{Missing, I}}, β; kwargs...) where {I <: Integer}
    @assert length(β) == 1
    β = β[1]
    mask = ismissing.(grid)
    grid = initializepotts(grid, mask, β)
    simulatepotts(grid, β; kwargs..., mask = mask)
end

function initializepotts(grid::AbstractMatrix{I}, mask, β) where {I <: Integer}
    return grid
end

function initializepotts(grid::AbstractMatrix{Union{Missing, I}}, mask, β) where {I <: Integer}

    # Avoid mutating input 
    mask = copy(mask)

    # Early return if no missing values
    sum_mask = sum(mask)
    sum_mask == 0 && return convert(Matrix{I}, grid)

    # Find the number of states and compute the critical inverse temperature
    states = unique(skipmissing(grid))
    num_states = length(states)
    β_crit = log(1 + sqrt(num_states))

    # choose a fraction of missing sites to seed randomly
    frac_seed = 0.01
    n_seed = ceil(Int, frac_seed * sum(mask))
    seed_idx = rand(findall(mask), n_seed)
    grid[seed_idx] .= rand(1:num_states, n_seed)
    mask[seed_idx] .= false

    if β < β_crit 
        # High temperature: 
        neigh_offsets = CartesianIndex.([(-1,0), (1,0), (0,-1), (0,1)])
        neigh_buf = Vector{Int}(undef, 4)   # neighbor labels
        counts = zeros(Int, num_states)

        # Iterate a few rounds to propagate information
        iterations = 0
        max_iterations = sum_mask * 3

        while any(mask) && iterations < max_iterations
            iterations += 1
            for idx in findall(mask)
                count = 0
                for offset in neigh_offsets
                    ni = idx + offset
                    checkbounds(Bool, grid, ni) || continue
                    @inbounds !mask[ni] || continue
                    @inbounds neigh_buf[count += 1] = grid[ni]
                end

                if count > 0
                    # Count neighbors
                    fill!(counts, 0)
                    @inbounds for j in 1:count
                        s = neigh_buf[j]
                        counts[s] += 1
                    end

                    # --- probabilistic update ---
                    weights = exp.(β .* counts[1:num_states])
                    probs = weights ./ sum(weights)

                    # sample a state according to probs
                    chosen = rand(Categorical(probs))
                    grid[idx] = chosen
                    mask[idx] = false
                end
            end
        end

        # Fallback: if anything remains unassigned, fill uniformly
        if any(mask)
            @inbounds grid[mask] .= rand(1:num_states, sum(mask))
        end
        
    else
        # Low temperature: iterative neighbor-based fill

        # Pre-allocate neighbor arrays and offsets
        neigh_offsets = CartesianIndex.([(-1,0), (1,0), (0,-1), (0,1)])
        neigh_buf = Vector{Int}(undef, 4)  # max 4 neighbors
        counts = zeros(Int, num_states)
        
        changed = true
        iterations = 0
        max_iterations = sum_mask * 2  # safety net
        
        while changed && any(mask) && iterations < max_iterations
            changed = false
            iterations += 1
            
            # Process all missing cells in each iteration
            for idx in findall(mask)
                count = 0
                # Check neighbors
                for offset in neigh_offsets
                    ni = idx + offset
                    checkbounds(Bool, grid, ni) || continue
                    @inbounds !mask[ni] || continue
                    @inbounds neigh_buf[count+=1] = grid[ni]
                end
                
                if count > 0
                    # Count neighbor states
                    fill!(counts, 0)
                    @inbounds for i in 1:count
                        s = neigh_buf[i]
                        counts[s] += 1
                    end
                    
                    # Find modes
                    maxcount = maximum(@view counts[1:num_states])
                    n_modes = 0
                    @inbounds for s in 1:num_states
                        if counts[s] == maxcount
                            neigh_buf[n_modes+=1] = s
                        end
                    end
                    
                    # Random tie-break
                    @inbounds grid[idx] = neigh_buf[rand(1:n_modes)]
                    mask[idx] = false
                    changed = true
                end
            end
        end
        
        # Fill any remaining missing values randomly
        remaining = sum(mask)
        if remaining > 0
            @inbounds grid[mask] .= rand(1:num_states, remaining)
        end
    end
    
    return convert(Matrix{I}, grid)
end

function initializepotts(nrows::Integer, ncols::Integer, num_states::Integer, β; rng::AbstractRNG = Random.default_rng())

    β_crit = log(1 + sqrt(num_states))
  
    if β < β_crit
        # Random initialization for high temperature
        grid = rand(rng, 1:num_states, nrows, ncols)
    else
        # Clustered initialization for low temperature
        cluster_size = max(1, min(nrows, ncols) ÷ 4)
        clustered_rows = ceil(Int, nrows / cluster_size)
        clustered_cols = ceil(Int, ncols / cluster_size)
        base_grid = rand(rng, 1:num_states, clustered_rows, clustered_cols)
        grid = repeat(base_grid, inner = (cluster_size, cluster_size))
        grid = grid[1:nrows, 1:ncols] # trim to exact dimensions

        # Add small random perturbations
        for i in 1:nrows, j in 1:ncols
            if rand(rng) < 0.05
                grid[i, j] = rand(1:num_states)
            end
        end
    end

    return grid
end


# ---- Hidden Potts ----

"""
	simulatehiddenpotts(Z::AbstractMatrix{Union{Missing, F}}, β, distributions; num_iterations::Integer = 100) where {F <: AbstractFloat}
	simulatehiddenpotts(nrows::Integer, ncols::Integer, β, distributions; kwargs...)
Gibbs sampling from a spatial hidden Potts model with inverse-temperature parameter `β`>0 and user-defined response `distributions`. 

Returns the full MCMC chains of the latent labels Y and the observations Z.

# Examples
```
using NeuralEstimators
using Distributions
using Plots 

## True Potts parameter and example vector of distributions
β = 0.8
distributions = [Normal(0, 1), Normal(2, 1)]

## Marginal simulation 
Y, Z = simulatehiddenpotts(100, 100, β, distributions)

## Conditional simulation 
Z₁  = removedata(Z, 0.2)                         
Y_imputed, Z_imputed = simulatehiddenpotts(Z₁, β, distributions)

## Multiple chains
num_chains = 4

## Visualize some fields
grids = [simulatehiddenpotts(100, 100, β, distributions).Z for β ∈ 0.3:0.1:1.2]
heatmaps = heatmap.(grids, legend = false, aspect_ratio=1)
Plots.plot(heatmaps...)
```
"""
function simulatehiddenpotts(nrows::Integer, ncols::Integer, β, distributions; rng::AbstractRNG = Random.default_rng(), kwargs...)
    
    # Simulate Potts field
    num_states = length(distributions)
    Y = simulatepotts(nrows, ncols, num_states, β; rng = rng, kwargs...)

    # Simulate data 
    Z = broadcast(y -> rand(rng, distributions[y]), Y)

    return (Y = Y, Z = Z)
end

function updateYZ!(
    Y::AbstractMatrix{I},
    Z::AbstractMatrix{F},
    β,
    distributions,
    neighbor_offsets,
    missing_idxs,
    observed_idxs,
    neighbour_counts,
    log_densities,
    probs,
    cum_probs,
    rng::AbstractRNG
) where {I<:Integer, F<:Real}

    num_states = length(distributions)

    # ---- Missing pixels: MH updates ----
    for i in missing_idxs
        @inbounds begin
            current = Y[i]

            # Propose new label (different from current) without allocation
            proposal = rand(rng, 1:(num_states-1))
            proposal += (proposal >= current)

            # Count neighbour agreements
            n_same_current = 0
            n_same_proposal = 0
            for offset in neighbor_offsets
                ni = i + offset
                checkbounds(Bool, Y, ni) || continue
                state = Y[ni]
                n_same_current  += (state == current)
                n_same_proposal += (state == proposal)
            end

            # MH acceptance (missing Z: joint proposal so likelihood cancels)
            ΔS = n_same_proposal - n_same_current
            accept = (ΔS >= 0.0) || (rand(rng) < exp(β * ΔS))
            if accept
                Y[i] = proposal
            end

            # Always update Z according to current (accepted or not) Y
            Z[i] = rand(rng, distributions[Y[i]])
        end
    end

    # ---- Observed pixels: Gibbs updates ----
    for i in observed_idxs
        @inbounds begin
            fill!(neighbour_counts, 0)
            for offset in neighbor_offsets
                ni = i + offset
                checkbounds(Bool, Y, ni) || continue
                state = Y[ni]
                neighbour_counts[state] += 1
            end

            # Observation contribution
            Zᵢ = Z[i]
            for s in 1:num_states
                log_densities[s] = logpdf(distributions[s], Zᵢ)
            end

            for s in 1:num_states
                probs[s] = β * neighbour_counts[s] + log_densities[s]
            end
            max_log = maximum(probs)
            @. probs = exp(probs - max_log)
            ssum = sum(probs)
            @. probs = probs / ssum

            # Sample new state with rng
            u = rand(rng)
            cumsum!(cum_probs, probs)
            Y[i] = searchsortedfirst(cum_probs, u)
        end
    end

    return nothing
end

# Single chain MCMC 
function simulatehiddenpotts(
    Z0::AbstractMatrix{Union{Missing, F}},
    β,
    distributions;
    num_iterations::Integer = 100, 
    rng::AbstractRNG = Random.default_rng()
) where {F <: AbstractFloat}

    @assert length(β) == 1
    β = β[1]
    Z = copy(Z0) # avoid mutating user's data
    num_states = length(distributions)

    # Compute missingness mask
    mask_missing = ismissing.(Z)

    # Check Dirac distributions and build dirac_mask
    is_dirac = [d isa Dirac for d in distributions]
    if any(is_dirac)
        dirac_values = [d.value for d in distributions if d isa Dirac]
        dirac_mask = map(Z) do z
            !ismissing(z) && (z ∈ dirac_values)
        end
    else
        dirac_mask = falses(size(Z))
    end

    # Initialize Y at observed pixels
    Y = map(Z) do z
        if ismissing(z)
            missing
        else 
            argmin(abs.(z .- mean.(distributions)))
             # choose most likely label under emission distributions 
            # likelihoods = pdf.(distributions, z)
            # rand(rng, Categorical(likelihoods ./ sum(likelihoods)))
        end
    end

    # Initialize Y at missing pixels using regular Potts initializer
    Y = initializepotts(Y, mask_missing, β)

    # Initialize missing Z conditional on initialized Y
    @inbounds Z[mask_missing] .= rand.(distributions[Y[mask_missing]])

    # Drop Missing from container
    # Y = convert(Matrix{nonmissingtype(eltype(Y))}, Y)
    Z = convert(Matrix{F}, Z)

    # Preallocation for looping (local buffers per chain)
    neighbor_offsets = CartesianIndex.([(0,1), (1,0), (0,-1), (-1,0)])
    update_idxs   = findall(.!dirac_mask)
    missing_idxs  = update_idxs[mask_missing[update_idxs]]
    observed_idxs = setdiff(update_idxs, missing_idxs)

    neighbour_counts = zeros(Int, num_states)
    log_densities = zeros(Float64, num_states)
    probs = zeros(Float64, num_states)
    cum_probs = zeros(Float64, num_states)

    # Store iterates
    Y_chain = Vector{typeof(Y)}(undef, num_iterations)
    Z_chain = Vector{typeof(Z)}(undef, num_iterations)
    for iter in 1:num_iterations
        updateYZ!(Y, Z, β, distributions, neighbor_offsets, missing_idxs, observed_idxs, neighbour_counts, log_densities, probs, cum_probs, rng)
        Y_chain[iter] = copy(Y)
        Z_chain[iter] = copy(Z)
    end

    return (Y = Y_chain, Z = Z_chain)
end


# ---- Parallel wrappers: run many independent chains in parallel ----

"""
simulatepotts_parallel(Y1, β; num_iterations=100, num_chains=4, seed=1234)

Runs `num_chains` independent chains in parallel using `Threads.@threads`.
"""
function simulatepotts_parallel(
    Y1::AbstractMatrix{Union{Missing, I}},
    β;
    num_iterations::Integer = 100,
    num_chains::Integer = 32,
    seed::Integer = 0
) where {I <: Integer}

    # Preallocate outputs
    nx, ny = size(Y1)
    Y_out = Array{Int}(undef, nx, ny, num_chains)

    # Launch chains in parallel; each thread/chain gets its own RNG
    @threads for c in 1:num_chains
        # create a chain-local RNG (different seed per chain)
        rng = MersenneTwister(UInt32(seed + c + Threads.threadid()))
        res = simulatepotts(Y1, β; num_iterations = num_iterations, rng = rng)
        # copy into preallocated arrays
        @inbounds begin
            Y_out[:, :, c] = res
        end
    end

    return Y_out
end

"""
simulatehiddenpotts_parallel(Z1, β, distributions; num_iterations=100, num_chains=4, seed=1234)

Runs `num_chains` independent chains in parallel using `Threads.@threads`.
Returns a NamedTuple with keys :Y and :Z each containing arrays sized (nx × ny × num_iterations × num_chains). 
"""
function simulatehiddenpotts_parallel(
    Z1::AbstractMatrix{Union{Missing, F}},
    β,
    distributions;
    num_iterations::Integer = 100,
    num_chains::Integer = 32,
    seed::Integer = 0
) where {F <: AbstractFloat}

    # Preallocate outputs
    nx, ny = size(Z1)
    Y_out = Array{Int}(undef, nx, ny, num_iterations, num_chains)
    Z_out = Array{F}(undef, nx, ny, num_iterations, num_chains)

    # Launch chains in parallel; each thread/chain gets its own RNG
    @threads for c in 1:num_chains
        # create a chain-local RNG (different seed per chain)
        rng = MersenneTwister(UInt32(seed + c + Threads.threadid()))
        res = simulatehiddenpotts(Z1, β, distributions; num_iterations = num_iterations, rng = rng)

        # res.Y and res.Z are vectors of matrices length num_iterations
        for iter in 1:num_iterations
            Y_mat = res.Y[iter]
            Z_mat = res.Z[iter]
            # copy into preallocated arrays
            @inbounds begin
                Y_out[:, :, iter, c] = Y_mat
                Z_out[:, :, iter, c] = Z_mat
            end
        end
    end

    return (Y = Y_out, Z = Z_out)
end

function simulatehiddenpotts_parallel(
    nx::Integer, ny::Integer, β, distributions;
    num_chains::Integer = 32,
    seed::Integer = 0,
    kwargs...
)
    # Preallocate outputs
    Y_out = Array{Int}(undef, nx, ny, num_chains)
    Z_out = Array{Float64}(undef, nx, ny, num_chains)

    # Launch chains in parallel
    @threads for c in 1:num_chains
        rng = MersenneTwister(UInt32(seed + c + Threads.threadid()))
        res = simulatehiddenpotts(nx, ny, β, distributions; rng = rng, kwargs...)
        # copy into preallocated arrays
        @inbounds begin
            Y_out[:, :, c] = res.Y
            Z_out[:, :, c] = res.Z
        end
    end

    return (Y = Y_out, Z = Z_out)
end

# using BenchmarkTools, Distributions, NeuralEstimators
# β = 0.8
# distributions = [Normal(0, 1), Normal(2, 1)]
# Y, Z = simulatehiddenpotts(100, 100, β, distributions)
# Z₁  = removedata(Z, 0.2)  
# @belapsed simulatehiddenpotts(Z₁, β, distributions; num_iterations = 1)   # 0.0016 
# @belapsed simulatehiddenpotts(Z₁, β, distributions; num_iterations = 100) # 0.0496
# @belapsed simulatehiddenpotts_parallel(Z₁, β, distributions; num_chains = 100) # 0.22

# ---- Examples ----

# using Distributions
# using Plots

# clean_heatmap(grid; title="", xlabel="") = heatmap(
#     grid, legend=false, aspect_ratio=1, axis=false, border=:none,
#     title=title, xlabel=xlabel
# )
# function remove_quarter_circle(Z)
#     data = Z[:, :]  # Extract the matrix
#     data = Matrix{Union{Missing, eltype(data)}}(data)  # Allow `missing` values in the matrix
#     n_rows, n_cols = size(data)  # Get matrix dimensions

#     # Calculate row and column midpoints (top-right quadrant)
#     row_mid = ceil(Int, n_rows / 2)
#     col_mid = ceil(Int, n_cols / 2)

#     # Determine the radius of the quarter circle (smallest half-dimension)
#     radius = min(row_mid, n_cols - col_mid)

#     # Create Cartesian indices for the top-right quadrant
#     for i in 1:row_mid  # Loop through the top half rows
#         for j in col_mid:n_cols  # Loop through the right half columns
#             # Calculate the distance from the top-right corner
#             dist = sqrt((i - 1)^2 + (j - n_cols)^2)

#             # If distance is less than or equal to the radius, mark for removal
#             if dist <= radius
#                 data[i, j] = missing  
#             end
#         end
#     end
#     return data[:, :]
# end

# # Visualize simulations for varying β 
# function visualizepotts(distributions, β_values = 0.3:0.1:1.2; kwargs...) 
#     grids = [simulatehiddenpotts(30, 30, β, distributions; kwargs...).Z for β ∈ β_values]
#     incomplete_grids = remove_quarter_circle.(grids)
#     imputed_grids = getindex.(getindex.(simulatehiddenpotts.(incomplete_grids, β_values, Ref(distributions); num_iterations = 100, kwargs...), :Z), 100)
#     heatmaps = [clean_heatmap(grid) for (i, grid) in enumerate(vcat(grids, incomplete_grids, imputed_grids))]
#     plot(heatmaps..., layout=(3, length(β_values)), size=(3 * 600, length(β_values) * 90))
# end

# # Visualize initializations/iterations for varying β 
# function visualizepotts2(distributions, β_values = 0.3:0.1:1.2; num_iters = [1, 10, 20, 50, 100, 200, 300, 1000]) 
#     col_titles = ["Original", "Incomplete"] ∪ ["Iter = $n" for n in num_iters]
#     all_heatmaps = []
#     for (row_idx, β) in enumerate(β_values)
#         # Complete and incomplete data
#         original = simulatehiddenpotts(50, 50, β, distributions).Z
#         incomplete = remove_quarter_circle(original)
#         for (col_idx, grid) in enumerate([original, incomplete])
#             title = row_idx == 1 ? col_titles[col_idx] : ""
#             xlabel = col_idx == 1 ? "β = $(round(β, digits=2))" : ""
#             push!(all_heatmaps, clean_heatmap(grid; title=title, xlabel=xlabel))
#         end

#         # Imputed for each iteration 
#         # Run one long chain up to the maximum number of iterations
#         max_iter = maximum(num_iters)
#         sim = simulatehiddenpotts(incomplete, β, distributions; num_iterations=max_iter)

#         # Extract Z after each n
#         for (col_idx, n) in enumerate(num_iters)
#             imputed = sim.Z[n]
#             title = row_idx == 1 ? col_titles[col_idx + 2] : ""
#             xlabel = ""
#             push!(all_heatmaps, clean_heatmap(imputed; title=title, xlabel=xlabel))
#         end
#     end
#     ncols = 2 + length(num_iters)
#     plot(
#         all_heatmaps...,
#         layout = (length(β_values), ncols),
#         size = (ncols * 200, length(β_values) * 200)
#     )
# end

# # Visualize initializations/iterations for fixed β and fixed incomplete data
# function visualizepotts3(distributions, β = log(1 + sqrt(length(distributions))) + 0.025; num_iters = [1, 10, 20, 50, 100, 200, 300, 1000], num_rep = 5)
#     original = simulatehiddenpotts(50, 50, β, distributions).Z
#     incomplete = remove_quarter_circle(original)
#     col_titles = ["Original", "Incomplete"] ∪ ["Iter = $n" for n in num_iters]
#     all_heatmaps = []
#     for row_idx in 1:num_rep

#         for (col_idx, grid) in enumerate([original, incomplete])
#             title = row_idx == 1 ? col_titles[col_idx] : ""
#             xlabel = col_idx == 1 ? "β = $(round(β, digits=2))" : ""
#             push!(all_heatmaps, clean_heatmap(grid; title=title, xlabel=xlabel))
#         end

#         # Imputed for each iteration 
#         # Run one long chain up to the maximum number of iterations
#         max_iter = maximum(num_iters)
#         sim = simulatehiddenpotts(incomplete, β, distributions; num_iterations=max_iter)

#         # Extract Z after each n
#         for (col_idx, n) in enumerate(num_iters)
#             imputed = sim.Z[n]   
#             title = row_idx == 1 ? col_titles[col_idx + 2] : ""
#             xlabel = ""
#             push!(all_heatmaps, clean_heatmap(imputed; title=title, xlabel=xlabel))
#         end
#     end
#     ncols = 2 + length(num_iters)
#     plot(
#         all_heatmaps...,
#         layout = (num_rep, ncols),
#         size = (ncols * 200, num_rep * 200)
#     )
# end

# # Visualize parallel sampler 
# function visualizepotts4(distributions, β = log(1 + sqrt(length(distributions))) + 0.1; num_iters = [1, 10, 20, 50, 100, 200, 300, 1000], num_chains = 5)
#     original = simulatehiddenpotts(50, 50, β, distributions).Z
#     incomplete = remove_quarter_circle(original)
#     col_titles = ["Original", "Incomplete"] ∪ ["Iter = $n" for n in num_iters]
#     all_heatmaps = []
#     max_iter = maximum(num_iters)
#     sim = simulatehiddenpotts_parallel(incomplete, β, distributions; num_iterations=max_iter, num_chains = num_chains)
#     for row_idx in 1:num_chains

#         for (col_idx, grid) in enumerate([original, incomplete])
#             title = row_idx == 1 ? col_titles[col_idx] : ""
#             xlabel = col_idx == 1 ? "β = $(round(β, digits=2))" : ""
#             push!(all_heatmaps, clean_heatmap(grid; title=title, xlabel=xlabel))
#         end

#         # Imputed for each iteration 
#         # Run one long chain up to the maximum number of iterations

#         # Extract Z after each n
#         for (col_idx, n) in enumerate(num_iters)
#             imputed = sim.Z[:, :, n, row_idx]   
#             title = row_idx == 1 ? col_titles[col_idx + 2] : ""
#             xlabel = ""
#             push!(all_heatmaps, clean_heatmap(imputed; title=title, xlabel=xlabel))
#         end
#     end
#     ncols = 2 + length(num_iters)
#     plot(
#         all_heatmaps...,
#         layout = (num_chains, ncols),
#         size = (ncols * 200, num_chains * 200)
#     )
# end

# distribution(λ::AbstractVector) = Normal(λ[1], λ[2])
# μ = [-1.0, 0.0, 1.0]
# σ = [0.35, 0.35, 0.35]
# λ = vcat(μ', σ')
# distributions = distribution.(eachcol(λ))
# inflated = false
# if inflated distributions = [distributions..., Dirac(0.0)] end
# visualizepotts(distributions)
# visualizepotts2(distributions) 
# visualizepotts3(distributions)
# visualizepotts4(distributions)

# distribution(λ::AbstractVector) = Beta(λ[1], λ[2])
# a = [2.0, 5]
# b = [5.0, 1.0]
# λ = vcat(a', b')
# distributions = distribution.(eachcol(λ))
# inflated = false
# if inflated distributions = [distributions..., Dirac(0.0)] end
# visualizepotts(distributions)
# visualizepotts2(distributions)
# visualizepotts3(distributions)
# visualizepotts4(distributions, 0.7)