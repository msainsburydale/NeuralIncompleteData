using Folds
using LinearAlgebra
using Statistics
using StatsBase

"""
    summary_statistic(Z, D; kwargs...)

Compute spatial dependence measures for various data structures. This is a helper function that wraps around `variograms()` for different data structures.

# Arguments
- `Z`: Input data, which can be:
  - `AbstractMatrix`: nsite × ntime matrix (single dataset)
  - `AbstractArray`: Multidimensional array (flattened to matrix)
  - `AbstractVector{AbstractArray}`: Vector of datasets (multiple realizations)
- `D`: nsite × nsite distance matrix
- `kwargs...`: Additional arguments passed to `variograms()`

# Returns
- For single dataset: Vector of length `5 * nbins` (flattened `variograms` output)
- For multiple datasets: Matrix of size `(5 * nbins) × nsim` where each column contains the flattened variogram output for one dataset

# Examples
```
using Distances
using LinearAlgebra
using NeuralEstimators
using Random

# Spatial data
pts = range(0.0, 1.0, 16)
S = expandgrid(pts, pts)
D = pairwise(Euclidean(), S, S, dims = 1)
Σ = exp.(-D)
L = cholesky(Σ).L
Z = simulategaussian(L, 150)

# Single dataset (matrix)
summary_statistic(Z, D)

# Multiple datasets (vector of matrices)
summary_statistic([Z, Z], D)
```
"""
function summary_statistic(Z::AbstractMatrix, D; kwargs...) 
    return vec(variograms(Z, D; kwargs...))
end

function summary_statistic(Z::AbstractArray, D; kwargs...) 
    summary_statistic(flatten(Z), D; kwargs...)
end

function summary_statistic(Z::AbstractVector{A}, D; kwargs...) where A <: AbstractArray
    T = Folds.map(Z) do z 
        summary_statistic(z, D; kwargs...)
    end
    T = reduce(hcat, T)
    return T
end

"""
    variograms(Z, D; u=0.95, nbins=10, dmin=0.05, dmax=0.5)

Compute multiple binned spatial dependence measures between sites.

Arguments:
- `Z`: nsite × ntime matrix (each row = site, each column = observation/time)
- `D`: nsite × nsite distance matrix
- `u`: quantile threshold for tail dependence (default 0.95)
- `nbins`: number of distance bins (default 10)
- `dmin`, `dmax`: range of bin centers

Returns: `nbins × 5` matrix containing binned averages of:
  - Column 1: η (tail dependence coefficient)
  - Column 2: χ (tail dependence coefficient)  
  - Column 3: variogram (0.5 × mean squared differences)
  - Column 4: MADO (0.5 × mean absolute differences)
  - Column 5: RODO (0.5 × mean square root of absolute differences)

# Examples
```
using Distances
using LinearAlgebra
using NeuralEstimators
using Random

# Spatial data
pts = range(0.0, 1.0, 16)
S = expandgrid(pts, pts)
D = pairwise(Euclidean(), S, S, dims = 1)
Σ = exp.(-D)
L = cholesky(Σ).L
Z = simulategaussian(L, 150)

# Spatial dependence measures
variograms(Z, D; u = 0.70)
variograms(Z, D, u = 0.85)
variograms(Z, D, u = 0.99)
```
"""
function variograms(
    Z::AbstractMatrix, D::AbstractMatrix; 
    u::Real=0.95, nbins::Int=10, dmin::Real=0.05, dmax::Real=0.5
    )

    nsite, ntime = size(Z)

    # --- Precompute ranks for all sites ---
    R = similar(Z, Float64)
    for i in 1:nsite
        mask = .!ismissing.(Z[i, :])
        x = Z[i, mask]
        N = length(x)
        r = ordinalrank(x) ./ (N + 1)
        R[i, mask] .= r
        R[i, .!mask] .= NaN
    end

    # --- Collect pairwise statistics ---
    npairs = nsite * (nsite - 1) ÷ 2
    eta_data = fill(NaN, npairs)
    chi_data = fill(NaN, npairs)
    vario_data = fill(NaN, npairs)
    mado_data = fill(NaN, npairs)
    rodo_data = fill(NaN, npairs)
    dist_data = fill(NaN, npairs)

    k = 1
    for i in 1:(nsite-1)
        r1 = R[i, :]
        z1 = Z[i, :]

        if all(ismissing.(z1))
            continue
        end

        for j in (i+1):nsite
            r2 = R[j, :]
            z2 = Z[j, :]

            if all(ismissing.(z2))
                continue
            end

            # common non-missing mask
            mask = .!(ismissing.(r1) .| ismissing.(r2) .| ismissing.(z1) .| ismissing.(z2))
            r1v, r2v = r1[mask], r2[mask]
            z1v, z2v = z1[mask], z2[mask]
            N = length(r1v)

            joint = sum((r1v .> u) .& (r2v .> u)) / N
            eta_data[k] = joint > 0 ? log(1 - u) / log(joint) : 0.0
            chi_data[k] = joint / (1 - u)

            vario_data[k] = 0.5 * mean((z1v .- z2v).^2)
            mado_data[k] = 0.5 * mean(abs.(z1v .- z2v))
            rodo_data[k] = 0.5 * mean(sqrt.(abs.(z1v .- z2v)))

            dist_data[k] = D[i, j]
            k += 1
        end
    end

    # --- Bin distances ---
    bin_centers = range(dmin, dmax; length=nbins)
    Δ = step(bin_centers)
    bin_edges = [bin_centers[1] - Δ/2; (bin_centers[1:end-1] .+ bin_centers[2:end]) ./ 2; bin_centers[end] + Δ/2]
    eta_binned = bin_average(dist_data, eta_data, bin_edges, bin_centers)
    chi_binned = bin_average(dist_data, chi_data, bin_edges, bin_centers)
    vario_binned = bin_average(dist_data, vario_data, bin_edges, bin_centers)
    mado_binned = bin_average(dist_data, mado_data, bin_edges, bin_centers)
    rodo_binned = bin_average(dist_data, rodo_data, bin_edges, bin_centers)

    return hcat(eta_binned, chi_binned, vario_binned, mado_binned, rodo_binned)
end

function bin_average(dist_data, val_data, bin_edges, bin_centers)
    mask = .!isnan.(dist_data) .&& .!isnan.(val_data)
    dist_data = dist_data[mask]
    val_data = val_data[mask]
    binned_vals = similar(bin_centers, Float64)
    for k in 1:length(bin_centers)
        mask = (dist_data .>= bin_edges[k]) .& (dist_data .< bin_edges[k+1])
        binned_vals[k] = any(mask) ? mean(filter(!isnan, val_data[mask])) : NaN 
    end
    return binned_vals
end