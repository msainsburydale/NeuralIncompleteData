import NeuralEstimators: EM
using DataFrames

# Helper function for visualizing EM iterates
function run_EM(em::EM, Z1, all_θ₀; parameter_names = nothing, all_nsims = [1, 10, 30], burnin::Integer = 10, kwargs...)
    if isa(parameter_names, String)
        parameter_names = [parameter_names]
    end
    dfs = []
    for (i, θ₀) in enumerate(all_θ₀)
        for nsims in all_nsims
            res = em(Z1, θ₀; nsims = nsims, burnin = burnin, tol = 1e-7, niterations = 100, kwargs...)

            estimates = res.iterates  
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
            df = DataFrame(
                iteration = repeat(1:niters, inner = nparams),
                parameter = repeat(parameter_names, outer = niters),
                estimate = vec(estimates),
                averaged_estimate = vec(averaged_estimates),
                theta_0 = fill(i, nparams * niters),
                nsims = fill(nsims, nparams * niters),
            )

            push!(dfs, df)
        end
    end

    return vcat(dfs...)
end