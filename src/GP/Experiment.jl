# Gaussian process simulation study (Section 3.2), pure-Julia implementation (Vecchia.jl in place of GpGp). This
# mirrors src/GP/Experiment.R, and uses the same trained networks and the same incomplete data sets, which are saved
# by src/GP/Experiment.R (so that script must be run first). Results are written to intermediates/GP/julia, and the
# tables and figures are produced by `Rscript src/GP/Results.R --julia`.
#
# Usage (from the top level of the repository):
#   julia --project=. --threads=32 src/GP/Experiment.jl [--quick]

quick = "--quick" in ARGS

# On x86-64, MKL is used for BLAS and LAPACK: OpenBLAS serialises concurrent calls from many Julia threads (which
# then spend most of their time waiting on a lock in the kernel), whereas MKL does not. MKL's sequential threading
# layer is used, since parallelism is over Julia threads: with its default OpenMP layer, every Julia thread that calls
# MKL starts its own team of threads. The layer must be chosen before MKL is loaded.
@static if Sys.ARCH === :x86_64
    ENV["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    using MKL
end
using NeuralEstimators
using NeuralEstimators: estimate
using Flux
using CUDA, cuDNN # cuDNN: Flux convolutions on NVIDIA GPUs
using BSON: @load
using BenchmarkTools
using CSV, DataFrames
using RData
using LinearAlgebra
using Random: Xoshiro
using Statistics: median

# Parallelism is over data sets (and conditional simulations), so BLAS uses a single thread
BLAS.set_num_threads(1)

include(joinpath(pwd(), "src", "Architecture.jl"))
include(joinpath(pwd(), "src", "GP", "Simulation.jl"))
include(joinpath(pwd(), "src", "EM.jl"))

r_path = joinpath("intermediates", "GP")
int_path = joinpath(r_path, "julia")
est_path = joinpath(int_path, "Estimates")
mkpath(est_path)

parameter_names = ["rho", "tau"]
d = length(parameter_names)

# ---- Load the trained networks ----

function loadnetwork(input_channels, folder)
    estimator = architecture(d, input_channels)
    path = joinpath(r_path, folder, "best_estimator.bson")
    @load path model_state
    Flux.loadmodel!(estimator, model_state)
    return estimator
end
neuralMAP = loadnetwork(1, "runs_EM")
maskedestimator = loadnetwork(2, "runs_masking")

# The networks are moved to the GPU (if there is one) once: batched_em and estimate then use them there without
# copying them, so the timings do not include copying the weights to the GPU
neuralMAP = Flux.gpu_device()(neuralMAP)
maskedestimator = Flux.gpu_device()(maskedestimator)

# ---- Estimation ----

# Hyperparameters of the EM approach (the same as in src/GP/Experiment.R)
sim = GPEMSimulator(nsims = quick ? 3 : 30)
em_kwargs = (niterations = quick ? 3 : 50, burnin = quick ? 1 : 5, tol = 0.01, nconsecutive = quick ? 1 : 3)

runtimes = DataFrame(set = String[], missingness = String[], estimator = String[], time = Float64[], n = Int[])

# Runs all estimators on the incomplete data sets Z1, and saves the estimates, EM diagnostics and run times
function run_estimators(Z1, set, missingness)
    println("\n$set set, $missingness data ($(length(Z1)) data sets)")
    path = joinpath(est_path, set)
    mkpath(path)
    record(estimator, t) = push!(runtimes, (set, missingness, estimator, t, length(Z1)))

    println("  Running the masking NBE...")
    UW = encodedata.(Z1)
    estimate(maskedestimator, UW[1:1]) # compilation
    t = @elapsed masked = estimate(maskedestimator, UW)
    record("masking", t)

    println("  Running the EM NBE...")
    # The EM algorithm starts from the prior mean (as does the MAP optimiser), as in Section 3.1 of the paper
    t = @elapsed em = batched_em(sim, neuralMAP, Z1, gp_prior_mean; em_kwargs..., rng = Xoshiro(1), verbose = true)
    record("neuralEM", t)
    CSV.write(joinpath(path, "EM_diagnostics_$missingness.csv"), DataFrame(k = eachindex(Z1), niterations = em.niterations, converged = em.converged))
    CSV.write(joinpath(path, "EM_timings_$missingness.csv"), DataFrame(component = ["simulation", "estimation"], time = [em.timings.simulation, em.timings.estimation]))

    println("  Computing MAP estimates...")
    t = @elapsed MAP = gp_MAP(Z1; rng = Xoshiro(1))
    record("MAP", t)

    return Dict("masking" => masked, "neuralEM" => em.estimates, "MAP" => MAP)
end

# Long-format data frame of estimates (the same format as src/GP/Experiment.R)
function estimates_dataframe(estimates, truth, k, j)
    vcat([DataFrame(
        m = 1,
        k = repeat(k, inner = d),
        j = repeat(j, inner = d),
        estimator = name,
        parameter = repeat(parameter_names, outer = size(truth, 2)),
        estimate = vec(estimates[name]),
        truth = vec(truth)
    ) for name in ("masking", "neuralEM", "MAP")]...)
end

loadZ1(set, missingness) = [convert(Array{Union{Missing, Float32}}, z) for z in RData.load(joinpath(r_path, "Estimates", set, "Z1_$missingness.rds"))]

# ---- Test set ----

θ_test = RData.load(joinpath(r_path, "Estimates", "Test", "theta_test.rds"))
K = size(θ_test, 2)
df = vcat(
    [insertcols!(estimates_dataframe(run_estimators(loadZ1("Test", mis), "Test", mis), θ_test, 1:K, ones(Int, K)), :missingness => mis)
     for mis in ("MCAR", "MNAR")]...
)
CSV.write(joinpath(est_path, "estimates_test.csv"), df)

# ---- Sampling distributions for selected parameter values ----

θ = RData.load(joinpath(r_path, "Estimates", "Scenarios", "theta.rds"))
K = size(θ, 2)
dfs = map(("MCAR", "MNAR")) do mis
    Z1 = loadZ1("Scenarios", mis)
    J = length(Z1) ÷ K
    truth = θ[:, repeat(1:K, outer = J)]
    df = estimates_dataframe(run_estimators(Z1, "Scenarios", mis), truth, repeat(1:K, outer = J), repeat(1:J, inner = K))
    insertcols!(df, :missingness => mis)
    CSV.write(joinpath(est_path, "estimates_$(mis)_scenarios.csv"), df)
    df
end
CSV.write(joinpath(est_path, "estimates_scenarios.csv"), vcat(dfs...))

CSV.write(joinpath(int_path, "runtime_total.csv"), runtimes)

# ---- Run times for a single data set ----

println("\nTiming the estimators for a single data set...")
Z1 = loadZ1("Test", "MCAR")[1]
df = DataFrame(estimator = String[], time = Float64[])

# MAP (likelihood evaluated in parallel over locations)
push!(df, ("MAP", @belapsed gp_MAP($Z1; threaded = true, rng = Xoshiro(1)) samples = 10 evals = 1))

# Masking NBE
UW1 = encodedata(Z1)
t_masking = @belapsed estimate($maskedestimator, $UW1)
push!(df, ("masking", t_masking))

# EM NBE (started from the prior mean)
em_single(Z1) = batched_em(sim, neuralMAP, Z1, gp_prior_mean; em_kwargs..., rng = Xoshiro(1))
em_result = em_single(Z1)
push!(df, ("EM", @belapsed em_single($Z1) samples = 10 evals = 1))
push!(df, ("EM_niterations", size(em_result.iterates, 2) - 1))

CSV.write(joinpath(int_path, "runtime_singledataset.csv"), df)
println(df)

# ---- Convergence of the EM algorithm for a single data set ----

data = RData.load(joinpath(r_path, "EM_iterates_data.rds"))
Z1 = Matrix{Union{Missing, Float32}}(data["Z1"])
df = run_EM(sim, neuralMAP, Z1, [[0.05, 0.1], [0.3, 0.7]]; parameter_names = parameter_names, all_nsims = [1, 10, 30], rng = Xoshiro(1))
CSV.write(joinpath(int_path, "EM_iterates.csv"), df)
