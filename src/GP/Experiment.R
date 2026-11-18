# Gaussian process simulation study (Section 3.2), R implementation. A pure-Julia implementation of the same study is
# in src/GP/Experiment.jl; it uses the incomplete data saved by this script, and the same neural networks.
#
# Usage (from the top level of the repository):
#   Rscript src/GP/Experiment.R [--quick] [--train]
#     --quick  a computationally inexpensive run, to check that the code works
#     --train  simulate training data and train the networks (otherwise, the trained networks are loaded)
#     --timings-only  only time the estimators for a single data set (the other results must already exist)
# The number of parallel workers is min(32, half the number of cores), or the environment variable NCORES.
# Results (tables and figures) are produced by src/GP/Results.R.

args <- commandArgs(trailingOnly = TRUE)
quick <- "--quick" %in% args
train_networks <- "--train" %in% args
timings_only <- "--timings-only" %in% args # only the run times for a single data set (after a full run)

# Parallelism is over data sets, with one BLAS/OpenMP thread per worker process. This R process may use ncores
# OpenMP threads in GpGp, which is used when estimating from a single data set (run times, convergence figure): the
# OpenMP runtime reads OMP_NUM_THREADS when GpGp is loaded, and the workers (launched afterwards) inherit one thread.
ncores <- as.integer(Sys.getenv("NCORES", min(32, parallel::detectCores() %/% 2)))
Sys.setenv(
  OPENBLAS_NUM_THREADS = "1",
  OMP_NUM_THREADS = as.character(ncores),
  MKL_NUM_THREADS = "1",
  JULIACONNECTOR_JULIAOPTS = "--project=."
)

suppressMessages({
  library("abc")
  library("abctools")
  library("dplyr")
  library("NeuralEstimators")
  library("JuliaConnectoR")
  library("readr")
  library("parallel")
  library("future")
  library("future.apply")
  options(dplyr.summarise.inform = FALSE)
})

# ABC (not reported in the paper) needs geoR::variog(); it is skipped if geoR cannot be loaded (e.g., on a Mac
# without XQuartz)
run_abc <- tryCatch(suppressMessages(requireNamespace("geoR", quietly = TRUE)), error = function(e) FALSE)
if (!run_abc) warning("geoR could not be loaded: skipping ABC")

source(file.path("src", "GP", "Simulation.R"))
source(file.path("src", "EM.R"))
source(file.path("src", "Utils.R"))

int_path <- file.path("intermediates", "GP")
est_path <- file.path(int_path, "Estimates")
dir.create(file.path(est_path, "Test"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(est_path, "Scenarios"), recursive = TRUE, showWarnings = FALSE)

parameter_names <- c("rho", "tau")
d <- as.integer(length(parameter_names))

# ---- Parallel workers ----

# Stateless parallel map (training-data simulation, MAP, ABC)
Sys.setenv(OMP_NUM_THREADS = "1")
plan(multisession, workers = ncores)
lapply_function <- function(...) future_lapply(..., future.seed = TRUE)

# Workers for the conditional simulations of the EM algorithm: each keeps the set-up of its data sets
cl <- makeCluster(ncores)
invisible(clusterEvalQ(cl, source(file.path("src", "GP", "Simulation.R"))))
clusterSetRNGStream(cl, 1)

juliaEval('using NeuralEstimators, Flux, CUDA, cuDNN, BenchmarkTools')
juliaEval('using BSON: @load, @save')
snk <- juliaEval('include(joinpath(pwd(), "src", "Architecture.jl"))')
architecture <- juliaFun("architecture")

# ---- Training ----

## Maximum number of epochs
epochs <- ifelse(quick, 10, 100)

## Path of the trained networks
network_path <- function(folder) file.path(int_path, folder, "best_estimator.bson")
savenetwork <- function(estimator, folder) {
  juliaLet('model_state = Flux.state(Flux.cpu(estimator)); @save path model_state', estimator = estimator, path = network_path(folder))
}

## Summary statistics for ABC: empirical variogram at these distances
S <- gridlocations(64)
u <- seq(0.04, 0.4, length.out = 10)
abc_path <- file.path(int_path, "ABC")
dir.create(abc_path, recursive = TRUE, showWarnings = FALSE)

if (train_networks) {
  cat("Simulating training data...\n")
  set.seed(1)
  K <- ifelse(quick, 1000, 25000)                # size of the training set
  theta_train <- sampler(K)                      # parameter vectors used in stochastic-gradient descent
  theta_val   <- sampler(K %/% 2)                # parameter vectors used to monitor performance
  tm <- system.time({
    Z_train <- simulator(theta_train, lapply_function = lapply_function)
    Z_val   <- simulator(theta_val, lapply_function = lapply_function)
  })
  saveRDS(tm, file = file.path(int_path, "sim_time.rds"))

  ## Data sets for the masking approach
  UW_train <- encodedata(lapply(Z_train, removedata))
  UW_val   <- encodedata(lapply(Z_val, removedata))

  ## Neural MAP estimator for the EM approach
  neuralMAP <- architecture(d, input_channels = 1L)
  neuralMAP <- train(neuralMAP, theta_train = theta_train, theta_val = theta_val, Z_train = Z_train, Z_val = Z_val,
                     epochs = epochs, savepath = file.path(int_path, "runs_EM"))
  savenetwork(neuralMAP, "runs_EM")

  ## Masked neural Bayes estimator
  maskedestimator <- architecture(d, input_channels = 2L)
  maskedestimator <- train(maskedestimator, theta_train = theta_train, theta_val = theta_val, Z_train = UW_train, Z_val = UW_val,
                           epochs = epochs, savepath = file.path(int_path, "runs_masking"))
  savenetwork(maskedestimator, "runs_masking")

  ## ABC: summary statistics of the training data
  if (run_abc) tm <- system.time({
    sumstats_train <- lapply_function(Z_train, function(z) geoR::variog(coords = S, data = c(z), uvec = u, messages = FALSE)$v)
  })
  if (run_abc) {
    sumstats_train <- do.call(cbind, sumstats_train)
    sumstats_model <- saABC(t(theta_train), t(sumstats_train))
    sumstats_train <- sumstats_model$B %*% sumstats_train
    saveRDS(tm, file = file.path(abc_path, "sumstats_time.rds"))
    saveRDS(sumstats_train, file = file.path(abc_path, "sumstats_train.rds"))
    saveRDS(sumstats_model, file = file.path(abc_path, "sumstats_model.rds"))
    saveRDS(theta_train, file = file.path(abc_path, "theta_train.rds"))
  }
  rm(Z_train, Z_val, UW_train, UW_val)
}

# ---- Load the trained networks and the ABC reference table ----

loadnetwork <- function(input_channels, folder) {
  juliaLet('
    estimator = architecture(d, input_channels)
    @load path model_state
    Flux.loadmodel!(estimator, model_state)
    estimator
  ', d = d, input_channels = as.integer(input_channels), path = network_path(folder))
}
neuralMAP <- loadnetwork(1, "runs_EM")
maskedestimator <- loadnetwork(2, "runs_masking")

sumstats_train <- readRDS(file.path(abc_path, "sumstats_train.rds"))
sumstats_model <- readRDS(file.path(abc_path, "sumstats_model.rds"))
theta_train <- readRDS(file.path(abc_path, "theta_train.rds"))

abc_mode <- function(z1, S, u, sumstats, sumstats_model, theta, method = "neuralnet") {

  # Flatten data and remove NAs from vector and spatial locations
  z1 <- c(z1)
  I1 <- which(!is.na(z1))
  z1 <- z1[I1]
  S1 <- S[I1, , drop = FALSE]

  # Observed summary statistics
  target <- geoR::variog(coords = S1, data = z1, uvec = u, messages = FALSE)$v
  target <- sumstats_model$B %*% target

  invisible(capture.output({suppressWarnings({suppressMessages({
    object <- abc(target = c(target), param = t(theta), sumstat = t(sumstats), tol = 0.01, method = method)
    if (method == "rejection") {
      mode <- apply(object$unadj.values, 2, abc:::getmode)
    } else {
      wt <- object$weights
      mode <- apply(object$adj.values, 2, abc:::getmode, wt / sum(wt))
    }
  })})}, type = "output"))

  return(mode)
}

# ---- Estimation ----

## Hyperparameters of the EM approach (the same for all simulation studies)
em_args <- list(
  nsims = ifelse(quick, 3, 30),
  niterations = ifelse(quick, 3, 50),
  burn_in = ifelse(quick, 1, 5),
  tol = 0.01,
  nconsecutive = ifelse(quick, 1, 3)
)

## Runs all estimators on the incomplete data sets Z1, and saves the estimates, EM diagnostics and run times
runtimes <- tibble(set = character(), missingness = character(), estimator = character(), time = numeric(), n = integer())
run_estimators <- function(Z1, set, missingness) {

  cat(sprintf("\n%s set, %s data (%d data sets)\n", set, missingness, length(Z1)))
  path <- file.path(est_path, set)
  saveRDS(Z1, file = file.path(path, paste0("Z1_", missingness, ".rds"))) # used by src/GP/Experiment.jl
  record <- function(estimator, time) {
    runtimes <<- runtimes %>% add_row(set = set, missingness = missingness, estimator = estimator, time = time, n = length(Z1))
  }

  cat("  Running the masking NBE...\n")
  t <- system.time(masked <- estimate(maskedestimator, encodedata(Z1)))[["elapsed"]]
  record("masking", t)

  cat("  Running the EM NBE...\n")
  # The EM algorithm starts from the prior mean (as does the MAP optimiser), as in Section 3.1 of the paper
  t <- system.time(em <- do.call(batched_EM, c(list(Z1, neuralMAP, theta_0 = prior_mean, cl = cl, verbose = TRUE), em_args)))[["elapsed"]]
  record("neuralEM", t)
  write.csv(data.frame(k = seq_along(Z1), niterations = em$niterations, converged = em$converged),
            file.path(path, paste0("EM_diagnostics_", missingness, ".csv")), row.names = FALSE)
  write.csv(data.frame(component = names(em$timings), time = em$timings),
            file.path(path, paste0("EM_timings_", missingness, ".csv")), row.names = FALSE)

  cat("  Computing MAP estimates...\n")
  t <- system.time(MAP_est <- do.call(cbind, lapply_function(Z1, MAP)))[["elapsed"]]
  record("MAP", t)

  if (run_abc) {
    cat("  Computing ABC MAP estimates...\n")
    t <- system.time(ABC_est <- do.call(cbind, lapply_function(Z1, abc_mode, S = S, u = u, sumstats = sumstats_train,
                                                                sumstats_model = sumstats_model, theta = theta_train)))[["elapsed"]]
    record("ABC MAP", t)
    saveRDS(ABC_est, file = file.path(path, paste0("ABCMAP_", missingness, ".rds")))
  }

  saveRDS(masked, file = file.path(path, paste0("masked_", missingness, ".rds")))
  saveRDS(em$estimates, file = file.path(path, paste0("EM_", missingness, ".rds")))
  saveRDS(MAP_est, file = file.path(path, paste0("MAP_", missingness, ".rds")))

  out <- list(masking = masked, neuralEM = em$estimates, MAP = MAP_est)
  if (run_abc) out[["ABC MAP"]] <- ABC_est
  out
}

## Long-format data frame of estimates: estimates[, i] is the estimate for data set i, with truth[, i] its true value
estimates_dataframe <- function(estimates, truth, k, j) {
  do.call(rbind, lapply(names(estimates), function(name) {
    data.frame(
      m = 1,
      k = rep(k, each = d),
      j = rep(j, each = d),
      estimator = name,
      parameter = rep(parameter_names, times = ncol(truth)),
      estimate = c(estimates[[name]]),
      truth = c(truth)
    )
  }))
}

if (!timings_only) {

# ---- Test set ----

## Test data: reused if available (so that results are comparable across runs and implementations)
if (!quick && file.exists(file.path(int_path, "Z_test.rds"))) {
  theta_test <- readRDS(file.path(int_path, "theta_test.rds"))
  Z_test <- readRDS(file.path(int_path, "Z_test.rds"))
} else {
  set.seed(1)
  K_test <- ifelse(quick, 100, 1000)
  theta_test <- sampler(K_test)
  Z_test <- simulator(theta_test, lapply_function = lapply_function)
  if (!quick) {
    saveRDS(theta_test, file = file.path(int_path, "theta_test.rds"))
    saveRDS(Z_test, file = file.path(int_path, "Z_test.rds"))
  }
}
saveRDS(theta_test, file = file.path(est_path, "Test", "theta_test.rds"))

## Incomplete data
set.seed(1)
Z1_MCAR <- lapply(Z_test, removedata, proportion = 0.2)
Z1_MNAR <- lapply(lapply(Z_test, remove_quarter_circle), add_singletons)

df <- rbind(
  estimates_dataframe(run_estimators(Z1_MCAR, "Test", "MCAR"), theta_test, k = seq_len(ncol(theta_test)), j = 1) %>% mutate(missingness = "MCAR"),
  estimates_dataframe(run_estimators(Z1_MNAR, "Test", "MNAR"), theta_test, k = seq_len(ncol(theta_test)), j = 1) %>% mutate(missingness = "MNAR")
)
write.csv(df, file.path(est_path, "estimates_test.csv"), row.names = FALSE)

# ---- Sampling distributions for selected parameter values ----

set.seed(1)
J <- ifelse(quick, 10, 50)
rho <- qunif(c(0.1, 0.5, 0.9), 0.05, 0.3)
tau <- qunif(c(0.1, 0.5, 0.9))
theta <- t(as.matrix(expand.grid(rho, tau)))
K <- ncol(theta)
Z <- unlist(lapply(1:J, function(j) simulator(theta, lapply_function = lapply_function)), recursive = FALSE)

Z1_MCAR <- lapply(Z, removedata, proportion = 0.2)
Z1_MNAR <- lapply(lapply(Z, remove_quarter_circle), add_singletons)
saveRDS(theta, file = file.path(est_path, "Scenarios", "theta.rds"))

## Save the first replicate for each parameter vector, for plotting
savedata <- function(Z, missingness) {
  z <- unlist(lapply(Z[1:K], c))
  n <- length(Z[[1]])
  write.csv(data.frame(Z = z, k = rep(1:K, each = n), j = 1), file.path(est_path, paste0("Z_", missingness, ".csv")), row.names = FALSE)
}
savedata(Z1_MCAR, "MCAR")
savedata(Z1_MNAR, "MNAR")

truth <- theta[, rep(1:K, times = J)]
kk <- rep(1:K, times = J)
jj <- rep(1:J, each = K)
MCAR_df <- estimates_dataframe(run_estimators(Z1_MCAR, "Scenarios", "MCAR"), truth, k = kk, j = jj) %>% mutate(missingness = "MCAR")
MNAR_df <- estimates_dataframe(run_estimators(Z1_MNAR, "Scenarios", "MNAR"), truth, k = kk, j = jj) %>% mutate(missingness = "MNAR")
write.csv(MCAR_df, file.path(est_path, "estimates_MCAR_scenarios.csv"), row.names = FALSE)
write.csv(MNAR_df, file.path(est_path, "estimates_MNAR_scenarios.csv"), row.names = FALSE)
write.csv(rbind(MCAR_df, MNAR_df), file.path(est_path, "estimates_scenarios.csv"), row.names = FALSE)

write.csv(runtimes, file.path(int_path, "runtime_total.csv"), row.names = FALSE)

}

# ---- Run times for a single data set ----

cat("\nTiming the estimators for a single data set...\n")
Z1 <- readRDS(file.path(est_path, "Test", "Z1_MCAR.rds"))[[1]]
nrep <- ifelse(quick, 1, 10)
elapsed <- function(expr) { # median elapsed time of nrep evaluations of expr
  e <- substitute(expr)
  env <- parent.frame()
  median(replicate(nrep, system.time(eval(e, env))[["elapsed"]]))
}
df <- tibble(estimator = character(), time = numeric())

# Warm up (compilation in Julia)
invisible(estimate(maskedestimator, encodedata(list(Z1))))
invisible(do.call(batched_EM, c(list(list(Z1), neuralMAP, prior_mean), em_args)))

df <- df %>% add_row(estimator = "MAP", time = elapsed(MAP(Z1)))
if (run_abc) df <- df %>% add_row(estimator = "ABC MAP", time = elapsed(abc_mode(Z1, S = S, u = u, sumstats = sumstats_train, sumstats_model = sumstats_model, theta = theta_train)))

# Masking NBE, with and without the overhead of calling Julia from R
df <- df %>% add_row(estimator = "masking_with_overhead", time = elapsed(estimate(maskedestimator, encodedata(list(Z1)))))
UW1 <- encodedata(list(Z1))
t_masking <- juliaLet('@belapsed estimate($estimator, $z)', estimator = maskedestimator, z = UW1)
df <- df %>% add_row(estimator = "masking_without_overhead", time = t_masking)

# EM NBE (started from the prior mean), with the overhead of calling Julia from R
t_em <- elapsed(do.call(batched_EM, c(list(list(Z1), neuralMAP, prior_mean), em_args)))
em_result <- do.call(batched_EM, c(list(list(Z1), neuralMAP, prior_mean), em_args))
df <- df %>% add_row(estimator = "EM_with_overhead", time = t_em)

# EM NBE without the overhead: the time of each component of an EM iteration, times the number of iterations
num_its <- em_result$niterations[1]
Z_complete <- array(readRDS(file.path(est_path, "Test", "Z1_MCAR.rds"))[[1]], dim = c(64, 64, 1, em_args$nsims))
Z_complete[is.na(Z_complete)] <- 0 # (only the run time matters here)
t_estimation <- juliaLet('@belapsed estimate($estimator, $z)', estimator = neuralMAP, z = Z_complete)
t_setup <- elapsed(vecchia_setup(Z1))
setup <- vecchia_setup(Z1)
t_condsim <- elapsed(simulate_conditional(setup, prior_mean, nsims = em_args$nsims))
df <- df %>% add_row(estimator = "EM_without_overhead", time = t_setup + num_its * (t_condsim + t_estimation))
df <- df %>% add_row(estimator = "EM_niterations", time = num_its)

write.csv(df, file.path(int_path, "runtime_singledataset.csv"), row.names = FALSE)
print(df)

if (!timings_only) {

# ---- Convergence of the EM algorithm for a single data set ----

set.seed(1)
theta <- t(t(c(0.15, 0.4)))
Z <- simulator(theta)[[1]][, , 1, 1]
Z1 <- removedata(Z, proportion = 0.2)
saveRDS(list(Z1 = Z1, theta = theta), file = file.path(int_path, "EM_iterates_data.rds")) # used by src/GP/Experiment.jl
df <- run_EM(Z1, neuralMAP, theta_0 = list(c(0.05, 0.1), c(0.3, 0.7)), all_nsims = list(1, 10, 30),
             parameter_names = parameter_names, niterations = ifelse(quick, 15, 100))
write.csv(df, file = file.path(int_path, "EM_iterates.csv"), row.names = FALSE)

}

stopCluster(cl)
