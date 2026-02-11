# Boolean indicating whether (TRUE) or not (FALSE) to quickly establish that the code is working properly
quick <- identical(commandArgs(trailingOnly=TRUE)[1], "--quick")

parallel <- TRUE

# Limit BLAS threads globally for parallelization
Sys.setenv(
  OPENBLAS_NUM_THREADS = "1",
  OMP_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1", 
  JULIACONNECTOR_JULIAOPTS = "--project=."
)

suppressMessages({
  library("abc")
  library("GpGp") # fast_Gp_sim(),  cond_sim(), 
  library("dplyr")
  library("fields")
  library("Matrix")
  library("NeuralEstimators")
  library("JuliaConnectoR")
  library("abctools")
  library("geoR")
  library("readr")
  options(dplyr.summarise.inform = FALSE) 
})

if (parallel) {
  library("future")
  library("future.apply")
  plan(multisession, workers = availableCores() %/% 2)
  lapply_function <- function(...) future_lapply(..., future.seed = TRUE)
} else {
  lapply_function <- lapply
}

juliaEval('using NeuralEstimators, Flux, CUDA, BenchmarkTools')
juliaEval('using BSON: @load')
snk <- juliaEval('include(joinpath(pwd(), "src", "Architecture.jl"))')
architecture <- juliaFun("architecture")

int_path <- file.path("intermediates", "GP")
img_path <- file.path("img", "GP")
dir.create(int_path, recursive = TRUE, showWarnings = FALSE)
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(int_path, "Estimates", "Test"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(int_path, "Estimates", "Scenarios"), recursive = TRUE, showWarnings = FALSE)

source(file.path("src", "Plotting.R"))
source(file.path("src", "EM.R"))
source(file.path("src", "Utils.R"))

parameter_labels <- c("rho" = expression(rho), "tau" = expression(tau))
parameter_names <- names(parameter_labels)
d <- as.integer(length(parameter_names))   

# ---- Define the model: prior, marginal data simulation ----

## Sampling from the prior distribution
## K: number of samples to draw from the prior
rho_limits <- c(0.03, 0.35)    # range parameter
tau_limits <- c(0.01, 1)       # fine-scale variance parameter
sampler <- function(K) { 
  rho <- runif(K, min = rho_limits[1], max = rho_limits[2])
  tau <- runif(K, min = tau_limits[1], max = tau_limits[2])
  theta <- matrix(c(rho, tau), nrow = 2, byrow = TRUE)
  return(theta)
}
prior_mean <- c(mean(rho_limits), mean(tau_limits))

## Marginal simulation from the statistical model
## theta: a matrix of parameters drawn from the prior
simulator <- function(theta, N = 64) {

  locs <- as.matrix(expand.grid(seq(1, 0, len = N), seq(0, 1, len = N)))
  
  Z <- lapply_function(1:ncol(theta), function(k) {
    rho <- theta[1, k]
    tau <- theta[2, k]
    Z <- fast_Gp_sim(c(1, rho, 1, tau^2), "matern_isotropic", locs)
    array(Z, dim = c(N, N, 1, 1))
  })
  
  return(Z)
}

# ---- Conditional simulation and MAP (MLE) estimation using Vecchia ----

# Setup for the Vecchia approximation (done once per y, regardless of theta)
vecchia_setup <- function(y, num_neighbours = 30, reorder = TRUE) {
  
  covfun_name <- "matern_isotropic"
  y <- drop(y)
  
  # All locations
  N <- nrow(y) # NB assumes square grid
  locs <- as.matrix(expand.grid(seq(1, 0, len = N), seq(0, 1, len = N)))
  
  # Get the non-missing data, the observed locations, and the unobserved locations
  non_na_indices <- which(!is.na(y))
  na_indices <- which(is.na(y))
  y_obs <- y[non_na_indices]
  locs_obs <- locs[non_na_indices, ]
  locs_pred <- locs[na_indices, ]
  
  # how many observations and predictions
  n_obs <- nrow(locs_obs)
  n_pred <- nrow(locs_pred)
  
  # get orderings
  if(reorder) {
    ord1 <- if( n_obs < 6e4 ) GpGp::order_maxmin(locs_obs) else sample(1:n_obs)
    ord2 <- if( n_pred < 6e4 ) GpGp::order_maxmin(locs_pred) else sample(1:n_pred)
  } else {
    ord1 <- 1:n_obs
    ord2 <- 1:n_pred
  }
  
  # reorder observations and locations
  yord_obs <- y_obs[ord1]
  locsord_obs <- locs_obs[ord1, , drop = FALSE]
  locsord_pred <- locs_pred[ord2, , drop = FALSE]
  
  # put all coordinates together
  locs_all <- rbind(locsord_obs, locsord_pred)
  inds1 <- 1:n_obs
  inds2 <- (n_obs+1):(n_obs+n_pred)
  
  # get nearest neighbor array (expensive operation)
  NNarray_all <- GpGp::find_ordered_nn(locs_all, m = num_neighbours, lonlat = get_linkfun(covfun_name)$lonlat)
  
  # Store everything needed for simulation
  setup <- list(
    y = y,
    yord_obs = yord_obs,
    locs_all = locs_all,
    NNarray_all = NNarray_all,
    inds1 = inds1,
    inds2 = inds2,
    ord2 = ord2,
    n_obs = n_obs,
    n_pred = n_pred,
    na_indices = na_indices,
    na_indices_cartesian = which(is.na(y), arr.ind = TRUE),
    covfun_name = covfun_name
  )
  
  return(setup)
}

# Conditional simulation (can be called multiple times with different theta)
simulate_conditional <- function(setup, theta, nsims = 1) {
  
  rho <- theta[1]
  tau <- theta[2]
  nu <- 1     # smoothness
  sigma2 <- 1 # marginal variance
  covparms <- c(sigma2, rho, nu, tau^2)
  
  if (any(is.na(covparms)) || any(!is.finite(covparms))) {
    stop(sprintf("Invalid covparms: rho=%.4f, tau=%.4f", rho, tau))
  }
  
  if (rho <= 0 || tau < 0) {
    stop(sprintf("Invalid parameter values: rho=%.4f (must be >0), tau=%.4f (must be >=0)", 
                 rho, tau))
  }
  
  # Extract precomputed objects
  y <- setup$y
  yord_obs <- setup$yord_obs
  locs_all <- setup$locs_all
  NNarray_all <- setup$NNarray_all
  inds1 <- setup$inds1
  inds2 <- setup$inds2
  ord2 <- setup$ord2
  n_obs <- setup$n_obs
  n_pred <- setup$n_pred
  na_indices_cartesian <- setup$na_indices_cartesian
  covfun_name <- setup$covfun_name
  
  # get entries of Linv for obs locations and pred locations
  Linv_all <- vecchia_Linv(covparms, covfun_name, locs_all, NNarray_all)
  
  # Conditional simulations
  condsim <- matrix(NA, n_pred, nsims)
  for(j in 1:nsims){
    # Unconditional simulation
    z <- L_mult(Linv_all, stats::rnorm(n_obs + n_pred), NNarray_all)

    # Conditional simulation
    y_withzeros <- c(yord_obs + z[inds1], rep(0, n_pred))
    v1 <- Linv_mult(Linv_all, y_withzeros, NNarray_all)
    v1[inds1] <- 0
    v2 <- -L_mult(Linv_all, v1, NNarray_all)
    condsim[ord2, j] <- c(v2[inds2]) - z[inds2]
  }

  # Combine conditional simulation with observed data
  y_array <- array(y, dim = c(nrow(y), ncol(y), nsims))
  for(j in 1:nsims){
    na_indices_cartesian_j <- cbind(na_indices_cartesian, j=j)
    y_array[na_indices_cartesian_j] <- condsim[, j]
  }

  # Return in format suitable for CNN architecture, a 4D array
  y_array <- array(y_array, c(dim(y_array)[1], dim(y_array)[2], 1, dim(y_array)[3]))

  return(y_array)
}

## Wrapper function for conditional simulation from scratch
# Sanity check:
# theta <- c("rho" = 0.3, "tau" = 0)
# covparms <- c(1, theta["rho"], 1, theta["tau"])
# N <- 64                                                            # number of points in each direction
# locs <- as.matrix(expand.grid((1:N)/N, (1:N)/N))                   # spatial locations
# y <- matrix(fast_Gp_sim(covparms, "matern_isotropic", locs), N, N) # complete data
# y[1:(N/4),] <- NA                                                  # incomplete data
# z <- simulateconditional(y, theta, nsims = 30)                    # conditionally-completed data
# par(mfrow = c(1, 3))
# zlim_range <- range(c(y, z[, , 1, 1], z[, , 1, 2]), na.rm = TRUE)
# fields::image.plot(y, zlim = zlim_range, main = "Incomplete data")
# fields::image.plot(z[, , 1, 1], zlim = zlim_range, main = "Conditional simulation 1")
# fields::image.plot(z[, , 1, 2], zlim = zlim_range, main = "Conditional simulation 2")
simulateconditional <- function(y, theta, nsims = 1, num_neighbours = 30, reorder = TRUE) {
  setup <- vecchia_setup(y, num_neighbours, reorder)
  simulate_conditional(setup, theta, nsims)
}

## MAP estimation
nll_fun <- function(par, setup) {
  log_rho <- par[1]
  log_tau <- par[2]
  covparms <- c(1, exp(log_rho), 1, exp(log_tau)^2)
  
  Linv <- vecchia_Linv(
    covparms,
    setup$covfun_name,
    setup$locs_obs,
    setup$NNarray_obs
  )
  
  v <- Linv_mult(Linv, setup$yord_obs, setup$NNarray_obs)
  quad_form <- sum(v^2)
  
  # Extract diagonal of L^{-1}
  # Linv[i, 1] is the diagonal entry of row i
  diag_vals <- Linv[, 1]
  
  # log|Σ| = -2 * log|L^{-1}| = -2 * sum(log(diag(L^{-1})))
  logdet_Sigma <- -2 * sum(log(abs(diag_vals)))
  
  # Log-likelihood (without constant)
  ll <- -0.5 * logdet_Sigma - 0.5 * quad_form
  
  return(-ll)  # Return negative log-likelihood for minimization
}

MAP <- function (y) {
  
  # Initial parameters for optimization
  par <- log(prior_mean)
  
  setup <- vecchia_setup(drop(y)) 
  
  setup_ll <-  list(
    yord_obs = setup$yord_obs,
    locs_obs = setup$locs_all[setup$inds1, , drop = FALSE],
    NNarray_obs = setup$NNarray_all[setup$inds1, ],
    covfun_name = setup$covfun_name
  )
  
  optim_result <- optim(
    par = par,
    fn = nll_fun, 
    setup = setup_ll, 
    lower = log(c(rho_limits[1], tau_limits[1])),
    upper = log(c(rho_limits[2], tau_limits[2])), 
    method = 'L-BFGS-B',
    control = list(parscale = c(rep(1, 2)))
  )
  
  # Extract parameter estimates
  par <- optim_result$par
  rho_est <- exp(par[1])
  tau_est <- exp(par[2])
  est <- c(rho_est, tau_est)
  
  return(est)
}

MAP_multiple <- function(Z) {
  thetahat <- lapply_function(Z, MAP)
  thetahat <- do.call(cbind, thetahat)
  return(thetahat)
}

abc_mode <- function(z1, S, u, sumstats, sumstats_model, theta, method = "neuralnet") {
  
  # Flatten data and remove NAs from vector and spatial locations
  z1 <- c(z1)
  I1 <- which(!is.na(z1))
  z1 <- z1[I1]
  S1 <- S[I1, , drop = FALSE]
  
  # Observed summary statistics
  target <- variog(coords = S1, data = z1, uvec = u, messages = FALSE)$v
  target <- sumstats_model$B %*% target

  # Suppress all output types to keep console clean
  invisible(capture.output({suppressWarnings({suppressMessages({
    
    # ABC sampling
    object <- abc(
      target = c(target),
      param = t(theta), 
      sumstat = t(sumstats),
      tol = 0.01, 
      method = method
    )
    
    # Compute approximate posterior mode
    if (method == "rejection") {
      samples <-  object$unadj.values
      mode <- apply(samples, 2, abc:::getmode)
    } else {
      samples <-  object$adj.values
      wt <- object$weights
      mode <- apply(samples, 2, abc:::getmode, wt/sum(wt))
    }
  
  })})}, type = "output"))
  
  return(mode)
}

abc_mode_multiple <- function(Z, ...) {
  thetahat <- lapply_function(Z, abc_mode, ...)
  thetahat <- do.call(cbind, thetahat)
  return(thetahat)
}


# ---- Training ----

cat("Simulating training data...\n")

## Simulate training data 
set.seed(1)
K <- ifelse(quick, 1000, 25000)    # size of the training set 
theta_train <- sampler(K)          # parameter vectors used in stochastic-gradient descent during training
theta_val   <- sampler(K %/% 2)       # parameter vectors used to monitor performance during training
tm <- system.time({
  Z_train <- simulator(theta_train)  # data used in stochastic-gradient descent during training
  Z_val   <- simulator(theta_val)    # data used to monitor performance during training
})
saveRDS(tm, file = file.path(int_path, "sim_time.rds"))

## Construct data sets for masking approach
UW_train <- encodedata(lapply(Z_train, removedata))
UW_val   <- encodedata(lapply(Z_val, removedata))

## Maximum number of epochs 
epochs <- ifelse(quick, 10, 100)

## Train the neural MAP estimator for use in the EM approach
neuralMAP <- architecture(d, input_channels = 1L) #NB should explicitly enforce prior bounds
neuralMAP <- train(
  neuralMAP,      
  theta_train = theta_train, 
  theta_val = theta_val, 
  Z_train = Z_train, 
  Z_val = Z_val, 
  epochs = epochs,
  savepath = file.path(int_path, "runs_EM")
)

## Train the masked neural Bayes estimator
maskedestimator <- architecture(d, input_channels = 2L) #NB should explicitly enforce prior bounds
maskedestimator <- train(
  maskedestimator,     
  theta_train = theta_train, 
  theta_val = theta_val, 
  Z_train = UW_train, 
  Z_val = UW_val, 
  epochs = epochs, 
  savepath = file.path(int_path, "runs_masking")
)

# ---- Load the trained neural networks ----

neuralMAP <- juliaLet('
  estimator = architecture(d, 1)
  loadpath  = joinpath(pwd(), int_path, "runs_EM", "ensemble.bson")
  @load loadpath model_state
  Flux.loadmodel!(estimator, model_state)
  estimator
', d = d, int_path = int_path)

maskedestimator <- juliaLet('
  estimator = architecture(d, 2)
  loadpath  = joinpath(pwd(), int_path, "runs_masking", "ensemble.bson")
  @load loadpath model_state
  Flux.loadmodel!(estimator, model_state)
  estimator
', d = d, int_path = int_path)


# ---- ABC prep ----

abc_path <- file.path(int_path, "ABC")
dir.create(abc_path, recursive = TRUE, showWarnings = FALSE)

# Prepare data to compute summary statistics
N <- dim(Z_train[[1]])[1]
S <- as.matrix(expand.grid(seq(1, 0, len = N), seq(0, 1, len = N)))
Z_train <- lapply(Z_train, c)
u <- seq(0.04, 0.4, length.out = 10) # distances used for binning

# Compute summary statistics from complete data
tm <- system.time({
  sumstats_train <- lapply_function(Z_train, function(z) {
    variog(coords = S, data = z, uvec = u, messages = FALSE)$v
  })
})
sumstats_train <- do.call(cbind, sumstats_train)
sumstats_model <- saABC(t(theta_train), t(sumstats_train))
sumstats_train <- sumstats_model$B %*% sumstats_train

saveRDS(tm, file = file.path(abc_path, "sumstats_time.rds"))
saveRDS(sumstats_train, file = file.path(abc_path, "sumstats_train.rds"))
saveRDS(sumstats_model, file = file.path(abc_path, "sumstats_model.rds"))
saveRDS(theta_train, file = file.path(abc_path, "theta_train.rds"))

# ---- Assessment with missing data ----

## Simulate testing data
set.seed(1)
K_test <- ifelse(quick, 100, 1000)
theta_test <- sampler(K_test)       
Z_test <- simulator(theta_test)
saveRDS(theta_test, file = file.path(int_path, "theta_test.rds"))
saveRDS(Z_test, file = file.path(int_path, "Z_test.rds"))

# Incomplete data
set.seed(1)
Z1_MCAR <- lapply(Z_test, removedata, proportion = 0.2)
Z1_MNAR <- lapply(Z_test, remove_quarter_circle)
Z1_MNAR <- lapply(Z1_MNAR, add_singletons)

# Encoded data set for masking approach
UW_MCAR <- encodedata(Z1_MCAR)
UW_MNAR <- encodedata(Z1_MNAR)

## Hyperparameters of EM approach
niterations <- ifelse(quick, 3, 50)
nsims <- ifelse(quick, 3, 30)
burn_in <- ifelse(quick, 1, 5)
theta_0 <- prior_mean

## Estimation over the test set
set.seed(1)
cat("Running masking NBE...\n")
masked_MCAR <- estimate(maskedestimator, UW_MCAR)
masked_MNAR <- estimate(maskedestimator, UW_MNAR)
cat("Running EM NBE...\n")
EM_MCAR <- EM_multiple(Z1_MCAR, setupconditionalsimulation = vecchia_setup, simulateconditional = simulate_conditional, estimator = neuralMAP, nsims = nsims, niterations = niterations, burn_in = burn_in, theta_0 = masked_MCAR)
EM_MNAR <- EM_multiple(Z1_MNAR, setupconditionalsimulation = vecchia_setup, simulateconditional = simulate_conditional, estimator = neuralMAP, nsims = nsims, niterations = niterations, burn_in = burn_in, theta_0 = masked_MNAR)
cat("Computing MAP estimates...\n")
MAP_MCAR <- MAP_multiple(Z1_MCAR)
MAP_MNAR <- MAP_multiple(Z1_MNAR)
cat("Computing ABC MAP estimates...\n")
ABCMAP_MCAR <- abc_mode_multiple(Z1_MCAR, S = S, u = u, sumstats = sumstats_train, sumstats_model = sumstats_model, theta = theta_train)
ABCMAP_MNAR <- abc_mode_multiple(Z1_MNAR, S = S, u = u, sumstats = sumstats_train, sumstats_model = sumstats_model, theta = theta_train)

saveRDS(masked_MCAR, file = file.path(int_path, "Estimates", "Test", "masked_MCAR.rds"))
saveRDS(masked_MNAR, file = file.path(int_path, "Estimates", "Test", "masked_MNAR.rds"))
saveRDS(EM_MCAR, file = file.path(int_path, "Estimates", "Test", "EM_MCAR.rds"))
saveRDS(EM_MNAR, file = file.path(int_path, "Estimates", "Test", "EM_MNAR.rds"))
saveRDS(MAP_MCAR, file = file.path(int_path, "Estimates", "Test", "MAP_MCAR.rds"))
saveRDS(MAP_MNAR, file = file.path(int_path, "Estimates", "Test", "MAP_MNAR.rds"))
saveRDS(ABCMAP_MCAR, file = file.path(int_path, "Estimates", "Test", "ABCMAP_MCAR.rds"))
saveRDS(ABCMAP_MNAR, file = file.path(int_path, "Estimates", "Test", "ABCMAP_MNAR.rds"))
saveRDS(theta_test, file = file.path(int_path, "Estimates", "Test", "theta_test.rds"))

estimates_dataframe <- function(estimates, truth, estimator_name, parameter_names) {
  
  d <- nrow(truth)
  K <- ncol(truth)
  
  data.frame(
    m = 1, 
    k = rep(1:K, times = d),
    j = 1,
    estimator = estimator_name, 
    parameter = rep(parameter_names, each = K),
    estimate = c(t(estimates)), 
    truth = c(t(truth))
  )
}

MCAR_df <- estimates_dataframe(masked_MCAR, theta_test, "masking", parameter_names) %>% 
  rbind(estimates_dataframe(EM_MCAR, theta_test, "neuralEM", parameter_names)) %>%
  rbind(estimates_dataframe(MAP_MCAR, theta_test, "MAP", parameter_names))  %>%
  rbind(estimates_dataframe(ABCMAP_MCAR, theta_test, "ABC MAP", parameter_names))  %>%
  mutate(missingness = "MCAR")

MNAR_df <- estimates_dataframe(masked_MNAR, theta_test, "masking", parameter_names) %>% 
  rbind(estimates_dataframe(EM_MNAR, theta_test, "neuralEM", parameter_names)) %>%
  rbind(estimates_dataframe(MAP_MNAR, theta_test, "MAP", parameter_names))  %>%
  rbind(estimates_dataframe(ABCMAP_MNAR, theta_test, "ABC MAP", parameter_names))  %>%
  mutate(missingness = "MNAR")

df <- rbind(MCAR_df, MNAR_df)
write.csv(df, file.path(int_path, "Estimates", "estimates_test.csv"), row.names = F)

rmse_df <- df %>%
  anti_join(df %>% filter(parameter == "rho", truth > 0.275) %>% distinct(k), by = "k") %>% # Remove very large rho, likelihood too flat in that region and analytical MAP becomes numerically unstable
  mutate(squared_error = (estimate - truth)^2) %>%
  group_by(estimator, missingness) %>% 
  dplyr::summarise(
    rmse = sqrt(mean(squared_error)),
    bias = mean(estimate - truth)
  )
write.csv(rmse_df, file.path(int_path, "Estimates", "rmse.csv"), row.names = F)

rmse_df <- df %>%
  mutate(squared_error = (estimate - truth)^2) %>%
  group_by(estimator, missingness, parameter) %>% 
  dplyr::summarise(rmse = sqrt(mean(squared_error)), bias = mean(estimate - truth)) 
write.csv(rmse_df, file.path(int_path, "Estimates", "rmse_by_parameter.csv"), row.names = F)

## Sampling distributions 
set.seed(1)
J <- ifelse(quick, 10, 50)
rho <- qunif(c(0.1, 0.5, 0.9), 0.05, 0.3)
tau <- qunif(c(0.1, 0.5, 0.9))
theta <- t(as.matrix(expand.grid(rho, tau)))
Z <- lapply(1:J, function(j) simulator(theta))  
Z <- unlist(Z, recursive = FALSE)

Z1_MCAR <- lapply(Z, removedata, proportion = 0.2)
Z1_MNAR <- lapply(Z, remove_quarter_circle)
Z1_MNAR <- lapply(Z1_MNAR, add_singletons)

UW_MCAR <- encodedata(Z1_MCAR)
UW_MNAR <- encodedata(Z1_MNAR)

savedata <- function(Z, theta, missingness) {
  K <- ncol(theta)
  Z <- Z[1:K] # just save the first data set for each field
  J <- length(Z) / K
  z <- lapply(Z, c) 
  z <- unlist(z)
  n <- prod(dim(Z[[1]]))
  k <- rep(rep(1:K, each = J), each = n)
  j <- rep(rep(1:J, each = n), times = K)
  df <- data.frame(Z = z, k = k, j = j)
  save_path <- file.path(int_path, "Estimates", paste0("Z_", missingness, ".csv"))
  write.csv(df, save_path, row.names = FALSE)
}
savedata(Z1_MCAR, theta, "MCAR")
savedata(Z1_MNAR, theta, "MNAR")

cat("Running masking NBE...\n")
masked_MCAR <- estimate(maskedestimator, UW_MCAR)
masked_MNAR <- estimate(maskedestimator, UW_MNAR)
cat("Running EM NBE...\n")
EM_MCAR <- EM_multiple(Z1_MCAR, setupconditionalsimulation = vecchia_setup, simulateconditional = simulate_conditional, estimator = neuralMAP, nsims = nsims, niterations = niterations, burn_in = burn_in, theta_0 = masked_MCAR)
EM_MNAR <- EM_multiple(Z1_MNAR, setupconditionalsimulation = vecchia_setup, simulateconditional = simulate_conditional, estimator = neuralMAP, nsims = nsims, niterations = niterations, burn_in = burn_in, theta_0 = masked_MNAR)
cat("Computing MAP estimates...\n")
MAP_MCAR <- MAP_multiple(Z1_MCAR)
MAP_MNAR <- MAP_multiple(Z1_MNAR)
cat("Computing ABC MAP estimates...\n")
ABCMAP_MCAR <- abc_mode_multiple(Z1_MCAR, S = S, u = u, sumstats = sumstats_train, sumstats_model = sumstats_model, theta = theta_train)
ABCMAP_MNAR <- abc_mode_multiple(Z1_MNAR, S = S, u = u, sumstats = sumstats_train, sumstats_model = sumstats_model, theta = theta_train)

saveRDS(masked_MCAR, file = file.path(int_path, "Estimates", "Scenarios", "masked_MCAR.rds"))
saveRDS(masked_MNAR, file = file.path(int_path, "Estimates", "Scenarios", "masked_MNAR.rds"))
saveRDS(EM_MCAR, file = file.path(int_path, "Estimates", "Scenarios", "EM_MCAR.rds"))
saveRDS(EM_MNAR, file = file.path(int_path, "Estimates", "Scenarios", "EM_MNAR.rds"))
saveRDS(MAP_MCAR, file = file.path(int_path, "Estimates", "Scenarios", "MAP_MCAR.rds"))
saveRDS(MAP_MNAR, file = file.path(int_path, "Estimates", "Scenarios", "MAP_MNAR.rds"))
saveRDS(ABCMAP_MCAR, file = file.path(int_path, "Estimates", "Scenarios", "ABCMAP_MCAR.rds"))
saveRDS(ABCMAP_MNAR, file = file.path(int_path, "Estimates", "Scenarios", "ABCMAP_MNAR.rds"))
saveRDS(theta, file = file.path(int_path, "Estimates", "Scenarios", "theta.rds"))

estimates_dataframe <- function(estimates, truth, estimator_name, parameter_names) {
  
  d <- nrow(truth)
  K <- ncol(truth)
  J <- ncol(estimates) / K
  
  truth_repeated <- matrix(rep(truth, J), nrow = nrow(truth))
  
  data.frame(
    m = 1, 
    k = rep(1:K, each = d),
    j = rep(1:J, each = d*K),
    estimator = estimator_name, 
    parameter = rep(parameter_names, times = K),
    estimate = c(estimates), 
    truth = c(truth_repeated)
  )
}

MCAR_df <- estimates_dataframe(masked_MCAR, theta, "masking", parameter_names) %>% 
  rbind(estimates_dataframe(MAP_MCAR, theta, "MAP", parameter_names)) %>% 
  rbind(estimates_dataframe(EM_MCAR, theta, "neuralEM", parameter_names)) %>%
  rbind(estimates_dataframe(ABCMAP_MCAR, theta, "ABC MAP", parameter_names)) %>%
  mutate(missingness = "MCAR")

MNAR_df <- estimates_dataframe(masked_MNAR, theta, "masking", parameter_names) %>% 
  rbind(estimates_dataframe(MAP_MNAR, theta, "MAP", parameter_names)) %>%
  rbind(estimates_dataframe(EM_MNAR, theta, "neuralEM", parameter_names))  %>%
  rbind(estimates_dataframe(ABCMAP_MNAR, theta, "ABC MAP", parameter_names))  %>%
  mutate(missingness = "MNAR")

write.csv(MCAR_df, file.path(int_path, "Estimates", "estimates_MCAR_scenarios.csv"), row.names = F)
write.csv(MNAR_df, file.path(int_path, "Estimates", "estimates_MNAR_scenarios.csv"), row.names = F)
write.csv(rbind(MCAR_df, MNAR_df), file.path(int_path, "Estimates", "estimates_scenarios.csv"), row.names = F)


# ---- Timings for a single data set ----

df <- tibble(estimator = character(), time = numeric())

# MAP
start_time <- Sys.time()
MAP(Z1_MCAR[[1]])
end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
df <- df %>% add_row(estimator = "MAP", time = elapsed_time)

# ABC
start_time <- Sys.time()
abc_mode(Z1_MCAR[[1]], S = S, u = u, sumstats = sumstats_train, sumstats_model = sumstats_model, theta = theta_train)
end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
df <- df %>% add_row(estimator = "ABC MAP", time = elapsed_time)

# EM
start_time <- Sys.time()
em_result <- EM(Z1_MCAR[[1]], estimator = neuralMAP, setupconditionalsimulation = vecchia_setup, simulateconditional = simulate_conditional, nsims = 30, niterations = 50, burn_in = 2, theta_0 = prior_mean)
end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
df <- df %>% add_row(estimator = "EM_with_overhead", time = elapsed_time)

# Get a more realistic time (without overhead) by accurately computing time for a 
# single estimate + single conditional simulation, and then multiply by the 
# total number of iterations
num_its <- ncol(em_result$iterates)
Z_complete <- Z_test[[1]]
Z_complete <- array(Z_complete, dim = c(dim(Z_complete)[1], dim(Z_complete)[2], 1, 30))
estimation_time <- juliaLet('@belapsed estimate($estimator, $z)', estimator = neuralMAP, z = Z_complete)
setup_time <- system.time(setup <- vecchia_setup(drop(Z1_MCAR[[1]])))["elapsed"]
condsim_time <- system.time(simulate_conditional(setup, prior_mean, nsims = 30))["elapsed"]
total_time <- setup_time + num_its * (condsim_time + estimation_time)
df <- df %>% add_row(estimator = "EM_without_overhead", time = total_time)

# Masking NBE
start_time <- Sys.time()
estimate(maskedestimator, UW_MCAR[[1]])
end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
df <- df %>% add_row(estimator = "masking_with_overhead", time = elapsed_time)
elapsed_time <- juliaLet('@belapsed estimate($estimator, $z)', estimator = maskedestimator, z = UW_MCAR[[1]])
df <- df %>% add_row(estimator = "masking_without_overhead", time = elapsed_time)

write.csv(df, file.path(int_path, "runtime_singledataset.csv"))

# ---- Results ----

estimates_path <- file.path(int_path, "Estimates")
missingness <- c("MCAR", "MNAR")
estimator_labels <- c(
  "MAP" = "MAP",
  "neuralEM" = "EM NBE",
  "masking" = "Masking NBE"#, 
  #"ABC MAP" = "ABC MAP"
)

parameter_labels = c(
  "τ" = expression(hat(tau)), 
  "tau" = expression(hat(tau)), 
  "ρ" = expression(hat(rho)),
  "rho" = expression(hat(rho))
)

loadestimates <- function(type) {
  df <- file.path(estimates_path, paste0("estimates_", type, "_scenarios.csv")) %>% read.csv
  df$missingness <- type
  df
}

loaddata <- function(type) {
  df <- file.path(estimates_path, paste0("Z_", type, ".csv")) %>% read.csv
  df$missingness <- type
  df
}

df <- loadestimates(missingness[1]) %>% 
  rbind(loadestimates(missingness[2])) %>% 
  filter(estimator %in% names(estimator_labels))

d <- sum(names(parameter_labels) %in% df$parameter)

zdf <- loaddata(missingness[1]) %>% rbind(loaddata(missingness[2]))

N <- 64 # nrow(Z_test[[1]])
zdf$x <- rep(seq(0, 1, len=N), each = N) 
zdf$y <- seq(0, 1, len=N)

figures <- lapply(unique(df$k), function(kk) {
  
  df  <- df  %>% filter(k == kk)
  zdf <- zdf %>% filter(k == kk)
  
  l <- length(missingness) # number of missingness patterns
  
  ## Data plots
  suppressMessages({
    data <- lapply(missingness, function(mis) {
      field_plot(filter(zdf, j == 1, missingness == mis), regular = T) + 
        scale_x_continuous(breaks = c(0.2, 0.5, 0.8), expand = c(0, 0)) +
        scale_y_continuous(breaks = c(0.2, 0.5, 0.8), expand = c(0, 0)) +
        labs(fill = "Z") +
        theme(legend.title.align = 0.25, legend.title = element_text(face = "bold"))
    })
  })
  data_legend <- get_legend(data[[1]])
  data <- lapply(data, function(gg) gg + theme(legend.position = "none") + coord_fixed())
  
  suppressMessages({
    data[-l] <- lapply(data[-l], function(gg) gg +
                         theme(axis.text.x = element_blank(),
                               axis.ticks.x = element_blank(),
                               axis.title.x = element_blank()) +
                         scale_x_continuous(breaks = c(0.2, 0.5, 0.8), expand = c(0, 0))
    )
  })
  
  ## Box plots
  df$estimator <- factor(df$estimator, levels = c("MAP", "neuralEM", "masking", "ABC MAP"))
  box <- lapply(missingness, function(mis) {
    plotdistribution(filter(df, missingness == mis), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1, return_list = TRUE, flip = TRUE)
  })
  d <- length(unique(df$parameter))
  box_split <- lapply(1:d, function(i) {
    lapply(1:length(box), function(j) box[[j]][[i]])
  })
  
  # Modify the axes
  for (i in 1:d) {
    box_split[[i]][[l]] <- box_split[[i]][[l]] + labs(y = box_split[[i]][[1]]$labels$x)
    
    # Remove axis labels from internal panels
    box_split[[i]][-l] <- lapply(box_split[[i]][-l], function(gg) gg +
                                   theme(axis.text.x = element_blank(),
                                         axis.ticks.x = element_blank(), 
                                         axis.title.x = element_blank()))
    
    # Ensure axis limits are consistent for all panels for a given parameter
    ylims <- df %>% filter(parameter == unique(df$parameter)[i]) %>% summarise(range(estimate)) %>% as.matrix %>% c
    box_split[[i]] <- lapply(box_split[[i]], function(gg) gg + ylim(ylims))
  }
  
  box <- do.call(c, box_split)
  suppressMessages({
    box <- lapply(box, function(gg) gg + scale_estimator_aesthetic(df))
    box_legend <- get_legend(box[[1]])
    box <- lapply(box, function(gg) {
      gg$facet$params$nrow <- 2
      gg$facet$params$strip.position <- "bottom"
      gg <- gg + theme(legend.position = "none", axis.title.y = element_blank()) #+ scale_estimator(df)
      gg
    })
  })
  
  nrow <- length(missingness)
  legends <- list(ggplotify::as.ggplot(data_legend), ggplotify::as.ggplot(box_legend))
  plotlist <- c(data, box, legends)
  ncol <- d + 2
  figure  <- egg::ggarrange(plots = plotlist, nrow = nrow, ncol = ncol, byrow = FALSE)
  
  ggsv(file = paste0("boxplots_k", kk), plot = figure, path = img_path, 
       width = 2 * (d+2), height = 4.5)
  
})


# ---- Convergence analysis ----

set.seed(1)
theta <- t(t(c(0.15, 0.4)))
Z <- simulator(theta)[[1]][, , 1, 1]
fields::image.plot(Z)
Z1 <- removedata(Z, proportion =  0.2)
fields::image.plot(Z1)

theta_0 <- list(c(0.05, 0.1), c(0.3, 0.7))
all_nsims <- list(1, 10, 30)
df <- run_EM(Z1)
write.csv(df, file = file.path(int_path, "EM_iterates.csv"))
df <- read_csv(file.path(int_path, "EM_iterates.csv"))
figure1 <- plot_EM_trajectories(df, parameter_labels)
figure2 <- plot_EM_trajectories(df, parameter_labels, H = 30)
ggsv("convergence_multipleMC", figure1, path = img_path, width = 7.5, height = 4)
ggsv("convergence_singleMC", figure2, path = img_path, width = 6.5, height = 3)