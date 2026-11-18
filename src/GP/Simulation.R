# Gaussian process model of Section 3.2: prior, marginal simulation, conditional simulation and MAP estimation with
# the Vecchia approximation of GpGp (maxmin ordering, 30 nearest neighbours). This file only defines functions and
# constants, so that it can be sourced by parallel workers.

suppressMessages({
  library("GpGp")
  library("Matrix")
})

# ---- Prior ----

rho_limits <- c(0.03, 0.35)    # range parameter
tau_limits <- c(0.01, 1)       # fine-scale (nugget) standard deviation
prior_mean <- c(mean(rho_limits), mean(tau_limits))

## Sampling from the prior distribution (K: number of samples)
sampler <- function(K) {
  rho <- runif(K, min = rho_limits[1], max = rho_limits[2])
  tau <- runif(K, min = tau_limits[1], max = tau_limits[2])
  matrix(c(rho, tau), nrow = 2, byrow = TRUE)
}

## Covariance parameters in GpGp's "matern_isotropic" parameterisation: (variance, range, smoothness, nugget)
covparms <- function(theta) {
  rho <- theta[1]
  tau <- theta[2]
  if (any(!is.finite(theta)) || rho <= 0 || tau < 0) {
    stop(sprintf("Invalid parameter values: rho=%.4f (must be >0), tau=%.4f (must be >=0)", rho, tau))
  }
  c(1, rho, 1, tau^2)
}

gridlocations <- function(N) as.matrix(expand.grid(seq(1, 0, len = N), seq(0, 1, len = N)))

# ---- Marginal simulation ----

## theta: a matrix of parameters drawn from the prior; returns a list of N × N × 1 × 1 arrays
simulator <- function(theta, N = 64, lapply_function = lapply) {
  locs <- gridlocations(N)
  lapply_function(1:ncol(theta), function(k) {
    Z <- fast_Gp_sim(covparms(theta[, k]), "matern_isotropic", locs)
    array(Z, dim = c(N, N, 1, 1))
  })
}

# ---- Conditional simulation ----

## Set-up for conditional simulation (done once per data set y, regardless of theta). The observed locations are
## ordered first (maxmin ordering) and the unobserved locations last (maxmin ordering), and each location conditions
## on its num_neighbours nearest previous locations.
vecchia_setup <- function(y, num_neighbours = 30, reorder = TRUE) {

  covfun_name <- "matern_isotropic"
  y <- drop(y)

  N <- nrow(y) # NB assumes square grid
  locs <- gridlocations(N)

  # Observed data, observed locations, and unobserved locations
  non_na_indices <- which(!is.na(y))
  na_indices <- which(is.na(y))
  y_obs <- y[non_na_indices]
  locs_obs <- locs[non_na_indices, , drop = FALSE]
  locs_pred <- locs[na_indices, , drop = FALSE]
  n_obs <- nrow(locs_obs)
  n_pred <- nrow(locs_pred)

  if (n_pred == 0) return(list(y = y, complete = TRUE))

  # Orderings
  if (reorder) {
    ord1 <- if (n_obs < 6e4) GpGp::order_maxmin(locs_obs) else sample(1:n_obs)
    ord2 <- if (n_pred < 6e4) GpGp::order_maxmin(locs_pred) else sample(1:n_pred)
  } else {
    ord1 <- 1:n_obs
    ord2 <- 1:n_pred
  }
  locs_all <- rbind(locs_obs[ord1, , drop = FALSE], locs_pred[ord2, , drop = FALSE])

  # Nearest-neighbour array (expensive)
  NNarray_all <- GpGp::find_ordered_nn(locs_all, m = num_neighbours, lonlat = get_linkfun(covfun_name)$lonlat)

  # Sparsity pattern of the rows of the inverse Cholesky factor L⁻¹ that belong to the unobserved locations,
  # split into the columns of the observed (B21) and unobserved (B22) locations
  n <- n_obs + n_pred
  NN2 <- NNarray_all[(n_obs + 1):n, , drop = FALSE]
  ok <- !is.na(NN2)
  i <- row(NN2)[ok]
  j <- NN2[ok]
  obs_col <- j <= n_obs

  list(
    y = y,
    complete = FALSE,
    yord_obs = y_obs[ord1],
    locs_all = locs_all,
    NNarray_all = NNarray_all,
    nz = ok,
    obs_col = obs_col,
    i21 = i[obs_col], j21 = j[obs_col],
    i22 = i[!obs_col], j22 = j[!obs_col] - n_obs,
    n_obs = n_obs,
    n_pred = n_pred,
    pred_indices = na_indices[ord2], # linear index (in y) of each unobserved location, in the simulation ordering
    covfun_name = covfun_name
  )
}

## Conditional simulation (can be called multiple times with different theta). Write the Vecchia approximation of
## the precision matrix of the ordered data (observed locations first) as B'B, with B = L⁻¹ lower triangular and
## sparse. Then Z₂ | Z₁ = z₁ ~ N(−B₂₂⁻¹B₂₁z₁, B₂₂⁻¹B₂₂⁻ᵀ), so that only the rows of B for the unobserved locations are
## needed: these are computed by vecchia_Linv(..., start_ind) and all nsims simulations are obtained with one sparse
## triangular solve. This is the same conditional distribution as that sampled by GpGp's cond_sim() (and by
## simulate_conditional_full() below), at a fraction of the cost.
## Returns an N × N × 1 × nsims array, the format used by the neural network.
simulate_conditional <- function(setup, theta, nsims = 1) {

  y <- setup$y
  if (isTRUE(setup$complete)) return(array(y, dim = c(dim(y), 1, nsims)))

  B <- conditional_factors(setup, theta)
  rhs <- matrix(stats::rnorm(setup$n_pred * nsims), setup$n_pred, nsims) - as.numeric(B$B21 %*% setup$yord_obs)
  condsim <- as.matrix(solve(B$B22, rhs))

  # Combine the conditional simulations with the observed data
  ymat <- matrix(y, length(y), nsims)
  ymat[setup$pred_indices, ] <- condsim
  array(ymat, dim = c(dim(y), 1, nsims))
}

## The blocks B21 (sparse) and B22 (sparse, lower triangular) of the rows of L⁻¹ for the unobserved locations
conditional_factors <- function(setup, theta) {
  n1 <- setup$n_obs
  n2 <- setup$n_pred
  Linv <- vecchia_Linv(covparms(theta), setup$covfun_name, setup$locs_all, setup$NNarray_all, start_ind = n1 + 1L)
  x <- Linv[(n1 + 1):(n1 + n2), , drop = FALSE][setup$nz]
  list(
    B21 = sparseMatrix(i = setup$i21, j = setup$j21, x = x[setup$obs_col], dims = c(n2, n1)),
    B22 = as(sparseMatrix(i = setup$i22, j = setup$j22, x = x[!setup$obs_col], dims = c(n2, n2)), "triangularMatrix")
  )
}

## Reference implementation: the algorithm of GpGp's cond_sim(), which uses all rows of L⁻¹ (used in tests only)
simulate_conditional_full <- function(setup, theta, nsims = 1, E = matrix(stats::rnorm((setup$n_obs + setup$n_pred) * nsims), ncol = nsims)) {
  n1 <- setup$n_obs
  n <- n1 + setup$n_pred
  Linv <- vecchia_Linv(covparms(theta), setup$covfun_name, setup$locs_all, setup$NNarray_all)
  ok <- !is.na(setup$NNarray_all)
  B <- as(sparseMatrix(i = row(setup$NNarray_all)[ok], j = setup$NNarray_all[ok], x = Linv[ok], dims = c(n, n)), "triangularMatrix")
  Z  <- solve(B, E)                                           # unconditional simulations
  Y0 <- rbind(setup$yord_obs + Z[1:n1, , drop = FALSE], matrix(0, n - n1, ncol(E)))
  V1 <- as.matrix(B %*% Y0)
  V1[1:n1, ] <- 0
  -as.matrix(solve(B, V1))[(n1 + 1):n, , drop = FALSE] - as.matrix(Z[(n1 + 1):n, , drop = FALSE])
}

## Wrapper for conditional simulation from scratch (see the sanity check in the comments below)
# theta <- c(0.3, 0.1); N <- 64
# y <- matrix(fast_Gp_sim(covparms(theta), "matern_isotropic", gridlocations(N)), N, N)
# y[1:(N/4), ] <- NA
# z <- simulateconditional(y, theta, nsims = 2)
# par(mfrow = c(1, 3)); fields::image.plot(y); fields::image.plot(z[, , 1, 1]); fields::image.plot(z[, , 1, 2])
simulateconditional <- function(y, theta, nsims = 1, num_neighbours = 30, reorder = TRUE) {
  simulate_conditional(vecchia_setup(y, num_neighbours, reorder), theta, nsims)
}

# ---- MAP estimation ----

## Set-up for the likelihood of the observed data: maxmin ordering and nearest neighbours of the observed locations
## (the same neighbour sets as the first block of vecchia_setup(), which orders the observed locations first)
vecchia_setup_obs <- function(y, num_neighbours = 30) {
  y <- drop(y)
  locs <- gridlocations(nrow(y))
  idx <- which(!is.na(y))
  locs_obs <- locs[idx, , drop = FALSE]
  ord <- GpGp::order_maxmin(locs_obs)
  locs_obs <- locs_obs[ord, , drop = FALSE]
  list(
    yord_obs = y[idx][ord],
    locs_obs = locs_obs,
    NNarray_obs = GpGp::find_ordered_nn(locs_obs, m = num_neighbours, lonlat = FALSE),
    covfun_name = "matern_isotropic"
  )
}

## Negative log-likelihood (without the constant n log(2π)/2) of the observed data, as a function of (log rho, log tau)
nll_fun <- function(par, setup) {
  Linv <- vecchia_Linv(covparms(exp(par)), setup$covfun_name, setup$locs_obs, setup$NNarray_obs)
  v <- Linv_mult(Linv, setup$yord_obs, setup$NNarray_obs)
  logdet_Sigma <- -2 * sum(log(abs(Linv[, 1]))) # Linv[i, 1] is the ith diagonal element of L⁻¹
  0.5 * logdet_Sigma + 0.5 * sum(v^2)
}

## MAP estimate: the prior is uniform, so this maximises the likelihood over the prior support
MAP <- function(y) {
  setup <- vecchia_setup_obs(y)
  optim_result <- optim(
    par = log(prior_mean),
    fn = nll_fun,
    setup = setup,
    lower = log(c(rho_limits[1], tau_limits[1])),
    upper = log(c(rho_limits[2], tau_limits[2])),
    method = "L-BFGS-B"
  )
  exp(optim_result$par)
}
