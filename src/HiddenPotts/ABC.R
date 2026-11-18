# Hidden Potts model (Section 3.3): approximate Bayesian computation (ABC) with summary statistics computed from the
# data. The reference table is the training data simulated by src/HiddenPotts/SimulateData.jl (the parameters and data
# used to train the NBEs), and the estimates are for its test and scenario data.
#
# Usage (from the top level of the repository, after src/HiddenPotts/SimulateData.jl):
#   Rscript src/HiddenPotts/ABC.R

suppressMessages({
  library("abc")
  library("abctools")
  library("dplyr")
  library("ggplot2")
  library("tidyr")
  library("purrr")
  library("future")
  library("future.apply")
  options(dplyr.summarise.inform = FALSE)
})
# Limit BLAS threads globally for parallel computation
Sys.setenv(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
# The number of parallel workers is min(32, half the number of cores), or the environment variable NCORES
ncores <- as.integer(Sys.getenv("NCORES", min(32, parallel::detectCores() %/% 2)))
plan(multisession, workers = ncores)
source(file.path("src", "Plotting.R"))
source(file.path("src", "Utils.R"))

int_path <- file.path("intermediates", "HiddenPotts")
img_path <- file.path("img", "HiddenPotts")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

q <- 3 # number of labels

parameter_labels = c(
  "β" = expression(beta),
  "μ1" = expression(mu[1]),
  "μ2" = expression(mu[2]),
  "μ3" = expression(mu[3]),
  "σ1" = expression(sigma[1]),
  "σ2" = expression(sigma[2]),
  "σ3" = expression(sigma[3])
)

add_singletons <- function(a) {
  dim(a) <- c(dim(a), 1, 1)
  a
}

removedata <- function(Z, proportion = runif(1, 0.01, 0.4)) {
  
  # Ensure proportion is between 0 and 1
  if (proportion < 0 || proportion > 1) stop("Proportion must be between 0 and 1")
  
  # Randomly sample indices to replace
  n <- length(Z)
  n_na <- round(proportion * n)
  na_indices <- sample(1:n, n_na)
  
  # Replace selected elements with NA
  Z[na_indices] <- NA
  
  return(Z)
}


remove_complex_missingness <- function(Z) {
  Z <- drop(Z)  # ensure it's a 2D matrix
  
  # Dimensions
  n_rows <- nrow(Z)
  n_cols <- ncol(Z)
  
  # Midpoints
  row_mid <- n_rows %/% 2
  col_mid <- n_cols %/% 2
  
  # Quarter-circle radius
  radius <- min(row_mid, n_cols - col_mid)
  
  # Store indices to NA
  indices_to_remove <- list()
  
  # --- Quarter circle (top-right) ---
  for (i in 1:row_mid) {           # top half rows
    for (j in col_mid:n_cols) {    # right half columns
      dist <- sqrt((i - 1)^2 + (j - n_cols)^2)
      if (dist <= radius) {
        indices_to_remove[[length(indices_to_remove) + 1]] <- c(i, j)
      }
    }
  }
  
  # --- Ellipse (bottom-left) ---
  for (i in (row_mid + 1):n_rows) {
    for (j in 1:col_mid) {
      di <- (i - n_rows)^2 / (0.3 * n_rows)^2
      dj <- (j - 1)^2 / (0.5 * n_cols)^2
      if (di + dj <= 1.0) {
        indices_to_remove[[length(indices_to_remove) + 1]] <- c(i, j)
      }
    }
  }
  
  # Apply missingness to matrix
  for (idx in indices_to_remove) {
    Z[idx[1], idx[2]] <- NA
  }
  
  return(Z)
}

# ---- Data simulated by src/HiddenPotts/SimulateData.jl ----

# Reads a data set of an HDF5 file written by Julia (same dimensions and element order), with rhdf5 or hdf5r
read_h5 <- function(file, name) {
  if (requireNamespace("rhdf5", quietly = TRUE)) return(rhdf5::h5read(file, name))
  if (requireNamespace("hdf5r", quietly = TRUE)) {
    f <- hdf5r::H5File$new(file, mode = "r")
    on.exit(f$close_all())
    return(f[[name]]$read())
  }
  stop("reading HDF5 files needs the R package rhdf5 or hdf5r")
}
as_list <- function(a) lapply(seq_len(dim(a)[3]), function(k) a[, , k]) # an n × n × K array as a list of K matrices
Z_train     <- as_list(read_h5(file.path(int_path, "training_data.h5"), "Z_train"))
theta_train <- read_h5(file.path(int_path, "training_data.h5"), "theta_train")
Z_test      <- as_list(read_h5(file.path(int_path, "test_data.h5"), "Z_test"))
Z_scenarios <- as_list(read_h5(file.path(int_path, "test_data.h5"), "Z_scenarios"))
theta_test  <- unname(as.matrix(read.csv(file.path(int_path, "theta_test.csv"), header = FALSE)))
stopifnot(ncol(theta_train) == length(Z_train), ncol(theta_test) == length(Z_test))

# ---- ABC ----

abc_path <- file.path(int_path, "ABC")
dir.create(abc_path, recursive = TRUE, showWarnings = FALSE)

# Bounded EM for 1D Gaussian mixtures 
# Enforces box priors: mu_k in [muL, muU], sigma_k in [sigL, sigU]
bounded_gmm_em <- function(
  y, K,
  mu_bounds, sigma_bounds,
  pi_init = NULL, mu_init = NULL, sigma_init = NULL,
  maxit = 200, tol = 1e-6,
  sigma_floor = 1e-6,
  alpha = 2.0,        # Dirichlet pseudo-counts for pi (per component)
  pi_min = 5e-3,      # minimum allowed pi after update (will renormalize)
  verbose = FALSE
) {
  y <- as.numeric(y); y <- y[is.finite(y)]
  n <- length(y)
  stopifnot(n > 0, K >= 1,
            all(dim(mu_bounds) == c(K,2)),
            all(dim(sigma_bounds) == c(K,2)))
  # --- init ---
  if (is.null(mu_init)) {
    qs <- stats::quantile(y, probs = seq(0, 1, length.out = K + 2)[2:(K+1)], names = FALSE)
    mu  <- pmin(pmax(qs, mu_bounds[,1]), mu_bounds[,2])
  } else mu <- pmin(pmax(as.numeric(mu_init), mu_bounds[,1]), mu_bounds[,2])

  if (is.null(sigma_init)) {
    rough <- rep(stats::IQR(y)/2, K); rough[!is.finite(rough)] <- stats::sd(y)
    sigma <- pmin(pmax(rough, sigma_bounds[,1]), sigma_bounds[,2]); sigma <- pmax(sigma, sigma_floor)
  } else {
    sigma <- pmin(pmax(as.numeric(sigma_init), sigma_bounds[,1]), sigma_bounds[,2])
    sigma <- pmax(sigma, sigma_floor)
  }

  if (is.null(pi_init)) pi <- rep(1/K, K) else { pi <- as.numeric(pi_init); pi <- pi/sum(pi) }

  ord <- order(mu); mu <- mu[ord]; sigma <- sigma[ord]; pi <- pi[ord]
  mu_bounds <- mu_bounds[ord,,drop=FALSE]; sigma_bounds <- sigma_bounds[ord,,drop=FALSE]

  loglik <- function(mu, sigma, pi) {
    ll_mat <- vapply(1:K, function(k) dnorm(y, mean = mu[k], sd = sigma[k], log = TRUE) + log(pi[k]), numeric(n))
    m <- apply(ll_mat, 1, max)
    sum(m + log(rowSums(exp(ll_mat - m))))
  }

  last_ll <- -Inf
  for (it in 1:maxit) {
    # --- E-step  ---
    # compute log weights 
    logw <- vapply(1:K, function(k)
      (dnorm(y, mean = mu[k], sd = sigma[k], log = TRUE) + log(pi[k])),
      numeric(n)
    )
    m <- apply(logw, 1, max)
    w <- exp(logw - m)
    r <- w / rowSums(w)

    # --- Nk with pseudo-counts (Dirichlet regularization) ---
    Nk <- colSums(r) + alpha

    # --- M-step with bounds (same as before, but using Nk above) ---
    mu_hat_un <- as.numeric(colSums(r * y) / (colSums(r) + .Machine$double.eps))
    mu_new <- pmin(pmax(mu_hat_un, mu_bounds[,1]), mu_bounds[,2])

    var_hat <- numeric(K)
    for (k in 1:K) {
      diff2 <- (y - mu_new[k])^2
      var_hat[k] <- sum(r[,k] * diff2) / (colSums(r)[k] + .Machine$double.eps)
    }
    sigma_un <- sqrt(pmax(var_hat, sigma_floor^2))
    sigma_new <- pmin(pmax(sigma_un, sigma_bounds[,1]), sigma_bounds[,2])
    sigma_new <- pmax(sigma_new, sigma_floor)

    # --- pi update with pseudo-counts then floor-enforce ---
    pi_new_raw <- Nk / sum(Nk)
    # enforce a minimum weight
    pi_new <- pmax(pi_new_raw, pi_min)
    pi_new <- pi_new / sum(pi_new)

    # reorder by mu to keep labels aligned
    ord <- order(mu_new)
    mu_new    <- mu_new[ord]
    sigma_new <- sigma_new[ord]
    pi_new    <- pi_new[ord]
    mu_bounds    <- mu_bounds[ord,,drop=FALSE]
    sigma_bounds <- sigma_bounds[ord,,drop=FALSE]

    # convergence check
    delta <- max(abs(mu_new - mu), abs(sigma_new - sigma), abs(pi_new - pi))
    mu <- mu_new; sigma <- sigma_new; pi <- pi_new

    ll <- loglik(mu, sigma, pi)
    if (verbose) cat(sprintf("iter %3d: ll=%.6f, Δ=%.3e\n", it, ll, delta))
    if (delta < tol || ll < last_ll + 1e-12) break
    last_ll <- ll
  }

  # final responsibilities
  logw <- vapply(1:K, function(k) dnorm(y, mean = mu[k], sd = sigma[k], log = TRUE) + log(pi[k]), numeric(n))
  m <- apply(logw, 1, max)
  w <- exp(logw - m)
  r_final <- w / rowSums(w)

  list(mu = mu, sigma = sigma, pi = pi,
       responsibilities = r_final, loglik = last_ll, iterations = it)
}

compute_neighbor_diff <- function(Y, sigma, normalize = FALSE) {
  n <- nrow(Y)
  
  diffs <- c()
  
  # neighbor offsets (4-neighborhood)
  offsets <- list(c(-1,0), c(1,0), c(0,-1), c(0,1))
  
  for (i in 1:n) {
    for (j in 1:n) {
      y_ij <- Y[i,j]
      if (is.na(y_ij)) next
      
      for (off in offsets) {
        ni <- i + off[1]; nj <- j + off[2]
        if (ni >= 1 && ni <= n && nj >= 1 && nj <= n) {
          y_nb <- Y[ni, nj]
          if (is.na(y_nb)) next
          
          # raw difference
          d <- abs(y_ij - y_nb)
          
          if (normalize) {
            # normalize by average sigma (robust to μ misestimation)
            d <- d / mean(sigma, na.rm = TRUE)
          }
          
          diffs <- c(diffs, d)
        }
      }
    }
  }
  
  # 1 / (1 + mean(diffs, na.rm = TRUE))
  mean(diffs, na.rm = TRUE)
}


compute_neighbor_agreement <- function(Y, mu, sigma, threshold = 0.05) {
  n <- nrow(Y)
  q <- length(mu)

  # Flatten data
  Y_vec <- c(Y)

  # Compute simple approximate label probabilities for each pixel: Pr(Y = y | Z, sigma, mu) = Gau(Z, mu[y], sigma[y])
  probs <- sapply(1:q, function(k) {
    dnorm(Y_vec, mean = mu[k], sd = sigma[k])
  })
  probs <- probs / rowSums(probs)  # normalize

  # Mask out uncertain pixels
  max_probs <- apply(probs, 1, max)
  valid <- which(max_probs > threshold)

  # Reshape into list of probability vectors
  prob_grid <- vector("list", n * n)
  for (idx in seq_along(Y_vec)) {
    prob_grid[[idx]] <- probs[idx, ]
  }

  # Neighbor offsets (4-neighborhood)
  offsets <- list(c(-1,0), c(1,0), c(0,-1), c(0,1))

  agreements <- c()

  for (i in 1:n) {
    for (j in 1:n) {
      idx <- (i-1)*n + j
      if (!(idx %in% valid)) next  # skip uncertain pixels
      p1 <- prob_grid[[idx]]

      for (off in offsets) {
        ni <- i + off[1]; nj <- j + off[2]
        if (ni >= 1 && ni <= n && nj >= 1 && nj <= n) {
          nidx <- (ni-1)*n + nj
          if (!(nidx %in% valid)) next
          p2 <- prob_grid[[nidx]]
          # Expected agreement = dot product of probability vectors
          agreements <- c(agreements, sum(p1 * p2))
        }
      }
    }
  }

  mean(agreements, na.rm = TRUE)
}

summary_statistic <- function(Y, K, mu_bounds, sigma_bounds, ...) {
  
  Y <- drop(Y)
  Y_vec <- as.numeric(Y)
  Y_vec <- Y_vec[!is.na(Y_vec)]

  # 1. Mixture-component statistics
  fit <- bounded_gmm_em(
    y = Y_vec, K = K,
    mu_bounds = mu_bounds,
    sigma_bounds = sigma_bounds,
    maxit = 300, tol = 1e-7, verbose = FALSE
  )
  means <- fit$mu
  sds   <- fit$sigma
  pi_k  <- fit$pi
  
  # fit is result from bounded_gmm_em (see earlier)
  # fit$mu, fit$sigma, fit$pi, fit$responsibilities (n x K)
  pooled_sigma_from_EM <- function(fit) {
    pi <- fit$pi
    sigma_k <- fit$sigma
    sqrt(sum(pi * (sigma_k^2)))
  }
  sigma <- pooled_sigma_from_EM(fit)
  
  # 2. Spatial dependence
  beta1 = compute_neighbor_agreement(Y, mu = means, sigma = sds)
  beta2 = compute_neighbor_diff(Y, sigma = sigma)
  
  
  # Combine into a single summary vector
  return(c(beta1, beta2, means, sds, pi_k[-1]))
}

## As the ABC reference table, use the same parameters and data used during training
theta <- theta_train
Z <- Z_train
prior_lower_bound <- apply(theta, 1, min)
prior_upper_bound <- apply(theta, 1, max)
mu_bounds <- cbind(prior_lower_bound[2:(q+1)], prior_upper_bound[2:(q+1)])
sigma_bounds <- cbind(prior_lower_bound[(q+2):(2*q+1)], prior_upper_bound[(q+2):(2*q+1)])

## Compute summary statistics from complete data (amortized)
tm <- system.time({
  sumstats <- future_lapply(
    Z, summary_statistic, 
    K = q, 
    mu_bounds = mu_bounds, 
    sigma_bounds = sigma_bounds, 
    future.seed = TRUE)
})
sumstats <- do.call(cbind, sumstats)

sumstats_model <- saABC(t(theta), t(sumstats)) 
sumstats <- sumstats_model$B %*% sumstats

saveRDS(tm, file = file.path(abc_path, "sumstats_time.rds"))
saveRDS(sumstats, file = file.path(abc_path, "sumstats.rds"))
saveRDS(theta, file = file.path(abc_path, "theta.rds"))

## Visualize the summary statistics
num_points <- min(ncol(theta), 1000)
df <- data.frame(
  parameter = rep(names(parameter_labels), num_points),
  truth = as.vector(theta[, 1:num_points]),
  sumstat = as.vector(sumstats[, 1:num_points])
)
param_names <- names(parameter_labels)
df <- df %>% mutate(parameter = factor(parameter, levels = param_names))
figure <- ggplot(df, aes(x = truth, y = sumstat)) +
  geom_point(alpha = 0.6) +
  facet_wrap(~ parameter, scales = "free", nrow = 1,
             labeller = labeller(parameter = as_labeller(parameter_labels, default = label_parsed))) +
  theme_minimal() +
  labs(x = expression(theta[i]),
       y = expression(S[i]))
ggsv(file = "ABC_summaries", plot = figure, path = img_path, width = 9.5, height = 4.5)

## ABC 
abc_mode <- function(z1, sumstats, theta, method = "neuralnet") {
  
  # Observed summary statistics
  target <- summary_statistic(z1, K = q, 
                              mu_bounds = mu_bounds, 
                              sigma_bounds = sigma_bounds)
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

## Complete data
abc_modes <- future_lapply(
  Z_test, abc_mode, sumstats = sumstats, theta = theta,
  future.seed = TRUE
)
abc_modes <- do.call(cbind, abc_modes)
abc_rmse <- sqrt(mean((abc_modes - theta_test)^2))
abc_rmse_beta <- sqrt(mean((abc_modes[1, ] - theta_test[1, ])^2))
saveRDS(abc_modes, file = file.path(abc_path, "estimates_complete.rds"))
saveRDS(abc_rmse, file = file.path(abc_path, "rmse_complete.rds"))
saveRDS(abc_rmse_beta, file = file.path(abc_path, "rmse_beta_complete.rds"))

## MCAR data 
Z1_MCAR <- lapply(Z_test, removedata, proportion = 0.2)
abc_modes <- future_lapply(
  Z1_MCAR, abc_mode, sumstats = sumstats, theta = theta,
  future.seed = TRUE
)
abc_modes <- do.call(cbind, abc_modes)
abc_rmse <- sqrt(mean((abc_modes - theta_test)^2))
abc_rmse_beta <- sqrt(mean((abc_modes[1, ] - theta_test[1, ])^2))
saveRDS(abc_modes, file = file.path(abc_path, "estimates_MCAR.rds"))
saveRDS(abc_rmse, file = file.path(abc_path, "rmse_MCAR.rds"))
saveRDS(abc_rmse_beta, file = file.path(abc_path, "rmse_beta_MCAR.rds"))

## MB data
Z1_MB <- lapply(Z_test, remove_complex_missingness)
abc_modes <- future_lapply(
  Z1_MB, abc_mode, sumstats = sumstats, theta = theta,
  future.seed = TRUE
)
abc_modes <- do.call(cbind, abc_modes)
abc_rmse <- sqrt(mean((abc_modes - theta_test)^2))
abc_rmse_beta <- sqrt(mean((abc_modes[1, ] - theta_test[1, ])^2))
saveRDS(abc_modes, file = file.path(abc_path, "estimates_MB.rds"))
saveRDS(abc_rmse, file = file.path(abc_path, "rmse_MB.rds"))
saveRDS(abc_rmse_beta, file = file.path(abc_path, "rmse_beta_MB.rds"))

## MCAR (scenarios)
Z1_MCAR <- lapply(Z_scenarios, removedata, proportion = 0.2)
abc_modes <- future_lapply(
  Z1_MCAR, abc_mode, sumstats = sumstats, theta = theta,
  future.seed = TRUE
)
abc_modes <- do.call(cbind, abc_modes)
saveRDS(abc_modes, file = file.path(abc_path, "estimates_MCAR_scenarios.rds"))

## MB (scenarios) 
Z1_MB <- lapply(Z_scenarios, remove_complex_missingness)
abc_modes <- future_lapply(
  Z1_MB, abc_mode, sumstats = sumstats, theta = theta,
  future.seed = TRUE
)
abc_modes <- do.call(cbind, abc_modes)
saveRDS(abc_modes, file = file.path(abc_path, "estimates_MB_scenarios.rds"))

## Timing for a single data set
start_time <- Sys.time()
z1 <- Z1_MCAR[[1]]
abc_modes <- abc_mode(z1, sumstats = sumstats, theta = theta)
end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
saveRDS(elapsed_time, file = file.path(abc_path, "runtime_singledataset.rds"))
