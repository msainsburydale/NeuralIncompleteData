# Boolean indicating whether (TRUE) or not (FALSE) to quickly establish that the code is working properly
quick <- identical(commandArgs(trailingOnly=TRUE)[1], "--quick")

## R and Julia packages for simulation and neural Bayes estimation
suppressMessages({
  library("abc")
  library("abctools")
  library("bayesImageS")
  library("dplyr")
  library("NeuralEstimators")
  library("JuliaConnectoR")
  library("truncnorm")
  library("ggplot2")
  library("tidyr")
  library("purrr")
  library("future")
  library("future.apply")
  options(dplyr.summarise.inform = FALSE)
})
# Limit BLAS threads globally for parallel simulation
Sys.setenv(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
plan(multisession)
source(file.path("src", "Plotting.R"))
source(file.path("src", "Utils.R"))
# NB first must set working directory to top-level of repo
Sys.setenv("JULIACONNECTOR_JULIAOPTS" = "--project=. --threads=auto")
juliaEval('using NeuralEstimators, Flux, CUDA')
snk <- juliaEval('include(joinpath(pwd(), "src", "Architecture.jl"))')
architecture <- juliaFun("architecture")

int_path <- file.path("intermediates", "HiddenPotts")
img_path <- file.path("img", "HiddenPotts")
dir.create(int_path, recursive = TRUE, showWarnings = FALSE)
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

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


# ---- Data simulation ----

q <- 3      # number of labels
n <- 64     # image size n x n
d <- as.integer(n^2)
K <- ifelse(quick, 1000, 25000)  # number of data sets
burn <- 100 # Swendsen-Wang burn-in

# Prior for Potts model
beta_crit <- log(1 + sqrt(q))
beta_max <- 1.5 
beta <- runif(K, 0.03, beta_max) 

# Priors for emission model (Gaussian)
sample_emission_parameters <- function(K, q, m_bounds = c(-1, 1)) {
  m_k <- seq(m_bounds[1], m_bounds[2], length.out = q)  # prior means for mu_k
  spacing <- diff(m_k)[1]
  
  mu_k    <- matrix(NA, nrow = K, ncol = q)
  sigma_k <- matrix(NA, nrow = K, ncol = q)
  for (i in 1:K) {
    mu_k[i, ]    <- sort(rtruncnorm(q, a = m_bounds[1], b = m_bounds[2], mean = m_k, sd = spacing * 0.2))
    # mu_k[i, ]    <- rnorm(q, mean = m_k, sd = spacing * 0.3)
    sigma_k[i, ] <- runif(q, min = 0.03, max = 1 / q)
  }
  
  list(mu_k = mu_k, sigma_k = sigma_k)
}
params <- sample_emission_parameters(K, q)
mu_k <- params$mu_k
sigma_k <- params$sigma_k

# Plot the sampled means
mu_df <- as.data.frame(mu_k)
colnames(mu_df) <- paste0("Class_", 1:q)
mu_long <- pivot_longer(mu_df, cols = everything(), names_to = "Class", values_to = "mu")
ggplot(mu_long, aes(x = mu, fill = Class)) +
  geom_histogram(position = "identity", alpha = 0.4, bins = 30) +
  labs(x = expression(mu[k]), y = "Count") +
  theme_minimal()

# Potts setup
mask <- matrix(1, n, n)
neigh <- getNeighbors(mask, c(2,2,0,0))
block <- getBlocks(mask, 2)

# Hidden Potts simulation
tm <- system.time({
  Z <- future_lapply(1:K, function(i) {
    be <- beta[i]
    r  <- swNoData(be, q, neigh, block, burn)
    labels <- matrix(max.col(r$z[-1, ]), nrow = n, ncol = n, byrow = TRUE)
    
    # Check number of labels (should equal q)
    observed_classes <- sort(unique(as.vector(labels)))
    if (length(observed_classes) != q) {
      warning(sprintf(
        "Only %d of %d classes observed (classes: %s) for beta = %.2f",
        length(observed_classes), q, paste(observed_classes, collapse = ", "), be
      ))
    }
    
    # Observed field: Z ~ N(mu_k[Z], sigma_k[Z]^2)
    mu_i    <- mu_k[i, ]
    sigma_i <- sigma_k[i, ]
    Z <- matrix(NA, n, n)
    for (k in 1:q) {
      Z[labels == k] <- rnorm(sum(labels == k), mean = mu_i[k], sd = sigma_i[k])
    }
    
    Z
  }, future.seed = TRUE)
})
saveRDS(tm, file = file.path(int_path, "sim_time.rds"))

# --- Combine parameters ---
theta <- rbind(t(beta), t(mu_k), t(sigma_k))
p <- nrow(theta)

# Compute (approximate) prior mean and bounds 
prior_mean          <- rowMeans(theta)
prior_lower_bound   <- apply(theta, 1, min)
prior_upper_bound   <- apply(theta, 1, max)
saveRDS(prior_mean, file = file.path(int_path, "prior_mean.rds"))
saveRDS(prior_lower_bound, file = file.path(int_path, "prior_lower_bound.rds"))
saveRDS(prior_upper_bound, file = file.path(int_path, "prior_upper_bound.rds"))

# --- Generate datasets for multiple parameter scenarios ---
beta_scenarios <- seq(0.4, floor(beta_max * 10) / 10, by = 0.1)
mu_sigma_scenarios <- sample_emission_parameters(length(beta_scenarios), q)
theta_scenarios <- rbind(
  t(beta_scenarios),
  t(mu_sigma_scenarios$mu_k),
  t(mu_sigma_scenarios$sigma_k)
)

J <- ifelse(quick, 5, 50) # number of data sets per scenario

# Each future worker handles one replicate j
Z_scenarios <- future_lapply(1:J, function(j) {
  lapply(1:ncol(theta_scenarios), function(i) {
    r <- swNoData(theta_scenarios[1, i], q, neigh, block, burn)
    labels <- matrix(max.col(r$z[-1, ]), nrow = n, ncol = n, byrow = TRUE)
    
    mu_i    <- theta_scenarios[2:(q + 1), i]
    sigma_i <- theta_scenarios[(q + 2):(2 * q + 1), i]
    
    Z <- matrix(NA, n, n)
    for (k in 1:q) {
      Z[labels == k] <- rnorm(sum(labels == k), mean = mu_i[k], sd = sigma_i[k])
    }
    Z
  })
}, future.seed = TRUE)

# Flatten to a simple list 
Z_scenarios <- do.call(c, Z_scenarios)


# ---- Training, validation, and test sets ----

## Coerce data and parameters to required format
Z <- lapply(Z, add_singletons)
Z_scenarios <- lapply(Z_scenarios, add_singletons)

## Partition the data into training, validation, and test sets
K <- length(Z)
K1 <- ceiling(0.8*K)           # size of the training set 
K3 <- ifelse(quick, 15, 1000)  # size of the test set 
K2 <- K - K1 - K3              # size of the validation set 
if (K1 + K2 + K3 != length(Z)) {
  stop("The sum of the sizes of the training, validation, and test sets does not equal the total number of data sets.")
}
idx <- 1:K 
idx_train <- idx[1:K1]
idx_val   <- idx[(K1 + 1):(K1 + K2)]
idx_test  <- idx[(K1 + K2 + 1):K]

Z_train <- Z[idx_train]
Z_val   <- Z[idx_val]
Z_test  <- Z[idx_test]

theta_train <- theta[, idx_train, drop = F]
theta_val   <- theta[, idx_val, drop = F]
theta_test  <- theta[, idx_test, drop = F]

# Save test set
saveRDS(Z_test, file = file.path(int_path, "Z_test.rds"))
saveRDS(theta_test, file = file.path(int_path, "theta_test.rds"))

saveRDS(Z_scenarios, file = file.path(int_path, "Z_scenarios.rds"))
saveRDS(theta_scenarios, file = file.path(int_path, "theta_scenarios.rds"))

# ---- Construct EM NBE ----

epochs <- ifelse(quick, 3, 100)

## Initialize NBE with 1 input channel, then train
estimator <- architecture(p, prior_lower_bound, prior_upper_bound, input_channels = 1L)
estimator <- train(
  estimator,
  theta_train = theta_train,
  theta_val = theta_val,
  Z_train = Z_train,
  Z_val = Z_val,
  epochs = epochs,
  savepath = file.path(int_path, "runs_EM")
)

## Load the trained NBE
estimator <- loadstate(estimator, file.path(int_path, "runs_EM", "ensemble.bson"))

## Assess the NBE
assessment <- assess(estimator, theta_test, Z_test, estimator_names = "EM NBE")
plotestimates(assessment$estimates %>% filter(parameter == "θ1"))
saveRDS(rmse(assessment), file = file.path(int_path, "rmse_complete_EM-NBE.rds"))

# ---- Construct masking NBE ----

## Construct data sets for masking approach
UW_train <- encodedata(lapply(Z_train, removedata))
UW_val   <- encodedata(lapply(Z_val, removedata))
UW_test  <- encodedata(lapply(Z_test, removedata))

## Initialize NBE with 2 input channels, then train
maskedestimator <- architecture(p, prior_lower_bound, prior_upper_bound, input_channels = 2L)
maskedestimator <- train(
  maskedestimator,
  theta_train = theta_train,
  theta_val = theta_val,
  Z_train = UW_train,
  Z_val = UW_val,
  epochs = epochs,
  savepath = file.path(int_path, "runs_masking")
)

## Load the trained estimator
maskedestimator <- loadstate(maskedestimator, file.path(int_path, "runs_masking", "ensemble.bson"))

## MB data 
Z1_MB <- lapply(Z_test, remove_complex_missingness)
Z1_MB <- lapply(Z1_MB, add_singletons)
UW_MB <- encodedata(Z1_MB)

assessment <- assess(maskedestimator, theta_test, UW_MB, estimator_names = "Masking NBE")
plotestimates(assessment)
saveRDS(rmse(assessment), file = file.path(int_path, "rmse_MB_masking.rds"))

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