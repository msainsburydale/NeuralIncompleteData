cat("Computing ABC posteriors...\n")

source(file.path("src", "Plotting.R"))
source(file.path("src", "Utils.R"))
int_path <- file.path("intermediates", "GH")
dir.create(int_path, recursive = TRUE, showWarnings = FALSE)
abc_path <- file.path(int_path, "ABC")
dir.create(abc_path, recursive = TRUE, showWarnings = FALSE)
img_path <- file.path("img", "GH")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

suppressMessages({
library("abc")
library("abctools")
library("readr")
library("future.apply")
Sys.setenv(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
plan(multisession, workers = availableCores() %/% 2)
})

parameter_labels = c(
  "γ" = expression(alpha),
  "ω" = expression(omega),
  "λ" = expression(lambda), 
  "ρ" = expression(rho),
  "ν" = expression(nu)
)

# Load the parameters and summary statistics
theta         <- as.matrix(read.csv(file.path(abc_path, "theta_train.csv"), header = F))
theta_test    <- as.matrix(read.csv(file.path(abc_path, "theta_test.csv"), header = F))
theta_scenarios    <- as.matrix(read.csv(file.path(abc_path, "theta_scenarios.csv"), header = F))
sumstats      <- as.matrix(read.csv(file.path(abc_path, "sumstats_train.csv"), header = F))
sumstats_test <- as.matrix(read.csv(file.path(abc_path, "sumstats_test_complete.csv"), header = F))
sumstats_MCAR <- as.matrix(read.csv(file.path(abc_path, "sumstats_MCAR_test.csv"), header = F))
sumstats_MB   <- as.matrix(read.csv(file.path(abc_path, "sumstats_MB_test.csv"), header = F))

sumstats_model <- saABC(t(theta), t(sumstats))
sumstats       <- sumstats_model$B %*% sumstats
sumstats_test  <- sumstats_model$B %*% sumstats_test
sumstats_MCAR  <- sumstats_model$B %*% sumstats_MCAR
sumstats_MB    <- sumstats_model$B %*% sumstats_MB

## Visualize the summary statistics 
parameter_labels = c(
  "γ" = expression(alpha),
  "ω" = expression(omega),
  "λ" = expression(lambda), 
  "ρ" = expression(rho),
  "ν" = expression(nu)
)
df <- data.frame(
  parameter = rep(names(parameter_labels), 1000),
  truth = as.vector(theta[, 1:1000]),
  sumstat = as.vector(sumstats[, 1:1000])
)
param_names <- names(parameter_labels)
df <- df %>% mutate(parameter = factor(parameter, levels = param_names))
figure <- ggplot(df, aes(x = truth, y = sumstat)) +
  geom_point(alpha = 0.6, size = 0.75) +
  facet_wrap(~ parameter, scales = "free", nrow = 1,
             labeller = labeller(parameter = as_labeller(parameter_labels, default = label_parsed))) +
  theme_bw() +
  labs(
    x = expression(theta[i]),
    y = expression(S[i])
    )
ggsv(file = "ABC_summaries", plot = figure, path = img_path, width = 8.5, height = 2.5)

abc_mode <- function(sumstats_observed, sumstats, theta, method = "neuralnet") {
  
  # Suppress all output types to keep console clean
  invisible(capture.output({suppressWarnings({suppressMessages({

  # ABC sampling
  object <- abc(
    target = c(sumstats_observed), 
    param = t(theta), # n x p
    sumstat = t(sumstats), # n x d
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

## Timing for this stage of inference (added to the summary stat computation)
start_time <- Sys.time()
est <- abc_mode(c(sumstats_test[, 1]), sumstats = sumstats, theta = theta)
end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
saveRDS(elapsed_time, file = file.path(int_path, "runtime_ABC_inference.rds"))

abc_modes <- future_lapply(
  as.list(as.data.frame(sumstats_test)), 
  abc_mode,
  sumstats = sumstats, theta = theta,
  future.seed = TRUE
)
abc_modes <- do.call(cbind, abc_modes)
abc_rmse <- sqrt(mean((abc_modes - theta_test)^2))
saveRDS(abc_rmse, file = file.path(abc_path, "rmse_complete.rds"))

abc_modes <- future_lapply(
  as.list(as.data.frame(sumstats_MCAR)), 
  abc_mode,
  sumstats = sumstats, theta = theta,
  future.seed = TRUE
)
abc_modes <- do.call(cbind, abc_modes)
abc_rmse <- sqrt(mean((abc_modes - theta_test)^2))
saveRDS(abc_rmse, file = file.path(abc_path, "rmse_MCAR.rds"))

abc_modes <- future_lapply(
  as.list(as.data.frame(sumstats_MB)), 
  abc_mode,
  sumstats = sumstats, theta = theta,
  future.seed = TRUE
)
abc_modes <- do.call(cbind, abc_modes)
abc_rmse <- sqrt(mean((abc_modes - theta_test)^2))
saveRDS(abc_rmse, file = file.path(abc_path, "rmse_MB.rds"))

# Empirical sampling distributions

estimates_dataframe <- function(estimates, truth, estimator_name, parameter_names, m = 1) {
  
  d <- nrow(truth)
  K <- ncol(truth)
  J <- ncol(estimates) / K
  
  truth_rep <- truth[, rep(seq_len(K), times = J), drop = FALSE]
  
  if (nrow(estimates) == nrow(truth)) {
    
    data.frame(
      m = m, 
      k = rep(rep(1:K, each = d), times = J),
      j = rep(1:J, each = K*d),
      estimator = estimator_name, 
      parameter = rep(parameter_names, times = J * d),
      estimate = c(estimates),
      truth = c(truth_rep)
    )
  } else {
    stop("Invalid dimensions: estimates must have either nrow(truth) rows")
  }
}

future::plan(multisession, workers = min(32, availableCores() %/% 2))
sumstats_MCAR <- as.matrix(read.csv(file.path(abc_path, "sumstats_MCAR_scenarios.csv"), header = F))
sumstats_MCAR  <- sumstats_model$B %*% sumstats_MCAR
abc_modes <- future_lapply(
  as.list(as.data.frame(sumstats_MCAR)), 
  abc_mode,
  sumstats = sumstats, theta = theta, 
  future.seed = TRUE
)
abc_modes <- do.call(cbind, abc_modes)
abc_df <- estimates_dataframe(abc_modes, theta_scenarios, "ABC MAP", names(parameter_labels), m = 150)
saveRDS(abc_modes, file = file.path(abc_path, "estimates_MCAR_scenarios.rds"))
write.csv(abc_df, file = file.path(abc_path, "estimates_MCAR_scenarios.csv"), row.names = FALSE)

sumstats_MB   <- as.matrix(read.csv(file.path(abc_path, "sumstats_MB_scenarios.csv"), header = F))
sumstats_MB    <- sumstats_model$B %*% sumstats_MB
abc_modes <- lapply(
  as.list(as.data.frame(sumstats_MB)), 
  abc_mode,
  sumstats = sumstats, theta = theta
)
abc_modes <- do.call(cbind, abc_modes)
abc_df <- estimates_dataframe(abc_modes, theta_scenarios, "ABC MAP", names(parameter_labels), m = 150)
saveRDS(abc_modes, file = file.path(abc_path, "estimates_MB_scenarios.rds"))
write.csv(abc_df, file = file.path(abc_path, "estimates_MB_scenarios.csv"), row.names = FALSE)

