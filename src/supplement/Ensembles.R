# Boolean indicating whether (TRUE) or not (FALSE) to quickly establish that the code is working properly
quick <- identical(commandArgs(trailingOnly=TRUE)[1], "--quick")

Sys.setenv(
  OPENBLAS_NUM_THREADS = "1",
  OMP_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1", 
  JULIACONNECTOR_JULIAOPTS = "--project=."
)

suppressMessages({
library("NeuralEstimators")
library("JuliaConnectoR")
library("ggplot2")
library("future.apply")
library("dplyr")
library("egg")
library("gtools")
Sys.setenv(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
plan(multisession, workers = availableCores() %/% 2)
juliaEval('using NeuralEstimators, Flux')
juliaEval('using BSON: @load')
snk <- juliaEval('include(joinpath(pwd(), "src", "Architecture.jl"))')
})

source(file.path("src", "Plotting.R"))

img_path <- file.path("img", "Ensemble")
int_path <- file.path("intermediates", "Ensemble")
dir.create(int_path, recursive = TRUE, showWarnings = FALSE)
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

# Data simulation 

# Sampling from the prior distribution
# K: number of samples to draw from the prior
prior <- function(K) { 
  theta <- 0.5 * runif(K) # draw from the prior distribution
  theta <- t(theta)       # reshape to matrix
  return(theta)
}

# Marginal simulation from the statistical model
# theta: a matrix of parameters drawn from the prior
# m: number of conditionally independent replicates for each parameter vector
# N: number of locations along each dimension of the grid
simulate <- function(theta, m = 1, N = 16) { 
  
  if (length(N) == 1) N <- c(N, N)
  
  # Spatial locations, a grid over the unit square, and spatial distance matrix
  # S <- expand.grid(seq(1, 0, len = N[1]), seq(0, 1, len = N[2]))
  
  # Define the original N to establish the reference spacing
  N_ref <- c(16, 16)
  
  # Calculate the grid spacing based on the reference grid
  spacing_x <- 1 / (N_ref[1] - 1)  # Spacing in the x-direction
  spacing_y <- 1 / (N_ref[2] - 1)  # Spacing in the y-direction
  
  # Calculate the new range for the grid to preserve the spacing
  range_x <- (N[1] - 1) * spacing_x
  range_y <- (N[2] - 1) * spacing_y
  
  # Construct the grid with the same spacing as the reference grid
  S <- expand.grid(seq(0, range_x, length.out = N[1]), 
                   seq(0, range_y, length.out = N[2]))
  
  
  # Spatial distance matrix
  D <- as.matrix(dist(S))         
  
  # Simulate conditionally independent replicates for each parameter vector
  Z <- future_lapply(1:ncol(theta), function(k) {
    Sigma <- exp(-D/theta[k])  # covariance matrix
    L <- t(chol(Sigma))        # lower Cholesky factor of Sigma
    n <- nrow(L)               # number of observation locations
    mm <- if (length(m) == 1) m else sample(m, 1) # allow for variable sample sizes
    z <- matrix(rnorm(n*mm), nrow = n, ncol = mm) # standard normal variates
    Z <- L %*% z               # conditionally independent replicates from the model
    Z <- array(Z, dim = c(N[1], N[2], 1, mm)) # reshape to multidimensional array
    Z
  }, future.seed = TRUE) 
  
  return(Z)
}


K <- ifelse(quick, 2500, 25000)  # size of the training set  
theta_train <- prior(K)          # parameter vectors used in stochastic-gradient descent during training
theta_val   <- prior(K %/% 2)    # parameter vectors used to monitor performance during training
Z_train <- simulate(theta_train) # data used in stochastic-gradient descent during training
Z_val   <- simulate(theta_val)   # data used to monitor performance during training


# Architectures 

architecture1 <- juliaEval('
function architecture1()
psi = Chain(
  Conv((3, 3), 1 => 32, relu),   
  MaxPool((2, 2)),               
  Conv((3, 3), 32 => 64, relu),  
  MaxPool((2, 2)),               
  Flux.flatten                   
)
phi = Chain(
  Dense(256, 512, relu),        
  Dense(512, 1)
)
# PointEstimator(DeepSet(psi, phi))
summary_network = DeepSet(psi, identity)
inference_network = phi
MAPEstimator(summary_network, inference_network)
end
')

architecture2 <- juliaEval('
function architecture2()
psi = Chain(
  Conv((10, 10), 1 => 64, relu),
  Conv((5, 5),  64 => 128,  relu),
  Conv((3, 3),  128 => 256, relu),
  Flux.flatten                
)
phi = Chain(
  Dense(256, 512, relu),        
  Dense(512, 1)
)
# PointEstimator(DeepSet(psi, phi))  
summary_network = DeepSet(psi, identity)
inference_network = phi
MAPEstimator(summary_network, inference_network)
end
')

architecture3 <- juliaEval('
function architecture3()
psi = Chain(
  Conv((3, 3), 1=>16, pad=1, bias=false), BatchNorm(16, relu),   # Initial Conv layer
  ResidualBlock((3, 3), 16 => 16),                               # Residual Block 1
  ResidualBlock((3, 3), 16 => 32, stride=2),                     # Residual Block 2
  ResidualBlock((3, 3), 32 => 64, stride=2),                     # Residual Block 3
  ResidualBlock((3, 3), 64 => 128, stride=2),                    # Residual Block 4
  GlobalMeanPool(),                                              # Global pooling
  Flux.flatten,
  Dense(128, 128)                                                # Fully connected layer
)
phi = Chain(
  Dense(128, 512, relu),        
  Dense(512, 1)
)
# PointEstimator(DeepSet(psi, phi))
summary_network = DeepSet(psi, identity)
inference_network = phi
MAPEstimator(summary_network, inference_network)
end
')

architectures <- list(architecture1, architecture2, architecture3)

# Number of parameters in each architecture
sapply(architectures, function(arch) juliaLet('nparams(arch())', arch = arch))
# 150913 638657 390321

# Constructing an ensemble of estimators: For each architecture, train J estimators independently
J <- 10 # number of ensemble components

## Maximum number of epochs 
epochs <- ifelse(quick, 5, 100)

all_estimators <- lapply(seq_along(architectures), function(i) {
  cat("Training estimators with architecture", i, "\n")
  lapply(1:J, function(j) {
      cat("Training estimator", j, "\n")
      train(
        architectures[[i]](),
        theta_train = theta_train,
        theta_val = theta_val,
        Z_train = Z_train,
        Z_val = Z_val, 
        savepath = file.path(int_path, paste0("architecture", i), paste0("estimator", j)), 
        epochs = epochs, 
        use_gpu = F
      )
  }) 
})

## Average training time for each architecture
sapply(seq_along(architectures), function(i) {
  train_times <- sapply(1:J, function(j) {
      file_path = file.path(int_path, paste0("architecture", i), paste0("estimator", j, "_L1"), "train_time.csv")
      read.csv(file_path, header = FALSE)[1, 1]
  }) 
  mean(train_times)
})
# 323.7637  439.1885  519.0239 1953.1654

loadbestmodel <- function(estimator, path) {
  juliaLet(
    '
using NeuralEstimators, Flux 
using BSON: @load
model_state = Flux.state(estimator)
model_path = joinpath(path, "MAP_estimator.bson")
@load  model_path model_state
Flux.loadmodel!(estimator, model_state)
estimator
',
    estimator = estimator, path = path
    )
}

# load the estimators (allows starting from here without retraining)
all_estimators <- lapply(seq_along(architectures), function(i) {
  lapply(1:J, function(j) {
    estimator <- architectures[[i]]()
    loadbestmodel(estimator, file.path(int_path, paste0("architecture", i), paste0("estimator", j)))
  })
})

Ensemble <- function(estimators) juliaLet('Ensemble(estimators)', estimators = estimators)
all_ensembles <- lapply(all_estimators, Ensemble)

# Results
set.seed(1)
K_test <- ifelse(quick, 50, 1000)
theta_test <- prior(K_test)
Z_test <- simulate(theta_test)


# log-likelihood 
# allows for (conditionally) independent replicates stored in the fourth 
# dimension of Z, but these replicates must have the same missingness pattern.
log_lik <- function(theta, Z) {
  
  # Spatial locations and distance matrix
  N <- nrow(Z) # number of locations along each dimension of the grid
  S <- expand.grid(seq(1, 0, len = N), seq(0, 1, len = N))
  D <- as.matrix(dist(S))  
  I <- which(!is.na(c(Z[, , 1, 1]))) # indices of observed elements (assumed constant between replicates)
  D <- D[I, I]
  
  # Covariance matrix and its Cholesky factor
  Sigma <- exp(-D / theta)
  L <- t(chol(Sigma))
  
  # Compute the log determinant term: 2 * sum(log(diag(L)))
  log_det_term <- 2 * sum(log(diag(L)))
  
  # log-likelihood for each replicate
  ll <- apply(Z, MARGIN = 4, FUN = function(z) {
      
      # Observed elements, converted to vector
      z <- c(z)[I] 
      n <- length(z)
      
      # Compute L^{-1} * z
      L_inv_z <- solve(L, z)  
      
      # Compute the quadratic term: (L^{-1}z)^T (L^{-1}z)
      quad_term <- sum(L_inv_z^2)
      
      # Compute the log-likelihood
      ll <- -0.5 * (n * log(2 * pi) + log_det_term + quad_term)
      
      return(ll)
  })

  
  return(sum(ll))
}

# MAP (under uniform prior)
MAP <- function(Z1, theta_0) {
  neg_log_lik <- function(theta, Z) -log_lik(theta, Z)
  optim(theta_0, neg_log_lik, Z = Z1, method = "Brent", lower = 0, upper = 0.5)$par
}

map_estimates <- unlist(future_lapply(Z_test, MAP, theta_0 = 0.25, future.seed = TRUE))
map_rmse <- sqrt(mean((theta_test - map_estimates)^2))
saveRDS(map_rmse, file.path(int_path, "map_rmse.rds"))


## Visualise performance of each architecture and the ensemble
results <- function(estimators) {
  ensemble <- Ensemble(estimators)
  assessment <- assess(
    c(estimators, list(ensemble)), 
    theta_test, 
    Z_test, 
    parameter_names = "θ",
    estimator_names = c(paste("Estimator", 1:J), "Ensemble")
  )
  df <- rmse(assessment)
  ensemble_rmse <- df$rmse[df$estimator == "Ensemble"]
  df$difference <- df$rmse - ensemble_rmse 
  df$relative_difference <- df$difference  / ensemble_rmse
  return(df)
}

dfs <- lapply(seq_along(all_estimators), function(i) {
  df <- results(all_estimators[[i]])
  df$architecture <- as.character(i)
  return(df)
})
df <- do.call(rbind, dfs)
write.csv(df, file = "df_architecture.csv", row.names = FALSE)
df$architecture <- as.character(df$architecture)

gg <- ggplot(df) + 
  geom_boxplot(aes(x = architecture, y = rmse, group = architecture)) + 
  geom_point(data = df %>% filter(estimator == "Ensemble"), 
             aes(x = architecture, y = rmse),  
             colour = "red") + 
  geom_hline(yintercept = map_rmse, lty = "dashed") +
  labs(x = "Architecture", y = "RMSE") + 
  theme_bw()

## Visualise performance as a function of number of ensemble components
## (average over the different permutations of estimators, i.e., for each ensemble size 
## j, average performance over multiple permutations of the estimators).

# List of length 3 (corresponding to architecture), with each element itself a 
# list of length J = 10 (ensemble components), with each element the matrix of 
# estimates from the corresponding ensemble component. That is:
# thetahat_test[[a]][[j]]
#  a = architecture index
#  j = base estimator index (1..J)
#  value = matrix of estimates from that estimator on Z_test
thetahat_test <- lapply(all_estimators, function(estimators) {
  lapply(estimators, function(estimator) {
    estimate(estimator, Z_test)
  })
})

ensemble_mean <- function(mats) {
  Reduce(`+`, mats) / length(mats)
}

rmse_matrix <- function(theta_hat, theta_true) {
  sqrt(mean((theta_hat - theta_true)^2))
}


results2 <- function(thetahat_arch, theta_test) {
  
  J <- length(thetahat_arch)
  
  rmse_mean  <- numeric(J)
  rmse_sd    <- numeric(J)
  rmse_se    <- numeric(J)
  rmse_best  <- numeric(J)
  rmse_worst <- numeric(J)
  rmse_5th <- numeric(J)
  rmse_95th <- numeric(J)
  
  for (j in seq_len(J)) {
    
    # All subsets of size j
    combs <- combn(J, j, simplify = FALSE)
    
    rmse_j <- vapply(combs, function(idx) {
      theta_hat_ens <- Reduce(`+`, thetahat_arch[idx]) / j
      rmse_matrix(theta_hat_ens, theta_test)
    }, numeric(1))
    
    rmse_mean[j]  <- mean(rmse_j)
    rmse_sd[j]    <- sd(rmse_j)
    rmse_se[j]    <- rmse_sd[j] / sqrt(length(rmse_j))
    rmse_best[j]  <- min(rmse_j)
    rmse_worst[j] <- max(rmse_j)
    rmse_5th[j]  <- quantile(rmse_j, 0.05)
    rmse_95th[j] <- quantile(rmse_j, 0.95)
  }
  
  data.frame(
    j = seq_len(J),
    rmse = rmse_mean,
    rmse_sd = rmse_sd,
    rmse_se = rmse_se,
    rmse_best = rmse_best,
    rmse_worst = rmse_worst, 
    rmse_5th = rmse_5th, 
    rmse_95th = rmse_95th
  )
}


dfs2 <- lapply(seq_along(thetahat_test), function(a) {
  df <- results2(thetahat_test[[a]], theta_test)
  df$architecture <- as.character(a)
  df
})

df2 <- do.call(rbind, dfs2)
write.csv(df2, "df_ensemblecomponents.csv", row.names = FALSE)

gg2 <- ggplot(df2, aes(x=j, y=rmse, group = architecture, lty = architecture)) +
  # geom_ribbon(
  #   aes(ymin = rmse_best, ymax = rmse_worst),
  #   alpha = 0.2,
  #   color = NA
  # ) +
  # geom_ribbon(
  #   aes(ymin = rmse - rmse_sd,
  #       ymax = rmse + rmse_sd),
  #   alpha = 0.2,
  #   colour = NA
  # ) + 
  geom_point() +
  geom_line() +
  theme_bw() +
  geom_hline(yintercept = map_rmse, lty = "dashed") +
  labs(x = "Number of ensemble components", y = "RMSE", lty = "Architecture") +
  scale_x_continuous(breaks = 1:J) +  # Only integer breaks
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.key.width = unit(0.5, "cm"))


## Final figure
figure <- egg::ggarrange(gg, gg2, nrow = 1)
ggsv(file = file.path(img_path, "ensemble"), plot = figure, width = 7.4, height = 3.25)

# 
# gg3 <- ggplot(df2, aes(x=j, y=rmse)) +
#   facet_wrap(~architecture, nrow = 1) + 
#   # geom_ribbon(
#   #   aes(ymin = rmse_5th, ymax = rmse_95th),
#   #   alpha = 0.2,
#   #   color = NA
#   # ) +
#   # geom_ribbon(
#   #   aes(ymin = rmse_best, ymax = rmse_worst),
#   #   alpha = 0.2,
#   #   color = NA
#   # ) +
#   # geom_ribbon(
#   #   aes(ymin = rmse - rmse_sd,
#   #       ymax = rmse + rmse_sd),
#   #   alpha = 0.2,
#   #   colour = NA
#   # ) + 
#   geom_point() +
#   geom_line() +
#   theme_bw() +
#   geom_hline(yintercept = map_rmse, lty = "dashed") +
#   labs(x = "Number of ensemble components", y = "RMSE") +
#   scale_x_continuous(breaks = 1:J) 
# gg3
