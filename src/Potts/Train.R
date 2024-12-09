# Boolean indicating whether (TRUE) or not (FALSE) to quickly establish that the code is working properly
quick <- identical(commandArgs(trailingOnly=TRUE)[1], "--quick")

## R and Julia packages for simulation and neural Bayes estimation
suppressMessages({
  library("bayesImageS")
  library("doParallel")
  library("dplyr")
  library("NeuralEstimators")
  library("JuliaConnectoR")
  options(dplyr.summarise.inform = FALSE) 
})
# NB first must set working directory to top-level of repo
Sys.setenv("JULIACONNECTOR_JULIAOPTS" = "--project=.") 
juliaEval('using NeuralEstimators, Flux, CUDA')
architecture <- juliaEval('include(joinpath(pwd(), "src", "Architecture.jl"))')

int_path <- file.path("intermediates", "Potts")
dir.create(int_path, recursive = TRUE, showWarnings = FALSE)

q <- 2      # number of labels for the Potts model (Ising has q=2)
p <- 1L     # number of parameters in the model
n <- 64     # size of the image will be n^2 pixels 
d <- as.integer(n^2) # size of the image
K <- ifelse(quick, 1000, 50000)    # number of independent data sets
burn <- 100 # iterations of Swendsen-Wang to discard as burn-in
maxC <- detectCores() - 1   # maximum number of parallel CPU cores to use

# Sample from the prior
beta <- runif(K, 0.03, 1.5)

# Setup for the Potts/Ising model
mask <- matrix(1, n, n)
neigh <- getNeighbors(mask, c(2,2,0,0))
block <- getBlocks(mask, 2)

# execute in parallel
nc <- min(detectCores(), maxC)
cl <- makeCluster(nc)
clusterSetRNGStream(cl)
registerDoParallel(cl)

tm <- system.time({
  Z <- foreach(i=1:K, .multicombine = TRUE,
               .packages=c('bayesImageS')) %dopar% {
                 r <- swNoData(beta[i], q, neigh, block, burn)
                 labels <- matrix(r$z[, 2], nrow = n, ncol = n, byrow = TRUE)
                 labels
               }
})
saveRDS(tm, file = file.path(int_path, "sim_time.rds"))

# Generate a data set to assess the sampling distributions of the estimators
theta_scenarios <- seq(0.4, 1.1, by = 0.1)
J <- ifelse(quick, 3, 100) # number of data sets for each beta 
Z_scenarios <- foreach(1:J, .multicombine = TRUE,
                       .packages=c('bayesImageS')) %dopar% {
                         lapply(theta_scenarios, function(be) {
                           r <- swNoData(be, q, neigh, block, burn)
                           labels <- matrix(r$z[,2], nrow = n, ncol = n, byrow = TRUE)
                           labels
                         })
                       }
Z_scenarios <- do.call(c, Z_scenarios)

stopCluster(cl)

# Add one to labels to reflect notation that labels take values in {1, ..., q}
Z <- lapply(Z, function(z) z+1)
Z_scenarios <- lapply(Z_scenarios, function(z) z+1)


# ---- Training, validation, and test sets ----

## Coerce data and parameters to required format 
Z <- lapply(Z, function(z) {
  dim(z) <- c(dim(z)[1], dim(z)[2], 1, 1) 
  z
})
Z_scenarios <- lapply(Z_scenarios, function(z) {
  dim(z) <- c(dim(z)[1], dim(z)[2], 1, 1) 
  z
})
theta <- t(beta)
theta_scenarios <- t(theta_scenarios)

## Partition the data into training, validation, and test sets
K <- length(Z)
K1 <- ceiling(0.8*K)           # size of the training set 
K3 <- ifelse(quick, 15, 1000)  # size of the test set 
K2 <- K - K1 - K3              # size of the validation set 
if (K1 + K2 + K3 != length(Z)) {
  stop("The sum of the sizes of the training, validation, and test sets does not equal the total number of data sets.")
}
idx <- sample(1:K) # shuffle, because the parameters are in increasing order based on beta
idx_train <- idx[1:K1]
idx_val   <- idx[(K1 + 1):(K1 + K2)]
idx_test  <- idx[(K1 + K2 + 1):K]

Z_train <- Z[idx_train]
Z_val   <- Z[idx_val]
Z_test  <- Z[idx_test]

theta_train <- theta[, idx_train, drop = F]
theta_val   <- theta[, idx_val, drop = F]
theta_test  <- theta[, idx_test, drop = F]

# Save test sets for assessing the estimator
saveRDS(Z_test, file = file.path(int_path, "Z_test.rds"))
saveRDS(theta_test, file = file.path(int_path, "theta_test.rds"))

saveRDS(Z_scenarios, file = file.path(int_path, "Z_scenarios.rds"))
saveRDS(theta_scenarios, file = file.path(int_path, "theta_scenarios.rds"))

# ---- Construct neural Bayes estimator for use in neural EM algorithm ----

epochs <- ifelse(quick, 3, 100)

## Initialise the estimator
estimator <- architecture(p, 1) # initialise NBE with 1 input channel, containing the complete data Z

## Train the estimator 
estimator <- train(
  estimator, 
  theta_train = theta_train, 
  theta_val = theta_val, 
  Z_train = Z_train, 
  Z_val = Z_val, 
  loss = tanhloss(0.1),
  epochs = epochs, 
  savepath = file.path("intermediates", "Potts", "runs_EM")
)

## Assess the estimator 
# assessment <- assess(estimator, theta_test, Z_test, parameter_names = "beta", estimator_names = "NBE")
# plotestimates(estimates, parameter_labels = c("beta" = expression(beta)))

# ---- Construct masked neural Bayes estimator ----

## Construct data sets for masking approach
removedata <- function(Z, proportion = runif(1, 0.1, 0.9)) {
  
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
UW_train <- encodedata(lapply(Z_train, removedata))
UW_val   <- encodedata(lapply(Z_val, removedata))
UW_test  <- encodedata(lapply(Z_test, removedata))

## Initialise the estimator
maskedestimator <- architecture(p, 2) # initialise NBE with 2 input channels, containing the augmented data U and missingness pattern W

## Train the estimator 
maskedestimator <- train(
  maskedestimator, 
  theta_train = theta_train, 
  theta_val = theta_val, 
  Z_train = UW_train, 
  Z_val = UW_val, 
  loss = tanhloss(0.1),
  epochs = epochs, 
  savepath = file.path("intermediates", "Potts", "runs_masking")
  )

## Assess the estimator 
# assessment <- assess(maskedestimator, theta_test, UW_test, parameter_names = "beta", estimator_names = "Masked NBE")
# plotestimates(estimates, parameter_labels = c("beta" = expression(beta)))