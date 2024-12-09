# Boolean indicating whether (TRUE) or not (FALSE) to quickly establish that the code is working properly
quick <- identical(commandArgs(trailingOnly=TRUE)[1], "--quick")

# ---- Simulate data ----

cat("Simulating training data...")

## R packages for data simulation
source(file.path("src", "Plotting.R"))

suppressMessages({
library("bayesImageS")
library("doParallel")
library("dplyr")
library("ggplot2")
})

int_path <- file.path("intermediates", "application", "sea_ice")
img_path <- file.path("img", "application", "sea_ice")
dir.create(int_path, recursive = TRUE, showWarnings = FALSE)
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

q <- 2      # number of labels for the Potts model (Ising has q=2)
p <- 1L     # number of parameters in the model
grid_dim <- c(199, 219)     
d <- prod(grid_dim)     # size of the image
K <- if (quick) 5e3 else 1e5  # number of independent datasets
burn <- 100             # iterations of Swendsen-Wang to discard as burn-in
maxC <- detectCores()   # maximum number of parallel CPU cores to use

# Sample from the prior
beta <- runif(K, 0.03, 1.5)

# Setup for the Potts/Ising model
mask  <- matrix(1, grid_dim[1], grid_dim[2])
neigh <- getNeighbors(mask, c(2,2,0,0))
block <- getBlocks(mask, 2)

# execute in parallel
nc <- min(detectCores(), maxC/2)
# showConnections(all = TRUE)
# closeAllConnections()
cl <- makeCluster(nc)
clusterSetRNGStream(cl)
registerDoParallel(cl)

tm <- system.time({
  Z <- foreach(i=1:K, .multicombine=TRUE,
               .packages=c('bayesImageS')) %dopar% {
                 r <- swNoData(beta[i],q,neigh,block,burn)
                 labels <- matrix(r$z[,2], nrow=grid_dim[1], ncol=grid_dim[2], byrow = TRUE)
                 labels
               }
})
saveRDS(tm, file = file.path(int_path, "sim_time.rds"))

stopCluster(cl)


# ---- Partition simulated data into training, validation, and test sets ----

## Coerce data and parameters to required format 
Z <- lapply(Z, function(z) {
  dim(z) <- c(dim(z)[1], dim(z)[2], 1, 1) 
  z
})
theta <- t(beta)

## Partition the data into training, validation, and test sets
K <- length(Z)
K1 <- ceiling(0.8*K)  # size of the training set 
K3 <- if (quick) 50 else 1000 # size of the test set 
K2 <- K - K1 - K3     # size of the validation set 
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

# ---- Construct neural Bayes estimator for use in neural EM algorithm ----

## R and Julia packages for simulation and neural Bayes estimation
library("NeuralEstimators")
library("JuliaConnectoR")
# NB first must set working directory to top-level of repo
Sys.setenv("JULIACONNECTOR_JULIAOPTS" = "--project=.") 
juliaEval('using NeuralEstimators, Flux, CUDA')
juliaEval('include(joinpath(pwd(), "src", "Architecture.jl"))')

## Initialise the estimator
estimator <- juliaLet('estimator = architecture(p)', p = p)

## Train the estimator 
estimator <- train(estimator, 
                   theta_train = theta_train, 
                   theta_val = theta_val, 
                   Z_train = Z_train, 
                   Z_val = Z_val, 
                   savepath = file.path(int_path, "NBE"))

## Assess the estimator 
assessment <- assess(estimator, theta_test, Z_test, parameter_names = "beta", estimator_names = "NBE")
estimates <- assessment$estimates 
gg <- ggplot(estimates) + 
  geom_point(aes(truth, estimate)) + 
  geom_abline(colour = "red") +
  theme_bw() +
  labs(x = expression(beta), y = expression(hat(beta))) + 
  coord_fixed()
ggsv(file.path(img_path, "NBE_assessment"), gg, width = 3.5, height = 3.5)  
