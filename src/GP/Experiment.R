# Boolean indicating whether (TRUE) or not (FALSE) to quickly establish that the code is working properly
quick <- identical(commandArgs(trailingOnly=TRUE)[1], "--quick")

train_networks <- FALSE

## R and Julia packages for simulation and neural Bayes estimation
suppressMessages({
  library("GpGp") # fast_Gp_sim(),  cond_sim(), 
  library("doParallel")
  library("dplyr")
  library("fields")
  library("Matrix")
  library("NeuralEstimators")
  library("JuliaConnectoR")
  options(dplyr.summarise.inform = FALSE) 
})
## NB first must set working directory to top-level of repo
Sys.setenv("JULIACONNECTOR_JULIAOPTS" = "--project=.") 
juliaEval('using NeuralEstimators, Flux, CUDA')
juliaEval('using BSON: @load')
juliaEval('include(joinpath(pwd(), "src", "Architecture.jl"))')
architecture <- juliaFun("architecture")

int_path <- file.path("intermediates", "GP")
img_path <- file.path("img", "GP")
dir.create(int_path, recursive = TRUE, showWarnings = FALSE)
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(int_path, "Estimates", "Test"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(int_path, "Estimates", "Scenarios"), recursive = TRUE, showWarnings = FALSE)

source(file.path("src", "Plotting.R"))

# ---- Define the model: prior, data simulation, likelihood function ----

## Sampling from the prior distribution
## K: number of samples to draw from the prior
sampler <- function(K) { 
  rho <- runif(K, min = 0.03, max = 0.35)
  tau <- runif(K)
  theta <- matrix(c(rho, tau), nrow = 2, byrow = TRUE)
  return(theta)
}
parameter_labels <- c("rho" = expression(rho), "tau" = expression(tau))
parameter_names <- names(parameter_labels)
p <- as.integer(length(parameter_names))   # number of parameters in the model

## Marginal simulation from the statistical model
## theta: a matrix of parameters drawn from the prior
simulator <- function(theta) { 
  
  # Spatial locations, a grid over the unit square 
  N <- 64 # number of locations along each dimension of the grid
  locs <- as.matrix(expand.grid(seq(1, 0, len = N), seq(0, 1, len = N)))
  
  # Simulate conditionally independent replicates for each parameter vector
  Z <- mclapply(1:ncol(theta), function(k) {
    rho <- theta[1, k]
    tau <- theta[2, k]
    nu <- 1     # smoothness
    sigma2 <- 1 # marginal variance
    Z <- fast_Gp_sim(c(sigma2, rho, nu, tau^2), "matern_isotropic", locs)
    Z <- array(Z, dim = c(N, N, 1, 1)) # reshape to multidimensional array
    Z
  }, mc.cores = detectCores() - 1)
  
  return(Z)
}

## Conditional simulation using GpGp
# library("GpGp")
# library("fields")
# covparms <- c(1, 0.3, 1, 0)                                        # covariance parameters
# N <- 64                                                            # number of points in each direction
# locs <- as.matrix(expand.grid((1:N)/N, (1:N)/N))                   # spatial locations
# y <- matrix(fast_Gp_sim(covparms, "matern_isotropic", locs), N, N) # complete data
# y[1:(N/4),] <- NA                                                  # incomplete data
# z <- simulateconditional(y, covparms, nsims = 30)                  # conditionally-completed data
# par(mfrow = c(1, 3))
# zlim_range <- range(c(y, z[, , 1], z[, , 2]), na.rm = TRUE)
# fields::image.plot(y, zlim = zlim_range, main = "Incomplete data")
# fields::image.plot(z[, , 1], zlim = zlim_range, main = "Conditional simulation 1")
# fields::image.plot(z[, , 2], zlim = zlim_range, main = "Conditional simulation 2")
simulateconditional <- function(y, covparms, nsims = 1, m = 30, reorder = TRUE){
  
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
    ord1 <- if( n_obs < 6e4 ) order_maxmin(locs_obs) else sample(1:n_obs)
    ord2 <- if( n_pred < 6e4 ) order_maxmin(locs_pred) else sample(1:n_pred)
  } else {
    ord1 <- 1:n_obs
    ord2 <- 1:n_pred
  }
  
  # reorder observations and locations
  yord_obs<- y_obs[ord1]
  locsord_obs <- locs_obs[ord1, , drop = FALSE]
  locsord_pred <- locs_pred[ord2, , drop = FALSE] 
  
  # put all coordinates together
  locs_all <- rbind(locsord_obs, locsord_pred)
  inds1 <- 1:n_obs
  inds2 <- (n_obs+1):(n_obs+n_pred)
  
  # get nearest neighbor array
  NNarray_all <- find_ordered_nn(locs_all, m = m, lonlat = get_linkfun(covfun_name)$lonlat)
  
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
    condsim[ord2,j] <- c(v2[inds2]) - z[inds2]
  }
  
  # Combine conditional simulation with observed data 
  y_array <- array(y, dim = c(nrow(y), ncol(y), nsims))
  na_indices_cartesian <- which(is.na(y), arr.ind = TRUE)
  for(j in 1:nsims){
    na_indices_cartesian_j <- cbind(na_indices_cartesian, j=j)
    y_array[na_indices_cartesian_j] <- condsim[, j]
  }
  
  return(y_array)
}


## MAP estimator using GpGp::fit_model
MAP <- function(Z, tau_0 = 0.5, rho_0 = 0.175) {
  
  Z <- drop(Z)
  
  # All locations
  N <- nrow(Z) # NB assumes square grid 
  locs <- as.matrix(expand.grid(seq(1, 0, len = N), seq(0, 1, len = N)))
  
  # Extract observed elements and their locations 
  Z1 <- Z[which(!is.na(Z))]
  Z1_locs <- locs[which(!is.na(Z)), ]
  
  thetahat <- GpGp::fit_model(
    Z1,
    Z1_locs,
    covfun_name = "matern_isotropic",
    start_parms = c(1, rho_0, 1, tau_0^2),
    fixed_parms = c(1, 3), 
    silent = TRUE, 
    m_seq = 30
  )
  
  thetahat <- exp(thetahat$logparms)
  thetahat[2] <- sqrt(thetahat[2]) # convert tau^2 to tau
  
  # Enforce prior bounds 
  thetahat[1] <- pmin(pmax(thetahat[1], 0.03), 0.4)
  thetahat[2] <- pmin(pmax(thetahat[2], 0), 1)
  
  return(thetahat)
}

MAP_multiple <- function(Z) {
  thetahat <- mclapply(Z, function(z) {
    MAP(z)
  }, mc.cores = detectCores() - 1)
  thetahat <- do.call(cbind, thetahat)
  return(thetahat)
}

EM <- function(Z1,                      # data (a matrix containing NAs)
               estimator,               # neural MAP estimator
               theta_0,                 # initial estimate
               niterations = 10,        # maximum number of iterations
               tolerance = 0.05,        # convergence tolerance
               nconsecutive = 1,        # number of consecutive iterations for which the convergence criterion must be met
               nsims = 1,               # Monte Carlo sample size
               verbose = FALSE,         # print current estimate to console if TRUE
               return_iterates = FALSE  # return all iterates if TRUE
) {
  
  if(verbose) print(paste("Initial estimate:", paste(as.vector(theta_0), collapse = ", ")))
  theta_l <- theta_0          # initial estimate
  convergence_counter <- 0    # initialise counter for consecutive convergence
  
  # Initialize a matrix to store all iterates as columns
  p <- length(theta_0)
  theta_all <- matrix(NA, nrow = p, ncol = niterations + 1)
  theta_all[, 1] <- theta_0
  
  for (l in 1:niterations) {
    # current parameters 
    rho <- theta_l[1]
    tau <- theta_l[2]
    nu <- 1     # smoothness
    sigma2 <- 1 # marginal variance
    covparms <- c(sigma2, rho, nu, tau^2)
    # complete the data by conditional simulation
    Z <- simulateconditional(Z1, covparms, nsims = nsims)
    Z <- array(Z, c(dim(Z)[1], dim(Z)[2], 1, dim(Z)[3])) # insert singleton dimension to create 4D array
    # compute the MAP estimate from the conditionally sampled replicates
    theta_l_plus_1 <- c(estimate(estimator, Z)) 
    # check convergence criterion
    if (max(abs(theta_l_plus_1 - theta_l) / abs(theta_l)) < tolerance) {
      # increment counter if condition is met
      convergence_counter <- convergence_counter + 1  
      # check if convergence criterion has been met for required number of iterations
      if (convergence_counter == nconsecutive) {
        if(verbose) print(paste0("Iteration ", l, ": ", paste(theta_l_plus_1, collapse = ", ")))
        if(verbose) message("The EM algorithm has converged")
        theta_all[, l + 1] <- theta_l_plus_1  # store the final iterate
        break
      }
    } else {
      # reset counter if condition is not met
      convergence_counter <- 0  
    }
    theta_l <- theta_l_plus_1  
    theta_all[, l + 1] <- theta_l  # store the iterate
    if(verbose) print(paste0("Iteration ", l, ": ", paste(theta_l, collapse = ", ")))
  }
  
  # Remove unused columns if convergence occurred before max iterations
  theta_all <- theta_all[, 1:(l + 1), drop = FALSE]
  
  # Return all iterates if return_iterates is TRUE, otherwise return the last iterate
  if (return_iterates) theta_all else theta_all[, ncol(theta_all)]
}

EM_multiple <- function(Z1, estimator, theta_0, ...) {
  
  # If theta_0 is a vector, replicate it for each element of Z1
  if (is.vector(theta_0)) {
    theta_0 <- matrix(rep(theta_0, length(Z1)), ncol = length(Z1))  # Replicate for each Z1
  }
  
  # Apply EM for each Z1 with corresponding theta_0 column
  # thetahat <- mclapply(seq_along(Z1), function(i) {
  #   EM(Z1[[i]], estimator, theta_0[, i], ...)
  # }, mc.cores = detectCores() - 1)
  thetahat <- lapply(seq_along(Z1), function(i) {
    EM(Z1[[i]], estimator, theta_0[, i], ...)
  })

  thetahat <- do.call(cbind, thetahat)
  return(thetahat)
}

# Removes data completely at random (i.e., generates MCAR data)
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

# ---- Training ----

if (train_networks) {
  
cat("Simulating training data...")

## Simulate training data 
K <- ifelse(quick, 1000, 25000)    # size of the training set 

  theta_train <- sampler(K)          # parameter vectors used in stochastic-gradient descent during training
  theta_val   <- sampler(K/10)       # parameter vectors used to monitor performance during training
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

## Train the neural MAP estimator for use in the neural EM algorithm
neuralMAP <- architecture(p, 1L) # initialise NBE with 1 input channel, containing the complete data Z
neuralMAP <- train(
  neuralMAP,      
  theta_train = theta_train, 
  theta_val = theta_val, 
  Z_train = Z_train, 
  Z_val = Z_val, 
  loss = tanhloss(0.1), 
  epochs = epochs, 
  savepath = file.path(int_path, "runs_EM")
)

## Train the masked neural Bayes estimator
maskedestimator <- architecture(p, 2L) # initialise NBE with 2 input channels, containing the augmented data U and missingness pattern W
maskedestimator <- train(
  maskedestimator,     
  theta_train = theta_train, 
  theta_val = theta_val, 
  Z_train = UW_train, 
  Z_val = UW_val, 
  loss = tanhloss(0.1), 
  epochs = epochs, 
  savepath = file.path(int_path, "runs_masking")
)
}

# ---- Load the trained neural networks ----

neuralMAP <- juliaLet('
  estimator = architecture(p, 1)
  loadpath  = joinpath(pwd(), int_path, "runs_EM", "ensemble.bson")
  @load loadpath model_state
  Flux.loadmodel!(estimator, model_state)
  estimator
', p = p, int_path = int_path)

maskedestimator <- juliaLet('
  estimator = architecture(p, 2)
  loadpath  = joinpath(pwd(), int_path, "runs_masking", "ensemble.bson")
  @load loadpath model_state
  Flux.loadmodel!(estimator, model_state)
  estimator
', p = p, int_path = int_path)


# ---- Assessment with missing data ----

## Simulate testing data
set.seed(1)
K_test <- ifelse(quick, 100, 1000)
theta_test <- sampler(K_test)       
Z_test <- simulator(theta_test)     

## Example
# N <- 64
# locs <- as.matrix( expand.grid( (1:N)/N, (1:N)/N ) )
# covparms <- c(4,0.3,1,0)
# y <- fast_Gp_sim(covparms, "matern_isotropic", locs)
# y <- matrix(y, N, N)
# fields::image.plot(y)
# fields::image.plot(remove_quarter_circle(y))
remove_quarter_circle <- function(Z) {
  
  Z <- drop(Z)
  
  # Get the dimensions of the matrix
  nrows <- nrow(Z)
  ncols <- ncol(Z)
  
  # Calculate the row and column midpoints (flooring for odd dimensions)
  row_mid <- ceiling(nrows / 2)
  col_mid <- ceiling(ncols / 2)
  
  # Determine the radius of the quarter circle (smallest half-dimension of the matrix)
  radius <- min(nrows - row_mid, ncols - col_mid)
  
  # Loop over the lower-right quadrant
  for (i in (row_mid + 1):nrows) {
    for (j in (col_mid + 1):ncols) {
      
      # Calculate the distance from the lower-right corner
      dist <- sqrt((i - nrows)^2 + (j - ncols)^2)
      
      # If the distance is less than or equal to the radius, set to NA
      if (dist <= radius) {
        Z[i, j] <- NA
      }
    }
  }
  
  return(Z)
}

# Incomplete data
set.seed(1)
Z1_MCAR <- lapply(Z_test, removedata, proportion = 0.2)
Z1_MNAR <- lapply(Z_test, remove_quarter_circle)

# Encoded data set for masking approach
UW_MCAR <- encodedata(Z1_MCAR)
UW_MNAR <- encodedata(Z1_MNAR)

## Initial estimates and number of conditional simulations in neural EM algorithm
theta_0 <- c(0.175, 0.5)
H <- 30 

## Estimation over the test set
set.seed(1)
cat("Running the masked NBE...")
masked_MCAR <- estimate(maskedestimator, UW_MCAR)
masked_MNAR <- estimate(maskedestimator, UW_MNAR)
cat("Running the neural EM algorithm...")
EM_MCAR <- EM_multiple(Z1_MCAR, theta_0 = theta_0, estimator = neuralMAP)
EM_MNAR <- EM_multiple(Z1_MNAR, theta_0 = theta_0, estimator = neuralMAP)
cat("Computing the MAP estimates...")
MAP_MCAR <- MAP_multiple(Z1_MCAR)
MAP_MNAR <- MAP_multiple(Z1_MNAR)
saveRDS(masked_MCAR, file = file.path(int_path, "Estimates", "Test", "masked_MCAR.rds"))
saveRDS(masked_MNAR, file = file.path(int_path, "Estimates", "Test", "masked_MNAR.rds"))
saveRDS(EM_MCAR, file = file.path(int_path, "Estimates", "Test", "EM_MCAR.rds"))
saveRDS(EM_MNAR, file = file.path(int_path, "Estimates", "Test", "EM_MNAR.rds"))
saveRDS(MAP_MCAR, file = file.path(int_path, "Estimates", "Test", "MAP_MCAR.rds"))
saveRDS(MAP_MNAR, file = file.path(int_path, "Estimates", "Test", "MAP_MNAR.rds"))
saveRDS(theta_test, file = file.path(int_path, "Estimates", "Test", "theta_test.rds"))

estimates_dataframe <- function(estimates, truth, estimator_name) {
  
  p <- nrow(truth)
  K <- ncol(truth)
  
  data.frame(
    m = 1, 
    k = rep(1:K, times = p),
    j = 1,
    estimator = estimator_name, 
    parameter = rep(parameter_names, each = K),
    estimate = c(t(estimates)), 
    truth = c(t(truth))
  )
}
 
MCAR_df <- estimates_dataframe(masked_MCAR, theta_test, "masking") %>% 
  rbind(estimates_dataframe(EM_MCAR, theta_test, "neuralEM")) %>%
  rbind(estimates_dataframe(MAP_MCAR, theta_test, "MAP")) 

MNAR_df <- estimates_dataframe(masked_MNAR, theta_test, "masking") %>% 
  rbind(estimates_dataframe(EM_MNAR, theta_test, "neuralEM")) %>%
  rbind(estimates_dataframe(MAP_MNAR, theta_test, "MAP")) 

write.csv(MCAR_df, file.path(int_path, "Estimates", "estimates_MCAR_test.csv"), row.names = F)
write.csv(MNAR_df, file.path(int_path, "Estimates", "estimates_MNAR_test.csv"), row.names = F)

#plotestimates(MCAR_df[sample(nrow(MNAR_df)), ], parameter_labels = parameter_labels)
#plotestimates(MNAR_df[sample(nrow(MNAR_df)), ], parameter_labels = parameter_labels)
#plotestimates(MCAR_df[sample(nrow(MNAR_df)), ] %>% filter(estimator=="masking"), parameter_labels = parameter_labels)
#plotestimates(MNAR_df[sample(nrow(MNAR_df)), ] %>% filter(estimator=="masking"), parameter_labels = parameter_labels)

rmse <- function(df) {
  
  df %>%
    group_by(estimator) %>% 
    # filter(parameter == "rho" & truth < 0.3 | parameter == "tau") %>% 
    mutate(squared_error = (estimate - truth)^2) %>%
      group_by(estimator) %>% 
    summarise(rmse = sqrt(mean(squared_error)), bias = mean(estimate - truth)) 
}


rmse_df1 <- rmse(MCAR_df) 
rmse_df2 <- rmse(MNAR_df) 
rmse_df1$missingness <- "MCAR"
rmse_df2$missingness <- "MNAR"
rmse_df <- rbind(rmse_df1, rmse_df2)
rmse_df
write.csv(rmse_df, file.path(int_path, "Estimates", "rmse.csv"), row.names = F)

## Sampling distributions - estimate many data sets for each parameter vector
J <- ifelse(quick, 10, 100)
rho <- qunif(c(0.1, 0.5, 0.9), 0.05, 0.3)
tau <- qunif(c(0.1, 0.5, 0.9))
theta <- t(as.matrix(expand.grid(rho, tau)))
Z <- lapply(1:J, function(j) simulator(theta))  
Z <- unlist(Z, recursive = FALSE)

Z1_MCAR <- lapply(Z, removedata, proportion = 0.2)
Z1_MNAR <- lapply(Z, remove_quarter_circle)
UW_MCAR <- encodedata(Z1_MCAR)
UW_MNAR <- encodedata(Z1_MNAR)

savedata <- function(Z, theta, missingness) {
  K <- ncol(theta)
  Z <- Z[1:K] # just save the first data set for each field
  J <- length(Z) / K
  z <- lapply(Z, c) 
  z <- unlist(z)
  d <- prod(dim(Z[[1]]))
  k <- rep(rep(1:K, each = J), each = d)
  j <- rep(rep(1:J, each = d), times = K)
  df <- data.frame(Z = z, k = k, j = j)
  save_path <- file.path(int_path, "Estimates", paste0("Z_", missingness, ".csv"))
  write.csv(df, save_path, row.names = FALSE)
}
savedata(Z1_MCAR, theta, "MCAR")
savedata(Z1_MNAR, theta, "MNAR")

cat("Running the masked NBE...")
masked_MCAR <- estimate(maskedestimator, UW_MCAR)
masked_MNAR <- estimate(maskedestimator, UW_MNAR)
cat("Running the neural EM algorithm...")
EM_MCAR <- EM_multiple(Z1_MCAR, theta_0 = theta_0, estimator = neuralMAP)
EM_MNAR <- EM_multiple(Z1_MNAR, theta_0 = theta_0, estimator = neuralMAP)
cat("Computing the MAP estimates...")
MAP_MCAR <- MAP_multiple(Z1_MCAR)
MAP_MNAR <- MAP_multiple(Z1_MNAR)
saveRDS(masked_MCAR, file = file.path(int_path, "Estimates", "Scenarios", "masked_MCAR.rds"))
saveRDS(masked_MNAR, file = file.path(int_path, "Estimates", "Scenarios", "masked_MNAR.rds"))
saveRDS(EM_MCAR, file = file.path(int_path, "Estimates", "Scenarios", "EM_MCAR.rds"))
saveRDS(EM_MNAR, file = file.path(int_path, "Estimates", "Scenarios", "EM_MNAR.rds"))
saveRDS(MAP_MCAR, file = file.path(int_path, "Estimates", "Scenarios", "MAP_MCAR.rds"))
saveRDS(MAP_MNAR, file = file.path(int_path, "Estimates", "Scenarios", "MAP_MNAR.rds"))

estimates_dataframe <- function(estimates, truth, estimator_name) {
  
  p <- nrow(truth)
  K <- ncol(truth)
  J <- ncol(estimates) / K
  
  truth_repeated <- matrix(rep(truth, J), nrow = nrow(truth))
  
  data.frame(
    m = 1, 
    k = rep(1:K, each = p),
    j = rep(1:J, each = p*K),
    estimator = estimator_name, 
    parameter = rep(parameter_names, times = K),
    estimate = c(estimates), 
    truth = c(truth_repeated)
  )
}

MCAR_df <- estimates_dataframe(masked_MCAR, theta, "masking") %>% 
  rbind(estimates_dataframe(MAP_MCAR, theta, "MAP")) %>% 
  rbind(estimates_dataframe(EM_MCAR, theta, "neuralEM")) 

MNAR_df <- estimates_dataframe(masked_MNAR, theta, "masking") %>% 
  rbind(estimates_dataframe(MAP_MNAR, theta, "MAP")) %>% 
  rbind(estimates_dataframe(EM_MNAR, theta, "neuralEM")) 

write.csv(MCAR_df, file.path(int_path, "Estimates", "estimates_MCAR_scenarios.csv"), row.names = F)
write.csv(MNAR_df, file.path(int_path, "Estimates", "estimates_MNAR_scenarios.csv"), row.names = F)

## ---- Timings for a single data set ----

df <- tibble(estimator = character(), time = numeric())

# MAP
start_time <- Sys.time()
MAP(Z1_MCAR[[1]])
end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
df <- df %>% add_row(estimator = "MAP", time = elapsed_time)

# Neural EM
start_time <- Sys.time()
EM(Z1_MCAR[[1]], theta_0 = theta_0, estimator = neuralMAP, nsims = H)
end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
df <- df %>% add_row(estimator = "neuralEM", time = elapsed_time)

# Masked neural Bayes estimator
start_time <- Sys.time()
estimate(maskedestimator, UW_MCAR[[1]])
end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
df <- df %>% add_row(estimator = "masking_with_overhead", time = elapsed_time)

elapsed_time <- juliaLet('
uw = gpu(uw)
maskedestimator = gpu(maskedestimator)
@elapsed maskedestimator(uw)
', maskedestimator = maskedestimator, uw = UW_MCAR[[1]])
df <- df %>% add_row(estimator = "masking", time = elapsed_time)

write.csv(df, file.path(int_path, "runtime_singledataset.csv"))

# ---- Results ----

estimates_path <- file.path(int_path, "Estimates")
missingness <- c("MCAR", "MNAR")
estimator_labels <- c(
  "MAP" = "MAP",
  "neuralEM" = "Neural EM",
  "masking" = "Masked NBE"
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

p <- sum(names(parameter_labels) %in% df$parameter)

zdf <- loaddata(missingness[1]) %>% rbind(loaddata(missingness[2]))

N <- nrow(Z_test[[1]])
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
        labs(fill = expression(Z[1])) +
        theme(legend.title.align = 0.25, legend.title = element_text(face = "bold"))
    })
  })
  data_legend <- get_legend(data[[1]])
  
  data <- lapply(data, function(gg) gg +
                   theme(legend.position = "none") +
                   theme(plot.title = element_text(hjust = 0.5) +
                           coord_fixed()))
  
  suppressMessages({
    data[-l] <- lapply(data[-l], function(gg) gg +
                         theme(axis.text.x = element_blank(),
                               axis.ticks.x = element_blank(),
                               axis.title.x = element_blank()) +
                         scale_x_continuous(breaks = c(0.2, 0.5, 0.8), expand = c(0, 0))
    )
  })
  
  
  
  ## Box plots
  df$estimator <- factor(df$estimator, levels = c("MAP", "neuralEM", "masking"))
  box <- lapply(missingness, function(mis) {
    plotdistribution(filter(df, missingness == mis), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1, return_list = TRUE, flip = TRUE)
  })
  p <- length(unique(df$parameter))
  box_split <- lapply(1:p, function(i) {
    lapply(1:length(box), function(j) box[[j]][[i]])
  })
  
  # Modify the axes
  for (i in 1:p) {
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
  ncol <- p + 2
  figure  <- egg::ggarrange(plots = plotlist, nrow = nrow, ncol = ncol, byrow = FALSE)
  
  ggsv(file = paste0("missing_boxplots_k", kk), plot = figure, path = img_path, 
       width = 2 * (p+2), height = 4.5)
  
})

