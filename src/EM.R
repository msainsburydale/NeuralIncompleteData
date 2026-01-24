library("progress")

# Returns:
# - Estimate, the averaged iterates, where the averaging starts from iteration burn_in + 1
# - The user-specified burn in period
# - Raw iterates (including initial value, theta_0)
EM <- function(Z1,                      # data (a matrix containing NAs)
               estimator,               # neural MAP estimator
               theta_0,                 # initial estimate
               simulateconditional,     # simulation function
               setupconditionalsimulation = NULL, 
               burn_in = 1,              # burn-in period
               niterations = 10,        # maximum number of iterations
               tol = 0.05,              # convergence tolerance
               nconsecutive = 1,        # number of consecutive iterations required for convergence
               nsims = 1,               # Monte Carlo sample size (integer or vector of length niterations)
               verbose = FALSE          # print current estimate if TRUE
) {
  
  # Validate nsims
  if (length(nsims) > 1) {
    if (length(nsims) != niterations) {
      stop("When nsims is a vector, its length must equal niterations (", niterations, ")")
    }
    if (any(nsims <= 0)) {
      stop("All elements of nsims must be positive")
    }
  } else {
    if (nsims <= 0) {
      stop("nsims must be positive")
    }
  }
  
  if (verbose)
    cat("Initial estimate:", paste(as.vector(theta_0), collapse = ", "), "\n")
  
  theta_l <- theta_0
  convergence_counter <- 0
  p <- length(theta_0)
  theta_all <- matrix(NA, nrow = p, ncol = niterations + 1)
  theta_all[, 1] <- theta_0
  bar_theta_l <- NULL
  
  Z1 <- drop(Z1)
  
  if (!is.null(setupconditionalsimulation)) {
    input <- setupconditionalsimulation(Z1)
  } else {
    input <- Z1
  }
  
  for (l in 1:niterations) {
    
    # Get current nsims value (either from vector or use scalar)
    nsims_current <- if (length(nsims) > 1) nsims[l] else nsims
    
    # complete the data by conditional simulation
    Z <- simulateconditional(input, theta_l, nsims = nsims_current)
    
    # MAP estimation
    theta_l_plus_1 <- c(estimate(estimator, Z))
    theta_all[, l + 1] <- theta_l_plus_1
    
    # convergence check (after burn-in)
    if (l > burn_in) {
      bar_theta_l_plus_1 <- rowMeans(theta_all[, (burn_in + 1):(l + 1), drop = FALSE])
      
      if (!is.null(bar_theta_l)) {
        rel_change <- max(abs(bar_theta_l_plus_1 - bar_theta_l) / (abs(bar_theta_l) + .Machine$double.eps))
        
        if (rel_change < tol) {
          convergence_counter <- convergence_counter + 1
          if (convergence_counter >= nconsecutive) {
            if (verbose) {
              cat("Iteration", l, "(nsims =", nsims_current, "):", 
                  paste(round(theta_l_plus_1, 4), collapse = ", "), "\n")
              message("The EM algorithm has converged.")
            }
            break
          }
        } else {
          convergence_counter <- 0
        }
      }
      
      bar_theta_l <- bar_theta_l_plus_1
    }
    
    theta_l <- theta_l_plus_1
    if (verbose) {
      cat("Iteration", l, "(nsims =", nsims_current, "):", 
          paste(round(theta_l, 4), collapse = ", "), "\n")
    }
    
    if (l == niterations && verbose)
      warning("The EM algorithm did not converge after the maximum number of iterations.")
  }
  
  # trim unused columns
  theta_all <- theta_all[, 1:(l + 1), drop = FALSE]
  
  # final averaged estimate
  estimate <- if (l > burn_in) rowMeans(theta_all[, (burn_in + 1):ncol(theta_all), drop = FALSE]) else theta_all[, ncol(theta_all)]
  
  # return list
  list(
    estimate = estimate,
    burn_in = burn_in,
    iterates = theta_all
  )
}

EM_multiple <- function(Z1, estimator, theta_0, ...) {
  
  # If theta_0 is a vector, replicate it for each element of Z1
  if (is.vector(theta_0)) {
    theta_0 <- matrix(rep(theta_0, length(Z1)), ncol = length(Z1))  # Replicate for each Z1
  }
  
  pb <- progress_bar$new(
    format = "[:bar] :current/:total (:percent) | Elapsed: :elapsed | ETA: :eta",
    total = length(Z1),
    clear = FALSE,
    width = 80
  )
  
  #NB Sometimes got segfaults when running parallel... just doing it sequentially for safety
  
  # plan(multisession, workers = min(8, availableCores() %/% 2))
  # plan(sequential)
  # plan(multicore, workers = 2)

  # Apply EM for each Z1 with corresponding theta_0 column
  thetahat <- lapply(seq_along(Z1), function(i) {
    # cat(sprintf("Processing dataset %d of %d\n", i, length(Z1)))
    pb$tick()
    estimate <- EM(Z1[[i]], estimator, theta_0[, i], ...)$estimate
    gc()
    return(estimate)
  })
  
  thetahat <- do.call(cbind, thetahat)
  return(thetahat)
}

# Helper function for visualizing EM iterates
run_EM <- function(Z1) {
  dfs <- lapply(seq_along(theta_0), function(i) {
    theta0 <- theta_0[[i]]
    burn_in <- 10
    
    do.call(rbind, lapply(all_nsims, function(nsims) {
      res <- EM(
        Z1, 
        estimator = neuralMAP, 
        simulateconditional = simulateconditional, 
        theta_0 = theta0, 
        verbose = FALSE, 
        burn_in = burn_in,
        tol = 0.0000001,
        nsims = nsims,
        niterations = 100
      )
      
      estimates <- res$iterates  
      
      # Compute post–burn-in running means for each parameter
      averaged_estimates <- apply(estimates, 1, function(param_values) {
        n <- length(param_values)
        avg <- rep(NA, n)
        if (n > burn_in) {
          for (t in (burn_in+1):n) {
            avg[t] <- mean(param_values[(burn_in+1):t])
          }
        }
        avg
      })
      
      # Convert to long data frame
      df <- data.frame(
        iteration = rep(1:ncol(estimates), each = nrow(estimates)),
        parameter = rep(parameter_names, times = ncol(estimates)),
        estimate = as.vector(estimates),
        averaged_estimate = as.vector(t(averaged_estimates)),
        theta_0 = i,
        nsims = nsims
      )
      
      df
    }))
  })
  df <- do.call(rbind, dfs)
  return(df)
}