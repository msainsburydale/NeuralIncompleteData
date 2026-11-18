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

# ---- Batched neural EM ----

# Lock-step neural EM for many data sets at once: the R counterpart of batched_em() in src/EM.jl.
#
# In each EM iteration, the active data sets are processed in chunks of `chunksize`: the conditional simulations of a
# chunk are run in parallel on the workers of the cluster `cl` (or in this R process if `cl` is NULL), and the neural
# MAP estimator is then applied to the whole chunk with a single call to Julia. Each worker of `cl` keeps the
# set-up objects (orderings, neighbour sets) of the data sets assigned to it, so that these are computed and sent
# once. The convergence assessment is identical to that of batched_em() (and NeuralEstimators.EM): with θ̄ₗ the
# average of the estimates from iteration burn_in to iteration l, a data set has converged once
# max_i |θ̄ₗ₊₁,ᵢ − θ̄ₗ,ᵢ| / (|θ̄ₗ,ᵢ| + ϵ) < tol for nconsecutive consecutive iterations, after which it leaves the batch.
#
# Workers must have the functions setupconditionalsimulation and simulateconditional (given by name), e.g., via
# clusterEvalQ(cl, source("src/GP/Simulation.R")).
#
# Returns a list with: estimates (d × K matrix), iterates (list of d × (l+1) matrices, including theta_0),
# niterations, converged, and timings (seconds spent in set-up, conditional simulation, and MAP estimation).
# Functions run on the workers of batched_EM(). They are defined at the top level so that, when serialised, they do
# not carry the data of the calling frame with them.
.em_worker_setup <- function(job, setupfun) {
  f <- get(setupfun)
  assign(".em_setups", setNames(lapply(job$Z, f), job$ks), envir = globalenv())
  NULL
}
.em_worker_simulate <- function(job, simfun, nsims) {
  sim <- get(simfun)
  setups <- get(".em_setups", envir = globalenv())
  lapply(seq_along(job$ks), function(i) sim(setups[[as.character(job$ks[i])]], job$thetas[, i], nsims = nsims))
}

batched_EM <- function(Z1, estimator, theta_0,
                       setupconditionalsimulation = "vecchia_setup",
                       simulateconditional = "simulate_conditional",
                       cl = NULL,
                       nsims = 30, niterations = 50, burn_in = 5, tol = 0.01, nconsecutive = 3,
                       chunksize = 256, verbose = FALSE) {

  stopifnot(burn_in >= 0, burn_in < niterations, chunksize > 0)
  K <- length(Z1)
  if (is.vector(theta_0)) theta_0 <- matrix(rep(theta_0, K), ncol = K)
  stopifnot(ncol(theta_0) == K)
  d <- nrow(theta_0)

  W <- if (is.null(cl)) 1L else length(cl)
  worker <- ((seq_len(K) - 1L) %% W) + 1L # worker holding the set-up of each data set

  # NB the timers below use gcFirst = FALSE: by default, system.time() first runs a full garbage collection, which
  # (outside the timed expression, but inside the EM) would cost about 0.1 s per call.

  # Set-up for conditional simulation, once per data set
  t_setup <- system.time({
    if (is.null(cl)) {
      setups <- lapply(Z1, get(setupconditionalsimulation))
    } else {
      jobs <- lapply(seq_len(W), function(w) list(ks = which(worker == w), Z = Z1[worker == w]))
      parallel::clusterApply(cl, jobs, .em_worker_setup, setupfun = setupconditionalsimulation)
    }
  }, gcFirst = FALSE)[["elapsed"]]

  simulate_chunk <- function(chunk, thetas) {
    if (is.null(cl)) {
      sim <- get(simulateconditional)
      return(lapply(seq_along(chunk), function(i) sim(setups[[chunk[i]]], thetas[, i], nsims = nsims)))
    }
    w <- worker[chunk]
    jobs <- lapply(seq_len(W), function(v) list(ks = chunk[w == v], thetas = thetas[, w == v, drop = FALSE]))
    out <- parallel::clusterApply(cl, jobs, .em_worker_simulate, simfun = simulateconditional, nsims = nsims)
    Z <- vector("list", length(chunk))
    for (v in seq_len(W)) Z[w == v] <- out[[v]]
    Z
  }

  theta_all <- lapply(seq_len(K), function(k) matrix(theta_0[, k], ncol = 1))
  bar_theta <- vector("list", K)
  counter <- integer(K)
  converged <- logical(K)
  done <- logical(K)
  nits <- integer(K)
  t_sim <- 0
  t_est <- 0

  active <- seq_len(K)
  for (l in seq_len(niterations)) {
    if (length(active) == 0) break
    for (chunk in split(active, ceiling(seq_along(active) / chunksize))) {

      thetas <- sapply(chunk, function(k) theta_all[[k]][, ncol(theta_all[[k]])])
      thetas <- matrix(thetas, nrow = d)

      # Conditional simulation, in parallel over data sets
      t_sim <- t_sim + system.time(Z <- simulate_chunk(chunk, thetas), gcFirst = FALSE)[["elapsed"]]

      # MAP estimation: one call for the chunk
      t_est <- t_est + system.time(thetahat <- estimate(estimator, Z), gcFirst = FALSE)[["elapsed"]]
      thetahat <- matrix(thetahat, nrow = d)

      # Convergence assessment (identical to batched_em in src/EM.jl)
      for (i in seq_along(chunk)) {
        k <- chunk[i]
        theta_all[[k]] <- cbind(theta_all[[k]], thetahat[, i])
        nits[k] <- l
        if (l > burn_in) {
          b <- rowMeans(theta_all[[k]][, (burn_in + 1):ncol(theta_all[[k]]), drop = FALSE])
          if (l > burn_in + 1 && max(abs(b - bar_theta[[k]]) / (abs(bar_theta[[k]]) + .Machine$double.eps)) < tol) {
            counter[k] <- counter[k] + 1
            if (counter[k] == nconsecutive) converged[k] <- done[k] <- TRUE
          } else {
            counter[k] <- 0
          }
          bar_theta[[k]] <- b
        }
      }
    }
    active <- active[!done[active]]
    if (verbose) cat(sprintf("EM iteration %d: %d of %d data sets still active (set-up %.1f s, simulation %.1f s, estimation %.1f s)\n",
                             l, length(active), K, t_setup, t_sim, t_est))
  }
  if (verbose && !all(converged)) warning(sprintf("The EM algorithm failed to converge for %d of %d data sets", sum(!converged), K))

  # Free the set-up objects held by the workers
  if (!is.null(cl)) parallel::clusterEvalQ(cl, rm(list = ".em_setups", envir = globalenv()))

  estimates <- sapply(seq_len(K), function(k) if (is.null(bar_theta[[k]])) theta_all[[k]][, ncol(theta_all[[k]])] else bar_theta[[k]])
  list(
    estimates = matrix(estimates, nrow = d),
    iterates = theta_all,
    niterations = nits,
    converged = converged,
    timings = c(setup = t_setup, simulation = t_sim, estimation = t_est)
  )
}

# Helper function for visualizing EM iterates: runs the EM algorithm for one incomplete data set Z1 from each initial
# value in theta_0 (a list) and each Monte Carlo sample size in all_nsims, with a negligible tolerance
run_EM <- function(Z1, estimator, theta_0, all_nsims = list(1, 10, 30), parameter_names, burn_in = 10, niterations = 100, ...) {
  dfs <- lapply(seq_along(theta_0), function(i) {
    do.call(rbind, lapply(all_nsims, function(nsims) {
      res <- batched_EM(list(Z1), estimator, theta_0[[i]], nsims = nsims, burn_in = burn_in,
                        tol = 1e-7, niterations = niterations, ...)
      estimates <- res$iterates[[1]]

      # Post–burn-in running means for each parameter
      averaged_estimates <- apply(estimates, 1, function(param_values) {
        n <- length(param_values)
        avg <- rep(NA, n)
        if (n > burn_in) for (t in (burn_in + 1):n) avg[t] <- mean(param_values[(burn_in + 1):t])
        avg
      })

      data.frame(
        iteration = rep(1:ncol(estimates), each = nrow(estimates)),
        parameter = rep(parameter_names, times = ncol(estimates)),
        estimate = as.vector(estimates),
        averaged_estimate = as.vector(t(averaged_estimates)),
        theta_0 = i,
        nsims = nsims
      )
    }))
  })
  do.call(rbind, dfs)
}
