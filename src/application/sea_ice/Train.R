# Boolean indicating whether (TRUE) or not (FALSE) to quickly establish that the code is working properly
# quick <- identical(commandArgs(trailingOnly=TRUE)[1], "--quick")
args <- commandArgs(trailingOnly = TRUE)

# Flags
quick <- "--quick" %in% args

# Domain argument (default "sub" if not provided)
domain_arg <- args[grepl("^--domain=", args)]
if (length(domain_arg) == 0) {
  domain <- "full"
} else {
  domain <- sub("^--domain=", "", domain_arg)
  if (!domain %in% c("sub", "full")) {
    stop("Invalid value for --domain. Must be 'sub' or 'full'.")
  }
}

cat("quick =", quick, "\n")
cat("domain =", domain, "\n")
source(file.path("src", "Plotting.R"))

# ---- Visualize prior ----

# Emission parameters
a1_support <- c(2, 5)
a2_support <- c(2, 5)
b1_support <- c(2, 5)
b2_support <- c(0.1, 0.9)

# Specify parameter sets to plot
components <- tibble(
  component = c("1", "1", "1", "1", "2", "2", "2", "2"),
  alpha = c(a1_support[1], a1_support[1],  a1_support[2], a1_support[2], 
            a2_support[1], a2_support[1],  a2_support[2], a2_support[2]),
  beta  = c(b1_support, b1_support, b2_support, b2_support),
  set = 1:8
) %>%
  mutate(label = paste0("α=", alpha, ", β=", beta))

# Create x values
x <- seq(0, 1, length.out = 500)

# Compute densities
df <- components %>%
  rowwise() %>%
  mutate(density = list(dbeta(x, alpha, beta))) %>%
  unnest(cols = c(density)) %>%
  mutate(x = rep(x, nrow(components)))

# Plot
ggplot(df, aes(x = x, y = density, color = label)) +
  geom_line() +
  facet_wrap(component ~ ., labeller = labeller(component = label_both)) +
  labs(
    x = "Z",
    y = "Density",
    color = "Parameters"
  ) +
  theme_bw(base_size = 14)

# ---- Simulate data ----

cat("Simulating training data...")

suppressMessages({
library("bayesImageS")
library("doParallel")
library("dplyr")
library("ggplot2")
})

int_path <- file.path("intermediates", "application", "sea_ice", domain)
img_path <- file.path("img", "application", "sea_ice", domain)
dir.create(int_path, recursive = TRUE, showWarnings = FALSE)
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

K <- if (quick) 5000 else 50000  # number of independent datasets

# Sample from the prior
n_beta <- 2        # number of beta mixture components
q <- 2 + n_beta    # number of labels
d <- 2*n_beta + 1  # number of parameters in the model

# Potts parameter
beta_crit <- log(1 + sqrt(q))
beta_lower <- 0.03
beta_upper <- 1.15 * beta_crit
beta <- runif(K, beta_lower, beta_upper)

# Store parameters in matrices: rows = components, cols = replicates (K)
a <- matrix(NA, nrow = n_beta, ncol = K)
b <- matrix(NA, nrow = n_beta, ncol = K)

for (i in 1:K) {
  # Sample until ordering constraint holds
  ok <- FALSE
  while (!ok) {
    # a_prop <- runif(n_beta, a_lower, a_upper)
    # b_prop <- runif(n_beta, b_lower, b_upper)
    # 
    # means <- a_prop / (a_prop + b_prop)
    # if (all(diff(means) > 0)) {
    #   a[, i] <- a_prop
    #   b[, i] <- b_prop
    #   ok <- TRUE
    # }
    
    a_prop <- c(runif(1, a1_support[1], a1_support[2]), runif(1, a2_support[1], a2_support[2]))
    b_prop <- c(runif(1, b1_support[1], b1_support[2]), runif(1, b2_support[1], b2_support[2]))
    
    means <- a_prop / (a_prop + b_prop)
    if (all(diff(means) > 0)) {
      a[, i] <- a_prop
      b[, i] <- b_prop
      ok <- TRUE
    }
  }
}
# hist(a[1, ])
# hist(a[2, ])
# hist(b[1, ])
# hist(b[2, ])

theta <- rbind(beta, a, b)
# prior_lower_bound <- c(beta_lower, rep(a_lower, n_beta), rep(b_lower, n_beta))
# prior_upper_bound <- c(beta_upper, rep(a_upper, n_beta), rep(b_upper, n_beta))
prior_lower_bound <- c(beta_lower, a1_support[1], a2_support[1], b1_support[1], b2_support[1])
prior_upper_bound <- c(beta_upper, a1_support[2], a2_support[2], b1_support[2], b2_support[2])
prior_mean <- rowMeans(theta)
saveRDS(prior_mean, file = file.path(int_path, "prior_mean.rds"))
saveRDS(prior_lower_bound, file = file.path(int_path, "prior_lower_bound.rds"))
saveRDS(prior_upper_bound, file = file.path(int_path, "prior_upper_bound.rds"))

# Setup grid dimensions, mask, neighbors, and blocks
grid_dim <- readRDS(file.path(int_path, "dim_grid.rds"))
mask  <- matrix(1, grid_dim[1], grid_dim[2])
neigh <- getNeighbors(mask, c(2,2,0,0))
block <- getBlocks(mask, 2)

# Parallel setup
nc <- detectCores() / 2
cl <- makeCluster(nc)
clusterSetRNGStream(cl)
registerDoParallel(cl)

# Hidden Potts simulation
tm <- system.time({
  Z <- foreach(i = 1:K, .multicombine = TRUE, .packages = c("bayesImageS", "fields")) %dopar% {
    
    # Latent labels
    burn <- 100 # Swendsen-Wang burn-in
    be <- beta[i]
    r <- swNoData(be, q, neigh, block, burn)
    Y <- matrix(max.col(r$z[-1, ]),
                     nrow = grid_dim[1],
                     ncol = grid_dim[2],
                     byrow = FALSE)
    fields::image.plot(Y)
    
    # Observed field
    Z <- matrix(NA, nrow(Y), ncol(Y))
    
    # Point masses
    Z[Y == 1] <- 0
    Z[Y == 2] <- 1
    
    # Beta components
    for (m in 1:n_beta) {
      mask <- Y == (2 + m)
      Z[mask] <- rbeta(sum(mask), shape1 = a[m, i], shape2 = b[m, i])
    }
    # fields::image.plot(Z)
    
    return(Z)
  }
})

saveRDS(tm, file = file.path(int_path, "sim_time.rds"))

stopCluster(cl)

fields::image.plot(Z[[which.min(beta)]])
fields::image.plot(Z[[which.min(abs(beta - (beta_crit - 0.05)))]])
fields::image.plot(Z[[which.min(abs(beta - (beta_crit - 0.00)))]])
fields::image.plot(Z[[which.min(abs(beta - (beta_crit + 0.05)))]])
fields::image.plot(Z[[which.max(beta)]])

# ---- Partition simulated data into training, validation, and test sets ----

## Coerce data to required format 
Z <- lapply(Z, function(z) {
  dim(z) <- c(dim(z)[1], dim(z)[2], 1, 1) 
  z
})

## Partition the data into training, validation, and test sets
K <- length(Z)
K1 <- ceiling(0.8*K)  # size of the training set 
K3 <- if (quick) 50 else 1000 # size of the test set 
K2 <- K - K1 - K3     # size of the validation set 
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

# ---- Construct NBE ----

## R and Julia packages for simulation and neural Bayes estimation
library("NeuralEstimators")
library("JuliaConnectoR")
# NB first must set working directory to top-level of repo
Sys.setenv("JULIACONNECTOR_JULIAOPTS" = "--project=.") 
juliaEval('using NeuralEstimators, Flux, CUDA')
juliaEval('include(joinpath(pwd(), "src", "Architecture.jl"))')
architecture <- juliaFun("architecture")

## Initialize the estimator
estimator <- juliaLet('architecture(d, a, b; input_channels = 1, J = 5)', d = d, a = prior_lower_bound, b = prior_upper_bound)

## Train the estimator 
estimator <- train(estimator, 
                   theta_train = theta_train, 
                   theta_val = theta_val, 
                   Z_train = Z_train, 
                   Z_val = Z_val, 
                   savepath = file.path(int_path, "NBE"))

## Load the estimator
estimator <- loadstate(estimator,  file.path(int_path, "NBE", "ensemble.bson"))

## Assess the estimator using recovery plots
assessment <- assess(estimator, theta_test, Z_test, estimator_names = "NBE")
df <- estimates <- assessment$estimates

parameter_labels = c(
  "θ1" = expression(beta), 
  "θ2" = expression(a[1]), 
  "θ3" = expression(a[2]), 
  "θ4" = expression(b[1]), 
  "θ5" = expression(b[2])
)

df <- dplyr::mutate_at(df, .vars = "parameter", .funs = factor, levels = names(parameter_labels), labels = parameter_labels)



df$beta_truth <- df$truth[df$parameter=="beta"][df$k]

beta_max <- beta_crit + 0.15
df_plot <- df %>%
  group_by(k) %>%
  filter(any(parameter == "beta" & truth < beta_max)) %>%
  ungroup()

df_plot <- df_plot[sample(nrow(df_plot)), ]

figure <- ggplot2::ggplot(df_plot) + 
  ggplot2::geom_point(ggplot2::aes(x=truth, y = estimate, colour = beta_truth), alpha = 0.6, size = 0.4) + 
  ggplot2::geom_abline(colour = "black", linetype = "dashed") +
  scale_colour_gradientn(
    colours = c("#0571B0", "#f0f0f0", "red"),
    values = scales::rescale(c(min(df_plot$beta_truth), beta_crit, max(df_plot$beta_truth))),
    name = expression(beta)
  ) +
  ggh4x::facet_grid2(estimator~parameter, scales = "free", independent = "y", labeller = label_parsed) + 
  ggplot2::labs(colour = expression(beta)) + 
  ggplot2::theme_bw() +
  theme(
    strip.text.y = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(size = 12)
    # legend.position = "none"
  )

ggsv(file.path(img_path, "NBE_assessment"), figure, width = 8.3, height = 2.4)