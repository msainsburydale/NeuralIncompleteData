suppressMessages({
source(file.path("src", "Plotting.R"))
library("latex2exp")
library("reshape2")
# install.packages("BiocManager")
# BiocManager::install("rhdf5")
library("rhdf5")
library("readr")
library("patchwork")
options(dplyr.summarise.inform = FALSE) 
})

map_Y_values <- function(Y) {
  # Define mapping (index = old value, element = new value)
  mapping <- c(2, 3, 1, 4)
  Y[] <- mapping[Y]   # important: using Y[] <- preserves the shape of Y
  return(Y)
}

if(!interactive()) pdf(NULL)
oldw <- getOption("warn")
options(warn = -1)

img_path <- file.path("img", "application", "sea_ice")
int_path <- file.path("intermediates", "application", "sea_ice")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

land_mask <- readRDS(file.path(int_path, "full", "land_mask.rds"))
land_border_mask <- readRDS(file.path(int_path, "full", "land_border_mask.rds"))

parameter_labels = c(
  "θ1" = expression(beta), 
  "θ2" = expression(a[1]), 
  "θ3" = expression(a[2]), 
  "θ4" = expression(b[1]), 
  "θ5" = expression(b[2])
)

year <- 1979:2023

read_estimates <- function(domain) {
  # Original estimates
  estimates <- read_csv(file.path(int_path, domain, "estimates.csv"), col_names = FALSE, show_col_types = FALSE)
  estimates <- t(as.matrix(estimates))
  colnames(estimates) <- names(parameter_labels)
  rownames(estimates) <- NULL
  estimates <- as.data.frame(estimates)
  estimates$year <- year
  estimates$domain <- domain
  estimates <- estimates %>%
    pivot_longer(
      cols = all_of(names(parameter_labels)),  
      names_to = "parameter",
      values_to = "estimate"
    )
  
  # Bootstrap estimates
  bs_estimates <- read.csv(file.path(int_path, domain, "bs_estimates_complete.csv"), header = F)
  bs_estimates <- as.matrix(bs_estimates)
  K <- length(year)
  B <- ncol(bs_estimates) / K
  p <- nrow(bs_estimates)
  bs_estimates <- data.frame(
    parameter = rep(names(parameter_labels), each = K * B),
    year      = rep(rep(1979:2023, each = B), times = p),
    value     = as.vector(t(bs_estimates))
  )
  
  # Compute bootstrap sd and Wald interval
  estimates <- bs_estimates %>%
    group_by(parameter, year) %>%
    summarise(sd_boot = sd(value), .groups = "drop") %>%
    left_join(estimates, by = c("parameter", "year")) %>%
    mutate(
      z = qnorm(0.975),
      lower = estimate - z * sd_boot,
      upper = estimate + z * sd_boot,
      med = estimate
    )
  
  return(estimates)
}
estimates <- lapply(c("full", "sub"), read_estimates)
estimates <- do.call(rbind, estimates)

# Plotting only Potts parameter, beta
p <- length(parameter_labels)
q <- (p-1)/2 + 2
beta_crit <- log(1 + sqrt(q))
prior_lower_bound <- readRDS(file.path(int_path, "full", "prior_lower_bound.rds"))
prior_upper_bound <- readRDS(file.path(int_path, "full", "prior_upper_bound.rds"))
ylims <- c(
  0.5,
 # prior_lower_bound[1],
  # NA,
  prior_upper_bound[1]
  )
beta_plot <- ggplot(estimates %>% filter(parameter == "θ1", domain == "full"), aes(x = year, y = estimate)) +
  geom_point() +
  geom_line() +
  geom_hline(aes(yintercept = beta_crit), linetype = "dashed") +
  geom_ribbon(aes(x = year, ymin = lower, ymax = upper), alpha = 0.3) +
  labs(y = expression(hat(beta))) +
  scale_x_continuous(expand = c(0, 0)) +
  # ylim(ylims) +
  theme_bw()

# Plotting all parameters
plot_parameters <- function(estimates) {
  estimates <- dplyr::mutate_at(
    estimates,
    .vars = "parameter",
    .funs = factor,
    levels = names(parameter_labels),
    labels = parameter_labels
  )

  # Compute limits for the parameters of the beta components
  first_param <- unique(estimates$parameter)[1]
  shared_limits <- estimates %>%
    filter(parameter != first_param) %>%
    summarise(
      ymin = min(lower, na.rm = TRUE),
      ymax = max(upper, na.rm = TRUE)
    )

  # check how many domains
  use_linetype <- nlevels(factor(estimates$domain)) > 1

  # set base aes
  base_aes <- aes(x = year, y = med)
  if (use_linetype) {
    base_aes <- aes(x = year, y = med, linetype = domain)
  }

  p <- ggplot(estimates, base_aes) +
    geom_line() +
    geom_point(size = 0.5) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.3) +
    facet_wrap(~ parameter, scales = "free_y", nrow = 1, labeller = label_parsed) +
    geom_hline(
      data = data.frame(parameter = first_param, yintercept = beta_crit),
      aes(yintercept = yintercept),
      linetype = "dashed", color = "black"
    ) +
    labs(y = "Estimate (Bootstrap CI)", x = "Year") +
    theme_bw() # +
    # ggh4x::facetted_pos_scales(
    #   y = list(
    #     parameter == first_param ~ scale_y_continuous(), # free scale for first
    #     TRUE ~ scale_y_continuous(
    #       limits = c(shared_limits$ymin, shared_limits$ymax)
    #     ) # shared for others
    #   )
    # )

  if (use_linetype) {
    p <- p + labs(linetype = "Domain")
  }

  return(p)
}
gg <- plot_parameters(estimates %>% filter(domain == "full"))
ggsv("all_parameters", gg, path = file.path(img_path, "full"), width = 11, height = 2.6)
gg <- plot_parameters(estimates %>% filter(domain == "sub"))
ggsv("all_parameters", gg, path = file.path(img_path, "sub"), width = 11, height = 2.6)
gg <- plot_parameters(estimates)
ggsv("all_parameters", gg, path = img_path, width = 11, height = 2.6)

# Sea-ice area
sie <- read.csv(file.path(int_path, "full", "sie.csv"), header = F)
sie <- as.matrix(sie)
df <- data.frame(year = year)
df$sie <-  apply(sie, 1, function(x) quantile(x, 0.5))
df$sie_lower <- apply(sie, 1, function(x) quantile(x, 0.025))
df$sie_upper <- apply(sie, 1, function(x) quantile(x, 0.975))

sie <- ggplot(df, aes(x = year, y = sie)) +
  geom_point() +
  geom_line() +
  geom_ribbon(aes(x = year, ymin = sie_lower, ymax = sie_upper), alpha = 0.3) +
  labs(y = "Sea-ice area") +
  scale_x_continuous(expand = c(0, 0)) +
  theme_bw()

fig <- egg::ggarrange(beta_plot, sie, nrow = 1)

## Conditional simulation at ice-sheet boundary
sea_ice <- readRDS(file.path(int_path, "full", "sea_ice.rds"))
sea_ice_1995 <- sea_ice[[17]]
sea_ice_1995 <- reshape2::melt(sea_ice_1995)
sea_ice_1995$year = 1995
sims1995 <- h5read(file.path(int_path, "full", "conditional_sims_1995.h5"), "Z")
probs <- apply(sims1995, c(1, 2), mean) # NB probabilities are much sharper when we use median rather than mean
probs <- reshape2::melt(probs)

gg1 <- ggplot(sea_ice_1995, aes(Var1, Var2, fill=value)) + 
  geom_raster() + 
  geom_tile(data = subset(reshape2::melt(land_border_mask), value == 1), aes(Var1, Var2), fill = "gray", alpha = 0.7) +
  scale_y_reverse(expand = c(0, 0)) + 
  scale_x_continuous(expand = c(0, 0)) + 
  theme_bw() + 
  scale_fill_gradient2(
    low = "darkblue", mid = "skyblue", high = "white", na.value = "red",
    midpoint = 0.5, 
  ) + 
  theme(axis.title = element_text(color = "transparent"),
        axis.text = element_text(color = "transparent"),
        axis.ticks = element_line(color = "transparent"))

# Zoom-in on region of interest
xmin=110; xmax=150
ymin=45; ymax=90

gg1 <- gg1 +
  geom_rect(aes(xmin=xmin, xmax=xmax + 2, ymin=ymin, ymax=ymax),
            size = 1.5, colour = "grey50", fill = "transparent")

window <- ggplot(probs, aes(Var1, Var2, fill = value)) + 
  geom_raster() + 
  # geom_tile(data = subset(reshape2::melt(land_border_mask), value == 1), aes(Var1, Var2), fill = "gray", alpha = 0.7) +
  theme_bw() +
  scale_fill_gradient2(
    low = "darkblue", mid = "skyblue", high = "white", na.value = "red",
    midpoint = 0.5, 
    name = "Predicted\nsea-ice\nproportion", 
    breaks = c(0.2, 0.5, 0.8)
  ) + 
  theme(axis.title = element_text(color = "transparent"),
        axis.text = element_text(color = "transparent"),
        axis.ticks = element_line(color = "transparent")) + 
  coord_fixed(xlim = c(xmin, xmax), ylim = c(ymax, ymin)) + 
  scale_y_reverse(expand = c(0, 0)) + 
  theme(plot.margin = unit(c(0, 30, 0, 0), "points")) + # padding around window (larger second entry moves window more to the left)
  theme(legend.position = "top") 

gg <- ggarrange(
  gg1 + 
    theme(legend.position = "none") + 
    guides(fill = guide_legend(override.aes = list(color = "black"))) + 
    coord_fixed(),
  window 
)

figure <- ggarrange(fig, gg, nrow = 1)

ggsv("beta_seaicearea_predictions", figure, path = img_path, width = 12.1, height = 3.2)


# ---- Plots of conditional simulations ----

year <- 1995
idx <- year - 1978
data <- readRDS(file.path(int_path, "full", "sea_ice.rds"))[[17]] 
Y <- h5read(file.path(int_path, "full", sprintf("conditional_sims_%d.h5", year)), "Y") %>% map_Y_values
Z <- h5read(file.path(int_path, "full", sprintf("conditional_sims_%d.h5", year)), "Z")

library("ggplot2")
library("patchwork")  # for plot arrangement
library("reshape2")   # for melting arrays

# 1. Main spatial plot (left panel)
main_plot <- ggplot(reshape2::melt(data), aes(Var1, Var2, fill = value)) + 
  geom_tile() + 
  theme_bw() + 
  geom_tile(data = subset(reshape2::melt(land_border_mask), value == 1), aes(Var1, Var2), fill = "gray", alpha = 0.7) +
  scale_y_reverse(expand = c(0, 0)) + 
  scale_x_continuous(expand = c(0, 0)) + 
  scale_fill_gradient2(
    low = "darkblue", mid = "skyblue", high = "white", na.value = "red",
    midpoint = 0.5, 
    name = "Sea-ice\nproportion", 
    breaks = c(0.2, 0.5, 0.8)
  ) + 
  theme(axis.title = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(), 
        legend.position = "none") + 
  geom_rect(aes(xmin=xmin, xmax=xmax + 2, ymin=ymin, ymax=ymax),
            size = 1.5, colour = "grey50", fill = "transparent")

create_simulation_plots <- function(sim_array, 
                                       n_cols = 4, n_rows = 1, 
                                       discrete = FALSE,
                                       legend_name = "") {
  # Determine how many plots we need
  n_plots <- n_cols * n_rows
  n_iter <- dim(sim_array)[3]
  
  # Select random iterations if we have more iterations than needed plots
  if (n_iter > n_plots) {
    selected_iters <- sample(1:n_iter, n_plots)
  } else {
    selected_iters <- 1:n_iter
  }
  
  plot_list <- list()
  
  for (i in 1:n_plots) {
    if (i <= length(selected_iters)) {
      iter <- selected_iters[i]
      
      # Convert array slice to data frame
      sim_df <- melt(sim_array[,,iter])
      names(sim_df) <- c("Var1", "Var2", "value")
      
      # Convert to factor if discrete
      if (discrete) {
        sim_df$value <- as.factor(sim_df$value)
      }
      
      p <- ggplot(sim_df, aes(Var1, Var2, fill = value)) + 
        geom_tile() + 
        # geom_tile(data = subset(reshape2::melt(land_border_mask), value == 1), aes(Var1, Var2), fill = "gray", alpha = 0.7) +
        theme_bw() + 
        theme(
          axis.title = element_blank(),
          axis.text = element_blank(),
          axis.ticks = element_blank(),
          plot.margin = margin(2, 2, 2, 2)
        ) + 
        coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymax, ymin)) + 
        scale_x_continuous(expand = c(0, 0)) + 
        scale_y_continuous(expand = c(0, 0)) 
      
      # Apply appropriate fill scale with legend
      if (!discrete) {
        p <- p + scale_fill_gradient2(
          low = "darkblue", 
          mid = "skyblue", 
          high = "white", 
          na.value = "red", 
          midpoint = 0.5,
          name = legend_name, 
          breaks = c(0.2, 0.5, 0.8)
        ) 
      } else {
        p <- p + 
        scale_fill_manual(
          # values = c("orange", "black", "#009E73", "#CC79A7"),
          values = c("darkblue", "#009E73", "orange", "white"),
          name = legend_name
        ) + 
        theme(legend.key = element_rect(color = "black", size = 0.5))
      }
      
    } else {
      # Create empty plot if we don't have enough iterations
      p <- ggplot() + theme_void()
    }
    
    plot_list[[i]] <- p
  }
  
  # Add an empty guide area for the legend
  plot_list[[n_plots + 1]] <- guide_area()
  
  # Arrange plots in grid with legend area
  wrap_plots(plot_list, ncol = n_cols + 1, nrow = n_rows, 
             guides = "collect") + 
    theme(legend.position = "right")
}

Z_sim_plots <- create_simulation_plots(Z[, , 1:4],
                                         discrete = FALSE,
                                         legend_name = "Z values")
Y_sim_plots <- create_simulation_plots(Y[, , 1:4],
                                         discrete = TRUE,
                                         legend_name = "Y labels")

# 4. Arrange all plots
final_plot <- main_plot + 
  (Z_sim_plots / Y_sim_plots) +  # / operator stacks plots vertically
  plot_layout(widths = c(1, 3))  # left panel takes 1/3, right panels take 2/3

ggsv("conditional_sims", final_plot, path = img_path, width = 10.7, height = 3)



# ---- Conditional simulations over whole field (used in presentation) ----

year <- 1995
idx <- year - 1978
data <- readRDS(file.path(int_path, "full", "sea_ice.rds"))[[17]] 
Z <- h5read(file.path(int_path, "full", sprintf("conditional_sims_%d.h5", year)), "Z")

library("ggplot2")
library("patchwork")  # for plot arrangement
library("reshape2")   # for melting arrays


plot_ice <- function(arr) {
  mat <- apply(arr, c(1, 2), mean)

  ggplot(reshape2::melt(mat), aes(Var1, Var2, fill = value)) + 
    geom_tile() + 
    theme_bw() + 
    geom_tile(data = subset(reshape2::melt(land_border_mask), value == 1), aes(Var1, Var2), fill = "gray", alpha = 0.7) +
    scale_y_reverse(expand = c(0, 0)) + 
    scale_x_continuous(expand = c(0, 0)) + 
    scale_fill_gradient2(
      low = "darkblue", mid = "skyblue", high = "white", na.value = "red",
      midpoint = 0.5, 
      name = "Sea-ice\nproportion", 
      breaks = c(0.2, 0.5, 0.8)
    ) + 
    theme(axis.title = element_blank(),
          axis.text = element_blank(),
          axis.ticks = element_blank(), 
          legend.position = "none") 
}


figure <- egg::ggarrange(
  plot_ice(data),
  plot_ice(Z[, , 1:3]),
  plot_ice(Z[, , 4:6]), 
  nrow = 1
)

ggsv("data_and_conditional_sims", figure, path = img_path, width = 10.4, height = 3.2)



# ---- Inference on observed pixels in open interval (0, 1) ----

# Mixture density function
dbetamix <- function(x, prob, a1, b1, a2, b2) {
  prob * dbeta(x, a1, b1) + (1 - prob) * dbeta(x, a2, b2)
}

# Histogram function returning tidy data frame
histogram_df <- function(Z, year = NULL, remove_0_and_1 = FALSE) {
  
  df <- data.frame(sea_ice = c(Z))

  # breaks: 28 interior + optional outer bins
  breaks <- seq(0.005, 1 - 0.005, length.out = 28)
  if (!remove_0_and_1) {
    breaks <- c(-0.03, breaks, 1.03)
  } else {
    Z <- Z[Z > 0.004 & Z < 0.996]
  }
  

  # build the plot object
  p <- ggplot(df, aes(x = sea_ice)) +
    geom_histogram(breaks = breaks)

  # extract binned data
  ggb <- ggplot_build(p)

  hist_df <- data.frame(
    mid   = ggb$data[[1]]$x,      # bin center
    count = ggb$data[[1]]$count,
    width = ggb$data[[1]]$xmax - ggb$data[[1]]$xmin  # bin width
  )

  # proportions
  hist_df$prop <- hist_df$count / sum(hist_df$count)
  hist_df$year <- year

  return(hist_df)
}

# Histogram 
histogram <- function(mat, 
                      title = element_blank(), 
                      mixture_params = NULL, 
                      show_components = FALSE,
                      show_parameters = FALSE,
                      ...) {
  
  hist_df <- histogram_df(mat, ...)
  
  # max proportion ignoring bin centered at 0 or 1
  max_nonzero <- max(hist_df$prop[hist_df$mid > 0 & hist_df$mid < 1])
  
  # Scale the continuous values so that they represent the true proportions
  scaling_factor <- sum(hist_df$prop[hist_df$mid > 0 & hist_df$mid < 1])
  
  
  # Determine plot title
  if ("year" %in% names(hist_df)) {
    unique_year <- unique(hist_df$year)
    if (length(unique_year) == 1) {
      plot_title <- as.character(unique_year)
    } else {
      plot_title <- title  # fallback if multiple years
    }
  } else {
    plot_title <- title
  }
  
  # Base histogram
  p <- ggplot(hist_df) +
    geom_col(aes(x = mid, y = prop, fill = mid), color = "black", width = hist_df$width) +
    scale_fill_gradient2(
      low = "darkblue", mid = "skyblue", high = "white",
      midpoint = 0.5,
      name = "Proportion of\nsea ice"
    ) +
    theme_bw() +
    labs(x = "Sea-ice proportion", y = "Proportion of observations", title = plot_title) +
    coord_cartesian(ylim = c(0, max_nonzero)) +
    scale_x_continuous(breaks = c(0.2, 0.5, 0.8)) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 10),
      legend.position = "none"
    )
  
  xgrid <- seq(0, 1, length.out = 500)
  
  # Overlay fitted beta mixture if parameters are supplied
  if (!is.null(mixture_params)) {
    dens_df <- data.frame(
      x = xgrid,
      density = dbetamix(xgrid, 
                         prob = mixture_params$prob,
                         a1 = mixture_params$a1,
                         b1 = mixture_params$b1,
                         a2 = mixture_params$a2,
                         b2 = mixture_params$b2)
    )
    dens_df$density <- dens_df$density * mean(hist_df$width) 
    
    # The beta mixture is only fit to the vlaues in (0, 1), so scale the 
    # proportions based on the point masses (if present)
    mass01 <- sum(hist_df$prop[hist_df$mid <= 0 | hist_df$mid >= 1])
    dens_df$density <- dens_df$density * (1 - mass01)
    
    p <- p + geom_line(data = dens_df, aes(x = x, y = density),
                       color = "red", size = 0.8, inherit.aes = FALSE)
    
    # Optional individual components
    if (show_components) {
      comp1 <- data.frame(
        x = xgrid,
        density = mixture_params$prob * dbeta(xgrid, mixture_params$a1, mixture_params$b1)
      )
      comp2 <- data.frame(
        x = xgrid,
        density = (1 - mixture_params$prob) * dbeta(xgrid, mixture_params$a2, mixture_params$b2)
      )
      comp1$density <- comp1$density * mean(hist_df$width) * (1 - mass01)
      comp2$density <- comp2$density * mean(hist_df$width) * (1 - mass01)
      
      p <- p +
        geom_line(data = comp1, aes(x = x, y = density),
                  color = "red", linetype = "dashed", size = 0.6) +
        geom_line(data = comp2, aes(x = x, y = density),
                  color = "red", linetype = "dashed", size = 0.6)
    }
    
    if (show_parameters) {
      # Add mixture parameters as text in top-left
      mixture_label <- sprintf(
        "atop(hat(a)[1] == %.2f *','~ hat(b)[1] == %.2f, hat(a)[2] == %.2f *','~ hat(b)[2] == %.2f *','~ pi == %.2f)",
        mixture_params$a1, mixture_params$b1,
        mixture_params$a2, mixture_params$b2,
        mixture_params$prob
      )
      
      p <- p +
        geom_text(aes(x = 0.02, y = max_nonzero*0.95, label = mixture_label),
                  hjust = 0, vjust = 1, inherit.aes = FALSE,
                  size = 3.5, color = "red", parse = TRUE)
      
    }
  }
  
  return(p)
}

make_histogram <- function(year, domain, ...) {
  
  # Z and Y
  idx <- year - 1978
  Y <- h5read(file.path(int_path, domain, sprintf("conditional_sims_%d.h5", year)), "Y") %>% map_Y_values
  Z <- readRDS(file.path(int_path, domain, "sea_ice.rds"))[[idx]]
  
  # Remove missing pixels 
  observed_idx <- !is.na(Z)
  Y <- apply(Y, 3, function(y) y[observed_idx])
  Z <- Z[observed_idx]

  # Estimate the probability of class 1 given that Y is class 1 or class 2
  prob <- sum(Y == 1) / sum(Y <= 2)
  # Sanity check: (1 - prob) == sum(Y == 2) / sum(Y <= 2)
  
  # Estimated mixture parameters
  estimates <- read_csv(file.path(int_path, domain, "estimates.csv"), col_names = FALSE, show_col_types = FALSE)
  estimates <- as.matrix(estimates)
  theta <- estimates[, idx]
  a1 <- theta[2]
  a2 <- theta[3]
  b1 <- theta[4]
  b2 <- theta[5]
  mixture_params <- list(prob = prob, a1 = a1, a2 = a2, b1 = b1, b2 = b2)
  
  

  histogram(Z, year = year, mixture_params = mixture_params, ...)
}

for (domain in c("full", "sub")) {
  years <- c(1990, 1993, 1995, 1999)
  hists <- lapply(years, make_histogram, show_components = TRUE, domain = domain, remove_0_and_1 = TRUE)
  figure <- patchwork::wrap_plots(hists, nrow = 1, axis_titles = "collect")
  print(figure)
  ggsv("histograms_and_mixture_fits", figure, path = file.path(img_path, domain), width = 11, height = 2.6)
}

# ---- Plots of pixel colours evolving in time ----

years <- c(1979, 1993, 1995, 2023)

## Data plot
sea_ice <- readRDS(file.path(int_path, "full", "sea_ice.rds"))
df <- lapply(years, function(year) {
  df <- reshape2::melt(sea_ice[[year - 1978]])
  df$year <- year
  df
})
df <- do.call(rbind, df)

data <- ggplot(df, aes(Var1, Var2, fill = value)) +
  geom_raster() +
  geom_tile(data = subset(reshape2::melt(land_border_mask), value == 1), aes(Var1, Var2), fill = "gray", alpha = 0.7) +
  scale_y_reverse(expand = c(0, 0)) + 
  scale_x_continuous(expand = c(0, 0)) +
  geom_tile(aes(x = 0, y = 0, colour = "Missing"), inherit.aes = FALSE) +   # dummy layer for "Missing" legend entry
  facet_wrap(~year, nrow = 1) +
  theme_bw() +
  scale_fill_gradient2(
    low = "darkblue", mid = "skyblue", high = "white", na.value = "red",
    midpoint = 0.5,
    name = "Sea-ice\nproportion\n", 
    limits = c(0, 1),
    breaks = c(0.2, 0.5, 0.8)
  ) +
  scale_colour_manual(
    name = NULL,
    values = c("Missing" = "red")
  ) +
  guides(
    fill = guide_colorbar(order = 1),
    colour = guide_legend(
      order = 2,
      override.aes = list(fill = "red", shape = 22) 
    )
  ) +
  theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(), 
    legend.spacing.y = unit(0.5, "pt")
  )


## Prediction plots
Z_pred_df <- lapply(years, function(year) {
  Z <- h5read(file.path(int_path, "full", sprintf("conditional_sims_%d.h5", year)), "Z")
  Z_pred <- apply(Z, c(1, 2), mean)# median)
  Z_pred <- reshape2::melt(Z_pred)
  Z_pred$year <- year
  Z_pred
})
Z_pred_df <- do.call(rbind, Z_pred_df)

Y_MAP_df <- lapply(years, function(year) {
  Y <- h5read(file.path(int_path, "full", sprintf("conditional_sims_%d.h5", year)), "Y") %>% map_Y_values
  Y_MAP <- apply(Y, c(1, 2), function(y) as.numeric(names(which.max(table(y)))))
  Y_MAP <- reshape2::melt(Y_MAP)
  Y_MAP$value <- as.factor(Y_MAP$value)
  Y_MAP$year <- year
  Y_MAP
})
Y_MAP_df <- do.call(rbind, Y_MAP_df)

Z_pred <- ggplot(Z_pred_df, aes(Var1, Var2, fill = value)) + 
  geom_raster() + 
  geom_tile(data = subset(reshape2::melt(land_border_mask), value == 1), aes(Var1, Var2), fill = "gray", alpha = 0.7) +
  scale_y_reverse(expand = c(0, 0)) + 
  scale_x_continuous(expand = c(0, 0)) + 
  theme_bw() + 
  facet_wrap(~year, nrow = 1) +
  theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(), 
    strip.background = element_blank(),
    strip.text.x = element_blank()
  ) + 
  scale_fill_gradient2(
    low = "darkblue", mid = "skyblue", high = "white", na.value = "red",
    midpoint = 0.5,
    name = "Predicted\nsea-ice\nproportion\n", 
    limits = c(0, 1),
    breaks = c(0.2, 0.5, 0.8)
  )

Y_MAP <- ggplot(Y_MAP_df, aes(Var1, Var2, fill = value)) + 
  geom_raster() + 
  geom_tile(data = subset(reshape2::melt(land_border_mask), value == 1), aes(Var1, Var2), fill = "gray", alpha = 0.7) +
  scale_y_reverse(expand = c(0, 0)) + 
  scale_x_continuous(expand = c(0, 0)) + 
  theme_bw() + 
  facet_wrap(~year, nrow = 1) + 
  scale_fill_manual(
    # values = c("orange", "black", "#009E73", "#CC79A7"),
    values = c("darkblue", "#009E73", "orange", "white"),
    name = "Y label\nMAP estimate"
  ) +
  theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_blank(), 
    legend.key = element_rect(color = "black", size = 0.5)  # Add border to legend keys
  )

figure <- egg::ggarrange(plots = list(data, Z_pred, Y_MAP), ncol = 1)

ggsv("predictions_and_Y", figure, path = img_path, width = 11.3, height = 7.7)

options(warn = oldw)
