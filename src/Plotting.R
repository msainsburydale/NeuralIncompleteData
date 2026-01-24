suppressMessages({
library("ggplotify")
library("dplyr")
library("ggpubr")
library("viridis")
library("tidyr")
library("xtable")
library("ggExtra")
library("NeuralEstimators")
library("ggplot2")
library("naniar")
library("gtable")
library("cowplot")
options(dplyr.summarise.inform = FALSE) 
})

if(!interactive()) pdf(NULL)

replace_unicode <- function(text) {
  text %>%
    str_replace("σ2", "sigma^2") %>%
    str_replace("ρ", "rho") %>%
    str_replace("ν", "nu") %>%
    str_replace("κ", "kappa") %>%
    str_replace("λ", "lambda") %>%
    str_replace("β", "beta") %>%
    str_replace("γ", "gamma") %>%
    str_replace("χ", "chi") %>%
    str_replace("μ", "mu") %>%
    str_replace("τ", "tau") %>%
    str_replace("ω1", "omega[1]") %>%
    str_replace("ω2", "omega[2]")
}

estimator_colours <- c(
  "MAP" = "gold",
  "neuralEM" = "chartreuse4",
  "EM" = "chartreuse4",
  "masking" = "deepskyblue3",
  "ABC MAP" = "red"
)

# Function for changing the aesthetics 
scale_estimator_aesthetic <- function(df, scale = "colour", values = estimator_colours, ...) {
  estimators <- unique(df$estimator)
  ggplot2:::manual_scale(
    scale,
    values = values[as.character(estimators)],        # allows for only a subset of estimators
    labels = estimator_labels[estimators],
    breaks = names(estimator_colours)[names(estimator_colours) %in% estimators], # specifies the order of the estimators in the plot
    ...
  )
}

# The simulations may be highly varied in magnitude, so we need to
# use an independent colour scale; this means that we can't use facet_wrap().
field_plot <- function(field, regular = TRUE, variable = "Z") {

  gg <- ggplot(field, aes(x = x, y = y))

  # Standard eval with ggplot2 without `aes_string()`: https://stackoverflow.com/a/55133909
  if (regular) {
    gg <- gg +
      geom_tile(aes(fill = !!sym(variable))) +
      scale_fill_viridis_c(option = "magma")
  } else {
    gg <- gg +
      geom_point(aes(colour = !!sym(variable))) +
      scale_colour_viridis_c(option = "magma")
  }

  gg <- gg +
    labs(fill = "", x = expression(s[1]), y = expression(s[2])) +
    theme_bw() +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0))

  return(gg)
}



plot_EM_trajectories <- function(
    df,
    parameter_labels,
    max_iteration = 50,
    H = NULL,       # which nsims to include, NULL for all
    theta_df = NULL,           # optional dataframe for reference lines
    MAP_df = NULL,             # optional dataframe for MAP reference lines
    colour_by_theta_0 = FALSE, 
    show_average = TRUE
) {
  
  if (min(df$iteration) == 1) {
    df$iteration <- df$iteration - 1
  }
  
  # Apply optional nsims filter
  if (!is.null(H)) {
    df <- df %>% filter(nsims %in% H)
  }
  
  # Convert parameter to factor with labels
  df <- df %>%
    mutate(
      parameter = factor(parameter, levels = names(parameter_labels), labels = parameter_labels),
      nsims = paste0("m == ", nsims)
    )
  
  # Base plot filtered by max_iteration
  p <- ggplot(df %>% filter(iteration <= max_iteration)) + 
    theme_bw()
  
  if (colour_by_theta_0) {
    p <- p + geom_line(aes(x = iteration, y = estimate, group = theta_0, colour = factor(theta_0)), lty = "dotdash") +
      theme(legend.position = "none")
      if (show_average) {
        p <- p + geom_line(aes(x = iteration, y = averaged_estimate, colour = factor(theta_0)), alpha = 0.5) 
      }
    p <- p + scale_colour_manual(values = c("1" = "#0072B2", "2" = "#E69F00"))
  } else {
    p <- p + geom_line(aes(x = iteration, y = estimate, group = theta_0), lty = "dotdash") 
      if (show_average) {
        p <- p + geom_line(aes(x = iteration, y = averaged_estimate, group = theta_0), colour = "red", alpha = 0.5)
      }
  }
  
  
  # Add optional horizontal lines if provided
  if (!is.null(theta_df)) {
    p <- p + geom_hline(data = theta_df, aes(yintercept = value), colour = "gray50", lty = "dashed")
  }
  if (!is.null(MAP_df)) {
    p <- p + geom_hline(data = MAP_df, aes(yintercept = MAP), colour = "black", lty = "dashed")
  }
  
  # Faceting
  if (!is.null(H) && length(H == 1)) {
    p <- p + facet_wrap(. ~ parameter, nrow = 1, scales = "free", labeller = label_parsed)
  } else {
    p <- p + facet_grid(parameter ~ nsims, scales = "free", labeller = label_parsed)
  }
  
  return(p)
}




ggsv <- function(filename, plot, device = NULL, ...) {
  suppressWarnings({
    # If a specific device is provided, use only that device
    if (!is.null(device)) {
      ggsave(plot, file = paste0(filename, ".", device), device = device, ...)
    } else {
      # Default behavior: loop over the devices
      # for (dev in c("pdf", "png")) {
      for (dev in c("pdf")) {
        ggsave(plot, file = paste0(filename, ".", dev), device = dev, ...)
      }
    }
  })
}
