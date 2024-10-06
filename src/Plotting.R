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

# the order of this controls the display order
estimator_colours <- c(
  "MAP" = "gold",
  "conventionalEM" = "darkorange",
  "neuralEM" = "chartreuse4",
  "masking" = "deepskyblue3"
)

# Function for changing the aesthetics 
scale_estimator_aesthetic <- function(df, scale = "colour", values = estimator_colours, ...) {
  estimators <- unique(df$estimator)
  ggplot2:::manual_scale(
    scale,
    values = values[estimators],        # allows for only a subset of estimators
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

ggsv <- function(filename, plot, ...) {
  suppressWarnings({
    for (device in c("pdf", "png")) {
      ggsave(plot, file = paste0(filename, ".", device), device = device, ...)
    }
  })
}
