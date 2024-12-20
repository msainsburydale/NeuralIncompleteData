suppressMessages({
  library("NeuralEstimators")
  library("ggplot2")
  library("dplyr")
  library("forcats")
  library("reshape2")
  library("latex2exp")
})
# Suppress summarise info
options(dplyr.summarise.inform = FALSE)

source(file.path("src", "plotting.R"))

rmfacet <- function(gg) gg + theme(
  strip.background = element_blank(),
  strip.text.x = element_blank()
)

img_path           <- file.path("img", "univariate")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

estimator_labels <- c(
  "MAP" = "MAP",
  "PosteriorMedian" = "Posterior median",
  "L1"    = expression(hat(theta)[NBE]("·") * " : " * L[1]),
  "k0.9"  = expression(hat(theta)[NBE]("·") * " : " * kappa == 0.9),
  "k0.7"  = expression(hat(theta)[NBE]("·") * " : " * kappa == 0.7),
  "k0.5"  = expression(hat(theta)[NBE]("·") * " : " * kappa == 0.5),
  "k0.3"  = expression(hat(theta)[NBE]("·") * " : " * kappa == 0.3),
  "k0.1"  = expression(hat(theta)[NBE]("·") * " : " * kappa == 0.1),
  "k0.05"  = expression(hat(theta)[NBE]("·") * " : " * kappa == 0.05), 
  "k0.01"  = expression(hat(theta)[NBE]("·") * " : " * kappa == 0.01)
)

# ---- Density plot  ----

df <- file.path("intermediates", "univariate", "Estimates", "estimates.csv")  %>% read.csv

estimator_colours <- c(
  "MAP" = "gold",
  "PosteriorMedian" = "darkorange",
  "k0.05" = "chartreuse4",
  "L1" = "deepskyblue3"
)

gg <- df %>%
  filter(estimator %in% names(estimator_colours)) %>%
  plotdistribution(type = "density", truth_colour = NA, estimator_labels = estimator_labels) %>%
  rmfacet
gg <- gg + labs(x = expression(hat(theta)))
ggsv(gg, file = "density", width = 7, height = 3.5, path = img_path)  

# ---- Difference with respect to the MAP ----

df <- df %>% filter(j <= 1000)
tmp <- split(df, df$estimator)
MAP <- tmp$MAP$estimate
tmp$MAP <- NULL
df_diff <- lapply(tmp, function(x) {
  x$estimate <- x$estimate - MAP
  x
  })
df_diff <- do.call(rbind, df_diff)
df_diff$estimator <- fct_rev(df_diff$estimator)
df_diff <- df_diff[df_diff$estimate > -0.2, ] # remove a couple of outliers  

gg <- rmfacet(plotdistribution(df_diff, type = "box", truth_colour = NA, estimator_labels = estimator_labels))
gg <- gg +
  geom_hline(aes(yintercept = 0), colour = "black", linetype = "dashed") +
  labs(y = expression(hat(theta) - hat(theta)[MAP])) +
  scale_y_continuous(limits = range(df_diff$estimate))
gg$layers[[1]]$geom_params$outlier.size <- 0.25

ggsv(gg, file = "boxplot", width = 6, height = 3.1, path = img_path)
