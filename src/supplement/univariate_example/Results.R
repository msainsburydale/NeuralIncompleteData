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

source(file.path("src", "Plotting.R"))

rmfacet <- function(gg) gg + theme(
  strip.background = element_blank(),
  strip.text.x = element_blank()
)

int_path <- file.path("intermediates", "univariate")
img_path <- file.path("img", "univariate")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

estimator_labels <- c(
  "MAP" = "MAP",
  "PosteriorMedian" = "Posterior median",
  "k0.9"  = expression(hat(theta)[NBE]("·") * " : Tanh loss, " * kappa == 0.9),
  "k0.5"  = expression(hat(theta)[NBE]("·") * " : Tanh loss, " * kappa == 0.5),
  "k0.3"  = expression(hat(theta)[NBE]("·") * " : Tanh loss, " * kappa == 0.3),
  "k0.2"  = expression(hat(theta)[NBE]("·") * " : Tanh loss, " * kappa == 0.2),
  "k0.1"  = expression(hat(theta)[NBE]("·") * " : Tanh loss, " * kappa == 0.1),
  "posteriorloss" = expression(hat(theta)[NBE]("·") * " : Approximate-posterior loss")
)

estimator_colours <- c(
  "MAP" = "#440154",
  "k0.1" = "#31688E",
  "posteriorloss" = "#BB3754",
  "PosteriorMedian" = "#FDE725"
)

# ---- Density plot  ----

df <- file.path(int_path, "estimates.csv")  %>% read.csv
gg <- df %>%
  filter(estimator %in% names(estimator_colours)) %>%
  plotdistribution(type = "density", truth_colour = NA, estimator_labels = estimator_labels) %>%
  rmfacet
gg <- gg + labs(x = expression(hat(theta))) + 
  scale_estimator_aesthetic(df)
ggsv(gg, file = "density", width = 7, height = 3.5, path = img_path)  

# ---- Difference with respect to the MAP ----

df <- df %>% filter(j <= 100, estimator %in% names(estimator_labels))
tmp <- split(df, df$estimator)
MAP <- tmp$MAP$estimate
tmp$MAP <- NULL
df_diff <- lapply(tmp, function(x) {
  x$estimate <- x$estimate - MAP
  x
  })
df_diff <- do.call(rbind, df_diff)
df_diff$estimator <- factor(df_diff$estimator, levels = names(estimator_labels))
df_diff <- df_diff[df_diff$estimate > -0.1, ] # remove a couple of outliers  

gg <- rmfacet(plotdistribution(df_diff, type = "box", truth_colour = NA, estimator_labels = estimator_labels))
gg <- gg +
  geom_hline(aes(yintercept = 0), colour = "black", linetype = "dashed") +
  labs(y = expression(hat(theta) - hat(theta)[MAP])) +
  scale_y_continuous(limits = range(df_diff$estimate)) #+  
  # scale_estimator_aesthetic(df)
gg$layers[[1]]$geom_params$outlier.size <- 0.1

ggsv(gg, file = "boxplot", width = 9, height = 3.1, path = img_path)


# ---- Approximate posterior density ----

library(ggplot2)
library(readr)

df <- file.path(int_path, "logdensity.csv")  %>% 
  read.csv %>% 
  mutate(estimated_density = exp(estimated_logdensity), 
         true_density = exp(true_logdensity)) %>% 
  filter(theta > 0.2, theta < 2.5)

# Calculate the arg max positions and values
est_max_idx <- which.max(df$estimated_logdensity)
true_max_idx <- which.max(df$true_logdensity)

est_max_point <- data.frame(
  theta = df$theta[est_max_idx],
  logdensity = df$estimated_logdensity[est_max_idx],
  density = df$estimated_density[est_max_idx]
)

true_max_point <- data.frame(
  theta = df$theta[true_max_idx],
  logdensity = df$true_logdensity[true_max_idx],
  density = df$true_density[true_max_idx]
)

# Plot log-density
log_density <- ggplot(df, aes(x = theta)) +
  geom_line(aes(y = estimated_logdensity, color = "Estimated"), 
            linewidth = 1.2, alpha = 0.8) +
  geom_line(aes(y = true_logdensity, color = "True"), 
            linewidth = 1.2, linetype = "longdash", alpha = 0.8) +
  # Add dots at the arg max positions
  geom_point(data = est_max_point,
             aes(x = theta, y = logdensity, color = "Estimated"),
             size = 4, shape = 16, alpha = 0.75) +
  geom_point(data = true_max_point,
             aes(x = theta, y = logdensity, color = "True"),
             size = 4, shape = 16, alpha = 0.75) +
  # Optional: Add vertical lines to emphasize the maxima
  geom_vline(aes(xintercept = theta[which.max(estimated_logdensity)], 
                 color = "Estimated"), 
             linetype = "dashed", linewidth = 0.5, alpha = 0.5) +
  geom_vline(aes(xintercept = theta[which.max(true_logdensity)], 
                 color = "True"), 
             linetype = "dashed", linewidth = 0.5, alpha = 0.5) +
  scale_color_manual(
    name = "",
    values = c("Estimated" = "#E69F00", "True" = "#0072B2"),
    labels = c("Estimate", "Truth")
  ) +
  labs(
    x = expression(theta),
    y = expression(log * " " * p(theta * " | " * Z))
  ) +
  theme_bw() +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    axis.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 11),
    panel.grid.minor = element_blank()
  )

# Plot density
density <- ggplot(df, aes(x = theta)) +
  geom_line(aes(y = estimated_density, color = "Estimated"), 
            linewidth = 1.2, alpha = 0.8) +
  geom_line(aes(y = true_density, color = "True"), 
            linewidth = 1.2, linetype = "longdash", alpha = 0.8) +
  # Add dots at the arg max positions
  geom_point(data = est_max_point,
             aes(x = theta, y = density, color = "Estimated"),
             size = 4, shape = 16, alpha = 0.75) +
  geom_point(data = true_max_point,
             aes(x = theta, y = density, color = "True"),
             size = 4, shape = 16, alpha = 0.75) +
  # Optional: Add vertical lines to emphasize the maxima
  geom_vline(aes(xintercept = theta[which.max(estimated_logdensity)], 
                 color = "Estimated"), 
             linetype = "dashed", linewidth = 0.5, alpha = 0.5) +
  geom_vline(aes(xintercept = theta[which.max(true_logdensity)], 
                 color = "True"), 
             linetype = "dashed", linewidth = 0.5, alpha = 0.5) +
  scale_color_manual(
    name = "",
    values = c("Estimated" = "#E69F00", "True" = "#0072B2"),
    labels = c("Estimate", "Truth")
  ) +
  labs(
    x = expression(theta),
    y = expression(p(theta * " | " * Z))
  ) +
  theme_bw() +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    axis.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 11),
    panel.grid.minor = element_blank()
  )

figure <- ggpubr::ggarrange(density, log_density, nrow = 1, common.legend = TRUE)
ggsv(figure, file = "approximateposterior", width = 8.5, height = 3.7, path = img_path)