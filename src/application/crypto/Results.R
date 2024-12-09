source(file.path("src", "Plotting.R"))
library("latex2exp")

if(!interactive()) pdf(NULL)
oldw <- getOption("warn")
options(warn = -1)

img_path <- file.path("img", "application", "crypto")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

df1 <- file.path(img_path, "empirical", "probability_estimates.csv") %>% read.csv %>% mutate(estimator = "empirical")
df2 <- file.path(img_path, "neuralEM", "probability_estimates.csv") %>% read.csv %>% mutate(estimator = "neuralEM")
df3 <- file.path(img_path, "encoding", "probability_estimates.csv") %>% read.csv %>% mutate(estimator = "encoding")
df4 <- file.path(img_path, "MAP", "probability_estimates.csv") %>% read.csv %>% mutate(estimator = "MAP")
df_est <- rbind(df1, df2, df3, df4)

df_ci1  <- file.path(img_path, "empirical", "probability_pointwise_intervals.csv") %>% read.csv %>% mutate(estimator = "empirical")
df_ci1  <- pivot_wider(df_ci1, names_from = bound, values_from = estimate)
df_ci2  <- file.path(img_path, "neuralEM", "probability_pointwise_intervals.csv") %>% read.csv %>% mutate(estimator = "neuralEM")
df_ci3  <- file.path(img_path, "encoding", "probability_pointwise_intervals.csv") %>% read.csv %>% mutate(estimator = "encoding")
df_ci4  <- file.path(img_path, "encoding", "probability_pointwise_intervals.csv") %>% read.csv %>% mutate(estimator = "MAP")
df_ci <- rbind(df_ci1, df_ci2, df_ci3, df_ci4)

df_est$pair <- gsub("1", "BITCOIN", df_est$pair)
df_est$pair <- gsub("2", "ETHEREUM", df_est$pair)
df_est$pair <- gsub("3", "AVALANCHE", df_est$pair)
df_est$pair <- gsub("-", " - ", df_est$pair)
df_ci$pair <- gsub("1", "BITCOIN", df_ci$pair)
df_ci$pair <- gsub("2", "ETHEREUM", df_ci$pair)
df_ci$pair <- gsub("3", "AVALANCHE", df_ci$pair)
df_ci$pair <- gsub("-", " - ", df_ci$pair)

 estimator_colours <- c(
  "empirical" = "black", 
  "neuralEM" = "green", 
  "encoding" = "blue"
)

makeplot_pointwise <- function(df_est, df_ci, tl) {
  
  df_est <- df_est %>% filter(tail == tl) %>% filter(estimator %in% names(estimator_colours))
  df_ci  <- df_ci %>% filter(tail == tl)  %>% filter(estimator %in% names(estimator_colours))

  ggplot() + 
    geom_line(data = df_est, aes(x = t, y = estimate, colour = estimator)) + 
    geom_ribbon(data = df_ci, aes(x = t, ymin = lower, ymax = upper, group = estimator, fill = estimator), alpha = 0.1) +
    scale_colour_manual(values = estimator_colours) + 
    scale_fill_manual(values = estimator_colours) + 
    labs(x = "Threshold u") + 
    facet_grid(tail ~ pair) + 
    coord_fixed() + 
    theme_bw() + 
    theme(
      legend.position = "none", 
      strip.background = element_blank(), 
      strip.text.y = element_blank()
    ) 
}

gg1 <- makeplot_pointwise(df_est, df_ci, "lower") + labs(y = TeX("$Pr(U_i < u, U_j < u)$")) + scale_x_continuous(breaks = c(0.02, 0.05, 0.08))
gg2 <- makeplot_pointwise(df_est, df_ci, "upper") + labs(y = TeX("$Pr(U_i > u, U_j > u)$")) + scale_x_continuous(breaks = c(0.92, 0.95, 0.98))
figure <- ggarrange(gg1, gg2, nrow = 2, align = "hv")

ggsv(file = "tail-probabilities-lower_pointwise", plot = gg1, width = 9.5, height = 3.5, path = img_path)
ggsv(file = "tail-probabilities-upper_pointwise", plot = gg2, width = 9.5, height = 3.5, path = img_path)
ggsv(file = "tail-probabilities_pointwise", plot = figure, width = 9.5, height = 6.5, path = img_path)

options(warn = oldw)

