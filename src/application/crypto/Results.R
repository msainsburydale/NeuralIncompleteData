source(file.path("src", "Plotting.R"))
library("latex2exp")

if(!interactive()) pdf(NULL)
oldw <- getOption("warn")
options(warn = -1)

img_path <- file.path("img", "application", "crypto")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

# ---- neural MAP vs analytic MAP ----

df <- file.path(img_path, "estimates_complete.csv") %>% read.csv

parameter_labels = c(
  "γ" = expression(hat(alpha)),
  "ω" = expression(hat(omega)),
  "λ" = expression(hat(lambda)),
  "Σ21" = TeX("$\\hat{Sigma}_{21,}$"),
  "Σ31" = TeX("$\\hat{Sigma}_{31,}$"),
  "Σ32" = TeX("$\\hat{Sigma}_{32,}$")
)

plotmarginals <- function(df, estimator_labels, parameter_labels) {
  
  if (length(unique(df$estimator)) != 2) stop("df should contain results from exactly two estimators")
  
  # subset estimator and parameter labels
  estimator_labels <- estimator_labels[names(estimator_labels) %in% df$estimator]
  parameter_labels <- parameter_labels[names(parameter_labels) %in% df$parameter]
  
  # convert to wide format based on the parameters and estimator names
  df <- df %>%
    pivot_wider(names_from = parameter, values_from = c("estimate", "truth")) %>%
    pivot_wider(names_from = estimator, values_from = paste("estimate", names(parameter_labels), sep = "_")) %>%
    as.data.frame
  
  parameters <- as.list(parameter_labels)
  estimators <- names(estimator_labels)
  p          <- length(parameter_labels)
  
  marginal <- lapply(1:p, function(i) {
    
    columns <- paste("estimate", names(parameter_labels)[i], estimators, sep = "_")
    lmts <- range(df[, columns])
    
    gg <- ggplot(df) +
      geom_point(aes(!!sym(columns[2]), !!sym(columns[1]))) +
      geom_abline(colour = "red") +
      theme_bw() +
      theme(strip.background = element_blank(), strip.text.x = element_blank()) +
      coord_fixed(xlim = lmts, ylim = lmts) + 
      labs(
        x = as.expression(bquote(.(parameters[[i]])[.(estimator_labels[2])])),
        y = as.expression(bquote(.(parameters[[i]])[.(estimator_labels[1])]))
      )
  })
  
  return(marginal)
}

marginal <- plotmarginals(
  df = df,
  estimator_labels = c("neuralMAP" = "neural MAP", "MAP" = "MAP"), 
  parameter_labels = parameter_labels
)

marginal <- egg::ggarrange(plots = marginal, nrow = 2)
ggsv(file = "scatterplot", plot = marginal, width = 8.5, height = 5.5, path = img_path)

# ---- inference with incomplete crypto data ----

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

figure <- makeplot_pointwise(df_est, df_ci, "lower") + labs(y = TeX("$Pr(U_i < u, U_j < u)$")) + scale_x_continuous(breaks = c(0.02, 0.05, 0.08))
ggsv(file = "tail-probabilities", plot = figure, width = 9.5, height = 3, path = img_path)

options(warn = oldw)

