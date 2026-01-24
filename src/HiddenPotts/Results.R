source(file.path("src", "Plotting.R"))

if(!interactive()) pdf(NULL)
oldw <- getOption("warn")
options(warn = -1)

library("readr")

int_path <- file.path("intermediates", "HiddenPotts")
img_path       <- file.path("img", "HiddenPotts")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

parameter_labels = c(
  "β" = expression(beta), 
  "μ1" = expression(mu[1]), 
  "μ2" = expression(mu[2]), 
  "μ3" = expression(mu[3]), 
  "σ1" = expression(sigma[1]),
  "σ2" = expression(sigma[2]),
  "σ3" = expression(sigma[3])
)

estimator_labels <- c(
  "neuralEM" = "EM NBE",
  "EM" = "EM NBE",
  "masking" = "Masking NBE"
)

missingness <- c("MCAR", "MB")

d <- length(parameter_labels)
q <- (d-1)/2
critical_value <- log(1 + sqrt(q))

# ---- Visualize MH acceptance probabilities ----


# values of beta to compare
betas <- c(0.2, 0.5, 0.8, 1.1)

# ΔS values
DeltaS <- -4:4

# build data frame of all combinations
df <- expand.grid(DeltaS = DeltaS, beta = betas)
df$accept_prob <- pmin(1, exp(df$beta * df$DeltaS))

# plot
figure <- ggplot(df, aes(x = DeltaS, y = accept_prob, color = factor(beta), group = beta)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  scale_color_brewer(palette = "Set1", name = expression(beta)) +
  labs(
    # title = "MH acceptance probability across ΔS",
    # x = expression(Delta*S),
    x = expression(S[i](Y[i] * "'" ) - S[i](Y[i])),
    y = "MH acceptance probability"
  ) +
  theme_bw(base_size = 14)
ggsv(file = "MH_acceptance_probability", plot = figure, path = img_path, width = 6.3, height = 3.5)


# ---- EM convergence ----

df <- read_csv(file.path(int_path, "EM_iterates.csv"), show_col_types = FALSE)
parameter_subset <- c("β", "μ1", "σ1")
figure1 <- plot_EM_trajectories(df %>% filter(parameter %in% parameter_subset), parameter_labels)
figure2 <- plot_EM_trajectories(df %>% filter(parameter %in% parameter_subset), parameter_labels, H = max(df$nsims))
ggsv("convergence_multipleMC", figure1, path = img_path, width = 7.5, height = 4)
ggsv("convergence_singleMC", figure2, path = img_path, width = 6.5, height = 3)

# ---- Global diagnostics ----

df <- lapply(missingness, function(set)  {
  df <- file.path(int_path, paste0("estimates_", set, "_test.csv")) %>% read.csv
  df$missingness <- set
  df
  })
df <- do.call(rbind, df)
df$estimator[df$estimator == "neuralEM"] <- "EM"

# RMSE
rmse <- df %>% group_by(estimator, missingness) %>% summarise(RMSE = sqrt(mean((estimate - truth)^2))) 
write.csv(rmse, row.names = F, file = file.path(int_path, "rmse.csv"))
rmse <- df %>% filter(parameter == "β") %>% group_by(estimator, missingness) %>% summarise(RMSE = sqrt(mean((estimate - truth)^2))) 
rmse_beta_MCAR <- readRDS(file.path(int_path, "ABC", "rmse_beta_MCAR.rds"))
rmse_beta_MB <- readRDS(file.path(int_path, "ABC", "rmse_beta_MB.rds"))
rmse <- rmse %>% rbind(data.frame(estimator = "ABC", missingness = missingness, RMSE = c(rmse_beta_MCAR, rmse_beta_MB)))
write.csv(rmse, row.names = F, file = file.path(int_path, "rmse_beta.csv"))

# Recovery plots
df <- dplyr::mutate_at(df, .vars = "parameter", .funs = factor, levels = names(parameter_labels), labels = parameter_labels)

figure <- ggplot2::ggplot(df[sample(nrow(df)), ] %>% filter(missingness == "MCAR")) + 
  ggplot2::geom_point(ggplot2::aes(x=truth, y = estimate, colour  = estimator), alpha = 0.6, size = 0.4) + 
  ggplot2::geom_abline(colour = "black", linetype = "dashed") +
  ggh4x::facet_grid2(estimator~parameter, scales = "free", independent = "y", labeller = label_parsed) + 
  ggplot2::labs(colour = "") + 
  ggplot2::theme_bw() +
  scale_estimator_aesthetic(df) + 
  theme(
    strip.text.y = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(size = 12) 
  )

ggsv(file = "recovery_plot_MICB", plot = figure, path = img_path, width = 11, height = 3)

figure <- ggplot2::ggplot(df[sample(nrow(df)), ] %>% filter(missingness == "MCAR")) + 
  ggplot2::geom_point(ggplot2::aes(x=truth, y = estimate, colour  = estimator), alpha = 0.6, size = 0.4) + 
  ggplot2::geom_abline(colour = "black", linetype = "dashed") +
  ggh4x::facet_grid2(estimator~parameter, scales = "free", independent = "y", labeller = label_parsed) + 
  ggplot2::labs(colour = "") + 
  ggplot2::theme_bw() +
  scale_estimator_aesthetic(df) + 
  theme(
    strip.text.y = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(size = 12) 
  )

ggsv(file = "recovery_plot_MCAR", plot = figure, path = img_path, width = 11, height = 3)

# ---- Sampling distributions ----

loadestimates <- function(type) {
  df <- file.path(int_path, paste0("estimates_", type, "_scenarios.csv")) %>% read.csv
  df$missingness <- type
  df
}

loaddata <- function(type) {
  df <- file.path(int_path, paste0("Z_", type, ".csv")) %>% read.csv
  df$missingness <- type
  df
}

df <- loadestimates(missingness[1]) %>% rbind(loadestimates(missingness[2]))
zdf <- loaddata(missingness[1]) %>% rbind(loaddata(missingness[2]))
df$estimator[df$estimator == "neuralEM"] <- "EM"


N <- zdf %>% filter(k==1 & j == 1 & missingness == "MCAR") %>% nrow %>% sqrt # NB assumes square grid
zdf$x <- rep(1:N, each = N)
zdf$y <- N:1

figures <- lapply(unique(df$k), function(kk) {

  df  <- df  %>% filter(k == kk) %>% filter(parameter == "β")
  zdf <- zdf %>% filter(k == kk)
  l <- length(missingness) # number of missingness patterns

  # ---- Data plots ----

  data <- lapply(missingness, function(mis) {

    field <- filter(zdf, j == 1, missingness == mis)

    gg <- ggplot(field, aes(x = x, y = y)) +
      geom_tile(aes(fill = Z)) +
      scale_fill_viridis_c(option = "magma") + 
      labs(fill = "Z", x = expression(s[1]), y = expression(s[2])) +
      theme_bw() +
      scale_x_continuous(expand = c(0, 0)) +
      scale_y_continuous(expand = c(0, 0)) +
      theme(legend.title.align = 0.25, legend.title = element_text(face = "bold")) + 
      theme(axis.text = element_blank(),
            axis.ticks = element_blank(),
            axis.title = element_blank())
  })
  data_legend <- get_legend(data[[1]])
  data <- lapply(data, function(gg) gg + theme(legend.position = "none") + coord_fixed())

  # ---- Box plots ----
  
  d <- length(unique(df$parameter))

  box <- lapply(missingness, function(mis) {
    plotdistribution(filter(df, missingness == mis), 
                     type = "box", 
                     parameter_labels = c("β" = expression(hat(beta))), 
                     estimator_labels = estimator_labels, truth_line_size = 1, return_list = TRUE, flip = TRUE)
  })

  if (d > 1) stop("Need to change the code that keeps the axes fixed between panels")
  est_lims <- range(df$estimate)

  box_split <- lapply(1:d, function(i) {
    lapply(1:length(box), function(j) box[[j]][[i]])
  })

  # Modify the axes
  for (i in 1:d) {
    box_split[[i]][[l]] <- box_split[[i]][[l]] + labs(y = box_split[[i]][[1]]$labels$x)

    # Remove axis labels from internal panels
    box_split[[i]][-l] <- lapply(box_split[[i]][-l], function(gg) gg +
                                   theme(axis.text.x = element_blank(),
                                         axis.ticks.x = element_blank(),
                                         axis.title.x = element_blank()))

    # Ensure axis limits are consistent for all panels for a given parameter
    ylims <- df %>% filter(parameter == unique(df$parameter)[i]) %>% summarise(range(estimate)) %>% as.matrix %>% c
    box_split[[i]] <- lapply(box_split[[i]], function(gg) gg + ylim(ylims))
  }

  box <- do.call(c, box_split)
  suppressMessages({
    box <- lapply(box, function(gg) gg + scale_estimator_aesthetic(df))
    box_legend <- get_legend(box[[1]])
    box <- lapply(box, function(gg) {
      gg$facet$params$nrow <- 2
      gg$facet$params$strip.position <- "bottom"
      gg <- gg + theme(legend.position = "none", axis.title.y = element_blank()) + scale_y_continuous(limits = est_lims, n.breaks = 4)
      gg
    })
  })

  legends <- list(ggplotify::as.ggplot(data_legend), ggplotify::as.ggplot(box_legend))

  # ---- Combine with global scatter plot ----

  suppressMessages({
  global_plots <- lapply(missingness, function(set) {

    df  <- file.path(int_path, paste0("estimates_", set, "_test.csv")) %>% read.csv
    df <- df %>% filter(parameter == "β")
    gg <- plotestimates(df[sample(nrow(df)), ], estimator_labels = estimator_labels) +
      scale_estimator_aesthetic(df) +
      scale_x_continuous(breaks = c(0.3, 0.6, 0.9, 1.2, 1.5)) +
      scale_y_continuous(breaks = c(0.3, 0.6, 0.9, 1.2, 1.5)) +
      labs(x = expression(beta), y = expression(hat(beta))) +
      theme(
        strip.background = element_blank(),
        strip.text.x = element_blank(),
        legend.position = "none"
      ) +
    geom_vline(xintercept = critical_value, linetype = "dashed")

    gg$layers[[1]]$aes_params$size <- 0.4

    gg
  })
  global_plots[[1]] <- global_plots[[1]] + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), axis.ticks = element_blank())
  global_plots[[1]] <- global_plots[[1]] + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), axis.ticks = element_blank())
  })

  plotlist <- c(data, box, global_plots, legends)
  nrow <- length(missingness)
  ncol <- d + 2 + 1
  figure  <- egg::ggarrange(plots = plotlist, nrow = nrow, ncol = ncol, byrow = FALSE, widths = c(1, 1, 1, 0.75))
  ggsv(file = paste0("boxplots_", kk), plot = figure, path = img_path, width = 2 * (d+3), height = 4.2)
})

options(warn = oldw)
