source(file.path("src", "Plotting.R"))

if(!interactive()) pdf(NULL)
oldw <- getOption("warn")
options(warn = -1)

int_path <- file.path("intermediates", "GH")
img_path       <- file.path("img", "GH")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

parameter_labels = c(
  "γ" = expression(hat(alpha)),
  "ω" = expression(hat(omega)),
  "λ" = expression(hat(lambda)), 
  "ρ" = expression(hat(rho)),
  "ν" = expression(hat(nu))
)

missingness <- c("MCAR", "MB")

estimator_labels <- c(
  "MAP" = "MAP",
  "EM" = "EM NBE",
  "masking" = "Masking NBE"
)

# ---- EM convergence ----

library("readr")
df <- read_csv(file.path(int_path, "EM_iterates.csv"), show_col_types = FALSE)
df_true <- read_csv(file.path(int_path, "EM_iterates_truth.csv"), show_col_types = FALSE)
figure1 <- plot_EM_trajectories(df, parameter_labels, colour_by_theta_0 = FALSE)
figure2 <- plot_EM_trajectories(df, parameter_labels, H = max(df$nsims), max_iteration = 10, colour_by_theta_0 = TRUE, show_average = FALSE) + scale_x_continuous(breaks = c(0, 5, 10))
ggsv("convergence_multipleMC", figure1, path = img_path, width = 8, height = 7)
ggsv("convergence_singleMC", figure2, path = img_path, width = 9.7, height = 2.6)

figure <- egg::ggarrange(figure1, figure2, ncol = 1, labels = c("A", "B"), heights = c(4, 1))
ggsv("convergence", figure, path = img_path, width = 8.8, height = 9.4)

# ---- RMSE ----

#TODO check these results

rmse <- data.frame()
for (set in missingness) {
  
  df  <- file.path(int_path, paste0("estimates_", set, "_test.csv")) %>% read.csv
  rmse_tmp <- df %>%
    mutate(loss = (estimate - truth)^2) %>%
    group_by(estimator) %>%
    summarise(RMSE = sqrt(mean(loss))) 
  rmse_tmp$missingness <- set
  rmse <- rbind(rmse, rmse_tmp)
}
rmse <- pivot_wider(rmse, id_cols = "missingness", names_from = "estimator", values_from = c("RMSE"))
write.csv(rmse, row.names = F, file = file.path(img_path, "rmse.csv"))

# ---- Sampling distributions ----

loadestimates <- function(type) {
  df <- file.path(int_path, paste0("estimates_", type, "_scenarios.csv")) %>% read.csv
  abc_df <- file.path(int_path, "ABC", paste0("estimates_", type, "_scenarios.csv")) %>% read.csv
  df <- rbind(df, abc_df)
  df$missingness <- type
  df
}

loaddata <- function(type) {
  df <- file.path(int_path, paste0("Z_", type, ".csv")) %>% read.csv
  df$missingness <- type
  df
}

df  <- loadestimates(missingness[1]) %>% rbind(loadestimates(missingness[2])) 
zdf <- loaddata(missingness[1]) %>% rbind(loaddata(missingness[2]))
d   <- sum(names(parameter_labels) %in% df$parameter)

N <- zdf %>% filter(k==1 & j == 1 & missingness == "MCAR") %>% nrow %>% sqrt # NB assumes square grid
zdf$x <- rep(seq(0, 1, len=N), each = N) 
zdf$y <- seq(0, 1, len=N)

figures <- lapply(unique(df$k), function(kk) {
  
  df  <- df  %>% filter(k == kk)
  zdf <- zdf %>% filter(k == kk)
  
  l <- length(missingness) # number of missingness patterns
  
  ## Data plots
  suppressMessages({
    data <- lapply(missingness, function(mis) {
      field_plot(filter(zdf, j == 1, missingness == mis), regular = T) + 
        scale_x_continuous(breaks = c(0.2, 0.5, 0.8), expand = c(0, 0)) +
        scale_y_continuous(breaks = c(0.2, 0.5, 0.8), expand = c(0, 0)) +
        labs(fill = expression(Z[1])) +
        theme(legend.title.align = 0.25, legend.title = element_text(face = "bold"))
    })
  })
  data_legend <- get_legend(data[[1]])
  
  data <- lapply(data, function(gg) gg +
                   theme(legend.position = "none") +
                   theme(plot.title = element_text(hjust = 0.5)) +
                           coord_fixed())
  
  suppressMessages({
    data[-l] <- lapply(data[-l], function(gg) gg +
                         theme(axis.text.x = element_blank(),
                               axis.ticks.x = element_blank(),
                               axis.title.x = element_blank()) +
                         scale_x_continuous(breaks = c(0.2, 0.5, 0.8), expand = c(0, 0))
    )
  })
  
  
  
  ## Box plots
  df$estimator <- factor(df$estimator, levels = c("MAP", "EM", "masking", "ABC MAP"))
  box <- lapply(missingness, function(mis) {
    plotdistribution(filter(df, missingness == mis), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1, return_list = TRUE, flip = TRUE)
  })
  d <- length(unique(df$parameter))
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
      gg <- gg + 
        theme(
          legend.position = "none", 
          axis.title.y = element_blank()
          ) 
      gg
    })
  })
  
  nrow <- length(missingness)
  legends <- list(ggplotify::as.ggplot(data_legend), ggplotify::as.ggplot(box_legend))
  plotlist <- c(data, box, legends)
  ncol <- d + 2
  figure  <- egg::ggarrange(plots = plotlist, nrow = nrow, ncol = ncol, byrow = FALSE)
  
  ggsv(file = paste0("boxplots_", kk), plot = figure, path = img_path, width = 10, height = 3.4)
})

options(warn = oldw)

