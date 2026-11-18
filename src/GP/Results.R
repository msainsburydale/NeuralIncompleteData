# Results of the Gaussian process simulation study (Section 3.2): RMSEs, box plots, and EM convergence figures.
#
# Usage (from the top level of the repository):
#   Rscript src/GP/Results.R          results of the R implementation (src/GP/Experiment.R)
#   Rscript src/GP/Results.R --julia  results of the Julia implementation (src/GP/Experiment.jl)

julia_results <- "--julia" %in% commandArgs(trailingOnly = TRUE)

source(file.path("src", "Plotting.R"))
if (!interactive()) pdf(NULL)
suppressMessages({
  library("dplyr")
  library("readr")
  options(dplyr.summarise.inform = FALSE)
})

int_path <- file.path("intermediates", "GP")
if (julia_results) int_path <- file.path(int_path, "julia")
estimates_path <- file.path(int_path, "Estimates")
img_path <- if (julia_results) file.path("img", "GP", "julia") else file.path("img", "GP")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

missingness <- c("MCAR", "MNAR")

# ---- RMSE ----

df <- read.csv(file.path(estimates_path, "estimates_test.csv"))

rmse_df <- df %>%
  anti_join(df %>% filter(parameter == "rho", truth > 0.275) %>% distinct(k), by = "k") %>% # Remove very large rho, likelihood too flat in that region and analytical MAP becomes numerically unstable
  mutate(squared_error = (estimate - truth)^2) %>%
  group_by(estimator, missingness) %>%
  summarise(rmse = sqrt(mean(squared_error)), bias = mean(estimate - truth))
write.csv(rmse_df, file.path(estimates_path, "rmse.csv"), row.names = FALSE)
print(as.data.frame(rmse_df))

rmse_df <- df %>%
  mutate(squared_error = (estimate - truth)^2) %>%
  group_by(estimator, missingness, parameter) %>%
  summarise(rmse = sqrt(mean(squared_error)), bias = mean(estimate - truth))
write.csv(rmse_df, file.path(estimates_path, "rmse_by_parameter.csv"), row.names = FALSE)

# ---- Sampling distributions ----

estimator_labels <- c(
  "MAP" = "MAP",
  "neuralEM" = "EM NBE",
  "masking" = "Masking NBE"#,
  #"ABC MAP" = "ABC MAP"
)

parameter_labels <- c(
  "τ" = expression(hat(tau)),
  "tau" = expression(hat(tau)),
  "ρ" = expression(hat(rho)),
  "rho" = expression(hat(rho))
)

loadestimates <- function(type) {
  df <- file.path(estimates_path, paste0("estimates_", type, "_scenarios.csv")) %>% read.csv
  df$missingness <- type
  df
}

loaddata <- function(type) {
  # The incomplete data are those of the R implementation
  df <- file.path("intermediates", "GP", "Estimates", paste0("Z_", type, ".csv")) %>% read.csv
  df$missingness <- type
  df
}

df <- loadestimates(missingness[1]) %>%
  rbind(loadestimates(missingness[2])) %>%
  filter(estimator %in% names(estimator_labels))

zdf <- loaddata(missingness[1]) %>% rbind(loaddata(missingness[2]))

N <- 64
zdf$x <- rep(seq(0, 1, len = N), each = N)
zdf$y <- seq(0, 1, len = N)

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
        labs(fill = "Z") +
        theme(legend.title.align = 0.25, legend.title = element_text(face = "bold"))
    })
  })
  data_legend <- get_legend(data[[1]])
  data <- lapply(data, function(gg) gg + theme(legend.position = "none") + coord_fixed())

  suppressMessages({
    data[-l] <- lapply(data[-l], function(gg) gg +
                         theme(axis.text.x = element_blank(),
                               axis.ticks.x = element_blank(),
                               axis.title.x = element_blank()) +
                         scale_x_continuous(breaks = c(0.2, 0.5, 0.8), expand = c(0, 0))
    )
  })

  ## Box plots
  df$estimator <- factor(df$estimator, levels = c("MAP", "neuralEM", "masking", "ABC MAP"))
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
      gg <- gg + theme(legend.position = "none", axis.title.y = element_blank())
      gg
    })
  })

  nrow <- length(missingness)
  legends <- list(ggplotify::as.ggplot(data_legend), ggplotify::as.ggplot(box_legend))
  plotlist <- c(data, box, legends)
  ncol <- d + 2
  figure  <- egg::ggarrange(plots = plotlist, nrow = nrow, ncol = ncol, byrow = FALSE)

  ggsv(file = paste0("boxplots_k", kk), plot = figure, path = img_path,
       width = 2 * (d+2), height = 4.5)

})

# ---- Convergence of the EM algorithm for a single data set ----

df <- read_csv(file.path(int_path, "EM_iterates.csv"), show_col_types = FALSE)
figure1 <- plot_EM_trajectories(df, parameter_labels)
figure2 <- plot_EM_trajectories(df, parameter_labels, H = 30)
ggsv("convergence_multipleMC", figure1, path = img_path, width = 7.5, height = 4)
ggsv("convergence_singleMC", figure2, path = img_path, width = 6.5, height = 3)
