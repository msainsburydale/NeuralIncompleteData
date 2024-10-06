model <- file.path("spatial", "Potts")

source(file.path("src", "Plotting.R"))

if(!interactive()) pdf(NULL)
oldw <- getOption("warn")
options(warn = -1)

estimates_path <- file.path("intermediates", model, "Estimates")
img_path       <- file.path("img", model)
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

parameter_labels = c(
  "β" = expression(hat(beta))
)

missingness <- c("MCAR", "MB")

estimator_labels <- c(
  "neuralEM" = "Neural EM",
  "neuralEncoding" = "Masked NBE"
)

# ---- Missing data: entire parameter space ----

rmse <- data.frame()
for (set in missingness) {

  df  <- file.path(estimates_path, paste0("estimates_", set, "_test.csv")) %>% read.csv
  
  ## RMSE
  rmse_tmp <- df %>%
    mutate(loss = (estimate - truth)^2) %>%
    group_by(estimator) %>%
    summarise(RMSE = sqrt(mean(loss))) 
  rmse_tmp$missingness <- set
  rmse <- rbind(rmse, rmse_tmp)

  plt <- plotestimates(df[sample(nrow(df)), ], estimator_labels = estimator_labels, parameter_labels = parameter_labels) + scale_estimator_aesthetic(df)
  ggsv(file = paste0("global", set), plot = plt, width = 6.5, height = 3.8, path = img_path)
}


plots <- lapply(missingness, function(set) {
  
  df  <- file.path(estimates_path, paste0("estimates_", set, "_test.csv")) %>% read.csv
  plotestimates(df[sample(nrow(df)), ], estimator_labels = estimator_labels, parameter_labels = parameter_labels) + 
    scale_estimator_aesthetic(df) + 
    labs(title = set, x = expression(beta), y = expression(hat(beta))) +
    theme(plot.title = element_text(hjust = 0.5)) + 
    theme(
      strip.background = element_blank(),
      strip.text.x = element_blank() 
      #axis.title.y = element_text(angle = 0, vjust = 0.5)
    ) 
    
})

plots[[1]] <- plots[[1]] + theme(legend.position = "none") 
plots[[2]] <- plots[[2]] + theme(axis.title.y = element_blank()) 

plt <- egg::ggarrange(plots = plots, nrow = 1)
ggsv(file = "global", plot = plt, width = 8.5, height = 3.8, path = img_path)

## Process the results for the RMSE
rmse <- pivot_wider(rmse, id_cols = "missingness", names_from = "estimator", values_from = c("RMSE"))
write.csv(rmse, row.names = F, file = file.path(img_path, "rmse.csv"))

# ---- Missing data: sampling distributions ----

loadestimates <- function(type) {
  df <- file.path(estimates_path, paste0("estimates_", type, "_scenarios.csv")) %>% read.csv
  df$missingness <- type
  df
}

loaddata <- function(type) {
  df <- file.path(estimates_path, paste0("Z_", type, ".csv")) %>% read.csv
  df$missingness <- type
  df
}

df <- loadestimates(missingness[1]) %>% rbind(loadestimates(missingness[2])) 
zdf <- loaddata(missingness[1]) %>% rbind(loaddata(missingness[2]))
p <- sum(names(parameter_labels) %in% df$parameter)

N <- 64 # TODO not ideal that this is hard coded; should probably save N, or x and y, in Julia
zdf$x <- rep(1:N, each = N) 
zdf$y <- N:1

figures <- lapply(unique(df$k), function(kk) {
  
  df  <- df  %>% filter(k == kk)
  zdf <- zdf %>% filter(k == kk)
  
  l <- length(missingness) # number of missingness patterns
  
  # ---- Data plots ----
  
  zdf$Z <- as.factor(zdf$Z)
  
  suppressMessages({
    data <- lapply(missingness, function(mis) {
      
      field <- filter(zdf, j == 2, missingness == mis)
      
      gg <- ggplot(field, aes(x = x, y = y)) + 
        geom_tile(aes(fill = Z)) +
        scale_fill_viridis(option = "magma", discrete = TRUE, na.value = "gray") +
        # scale_fill_viridis(option = "plasma", discrete = TRUE, na.value = "gray") +
        # scale_fill_manual(values = c("1" = "darkblue", "2" = "white"), na.value = "red") + 
        labs(fill = "Z", x = expression(s[1]), y = expression(s[2])) +
        theme_bw() +
        scale_x_continuous(expand = c(0, 0)) +
        scale_y_continuous(expand = c(0, 0)) + 
        theme(legend.title.align = 0.25, legend.title = element_text(face = "bold"))
    })
  })
  data_legend <- get_legend(data[[1]])
  
  data <- lapply(data, function(gg) gg +
                   theme(legend.position = "none") +
                   theme(plot.title = element_text(hjust = 0.5)) 
  )
  
  data <- lapply(data, function(gg) gg + coord_fixed())
  
  suppressMessages({
    data[-l] <- lapply(data[-l], function(gg) gg +
                         theme(axis.text.x = element_blank(),
                               axis.ticks.x = element_blank(),
                               axis.title.x = element_blank()) + 
                         scale_x_continuous(breaks = c(0.2, 0.5, 0.8), expand = c(0, 0))
    )
  })
  
  # ---- Box plots ----
  
  box <- lapply(missingness, function(mis) {
    plotdistribution(filter(df, missingness == mis), type = "box", parameter_labels = parameter_labels, estimator_labels = estimator_labels, truth_line_size = 1, return_list = TRUE, flip = TRUE)
  })
  
  #NB this will break if p > 1
  if (p > 1) stop("Need to change the code that keeps the axes fixed between panels") 
  est_lims <- range(df$estimate)
  
  
  p <- length(unique(df$parameter))
  box_split <- lapply(1:p, function(i) {
    lapply(1:length(box), function(j) box[[j]][[i]])
  })
  

  # Modify the axes 
  for (i in 1:p) {
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
  })
  box_legend <- get_legend(box[[1]])
  suppressMessages({
    box <- lapply(box, function(gg) {
      gg$facet$params$nrow <- 2
      gg$facet$params$strip.position <- "bottom"
      gg <- gg + theme(legend.position = "none", axis.title.y = element_blank()) + scale_y_continuous(limits = est_lims, n.breaks = 4)
      gg
    })
  })
  
  legends <- list(ggplotify::as.ggplot(data_legend), ggplotify::as.ggplot(box_legend))
  
  # ---- Combine with global scatter plot ----
  
  global_plots <- lapply(missingness, function(set) {
    
    df  <- file.path(estimates_path, paste0("estimates_", set, "_test.csv")) %>% read.csv
    gg <- plotestimates(df[sample(nrow(df)), ], estimator_labels = estimator_labels, parameter_labels = parameter_labels) + 
      scale_estimator_aesthetic(df) + 
      scale_x_continuous(breaks = c(0.3, 0.6, 0.9, 1.2, 1.5)) + 
      scale_y_continuous(breaks = c(0.3, 0.6, 0.9, 1.2, 1.5)) + 
      labs(x = expression(beta), y = expression(hat(beta))) +
      theme(
        strip.background = element_blank(),
        strip.text.x = element_blank(),
        legend.position = "none"
      ) 
    
    gg$layers[[1]]$aes_params$size <- 0.4
    
    gg
    
  })
  global_plots[[1]] <- global_plots[[1]] + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), axis.ticks = element_blank())
  global_plots[[1]] <- global_plots[[1]] + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), axis.ticks = element_blank())
  
  plotlist <- c(data, box, global_plots, legends)
  nrow <- length(missingness)
  ncol <- p + 2 + 1
  figure  <- egg::ggarrange(plots = plotlist, nrow = nrow, ncol = ncol, byrow = FALSE, widths = c(1, 1, 1, 0.75))
  ggsv(file = paste0("missing_boxplots_k", kk), plot = figure, path = img_path, width = 2 * (p+3), height = 4.2)

})

options(warn = oldw)

