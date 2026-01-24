suppressMessages({
  library("tidyverse")
  library("dplyr")
  library("ggplot2")
  library("fs")
})

methods <- c("EM", "masking")

models <- c(
  "GP" = "GP", 
  "HiddenPotts" = "Hidden Potts", 
  "GH" = "GH"#,
  # "Potts" = "Ising"
)

# ---- Total training times ----

for (model in names(models)) {
  for (method in methods) {
    path <- file.path("intermediates", model, paste("runs", method, sep = "_"))
    all_files <- list.files(path, recursive = TRUE) 
    all_files <- all_files[grep("train_time.csv", all_files)]
    all_files <- all_files[all_files != "train_time.csv"] # remove train_time.csv in top-level directory (this corresponds to a previous calculation of the total training time)
    all_times <- lapply(all_files, function(x) read.csv(file.path(path, x), header = F))
    all_times <- do.call(cbind, all_times)
    total_time <- sum(all_times) 
    df <- data.frame(seconds = total_time, minutes = total_time / 60, hours = total_time / 3600)
    write.csv(df, file = file.path(path, "train_time.csv"), row.names = F)  
  }
}

# ---- Risk profiles ----

load_risk_profile <- function(model, method, run_id, loss) {
  int_path <- file.path("intermediates", model)
  path <- file.path(int_path, paste0("runs_", method), paste0("estimator", run_id, "_", loss), "loss_per_epoch.csv")
  risk_df <- read_csv(path, col_names = c("Training set", "Validation set"), show_col_types = FALSE)
  risk_df <- risk_df %>%
    mutate(epoch = row_number(),
           run = paste0("estimator", run_id),
           method = method,
           model = model,
           loss = loss)
  risk_df$`Training set`[1] <- risk_df$`Validation set`[1]
  risk_df$method[risk_df$method == "EM"] <- "EM NBE"
  risk_df$method[risk_df$method == "masking"] <- "Masking NBE"
  return(risk_df)
}

all_risk <- map_dfr(names(models), function(model) {  
    map_dfr(methods, function(method) {
    int_path <- file.path("intermediates", model)
    method_dir <- file.path(int_path, paste0("runs_", method))
    if (!dir.exists(method_dir)) return(NULL)
    
    run_dirs <- dir_ls(method_dir, type = "directory")
    
    run_info <- tibble(run_dir = run_dirs) %>%
      mutate(
        run_id = str_extract(run_dir, "estimator\\d+") %>%
          str_remove("estimator") %>%
          as.integer(),
        run_loss = str_extract(run_dir, "_L[01]$") %>%
          str_remove("_")
      )
    
    map_dfr(seq_len(nrow(run_info)), function(i) {
      load_risk_profile(
        model = model,
        method   = method,
        run_id   = run_info$run_id[i],
        loss     = run_info$run_loss[i]
      )
    })
  })
})

risk_long <- all_risk %>%
  pivot_longer(cols = c("Training set", "Validation set"),
               names_to = "set",
               values_to = "risk") %>%
  mutate(epoch = epoch - 1) %>%
  mutate(model = factor(model, 
                        levels = names(models),  # Order based on names(models)
                        labels = models))        # Labels based on values of models

max_epochs_L1 <- risk_long %>%
  filter(loss == "L1") %>%
  group_by(method, model) %>%
  summarise(max_epoch_L1 = max(epoch), .groups = "drop")

risk_L1_extended <- risk_long %>%
  filter(loss == "L1") %>%
  left_join(max_epochs_L1, by = c("model", "method")) %>%
  group_by(model, method, run, set) %>%
  arrange(epoch) %>%
  tidyr::complete(
    epoch = seq(min(epoch), first(max_epoch_L1)),
    fill = list(risk = NA)
  ) %>%
  tidyr::fill(risk, .direction = "down") %>%
    mutate(epoch = ifelse(loss == "L1",
                          epoch - max_epoch_L1,
                          epoch)) %>%
  ungroup() %>%
  select(-max_epoch_L1)

risk_long_extended <- bind_rows(
  risk_L1_extended,
  risk_long %>% filter(loss != "L1")
)

# Create a combined variable for the legend
risk_long_extended <- risk_long_extended %>%
  mutate(combined_legend = paste(loss, set, sep = ": "))

create_risk_plot <- function(data, include_training = TRUE) {
  
  # Filter data if needed
  if (!include_training) {
    data <- data %>% filter(grepl("Validation set", combined_legend))
  }
  
  # Set legend order and labels
  if (include_training) {
    legend_order <- c("L1: Validation set", "L1: Training set", "L0: Validation set", "L0: Training set")
    legend_labels <- c(
      "L1: Training set" = expression(L[1]~"loss: training"),
      "L1: Validation set" = expression(L[1]~"loss: validation"),
      "L0: Training set" = "0-1 loss: training",
      "L0: Validation set" = "0-1 loss: validation"
    )
  } else {
    legend_order <- c("L1: Validation set", "L0: Validation set")
    legend_labels <- c(
      "L1: Validation set" = expression(L[1]~"loss"),
      "L0: Validation set" = "0-1 loss"
    )
  }
  
  # Create plot
  ggplot(data) +
    geom_line(
      aes(x = epoch, y = risk, group = interaction(run, set, loss), 
          linetype = combined_legend, colour = combined_legend), 
      alpha = 0.5, na.rm = TRUE
    ) +
    scale_color_manual(
      name = NULL,
      values = c("L1: Training set" = "red", 
                 "L1: Validation set" = "red",
                 "L0: Training set" = "black", 
                 "L0: Validation set" = "black"),
      labels = legend_labels,
      limits = legend_order
    ) +
    scale_linetype_manual(
      name = NULL,
      values = c("L1: Training set" = "dashed", 
                 "L1: Validation set" = "solid",
                 "L0: Training set" = "dashed", 
                 "L0: Validation set" = "solid"),
      labels = legend_labels,
      limits = legend_order
    ) +
    scale_x_continuous(limits = c(min(data$epoch), 50)) + 
    # facet_wrap(~method, ncol = 2) +
    ggh4x::facet_grid2(rows = vars(model), cols = vars(method), independent = "x", scales = "free") +  
    labs(y = "Bayes risk", x = "Epoch") +
    theme_bw() #+ 
    # guides(
    #   colour   = guide_legend(nrow = 2, byrow = FALSE),
    #   linetype = guide_legend(nrow = 2, byrow = FALSE)
    # ) +
    # theme(
    #   legend.position = "top",
    #   legend.box = "horizontal"
    # )
}

figure <- create_risk_plot(risk_long_extended, include_training = TRUE)
ggsave(file = "risk_profiles_with_training.pdf", device = "pdf", plot = figure, path = "img", width = 7, height = 1.5 + length(models))

figure <- create_risk_plot(risk_long_extended, include_training = FALSE)
ggsave(file = "risk_profiles_without_training.pdf", device = "pdf", plot = figure, path = "img", width = 7, height = 1.5 + length(models))
