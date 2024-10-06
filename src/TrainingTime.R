library("optparse")
option_list <- list(
  make_option("--model", type="character", default=NULL, metavar="character")
)
opt_parser  <- OptionParser(option_list=option_list)
model       <- parse_args(opt_parser)$model
model       <- gsub("/", .Platform$file.sep, model)

# Find all training time files 
for (estimator in c("EM", "masking")) {
  path <- file.path("intermediates", model, paste("runs", estimator, sep = "_"))
  all_files <- list.files(path, recursive = TRUE) 
  all_files <- all_files[grep("train_time.csv", all_files)]
  all_files <- all_files[all_files != "train_time.csv"] # remove train_time.csv in top-level directory (this corresponds to a previous calculation of the total training time)
  all_times <- lapply(all_files, function(x) read.csv(file.path(path, x), header = F))
  all_times <- do.call(cbind, all_times)
  total_time <- sum(all_times) 
  df <- data.frame(seconds = total_time, minutes = total_time / 60, hours = total_time / 3600)
  write.csv(df, file = file.path(path, "train_time.csv"), row.names = F)  
}
