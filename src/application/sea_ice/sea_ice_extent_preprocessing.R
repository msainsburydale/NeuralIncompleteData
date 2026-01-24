oldw <- getOption("warn")
options(warn = -1)

source(file.path("src", "Plotting.R"))

# https://rpubs.com/boyerag/297592
suppressMessages({
  library("ncdf4")
  library("reshape2")
  library("ggplot2")
})

# Ancillary data (contains information on longitude and latitude)
anc_data <- nc_open(file.path("data", "sea_ice", "G02202-cdr-ancillary-nh.nc"))
dir.create(file.path("data", "sea_ice", "meta_data"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path("img", "application", "sea_ice"), recursive = TRUE, showWarnings = FALSE)
sink(file.path("data", "sea_ice", "G02202-cdr-ancillary-nh.txt"))
print(anc_data)
sink()
longitude <- ncvar_get(anc_data, "longitude")
latitude <- ncvar_get(anc_data, "latitude")

all_files <- list.files(file.path("data", "sea_ice", "raw_data"))

sea_ice <- lapply(all_files, function(file) {
  # Open the file
  nc_data <- nc_open(file.path("data", "sea_ice", "raw_data", file))
  
  # Extract meta data to a text file
  dir.create(file.path("data", "sea_ice", "meta_data"), recursive = TRUE, showWarnings = FALSE)
  file.path("data", "sea_ice", "meta_data", paste0(gsub("\\..*", "", file), ".txt"))
  sink()
  
  # Extract key variables 
  sea_ice <- ncvar_get(nc_data, "cdr_seaice_conc")
  missing1 <- ncvar_get(nc_data, "spatial_interpolation_flag")
  missing2 <- ncvar_get(nc_data, "temporal_interpolation_flag")
  
  missing1[is.na(missing1)] <- 0
  missing2[is.na(missing2)] <- 0
  missing1[missing1 != 0] <- 1
  missing2[missing2 != 0] <- 1
  missing_combined <- pmin(missing1 + missing2, 1)
  missing <- if (sum(missing_combined) > 10000) missing1 else missing_combined
  # missing <- missing_combined
  
  spat_interp <- ggplot(reshape2::melt(missing1), aes(Var1, Var2, fill=value)) + geom_raster() + theme_bw() + scale_x_continuous(expand = c(0, 0)) + scale_y_continuous(expand = c(0, 0)) + scale_fill_gradient2(low = "darkblue", high = "white", midpoint = 0.5)
  temp_interp <- ggplot(reshape2::melt(missing2), aes(Var1, Var2, fill=value)) + geom_raster() + theme_bw() + scale_x_continuous(expand = c(0, 0)) + scale_y_continuous(expand = c(0, 0))
  sea_ice[sea_ice > 1] <- 0
  sea_ice[missing == 1] <- NA
  
  sea_ice
})

# Original grid size
dims <- dim(sea_ice[[1]]) # 304 448
N <- prod(dims)           # 136192

# Subset the data to maximum extent of the sea ice over all years
first_nonzero_col <- sapply(sea_ice, function(mat) head(which(apply(mat, 2, function(x) any(x != 0))), 1))
last_nonzero_col  <- sapply(sea_ice, function(mat) tail(which(apply(mat, 2, function(x) any(x != 0))), 1))
first_nonzero_row <- sapply(sea_ice, function(mat) head(which(apply(mat, 1, function(x) any(x != 0))), 1))
last_nonzero_row  <- sapply(sea_ice, function(mat) tail(which(apply(mat, 1, function(x) any(x != 0))), 1))
first_nonzero_col <- min(first_nonzero_col[first_nonzero_col != 39]) # ignore strange outlier
last_nonzero_col  <- max(last_nonzero_col)
first_nonzero_row <- min(first_nonzero_row)
last_nonzero_row  <- max(last_nonzero_row)
sea_ice <- lapply(sea_ice, function(mat) mat[first_nonzero_row:last_nonzero_row, first_nonzero_col:last_nonzero_col])
longitude <- longitude[first_nonzero_row:last_nonzero_row, first_nonzero_col:last_nonzero_col]
latitude <- latitude[first_nonzero_row:last_nonzero_row, first_nonzero_col:last_nonzero_col]


fields::image.plot(sea_ice[[1]])
fields::image.plot(longitude)
fields::image.plot(latitude)


# Impute missing values using a simple mean imputation, whereby, before 
# thresholding, we simply compute the average among the neighbours
sea_ice_imputed <- lapply(sea_ice, function(mat) {
  # Get dimensions
  rows <- nrow(mat)
  cols <- ncol(mat)
  
  # Create a copy to store imputed values
  imputed_mat <- mat
  
  # Iterate until no more imputations can be made
  max_iterations <- 100  # Safeguard against infinite loops
  iteration <- 0
  imputed_count <- 1  # Initialize to enter the loop
  
  while (imputed_count > 0 && iteration < max_iterations) {
    iteration <- iteration + 1
    imputed_count <- 0
    
    # Find indices of missing values
    missing_indices <- which(is.na(imputed_mat), arr.ind = TRUE)
    
    # Skip if no missing values
    if (nrow(missing_indices) == 0) break
    
    # For each missing value
    for (i in 1:nrow(missing_indices)) {
      row <- missing_indices[i, 1]
      col <- missing_indices[i, 2]
      
      # Define neighbor boundaries (8-connected neighborhood)
      row_min <- max(1, row - 1)
      row_max <- min(rows, row + 1)
      col_min <- max(1, col - 1)
      col_max <- min(cols, col + 1)
      
      # Extract neighborhood (excluding the center cell itself)
      neighborhood <- imputed_mat[row_min:row_max, col_min:col_max]
      
      # Compute mean of non-missing neighbors
      neighbor_values <- as.vector(neighborhood)
      neighbor_values <- neighbor_values[!is.na(neighbor_values)]
      
      # Impute with mean if there are any non-missing neighbors
      if (length(neighbor_values) > 0) {
        imputed_mat[row, col] <- mean(neighbor_values)
        imputed_count <- imputed_count + 1
      }
    }
  }
  
  imputed_mat
})
fields::image.plot(sea_ice[[1]])
fields::image.plot(sea_ice_imputed[[1]])

# Save the original and imputed data
save_path <- file.path("data", "sea_ice", "preprocessed")
dir.create(save_path, recursive = TRUE, showWarnings = FALSE)
simplifyto4Darray <- function(x) {
  x <- simplify2array(x)
  array(x, dim = c(dim(x)[1], dim(x)[2], 1, dim(x)[3]))
}
saveRDS(simplifyto4Darray(sea_ice), file.path(save_path, "sea_ice_proportion.rds"))
saveRDS(simplifyto4Darray(sea_ice_imputed), file.path(save_path, "sea_ice_proportion_complete.rds"))

# Threshold the data (> 15% considered ice)
threshold_data <- function(mat) {
  mat[mat >= 0.15] <- 1
  mat[mat < 0.15] <- 0
  mat
}
sea_ice <- lapply(sea_ice, threshold_data)
sea_ice_imputed <- lapply(sea_ice_imputed, threshold_data)

saveRDS(simplifyto4Darray(sea_ice), file.path(save_path, "sea_ice_extent.rds"))
saveRDS(simplifyto4Darray(sea_ice_imputed), file.path(save_path, "sea_ice_extent_complete.rds"))

# Plot the data
idx <- c(1, 15, 17, 45)
df <- lapply(idx, function(i) {
  df <- reshape2::melt(sea_ice[[i]])
  df$year <- i + 1978
  df
})
df <- do.call(rbind, df)
df$value <- factor(df$value, levels = c("0", "1", "Missing"))
df$value[is.na(df$value)] <- "Missing"

#TODO some things to update in this figure! See the current draft of the manuscript
ggplot(df, aes(Var1, Var2, fill = value)) + 
  geom_raster() + 
  facet_wrap(~year, nrow = 1) +
  theme_bw() + 
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(expand = c(0, 0)) + 
  scale_fill_manual(values = c("0" = "darkblue", "1" = "white", "Missing" = "red"), 
                    labels = c("0" = "Not ice", "1" = "Ice", "Missing" = "Missing"), 
                    name = "") + 
  theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  ) + 
  guides(fill = guide_legend(override.aes = list(color = "black")))

options(warn = oldw)
