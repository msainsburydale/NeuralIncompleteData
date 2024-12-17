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

# Double check that information aligns with the user manual
# range(longitude)
# range(latitude)

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

# dim(longitude)
# dim(latitude)
# fields::image.plot(longitude)
# fields::image.plot(latitude)

# Threshold the data (> 15% considered ice)
sea_ice <- lapply(sea_ice, function(mat) {
  mat[mat >= 0.15] <- 1
  mat[mat < 0.15] <- 0
  mat
})

# Save the data
saveRDS(sea_ice, file.path("data", "sea_ice", "sea_ice.rds"))
saveRDS(simplify2array(sea_ice), file.path("data", "sea_ice", "sea_ice_3Darray.rds"))

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

gg <- ggplot(df, aes(Var1, Var2, fill = value)) + 
  geom_tile() +
  geom_raster() + 
  facet_wrap(~year, nrow = 1) +
  theme_bw() + 
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(expand = c(0, 0)) + 
  scale_fill_manual(values = c("0" = "darkblue", "1" = "white", "Missing" = "red"), 
                    labels = c("0" = "Not ice", "1" = "Ice", "Missing" = "NA"), 
                    name = "") + 
  theme(axis.title = element_blank()) + 
  guides(fill = guide_legend(override.aes = list(color = "black")))

ggsv(file.path("img", "application", "sea_ice", "sea_ice"), gg, width = 10, height = 2.8)

# Assess percentage of missingness
dims <- dim(sea_ice[[1]]) # 199x219
N <- prod(dims) # 43581
# range((sapply(sea_ice, function(mat) sum(is.na(mat))) / N) * 100) # 0.0619536% 6.0530965%

options(warn = oldw)
