oldw <- getOption("warn")
options(warn = -1)

source(file.path("src", "Plotting.R"))

# https://rpubs.com/boyerag/297592
suppressMessages({
  library("ncdf4")
  library("reshape2")
  library("rnaturalearth")
  library("rnaturalearthdata")
  library("rnaturalearthhires")
  library("sf")
})

# Directories to save the full and subdomain data
int_path_full <- file.path("intermediates", "application", "sea_ice", "full")
int_path_sub <- file.path("intermediates", "application", "sea_ice", "sub")
dir.create(int_path_full, recursive = TRUE, showWarnings = FALSE)
dir.create(int_path_sub, recursive = TRUE, showWarnings = FALSE)

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
  # missing <- missing_combined
  
  spat_interp <- ggplot(reshape2::melt(missing1), aes(Var1, Var2, fill=value)) + geom_raster() + theme_bw() + scale_x_continuous(expand = c(0, 0)) + scale_y_continuous(expand = c(0, 0)) + scale_fill_gradient2(low = "darkblue", high = "white", midpoint = 0.5)
  temp_interp <- ggplot(reshape2::melt(missing2), aes(Var1, Var2, fill=value)) + geom_raster() + theme_bw() + scale_x_continuous(expand = c(0, 0)) + scale_y_continuous(expand = c(0, 0))
  # dir.create("data/sea_ice/images/spatial_interpolation/", recursive = TRUE, showWarnings = FALSE)
  # dir.create("data/sea_ice/images/temporal_interpolation/", recursive = TRUE, showWarnings = FALSE)
  # ggsave(paste0("data/sea_ice/images/spatial_interpolation/", file, ".pdf"), spat_interp, device = "pdf")
  # ggsave(paste0("data/sea_ice/images/temporal_interpolation/", file, ".pdf"), temp_interp, device = "pdf")
  
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

# # par(mfrow = c(1, 3))
# # fields::image.plot(sea_ice[[1]])
# # fields::image.plot(longitude)
# # fields::image.plot(latitude)
# 
# rotate_90_ccw <- function(x) apply(t(x), 2, rev)
# flip_horizontal <- function(x) apply(x, 2, rev)  
# # par(mfrow = c(1, 3))
# # fields::image.plot(x)
# # fields::image.plot(x %>% flip_horizontal)
# # fields::image.plot(x %>% flip_horizontal %>% rotate_90_ccw)
# 
# reorient <- function(x) x %>% flip_horizontal %>% rotate_90_ccw
# sea_ice <- lapply(sea_ice, reorient)
# longitude <- longitude %>% reorient
# latitude <- latitude %>% reorient
# # par(mfrow = c(1, 3))
# # fields::image.plot(sea_ice[[1]])
# # fields::image.plot(longitude)
# # fields::image.plot(latitude)

out_dir_window  <- "img/application/sea_ice/raw_data"
dir.create(out_dir_window,  recursive = TRUE, showWarnings = FALSE)

# Histogram function returning tidy data frame 
histogram_df <- function(mat, 
                         xmin = 1, xmax = nrow(mat),
                         ymin = 1, ymax = ncol(mat),
                         remove_0_and_1 = FALSE, 
                         year = NULL
) {
  submat <- mat[xmin:xmax, ymin:ymax]
  df <- data.frame(sea_ice = c(submat))
  
  # breaks: 28 interior + optional outer bins
  breaks <- seq(0.005, 1 - 0.005, length.out = 28)
  if (!remove_0_and_1) {
    breaks <- c(-0.03, breaks, 1.03)
  }
  
  # build the plot object
  p <- ggplot(df, aes(x = sea_ice)) +
    geom_histogram(breaks = breaks)
  
  # extract binned data
  ggb <- ggplot_build(p)
  
  hist_df <- data.frame(
    mid   = ggb$data[[1]]$x,      # bin center
    count = ggb$data[[1]]$count,
    width = ggb$data[[1]]$xmax - ggb$data[[1]]$xmin  # bin width
  )
  
  # proportions
  hist_df$prop <- hist_df$count / sum(hist_df$count)
  hist_df$year <- year
  
  return(hist_df)
}

histogram <- function(mat, title = "Entire domain", ...) {
  
  hist_df <- histogram_df(mat, ...)
  
  # max count ignoring bin centered at 0 or 1
  max_nonzero <- max(hist_df$count[hist_df$mid > 0 & hist_df$mid < 0.997])
  
  # Plot 
  ggplot(hist_df, aes(x = mid, y = prop, fill = mid)) +
    geom_col(color = "black", width = hist_df$width) +
    scale_fill_gradient2(
      low = "darkblue", mid = "skyblue", high = "white", 
      midpoint = 0.5,
      name = "Sea-ice\nproportion"
    ) +
    theme_bw() +
    labs(x = "Sea-ice proportion", y = "Proportion of observations", title = title) +
    coord_cartesian(ylim = c(0, max_nonzero/sum(hist_df$count))) +
    theme(
      plot.title = element_text(hjust = 0.5), 
      legend.position = "none"
      )
}

# Region of interest
xmin=15; xmax=75
ymin=100; ymax=200

years <- 1979:2023
names(sea_ice) <- as.character(years)

histogram_facet <- function(sea_ice) {
  # Combine all years into one tidy df
  all_hist_df <- do.call(
    rbind,
    lapply(seq_along(sea_ice), function(i) {
      histogram_df(sea_ice[[i]], year = names(sea_ice)[i],
                   # xmin = xmin, xmax = xmax,
                   # ymin = ymin, ymax = ymax,
                   remove_0_and_1 = TRUE)
    })
  )
  
  # Plot with facets
  ggplot(all_hist_df, aes(x = mid, y = prop, fill = mid)) +
    geom_col(color = "black", width = all_hist_df$width) +
    scale_fill_gradient2(low = "darkblue", mid = "skyblue", high = "white", midpoint = 0.5) +
    facet_wrap(~ year) +   
    theme_bw() +
    scale_x_continuous(breaks = c(0.2, 0.5, 0.8)) + 
    theme(
      legend.position = "none",
      strip.text = element_text(size = 8)
    ) +
    labs(x = "Sea-ice proportion", y = "Proportion of observations")
}


# histogram_facet(sea_ice)
figure <- histogram_facet(sea_ice[c("1990", "1993", "1999")])
ggsave(file.path("img/application/sea_ice/multimodal_histograms.pdf"), figure, device = "pdf", width = 8, height = 2.7)


# ---- Land mask based on longitude and latitude ----

# Get clean, validated world data
world_sf <- ne_countries(scale = "medium", returnclass = "sf") 

# Create all points at once
all_coords <- data.frame(
  i = rep(1:nrow(longitude), ncol(longitude)),
  j = rep(1:ncol(longitude), each = nrow(longitude)),
  lon = as.vector(longitude),
  lat = as.vector(latitude)
) %>% filter(!is.na(lon) & !is.na(lat))

# Convert to sf points
points_sf <- st_as_sf(all_coords, coords = c("lon", "lat"), crs = st_crs(world_sf))

# Batch intersection check
intersections <- st_intersects(points_sf, world_sf, sparse = FALSE)

# Create land mask
land_mask <- matrix(0, nrow = nrow(longitude), ncol = ncol(longitude))
land_mask[cbind(all_coords$i, all_coords$j)] <- as.integer(rowSums(intersections) > 0)

# par(mfrow = c(1,2))
# fields::image.plot(longitude, main = "Longitude")
# fields::image.plot(latitude,  main = "Latitude")
# fields::image.plot(mat, main = "Sea-ice concentration")
# fields::image.plot(land_mask, main = "Land mask")

# Function to detect borders
detect_borders <- function(mat) {
  borders <- matrix(0, nrow = nrow(mat), ncol = ncol(mat))
  
  for(i in 2:(nrow(mat)-1)) {
    for(j in 2:(ncol(mat)-1)) {
      # If current cell is land but any neighbor is not land, it's a border
      if(mat[i,j] == 1 && (mat[i-1,j] == 0 || mat[i+1,j] == 0 || 
                           mat[i,j-1] == 0 || mat[i,j+1] == 0)) {
        borders[i,j] <- 1
      }
    }
  }
  return(borders)
}

# Detect land borders
land_border_mask <- detect_borders(land_mask)

# Save the land and land border masks 
saveRDS(land_mask, file.path(int_path_full, "land_mask.rds"))
saveRDS(land_border_mask, file.path(int_path_full, "land_border_mask.rds"))

# ---- Data plots ----

snk <- lapply(years, function(year) {
  mat <- sea_ice[[year - 1978]]
  
  suppressMessages({
  
  # Entire spatial domain
  ice_plot <- ggplot(reshape2::melt(mat), aes(Var1, Var2, fill = value)) +
    geom_raster() +
    geom_tile(data = subset(reshape2::melt(land_border_mask), value == 1), aes(Var1, Var2), fill = "gray", alpha = 0.7) +
    scale_y_reverse(expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_gradient2(
      low = "darkblue", mid = "skyblue", high = "white", na.value = "red",
      midpoint = 0.5,
      name = "Sea-ice\nproportion",
      breaks = c(0.2, 0.5, 0.8)
    ) +
    theme_bw() +
    theme(axis.title = element_blank(),
          axis.text = element_blank(),
          axis.ticks = element_blank())

  # Zoom in on region of interest
  window <- ice_plot +
    theme(plot.margin = unit(c(20, 20, 20, 20), "points")) +
    theme(legend.position = "top") +
    coord_fixed(xlim = c(xmin, xmax), ylim = c(ymax, ymin)) +
    scale_y_reverse(expand = c(0, 0))

  # Add rectangle highlighting region of interest in the main plot
  ice_plot <- ice_plot +
    geom_rect(aes(xmin=xmin, xmax=xmax + 2, ymin=ymin, ymax=ymax),
              size = 1.5, colour = "grey50", fill = "transparent") +
    theme(legend.position = "none") +
    coord_fixed()

  # Plot histogram over the entire region and over the zoomed part (so, we will have four panels in the plot)
  ice_hist <- histogram(mat, remove_0_and_1 = TRUE, title = "Entire domain")
  ice_hist_window <- histogram(mat, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                               title = "Subdomain of interest",
                               remove_0_and_1 = TRUE)
  proportion_range <- range(c(ice_hist$data$prop, ice_hist_window$data$prop))
  ice_hist <- ice_hist +
    coord_cartesian(ylim = proportion_range)
  ice_hist_window <- ice_hist_window +
    coord_cartesian(ylim = proportion_range) +
    theme(axis.title.y = element_text(color = "transparent"))

  figure <- ggpubr::ggarrange(
    ice_plot,
    window,
    ice_hist,
    ice_hist_window,
    nrow = 1
  )
  
  })
  
  ggsv(file.path(out_dir_window, paste0("sea_ice_", year)), figure, device = "pdf", width = 10, height = 3.3)

  invisible(NULL)
})

# Save the full data
names(sea_ice) <- NULL 
saveRDS(sea_ice, file.path(int_path_full, "sea_ice.rds"))
saveRDS(simplify2array(sea_ice), file.path(int_path_full, "sea_ice_3Darray.rds"))
saveRDS(dim(sea_ice[[1]]), file.path(int_path_full, "dim_grid.rds"))

# Save the subdomain of interest
sea_ice_subdomain <- lapply(sea_ice, function(x) x[xmin:xmax, ymin:ymax])
# Sanity check: sea_ice_subdomain[[15]] %>% dim
# Sanity check: sea_ice_subdomain[[15]][sea_ice_subdomain[[15]] != 0] %>% c %>% hist
# Sanity check: fields::image.plot(sea_ice_subdomain[[15]])
saveRDS(sea_ice_subdomain, file.path(int_path_sub, "sea_ice.rds"))
saveRDS(simplify2array(sea_ice_subdomain), file.path(int_path_sub, "sea_ice_3Darray.rds"))
saveRDS(dim(sea_ice_subdomain[[1]]), file.path(int_path_sub, "dim_grid.rds"))

# Plot the data
years <- c(1979, 1993, 1995, 2023)
df <- lapply(years, function(year) {
  df <- reshape2::melt(sea_ice[[year - 1978]])
  df$year <- year
  df
})
df <- do.call(rbind, df)

na_colour = "red"
gg <- ggplot(df, aes(Var1, Var2, fill = value)) +
  geom_raster() +
  geom_tile(data = subset(reshape2::melt(land_border_mask), value == 1), aes(Var1, Var2), fill = "gray", alpha = 0.7) +
  geom_tile(aes(x = 0, y = 0, colour = "Missing"), inherit.aes = FALSE) + # dummy layer for "Missing" legend entry
  facet_wrap(~year, nrow = 1) +
  theme_bw() +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_reverse(expand = c(0, 0)) +
  scale_fill_gradient2(
    low = "darkblue", mid = "skyblue", high = "white", na.value = na_colour,
    midpoint = 0.5,
    name = "Sea-ice\nproportion\n", 
    limits = c(0, 1),
    breaks = c(0.2, 0.5, 0.8)
  ) +
  scale_colour_manual(
    name = NULL,
    values = c("Missing" = na_colour)
  ) +
  guides(
    fill = guide_colorbar(order = 1),
    colour = guide_legend(
      order = 2,
      override.aes = list(fill = na_colour, shape = 22) 
    )
  ) +
  theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(), 
    legend.spacing.y = unit(0.5, "pt")
  )

ggsv(file.path("img", "application", "sea_ice", "sea_ice"), gg, width = 10, height = 2.8)

# Assess percentage of missingness
# dims <- dim(sea_ice[[1]]) # 199x219
# N <- prod(dims) # 43581
# range((sapply(sea_ice, function(mat) sum(is.na(mat))) / N) * 100) # 0.0619536% 6.0530965%

options(warn = oldw)
