
add_singletons <- function(a) {
  dim(a) <- c(dim(a), 1, 1)
  a
}

# Removes data completely at random (i.e., generates MCAR data)
removedata <- function(Z, proportion = runif(1, 0.01, 0.4)) {
  
  # Ensure proportion is between 0 and 1
  if (proportion < 0 || proportion > 1) stop("Proportion must be between 0 and 1")
  
  # Randomly sample indices to replace
  n <- length(Z)
  n_na <- round(proportion * n)
  na_indices <- sample(1:n, n_na)
  
  # Replace selected elements with NA
  Z[na_indices] <- NA
  
  return(Z)
}


## Example
# N <- 64
# locs <- as.matrix( expand.grid( (1:N)/N, (1:N)/N ) )
# covparms <- c(4,0.3,1,0)
# y <- fast_Gp_sim(covparms, "matern_isotropic", locs)
# y <- matrix(y, N, N)
# fields::image.plot(y)
# fields::image.plot(remove_quarter_circle(y))
remove_quarter_circle <- function(Z) {
  
  Z <- drop(Z)
  
  # Get the dimensions of the matrix
  nrows <- nrow(Z)
  ncols <- ncol(Z)
  
  # Calculate the row and column midpoints (flooring for odd dimensions)
  row_mid <- ceiling(nrows / 2)
  col_mid <- ceiling(ncols / 2)
  
  # Determine the radius of the quarter circle (smallest half-dimension of the matrix)
  radius <- min(nrows - row_mid, ncols - col_mid)
  
  # Loop over the lower-right quadrant
  for (i in (row_mid + 1):nrows) {
    for (j in (col_mid + 1):ncols) {
      
      # Calculate the distance from the lower-right corner
      dist <- sqrt((i - nrows)^2 + (j - ncols)^2)
      
      # If the distance is less than or equal to the radius, set to NA
      if (dist <= radius) {
        Z[i, j] <- NA
      }
    }
  }
  
  return(Z)
}




