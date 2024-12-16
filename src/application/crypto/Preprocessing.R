suppressMessages({
  
source("src/Plotting.R")
library("readxl")
library("readr")
library("ggplot2")
library("dplyr")
library("reshape2")
library("tidyr")
library("GGally")
library("ggpubr")
library("combinat")
library("zoo")
library("crypto2")
  
# Pre-whiten using time series models
# See the following vignette for fitting ARMA(1, 1)-GARCH(1, 1) models:
# https://cran.r-project.org/web/packages/qrmtools/vignettes/ARMA_GARCH_VaR.html
library("rugarch")
  
})
if(!interactive()) pdf(NULL)

oldw <- getOption("warn")
options(warn = -1)

img_path  <- "img/application/crypto"
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

scatterplot_data <- function(df, lims = NA, origin = FALSE, geom = geom_point, axis_labs = NULL, ...) {
  
  # all pairs
  combinations <- df %>% names %>% combinat::combn(2) %>% as.matrix
  
  # Generate the scatterplots
  apply(combinations, 2, function(p) {
    gg <- ggplot(data = df) +
      geom(aes(x = !!sym(p[1]), y = !!sym(p[2])), ...) +
      theme_bw() 
    
    
    if (!is.null(axis_labs)) {
      gg <- gg + labs(x = axis_labs[[p[1]]], y = axis_labs[[p[2]]]) 
    }
    
    if (!all(is.na(lims))) {
      gg <- gg + 
        xlim(lims[1], lims[2]) + 
        ylim(lims[1], lims[2])
    }
    
    if (origin) {
      gg <- gg + 
        geom_vline(xintercept = 0, linewidth = 0.5, alpha = 0.3) +
        geom_hline(yintercept = 0, linewidth = 0.5, alpha = 0.3)
    }
    
    gg
  })
}


prewhiten <- function(X, replace_missing = c("persistence", "mean")) {
  
  replace_missing <- match.arg(replace_missing)
  
  ## Convert to vector
  X <- as.vector(X)

  ## Handle missing data
  idx <- is.na(X)
  if (replace_missing == "mean") {
    X[idx] <- mean(X, na.rm = TRUE)
  } else if (replace_missing == "persistence") {
    X[idx] <- zoo::na.locf(X, na.rm = FALSE)  # Fill forward
    X[idx & is.na(X)] <- zoo::na.locf(X, fromLast = TRUE)  # Fill backward if necessary
  }
  
  ## Fit an ARMA(1,1)-GARCH(1,1) model
  armaOrder  <- c(1,1) # ARMA order
  garchOrder <- c(1,1) # GARCH order
  varModel <- list(model = "sGARCH", garchOrder = garchOrder)
  spec <- ugarchspec(
    varModel,
    mean.model = list(armaOrder = armaOrder),
    distribution.model = "norm"
  )
  fit <- ugarchfit(spec, data = X)
  
  ## Standardised residuals
  Z <- as.numeric(residuals(fit, standardize = TRUE))
  
  ## Re-introduce missingness
  Z[idx] <- NA
  
  return(Z)
}


# ---- Crypto data ----

# df <- crypto2::crypto_list() %>% 
#   filter(name %in% c("Bitcoin", "Ethereum", "Avalanche")) %>% 
#   crypto2::crypto_history()
# write_csv(df, "data/crypto/raw_data.csv")

df <- read_csv("data/crypto/raw_data.csv", show_col_types = F)
df <- df %>%
  rename(Date = timestamp, Currency = name, Close = close) %>% 
  mutate(Date = as.Date(Date)) %>% 
  select(Date, Close, Currency) %>%
  rename(value = Close) %>%
  mutate(value = as.numeric(value), day = weekdays(Date)) 

# range(df$Date) # "2013-04-28" "2024-12-10"
# length(unique(df$Date)) # 4250
# df %>% group_by(Currency) %>% summarise(min(Date))

# Compute the log returns
df <- df %>%
  group_by(Currency) %>%
  mutate(logreturn = c(NA, diff(log(value))))

# Check NAs
idx <- which(is.na(df$logreturn))
# table(df$Currency[idx]) # Avalanche: 71   Bitcoin: 3  Ethereum: 3
# table(df$day[idx])
# length(idx) # 75

# Plotting order 
df$Currency <- factor(df$Currency, levels = c("Bitcoin", "Ethereum", "Avalanche"))

# Closing prices: time-series
add_dollar <- function(x, ...) format(paste0("$", x), ...)
closing_price_labeller <- function(variable, value){
  titles <- paste("Daily closing prices Yₜ:", value)
  lst <- lapply(titles, identity)
  names(lst) <- value
  return(lst)
}

p1a <- ggplot(df) +
  geom_line(aes(x = Date, y = value)) +
  facet_wrap(~ Currency, ncol = 1, scales = "free_y", labeller = closing_price_labeller) +
  scale_y_continuous(labels = function(x) paste0("$", format(x, scientific = FALSE))) + 
  scale_x_date(date_breaks = "2 year", date_labels =  "%b %Y") +
  theme_bw() +
  theme(
    strip.text = element_text(face = "bold", hjust = 0),
    strip.background = element_blank(),
    axis.title = element_blank()
    )

# log-daily returns: time series
returns_labeller <- function(variable,value){
  titles <- paste("Log-daily returns rₜ:", value)
  lst <- lapply(titles, identity)
  names(lst) <- value
  return(lst)
}
p1b <- ggplot(df) +
  geom_line(aes(x = Date, y = logreturn)) +
  facet_wrap(~ Currency, ncol = 1, labeller = returns_labeller) +
  scale_x_date(date_breaks = "2 year", date_labels =  "%b %Y") +
  theme_bw() +
  theme(
    strip.text = element_text(face = "bold", hjust = 0),
    strip.background = element_blank(),
    axis.title = element_blank()
  )

p1 <- ggarrange(p1a, p1b, align = "hv")

# log-daily returns: pairs plots
widedf <- df %>%
  select(Date, Currency, logreturn) %>%
  pivot_wider(names_from = Currency, values_from = logreturn)

p2 <- ggpairs(
  widedf, columns = 2:ncol(widedf),
  lower = list(continuous = wrap("points", alpha = 0.5))
  ) +
  theme_bw() +
  theme(
    strip.text = element_text(face = "bold"),
    strip.background = element_blank(),
  )


# log-daily returns: only bivariate plots
plots = scatterplot_data(
  widedf[, -1],
  lims = range(df$logreturn, na.rm = TRUE),
  origin = TRUE,
  alpha = 0.5
  )
p2a <- ggarrange(plotlist = plots, nrow = 1)

df <- df %>% group_by(Currency) %>% mutate(stdresiduals = prewhiten(logreturn))

# Standardised residuals: time series
residuals_labeller <- function(variable,value){
  titles <- paste("Standardised residuals Zₜ from log-daily returns rₜ:", value)
  lst <- lapply(titles, identity)
  names(lst) <- value
  return(lst)
}
p3b <- ggplot(df) +
  geom_line(aes(x = Date, y = stdresiduals)) +
  facet_wrap(~ Currency, ncol = 1, labeller = residuals_labeller) +
  scale_x_date(date_breaks = "2 year", date_labels =  "%b %Y") +
  theme_bw() +
  theme(
    strip.text = element_text(face = "bold", hjust = 0),
    strip.background = element_blank(),
    axis.title = element_blank()
  )

p3 <- ggarrange(p1a, p3b, align = "hv")


# Standardised residuals: pairs plots
widedf <- df %>%
  select(Date, Currency, stdresiduals) %>%
  pivot_wider(names_from = Currency, values_from = stdresiduals)

p4 <- ggpairs(
  widedf, columns = 2:ncol(widedf),
  lower = list(continuous = wrap("points", alpha = 0.5))
) +
  theme_bw() +
  theme(
    strip.text = element_text(face = "bold"),
    strip.background = element_blank(),
  )

# Standardised residuals: only bivariate plots
plots = scatterplot_data(
  widedf[, -1],
  lims = range(df$stdresiduals, na.rm = TRUE),
  origin = TRUE,
  alpha = 0.5
)
p4a <- ggarrange(plotlist = plots, nrow = 1)


# Save the plots
ggsv(p1, file = "log-returns-time-series", width = 12, height = 5, path = img_path)
ggsv(p2, file = "log-returns-pairs", width = 7, height = 5.5, path = img_path)
ggsv(p2a, file = "log-returns-pairs-scatteronly", width = 9, height = 2.5, path = img_path)
ggsv(p3, file = "std-residuals-time-series", width = 12, height = 5, path = img_path)
ggsv(p4, file = "std-residuals-pairs", width = 7, height = 5.5, path = img_path)
ggsv(p4a, file = "std-residuals-pairs-scatteronly", width = 9, height = 2.5, path = img_path)


## Save the standardised data for inference in Julia
widedf <- df %>%
  select(Date, Currency, stdresiduals) %>%
  pivot_wider(names_from = Currency, values_from = stdresiduals)
widedf$Date <- NULL
# remove entries that are all NA
retain_idx <- apply(widedf, 1, function(x) !all(is.na(x)))
widedf <- widedf[retain_idx, ]

write.csv(widedf, file = "data/crypto/standardised_data.csv", row.names = F)

options(warn = oldw)

