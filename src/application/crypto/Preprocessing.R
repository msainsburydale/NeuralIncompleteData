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
  
# Pre-whiten using time series models
# See the following vignette for fitting ARMA(1, 1)-GARCH(1, 1) models:
# https://cran.r-project.org/web/packages/qrmtools/vignettes/ARMA_GARCH_VaR.html
library("rugarch")
  
})
if(!interactive()) pdf(NULL)

oldw <- getOption("warn")
options(warn = -1)

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


prewhiten <- function(X) {

  ## If there are missing data, replace it by the mean
  X <- as.vector(X)
  idx <- is.na(X)
  X[idx] <- mean(X, na.rm = TRUE)

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


# ---- Crypto ----

# Data obtained from: yahoo finance

img_path  <- "img/application/crypto"
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

df1 <- read_csv("data/crypto/BTC-USD.csv", show_col_types = F) %>% mutate(Currency = "Bitcoin")
df2 <- read_csv("data/crypto/ETH-USD.csv", show_col_types = F) %>% mutate(Currency = "Ethereum")
df3 <- read_csv("data/crypto/AVAX-USD.csv", show_col_types = F) %>% mutate(Currency = "Avalanche")
df <- rbind(df1, df2, df3)
# ETH_start_date <- min(df2$Date) # consider data beginning from Ethereum's introduction
df <- df %>%
  select(Date, Close, Currency) %>%
  rename(value = Close) %>%
  mutate(value = as.numeric(value), day = weekdays(Date)) #%>%
  # filter(Date > ETH_start_date)

# range(df$Date) # "2014-09-17" "2023-04-02"
# length(unique(df$Date)) # 1970

# Compute the log returns
df <- df %>%
  group_by(Currency) %>%
  mutate(logreturn = c(NA, diff(log(value))))

# Check NAs
idx <- which(is.na(df$logreturn))
# table(df$Currency[idx]) # Avalanche: 71   Bitcoin: 3  Ethereum: 3
# table(df$day[idx])
# length(idx) # 75

# Closing prices: time-series
add_dollar <- function(x, ...) format(paste0("$", x), ...)
closing_price_labeller <- function(variable, value){
  titles <- paste("Daily closing prices:", value)
  lst <- lapply(titles, identity)
  names(lst) <- value
  return(lst)
}
p1a <- ggplot(df) +
  geom_line(aes(x = Date, y = value)) +
  facet_wrap(~ Currency, ncol = 1, scales = "free_y", labeller = closing_price_labeller) +
  scale_y_continuous(labels = add_dollar) +
  scale_x_date(date_breaks = "1 year", date_labels =  "%b %Y") +
  theme_bw() +
  theme(
    strip.text = element_text(face = "bold", hjust = 0),
    strip.background = element_blank(),
    axis.title = element_blank()
    )

# log-daily returns: time series
returns_labeller <- function(variable,value){
  titles <- paste("Log-daily returns:", value)
  lst <- lapply(titles, identity)
  names(lst) <- value
  return(lst)
}
p1b <- ggplot(df) +
  geom_line(aes(x = Date, y = logreturn)) +
  facet_wrap(~ Currency, ncol = 1, labeller = returns_labeller) +
  scale_x_date(date_breaks = "1 year", date_labels =  "%b %Y") +
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
  titles <- paste("Standardised residuals from log-daily returns:", value)
  lst <- lapply(titles, identity)
  names(lst) <- value
  return(lst)
}
p3b <- ggplot(df) +
  geom_line(aes(x = Date, y = stdresiduals)) +
  facet_wrap(~ Currency, ncol = 1, labeller = residuals_labeller) +
  scale_x_date(date_breaks = "1 year", date_labels =  "%b %Y") +
  theme_bw() +
  theme(
    strip.text = element_text(face = "bold", hjust = 0),
    strip.background = element_blank(),
    axis.title = element_blank()
  )

(p3 <- ggarrange(p1a, p3b, align = "hv"))


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
ggsave(
  p1, file = "log-returns-time-series.pdf",
  width = 12, height = 5, device = "pdf", path = img_path
)

ggsave(
  p2, file = "log-returns-pairs.pdf",
  width = 7, height = 5.5, device = "pdf", path = img_path
)

ggsave(
  p2a, file = "log-returns-pairs-scatteronly.pdf",
  width = 9, height = 2.5, device = "pdf", path = img_path
)


ggsave(
  p3, file = "std-residuals-time-series.pdf",
  width = 12, height = 5, device = "pdf", path = img_path
)

ggsave(
  p4, file = "std-residuals-pairs.pdf",
  width = 7, height = 5.5, device = "pdf", path = img_path
)

ggsave(
  p4a, file = "std-residuals-pairs-scatteronly.pdf",
  width = 9, height = 2.5, device = "pdf", path = img_path
)


## Save the standardised data for inference in Julia
widedf <- df %>%
  select(Date, Currency, stdresiduals) %>%
  pivot_wider(names_from = Currency, values_from = stdresiduals)
widedf$Date <- NULL
# remove entries that are all NA
retain_idx <- apply(widedf, 1, function(x) !all(is.na(x)))
widedf <- widedf[retain_idx, ]

write.csv(widedf, file = "data/crypto/standardised_crypto_data.csv", row.names = F)

options(warn = oldw)
