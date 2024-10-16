source(file.path("src", "Plotting.R"))
suppressMessages({
library("latex2exp")
library("reshape2")
# install.packages("BiocManager")
# BiocManager::install("rhdf5")
library("rhdf5")
options(dplyr.summarise.inform = FALSE) 
})

if(!interactive()) pdf(NULL)
oldw <- getOption("warn")
options(warn = -1)

img_path <- file.path("img", "application", "sea_ice")
int_path <- file.path("intermediates", "application", "sea_ice")
dir.create(img_path, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(file.path(int_path, "estimates.csv"))
df$year <- 1979:2023

bs <- read.csv(file.path(int_path, "bs_estimates.csv"), header = F)
sie <- read.csv(file.path(int_path, "sie.csv"), header = F)
bs <- as.matrix(bs)
sie <- as.matrix(sie)

df$beta_med   <- apply(bs, 1, function(x) quantile(x, 0.5))
df$beta_lower <- apply(bs, 1, function(x) quantile(x, 0.025))
df$beta_upper <- apply(bs, 1, function(x) quantile(x, 0.975))

df$sie_lower <- apply(sie, 1, function(x) quantile(x, 0.025))
df$sie_upper <- apply(sie, 1, function(x) quantile(x, 0.975))

bet <- ggplot(df, aes(x = year, y = beta_med)) + 
  geom_point() + 
  geom_line() + 
  geom_ribbon(aes(x = year, ymin = beta_lower, ymax = beta_upper), alpha = 0.1) +
  labs(y = expression(hat(beta))) + 
  theme_bw()

sie <- ggplot(df, aes(x = year, y = sie)) + 
  geom_point() +
  geom_line() + 
  geom_ribbon(aes(x = year, ymin = sie_lower, ymax = sie_upper), alpha = 0.1) +
  labs(y = "Sea-ice extent") + 
  theme_bw()

fig <- egg::ggarrange(bet, sie, nrow = 1)

## Uncertainty at ice-sheet boundary
sea_ice <- readRDS(file.path("data", "sea_ice", "sea_ice.rds"))
sea_ice_1995 <- sea_ice[[17]]
sea_ice_1995 <- reshape2::melt(sea_ice_1995)
sea_ice_1995$value <- factor(sea_ice_1995$value, levels = c("0", "1", "Missing"))
sea_ice_1995$value[is.na(sea_ice_1995$value)] <- "Missing"
sea_ice_1995$year = 1995

sims1995 <- h5read(file.path(int_path, "sims1995.h5"), "dataset")
probs <- apply(sims1995, c(1, 2), mean)
probs <- reshape2::melt(probs)

gg1 <- ggplot(sea_ice_1995, aes(Var1, Var2, fill=value)) + 
  geom_tile() +
  geom_raster() + 
  theme_bw() + 
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(expand = c(0, 0)) + 
  scale_fill_manual(values = c("0" = "darkblue", "1" = "white", "Missing" = "red"), 
                    labels = c("0" = "Not ice", "1" = "Ice", "Missing" = "NA"), 
                    name = "") + 
  theme(axis.title = element_blank()) + 
  guides(fill = guide_legend(override.aes = list(color = "black")))


# Zoom in on the larger sample sizes.
xmin=110; xmax=150
ymin=45; ymax=90

gg1 <- gg1 +
  geom_rect(aes(xmin=xmin, xmax=xmax + 2, ymin=ymin, ymax=ymax),
            size = 1.5, colour = "grey50", fill = "transparent")

gg2 <- ggplot(probs, aes(Var1, Var2, fill = value)) + 
  geom_tile() +
  geom_raster() + 
  theme_bw() + 
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(expand = c(0, 0)) + 
  scale_fill_gradient(low = "darkblue", 
                      high = "white",
                      name = "Predictive\nProbability\nof Ice") + 
  theme(axis.title = element_blank()) 

suppressMessages({
window <- gg2 +
  scale_y_continuous(position = "right") +
  coord_cartesian(xlim = c(xmin, xmax), ylim = c(ymin, ymax)) + 
  theme(plot.margin = unit(c(10, 10, 100, 10), "points")) # padding around window  

gg <- ggarrange(
  gg1 + 
    theme(legend.position = "none") + 
    guides(fill = guide_legend(override.aes = list(color = "black"))) + 
    coord_fixed(),
  window + theme(legend.position = "top") + coord_fixed(xlim = c(xmin, xmax), ylim = c(ymin, ymax))
)
})

figure <- ggarrange(fig, gg, nrow = 1)

ggsv("sea_ice_extent", figure, path = img_path, width = 12.1, height = 3.2)

options(warn = oldw)