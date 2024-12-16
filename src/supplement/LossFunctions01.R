suppressMessages({
  library("ggplot2")
  library("latex2exp")
  library("forcats")
  library("egg")
})
if(!interactive()) pdf(NULL)

strip_font_size <- 11

displacement <- c(
  seq(-1.2, -0.05, by = 0.01),
  seq(-0.05, 0.05, by = 0.0005),
  seq(0.05, 1.2, by = 0.01)
)

xbreaks <- c(-0.75, 0, 0.75)

# ---- power loss ---- 

beta <- c(0, 0.1, 0.2, 0.5, 1)

L_beta <- function(displacement, beta, delta = 0) (abs(displacement) + delta)^beta - delta^beta
L_beta_gradient <- function(displacement, beta, delta = 0) (-1)^(displacement < 0) * beta * (abs(displacement) + delta)^(beta-1)

beta <- c(0.05, 0.1, 0.5, 1)
displacement <- c(
  seq(-1.2, -0.05, by = 0.01),
  seq(-0.05, 0.05, by = 0.001),
  seq(0.05, 1.2, by = 0.001)
)
delta <- c(0, 0.1)

df <- expand.grid(beta = beta, displacement = displacement, delta = delta)
df$L_beta <- L_beta(df$displacement, df$beta, df$delta)
df$L_beta_gradient <- L_beta_gradient(df$displacement, df$beta, df$delta)
# The gradient diverges, so we get infinty; replace this by some large number
idx <- which(df$L_beta_gradient==Inf)
tmp <- df[idx, ] 
tmp$L_beta_gradient <- -100
df <- rbind(df, tmp)
tmp$L_beta_gradient <- 100
df <- rbind(df, tmp)

# "Open" and "closed" points to denote the discontinuity
points <- lapply(beta, function(b) {
  grad <- df[df$beta == b & df$delta == 0.1, "L_beta_gradient"]
  range(grad)
  point1 <- data.frame(x = 0, y = max(grad), beta = b, delta = 0.1, point = "upper")
  point2 <- data.frame(x = 0, y = min(grad), beta = b, delta = 0.1, point = "lower")
  rbind(point1, point2)
})
points <- do.call(rbind, points)

## Loss plot
df$delta <- factor(df$delta)
df$beta <- factor(df$beta)
levels(df$beta) <- sapply(paste("$\\beta$ = ", levels(df$beta)), TeX)
figure_1a <- ggplot(df) +
  geom_line(aes(x = displacement, y = L_beta, colour = delta, group = delta)) +
  facet_wrap(~fct_rev(beta), labeller = label_parsed, nrow = 1) +
  labs(
    x = expression(hat(theta) - theta),
    y = TeX(r'($L_{POW}(\hat{\theta}, \theta; \, \beta, \delta)$)'), 
    colour = expression(delta)
  ) +
  scale_x_continuous(breaks = xbreaks) +
  theme_bw() + 
  theme(strip.background = element_blank(), strip.text.x = element_text(size = strip_font_size))

## Gradient plot
levels(df$delta) <- sapply(paste("$\\delta$ = ", levels(df$delta)), TeX)
points$beta <- factor(points$beta)
points$delta <- factor(points$delta)
levels(points$beta) <- sapply(paste("$\\beta$ = ", levels(points$beta)), TeX)
levels(points$delta) <- sapply(paste("$\\delta$ = ", levels(points$delta)), TeX)

figure_1b <- ggplot(df) +
  geom_line(aes(x = displacement, y = L_beta_gradient, group = displacement < 0)) +
  geom_point(data = points, aes(x = x, y = y, fill = point), shape = 21) + 
  facet_grid(delta~fct_rev(beta), labeller = label_parsed, scales = "free") +
  labs(
    x = expression(hat(theta) - theta),
    y = TeX(r'($\frac{d L_{POW}(\hat{\theta}, \theta; \, \beta, \delta)}{d (\hat{\theta} - \theta)}$)')
  ) +
  scale_x_continuous(breaks = xbreaks) +
  scale_fill_manual(values = c("black", "white")) + 
  theme_bw() +
  theme(
    strip.background = element_blank(), 
    strip.text.x = element_text(size = strip_font_size), 
    legend.position = "none"
    )

figure_1a <- figure_1a + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), axis.ticks.x = element_blank())
figure_1b <- figure_1b + theme(strip.background = element_blank(), strip.text.x = element_blank())
figure_1 <- egg::ggarrange(figure_1a, figure_1b, heights = c(1, 2))

# ---- tanh loss ----

L_kappa <- function(displacement, kappa)  abs(tanh(displacement/kappa))
L_kappa_gradient <- function(displacement, kappa) (-1)^(displacement < 0) * (1 - tanh(displacement/kappa)^2) / kappa

kappa <- c(0.05, 0.1, 0.5, 1)

df <- expand.grid(kappa = kappa, displacement = displacement)
df$L_kappa <- L_kappa(df$displacement, df$kappa)
df$L_kappa_gradient <- L_kappa_gradient(df$displacement, df$kappa)

# "Open" and "closed" points to denote the discontinuity
points <- lapply(kappa, function(k) {
  grad <- df[df$kappa == k, "L_kappa_gradient"]
  range(grad)
  point1 <- data.frame(x = 0, y = max(grad), kappa = k, point = "upper")
  point2 <- data.frame(x = 0, y = min(grad), kappa = k, point = "lower")
  rbind(point1, point2)
})
points <- do.call(rbind, points)

## Loss plot
df$kappa <- factor(df$kappa)
levels(df$kappa) <- sapply(paste("$\\kappa$ = ", levels(df$kappa)), TeX)
figure_2a <- ggplot(df) +
  geom_line(aes(x = displacement, y = L_kappa)) +
  facet_wrap(~fct_rev(kappa), labeller = label_parsed, nrow = 1) +
  labs(
    x = expression(hat(theta) - theta),
    y = TeX(r'($L_{TANH}(\hat{\theta}, \theta; \, \kappa)$)')
  ) +
  theme_bw() + 
  scale_x_continuous(breaks = xbreaks) + 
  theme(strip.background = element_blank(), strip.text.x = element_text(size = strip_font_size))


## Gradient plot
points$kappa <- factor(points$kappa)
levels(points$kappa) <- sapply(paste("$\\kappa$ = ", levels(points$kappa)), TeX)
figure_2b <- ggplot(df) +
  geom_line(aes(x = displacement, y = L_kappa_gradient, group = displacement < 0)) +
  geom_point(data = points, aes(x = x, y = y, fill = point), shape = 21) + 
  facet_wrap(~fct_rev(kappa), labeller = label_parsed, nrow = 1) +
  labs(
    x = expression(hat(theta) - theta),
    y = TeX(r'($\frac{d L_{TANH}(\hat{\theta}, \theta; \, \kappa)}{d (\hat{\theta} - \theta)}$)')
  ) +
  theme_bw() + 
  scale_x_continuous(breaks = xbreaks) + 
  scale_fill_manual(values = c("black", "white")) + 
  theme(
    strip.background = element_blank(), 
    strip.text.x = element_text(size = strip_font_size), 
    legend.position = "none"
  )

figure_2a <- figure_2a + theme(axis.title.x = element_blank(), axis.text.x = element_blank(), axis.ticks.x = element_blank())
figure_2b <- figure_2b + theme(strip.background = element_blank(), strip.text.x = element_blank())
figure_2 <- egg::ggarrange(figure_2a, figure_2b, heights = c(1, 1))

figure <- egg::ggarrange(figure_1a, figure_1b, figure_2a, figure_2b, ncol = 1, labels = c("A", "", "B", ""))

#ggsave(figure, file = "loss_and_gradient.png", width = 8, height = 8, path = "img", device = "png", dpi = 600)
ggsave(figure, file = "loss_and_gradient.pdf", width = 8, height = 8, path = "img", device = "pdf")

# ---- Visualise tanh loss ----

strip_font_size <- 13

# p = 1 parameter

kappa <- c(0, 0.1, 0.2, 0.5, 1)
displacement <- c(
  seq(-1.2, -0.05, by = 0.01),
  seq(-0.05, 0.05, by = 0.001),
  seq(0.05, 1.2, by = 0.01)
)
L_kappa <- function(displacement, kappa) tanh(abs(displacement)/kappa)

df <- expand.grid(kappa = kappa, displacement = displacement)
df$L_kappa <- L_kappa(df$displacement, df$kappa)

df$kappa <- factor(df$kappa)
levels(df$kappa) <- sapply(paste("$\\kappa$ = ", levels(df$kappa)), TeX)
levels(df$kappa)[1] <- TeX("$\\kappa \\to 0$")

point1 <- data.frame(x = 0, y = 1, kappa = "kappa %->% 0")
point2 <- data.frame(x = 0, y = 0, kappa = "kappa %->% 0")

figure_1a <- ggplot(df) +
  geom_line(aes(x = displacement, y = L_kappa)) +
  geom_point(data = point1, aes(x = x, y = y), shape = 21, fill = "white") + 
  geom_point(data = point2, aes(x = x, y = y), shape = 21, fill = "black") + 
  facet_wrap(~fct_rev(kappa), nrow = 1, labeller = label_parsed) +
  labs(
    x = expression(hat(theta) - theta),
    # y = expression(L[kappa](theta, hat(theta)))
    y = TeX(r'($L_{TANH}(\hat{\theta}, \theta; \, \kappa)$)')
  ) +
  scale_x_continuous(breaks = c(-1, 0, 1)) +
  theme_bw() + 
  theme(strip.background = element_blank(), strip.text.x = element_text(size = strip_font_size))

# ggsave(figure_1a, file = "loss_1d.pdf", width = 8, height = 3, path = "img", device = "pdf")

# p = 2 parameters

thetahat1 <- thetahat2 <- round(seq(min(displacement), max(displacement), by = 0.05), 10)
L_joint     <- function(thetahat1, thetahat2, kappa) tanh(norm(as.matrix(c(thetahat1, thetahat2)), type = "1")/kappa)
df <- expand.grid(thetahat1 = thetahat1, thetahat2 = thetahat2, kappa = kappa)
n <- nrow(df)
df$L <- apply(df, 1, function(x) L_joint(x[1], x[2], x[3]))

idx <- df$kappa == 0 & df$thetahat1 == 0 & df$thetahat2 == 0
df$L[idx] <- 0

df$kappa <- factor(df$kappa)
levels(df$kappa) <- sapply(paste("$\\kappa$ = ", levels(df$kappa)), TeX)
levels(df$kappa)[1] <- TeX("$\\kappa \\to 0$")

figure_1b <- ggplot(df) +
  geom_tile(aes(x = thetahat1, y = thetahat2, fill = L)) +
  scale_fill_distiller(palette = "Spectral") +
  facet_grid(~ fct_rev(kappa), labeller = label_parsed) +
  labs(
    x = expression(hat(theta)[1] - theta[1]),
    y = expression(hat(theta)[2] - theta[2]),
    #fill = expression(L[kappa](bold("\U03B8"), hat(bold("\U03B8")))) # This doesn't work when saving using PDF
    fill = bquote(paste(L[TANH], "(", bold("\U03B8"), ", ", hat(bold("\U03B8")), "; ",  kappa, ")")) # This doesn't work when saving using PDF
    # fill = TeX(r'($L_{TANH}(\hat{\boldsymbol{\theta}}, \boldsymbol{\theta}; \, \kappa)$)')
  ) +
  scale_x_continuous(expand = c(0, 0), breaks = c(-1, 0, 1)) +
  scale_y_continuous(expand = c(0, 0), breaks = c(-1, 0, 1)) +
  theme_bw() +
  theme(strip.background = element_blank(), 
        strip.text.x = element_text(size = strip_font_size), 
        text = element_text(size = strip_font_size)
  )

# NB bold \theta doesn't save properly when saving as pdf:
figure_1 <- egg::ggarrange(figure_1a, figure_1b)
# ggsave(figure_1, file = "loss_1d_2d.pdf", width = 10.2, height = 6, path = "img", device = "pdf")
# ggsave(figure_1, file = "loss_1d_2d.png", width = 10.2, height = 6, path = "img", device = "png", dpi=600)

# Save 2d plot only
figure_1b <- figure_1b + coord_fixed()
ggsave(figure_1b, file = "loss_2d.pdf", width = 10.2, height = 2.8, path = "img", device = "pdf")
# ggsave(figure_1b, file = "loss_2d.png", width = 10.2, height = 2.8, path = "img", device = "png", dpi=600)


