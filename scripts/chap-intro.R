library(rpart)

# Load mushroom edibility data frame
mushroom <- treemisc::mushroom

# Simulate some quadratic linear model data
N <- 100
set.seed(1117)  # for reproducibility
x <- runif(N, min = 0, max = 1)
y <- rnorm(N, mean = 1 + 0.5 * x ^ 2, sd = 0.1)

# Fit three models: too simple, too complex, just right
fit1 <- lm(y ~ x)
fit2 <- lm(y ~ poly(x, degree = 2))
fit3 <- lm(y ~ poly(x, degree = 20))

# Plot results
palette("Okabe-Ito")
par(
  mfrow = c(1, 3),
  mar = c(4, 4, 1.5, 0.1), 
  cex.lab = 0.95, 
  cex.axis = 0.8,  # was 0.9
  mgp = c(2, 0.7, 0), 
  tcl = -0.3, 
  las = 1
)
investr::plotFit(fit1, data = data.frame(x, y), col.fit = 2, lty.fit = 1, 
                 lwd.fit = 2, hide = FALSE, main = "Underfitting")
investr::plotFit(fit2, data = data.frame(x, y), col.fit = 3, lty.fit = 1, 
                 lwd.fit = 2, hide = FALSE, main = "Just right?")
investr::plotFit(fit3, data = data.frame(x, y), col.fit = 4, lty.fit = 1, 
                 lwd.fit = 2, hide = FALSE, main = "Overfitting")
palette("default")

## N <- 100
## set.seed(1117)  # for reproducibility
## x <- runif(N, min = 0, max = 1)
## y <- rnorm(N, mean = 1 + 0.5 * x ^ 2, sd = 0.1)
## newx <- seq(from = 0, to = 1, length = 999)
## newy <- rnorm(length(xx), mean = 1 + 0.5 * xx ^ 2, sd = 0.1)
## fits <- lapply(1:50, FUN = function(k) {
##   FNN::knn.reg(data.frame(x = x), y = y, test = data.frame(x = xx),
##                k = k)
## })
## 
## par(
##   mfrow = c(1, 3),
##   mar = c(4, 4, 1.5, 0.1),
##   cex.lab = 0.95,
##   cex.axis = 0.8,  # was 0.9
##   mgp = c(2, 0.7, 0),
##   tcl = -0.3,
##   las = 1
## )
## plot(x, y)
## lines(newx, fits[[2]]$pred, col = 2, lwd = 2)
## plot(x, y)
## lines(newx, fits[[11]]$pred, col = 2, lwd = 2)
## plot(x, y)
## lines(newx, fits[[50]]$pred, col = 2, lwd = 2)
## 
## rmses <- sapply(fits, FUN = function(fit) {
##   sqrt(mean((newy - fit$pred) ^ 2))
## })
## plot(rmses, xlab = expression(k), ylab = "Test RMSE", type = "b", pch = 19)

library(mlbench)
library(treemisc)

# Generate data from the twonorm benchmark problem
set.seed(1050)
trn <- as.data.frame(mlbench.twonorm(500, d = 2))

palette("Okabe-Ito")
plot(x.2 ~ x.1, data = trn, col = as.integer(trn$classes) + 1,
     xlab = expression(x[1]), ylab = expression(x[2]))
palette("default")

# Construct lattice of test points
x <- seq(from = -4.5, to = 4.5, length = 999)
tst <- expand.grid("x.1" = x, "x.2" = x)

# Fit a one-nearest neighbor (knn1) model; overfitting
knn1 <- caret::knn3(classes ~ ., data = trn, k = 1, prob = TRUE)

# Add decision boundary to previous plot
pfun <- function(object, newdata) {
  predict(object, newdata = newdata)[, 1]
}
decision_boundary(knn1, train = trn, y = "classes", x1 = "x.1", x2 = "x.2", 
                  pfun = pfun)

# Add Bayesian decision boundary
pnormal <- function(data, neg = TRUE) {
  mu <- 2 / sqrt(2)
  if (isTRUE(neg)) mu <- -mu
  dnorm(data$x.1, mean = mu, sd = 1) * dnorm(data$x.2, mean = mu, sd = 1)
}
z <- matrix(ifelse(pnormal(tst) > pnormal(tst, neg = FALSE), 0, 1), nrow = 999,
            byrow = TRUE)
contour(sort(unique(tst$x.1)), sort(unique(tst$x.2)), z, add = TRUE, 
        drawlabels = FALSE, levels = 0.5, lty = 2)
legend("topright", legend = c("1-NN", "Bayes"), lty = 1:2, inset = 0.01,
       bty = "n")

library(rpart)

# Load the titanic passenger data and remove unwanted columns
t3 <- read.csv("https://hbiostat.org/data/repo/titanic3.csv",
               stringsAsFactors = TRUE)
keep <- c("survived", "pclass", "age", "sex", "sibsp", "parch")
t3 <- t3[, keep]  # only retain key variables

# Fit a simple classification tree
ctrl <- rpart.control(minsplit = 2, cp = 0)
# ctrl <- rpart.control(maxdepth = 2)
set.seed(1446)  # for reproducibility
t3.cart <- rpart(survived ~ ., data = t3, control = ctrl)
t3.cart.1se <- treemisc::prune_se(t3.cart, se = 1)

# Display the corresponding tree diagram
treemisc::tree_diagram(t3.cart.1se)

## par(
##   mar = c(4, 4, 0.1, 0.1),  # may be different for a handful of figures
##   cex.lab = 0.95,
##   cex.axis = 0.8,
##   mgp = c(2, 0.7, 0),
##   tcl = -0.3,
##   las = 1
## )

bn <- treemisc::banknote
cols <- palette.colors(3, palette = "Okabe-Ito")
pairs(bn[, 1L:6L], col = adjustcolor(cols[bn$y + 2], alpha.f = 0.5),
      pch = c(1, 2)[bn$y + 1], cex = 0.7)

aq <- datasets::airquality
color <- adjustcolor("forestgreen", alpha.f = 0.5)
ps <- function(x, y, ...) {  # custom panel function
  panel.smooth(x, y, col = color, col.smooth = "black", 
               cex = 0.7, lwd = 2)
}
pairs(aq, cex = 0.7, upper.panel = ps, col = color)

set.seed(943)  # for reproducibility
treemisc::gen_friedman1(5, nx = 7, sigma = 0.1)

mushroom <- treemisc::mushroom
mosaicplot(~ Edibility + odor, data = mushroom, color = TRUE,
           las = 1, main = "", cex.axis = 0.6)

data(spam, package = "kernlab")

# Distribution of ham and spam
table(spam$type)

# Compute average relative frequency of different words and characters
aggregate(cbind(remove, charDollar, hp, parts, direct) ~ type,
          data = spam, FUN = mean)

library(rpart)
library(treemisc)

# Split into train/test sets using a 70/30 split
set.seed(852)  # for reproducibility
id <- sample.int(nrow(spam), size = floor(0.7 * nrow(spam)))
spam.trn <- spam[id, ]  # training data
spam.tst <- spam[-id, ]  # test data

# Fit a simple classification tree
loss <- matrix(c(0, 1, 5, 0), nrow = 2)  # misclassification costs
spam.cart <- rpart(type ~ ., data = spam.trn, cp = 0,
                   parms = list("loss" = loss))
cp <- spam.cart$cptable
cp <- cp[cp[, "nsplit"] == 3, "CP"]  # CP associated with 3 splits
spam.cart.pruned <- prune(spam.cart, cp = cp)  # grab smaller subtree

# Display tree diagram
tree_diagram(spam.cart.pruned)

data(attrition, package = "modeldata")

# Distribution of class outcomes
table(attrition$Attrition)

ames <- as.data.frame(AmesHousing::make_ames())
ames$Sale_Price <- ames$Sale_Price / 1000  # rescale response
set.seed(2101)  # for reproducibility
trn.id <- sample.int(nrow(ames), size = floor(0.7 * nrow(ames)))
ames.trn <- ames[trn.id, ]  # training data/learning sample
ames.tst <- ames[-trn.id, ]  # test data

plot(Sale_Price ~ Gr_Liv_Area, data = ames.trn,
     col = adjustcolor(1, alpha.f = 0.5),
     xlab = "Above grade square footage",
     ylab = "Sale price / 1000")

wine <- treemisc::wine
xtabs(~ type + quality, data = wine)

library(survival)

pbc2 <- pbc[!is.na(pbc$trt), ]  # omit non-randomized subjects
pbc2$id <- NULL  # remove ID column
# Consider transplant patients to be censored at day of transplant 
pbc2$status <- ifelse(pbc2$status == 2, 1, 0)  

# Look at frequency of death and censored observations
table(pbc2$status)

samp <- head(pbc2[, 1:2], n = 10)
palette("Okabe-Ito")
plot(samp$time, seq_along(samp$time), pch = as.character(samp$status), 
     xlim = c(0, max(samp$time)), xlab = "Time (days)", ylab = "Subject",
     yaxt = "n", col = samp$status + 2, ylim = c(0, 11))
axis(2, at = seq_along(samp$time), las = 1)
for (i in seq_along(samp$time)) {
  lines(c(0, samp$time[i]), c(i, i), lty = 2, 
        col = adjustcolor(1, alpha.f = 0.3))
}
legend("topright", inset = 0.01, bty = "n", text.col = 2:3,
       legend = c("0: censored", "1: death"))
palette("default")

palette("Okabe-Ito")
plot(survfit(Surv(time, status) ~ trt, data = pbc2), col = 2:3,
     conf.int = FALSE, las = 1, xlab = "Days until death", 
     ylab = "Estimated survival probability")
legend("bottomleft", legend = c("Penicillmain", "Placebo"), 
       lty = 1, col = 2:3, text.col = 2:3, inset = 0.01, bty = "n")
palette("default")
