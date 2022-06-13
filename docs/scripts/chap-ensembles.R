library(ggplot2)

# Set the plotting theme
theme_set(theme_bw())

# Custom color scale
scale_oi <- function(n = 2, fill = FALSE, alpha = 1, ...) {
  okabe.ito <- palette.colors(n, palette = "Okabe-Ito", alpha = alpha)
  if (isFALSE(fill)) {
    scale_colour_manual(values = unname(okabe.ito), ...)
  } else {
    scale_fill_manual(values = unname(okabe.ito), ...)
  }
}

# Colorblind-friendly palette
cb.cols <- unname(palette.colors(8, palette = "Okabe-Ito"))

# Helper functions
err <- function(pred, obs) {
  1 - sum(diag(table(pred, obs))) / length(obs) 
}

library(rpart)

# Simulate sine wave data
gen_sine <- function(n = 500, sigma = 0.3) {
  x <- runif(n, min = 0, max = 2 * pi)
  data.frame(x, y = sin(x) + rnorm(n, sd = sigma))
}
set.seed(1503)  # for reproducibility
trn <- gen_sine()
tst <- gen_sine(10000)

# Helper function
MSE <- function(pred, obs) {
  mean((pred - obs) ^ 2)
}

# Generate finer grid of predictor values
xgrid <- data.frame(x = seq(from = 0, to = 2 * pi, length = 500))

# Single (overgrown) decision tree
tree <- rpart(y ~ x, data = trn, 
              control = rpart.control(minsplit = 2, cp = 0, xval = 0))
pred.tree <- predict(tree, newdata = xgrid)

# Bagged tree ensemble
set.seed(1013)  # for reproducibility
bag <- ipred::bagging(y ~ x, data = trn, nbagg = 1000)
pred.bag <- predict(bag, newdata = xgrid)

# Compute MSE on test set
mse.tree <- MSE(predict(tree, newdata = tst), obs = tst$y)
mse.bag <- MSE(predict(bag, newdata = tst), obs = tst$y)

# Plot results
p1 <- ggplot(trn, aes(x, y)) +
  geom_point(shape = 1, alpha = 0.3) +
  stat_function(fun = sin, colour = "black") +
  geom_line(data = data.frame(x = xgrid, y = pred.tree),
            color = cb.cols[2]) +
  ggtitle("Overgrown decision tree")
p2 <- ggplot(trn, aes(x, y)) +
  geom_point(shape = 1, alpha = 0.3) +
  stat_function(fun = sin, colour = "black") +
  geom_line(data = data.frame(x = xgrid, y = pred.bag),
            color = cb.cols[2]) +
  ggtitle("Bagged tree ensemble")
gridExtra::grid.arrange(p1, p2, nrow = 1)

knitr::include_graphics("diagrams/chap-ensembles-bagging.png", error = FALSE)

data(spam, package = "kernlab")
set.seed(852)  # for reproducibility
id <- sample.int(nrow(spam), size = floor(0.7 * nrow(spam)))
spam.trn <- spam[id, ]  # training data
spam.tst <- spam[-id, ]  # test data

library(rpart)

B <- 500  # number of trees in ensemble
ctrl <- rpart.control(minsplit = 2, cp = 0, xval = 0)
N <- nrow(spam.trn)  # number of training observations
spam.bag <- vector("list", length = B)  # to store trees
set.seed(900)  # for reproducibility
for (b in seq_len(B)) {  # fit trees to independent bootstrap samples
  boot.id <- sample.int(N, size = N, replace = TRUE)
  boot.df <- spam.trn[boot.id, ]  # bootstrap sample
  spam.bag[[b]] <- rpart(type ~ ., data = boot.df, control = ctrl)
}

vote <- function(x) names(which.max(table(x)))
err <- function(pred, obs) 1 - sum(diag(table(pred, obs))) / 
  length(obs) 

# Obtain (N x B) matrix of individual tree predictions
spam.bag.preds <- sapply(spam.bag, FUN = function(tree) {
  predict(tree, newdata = spam.tst, type = "class")
})

# Compute test error as a function of number of trees
spam.bag.err <- sapply(seq_len(B), FUN = function(b) {
  agg.pred <- apply(spam.bag.preds[, seq_len(b), drop = FALSE], 
                    MARGIN = 1, FUN = vote)
  err(agg.pred, obs = spam.tst$type)
})
# min(spam.bag.err)  # minimum misclassification error

set.seed(1023)  # for reproduicbility
spam.tree <- rpart(type ~ ., data = spam.trn, cp = 0)
spam.tree <- treemisc::prune_se(spam.tree, se = 1)
spam.tree.preds <- predict(spam.tree, newdata = spam.tst, type = "class")
spam.tree.err <- err(spam.tree.preds, obs = spam.tst$type)

## vote <- function(x) names(which.max(table(x)))
## err <- function(pred, obs) 1 - sum(diag(table(pred, obs))) /
##   length(obs)

## spam.bag.preds <- sapply(spam.bag, FUN = function(tree) {
##   predict(tree, newdata = spam.tst, type = "class")
## })  # N x B matrix of individual tree predictions
## 
## # Compute test error as a function of number of trees
## spam.bag.err <- sapply(seq_len(B), FUN = function(b) {
##   agg.pred <- apply(spam.bag.preds[, seq_len(b), drop = FALSE],
##                     MARGIN = 1, FUN = vote)  # aggregate trees 1:b
##   err(agg.pred, obs = spam.tst$type)  # compute test error
## })
## min(spam.bag.err)  # minimum misclassification error
## 
## #> [1] 0.0485

palette("Okabe-Ito")
plot(spam.bag.err, type = "l", xlab = "Number of trees", 
     ylab = "Test error", col = 1)
abline(h = min(spam.bag.err), lty = 2, col = 3)
palette("default")

B <- 500  # number of trees in ensemble
ctrl <- rpart.control(minsplit = 2, cp = 0, xval = 0)
N <- nrow(spam.trn)  # number of training observations
spam.bag.sub <- vector("list", length = B)
set.seed(900)  # for reproducibility
for (b in seq_len(B)) {
  boot.id <- sample.int(N, size = floor(N / 2), replace = FALSE)  # only change required
  boot.df <- spam.trn[boot.id, ]  # bootstrap sample
  spam.bag.sub[[b]] <- rpart(type ~ ., data = boot.df, control = ctrl)
}

# Helper functions
vote <- function(x) names(which.max(table(x)))
err <- function(pred, obs) 1 - sum(diag(table(pred, obs))) / length(obs) 

# Obtain (N x B) matrix of un-aggregated predictions
spam.bag.sub.preds <- sapply(spam.bag.sub, FUN = function(tree) {
  predict(tree, newdata = spam.tst, type = "class")
})

# Compute test error as a function of number of trees
spam.bag.sub.err <- sapply(seq_len(B), FUN = function(b) {
  agg.pred <- apply(spam.bag.sub.preds[, seq_len(b), drop = FALSE], 
                    MARGIN = 1, FUN = vote)
  err(agg.pred, obs = spam.tst$type)
})
# min(spam.bag.sub.err)  # minimum misclassification error

spam.trn$type <- ifelse(spam.trn$type == "spam", 1, -1)
spam.tst$type <- ifelse(spam.tst$type == "spam", 1, -1)
spam.xtrn <- subset(spam.trn, select = -type)  # feature columns only
spam.xtst <- subset(spam.tst, select = -type)  # feature columns only

library(rpart)

# Helper function to coerce factors to numeric
fac2num <- function(x) as.numeric(as.character(x))

# Apply AdaBoost.M1 algorithm
B <- 500  # number of trees in ensemble
ctrl <- rpart.control(maxdepth = 10, xval = 0)
N <- nrow(spam.trn)  # number of training observations
w <- rep(1 / N, times = N)  # initialize weights
spam.ada <- vector("list", length = B)  # to store sequence of trees
alpha <- numeric(B)  # to hold coefficients
for (i in seq_len(B)) {  # for b = 1, 2, ..., B
  spam.ada[[i]] <- rpart(type ~ ., data = spam.trn, weights = w,
                         control = ctrl, method = "class")
  # Compute predictions and coerce factor output to +1/-1
  pred <- fac2num(predict(spam.ada[[i]], type = "class"))
  err <- sum(w * (pred != spam.trn$type)) / sum(w)  # weighted error
  if (err == 0 | err == 1) {  # to avoid log(0) and dividing by 0
    err <- (1 - err) * 1e-06 + err * 0.999999
  }
  alpha[i] <- log((1 / err) - 1)  # coefficient from step 2) (c)
  w <- w * exp(alpha[i] * (pred != spam.trn$type))  # update weights
}

err <- function(pred, obs) 1 - sum(diag(table(pred, obs))) / 
  length(obs) 

spam.ada.preds <- sapply(seq_len(B), FUN = function(i) {
  class.labels <- predict(spam.ada[[i]], newdata = spam.tst, type = "class")
  alpha[i] * fac2num(class.labels)
})  # (N x B) matrix of un-aggregated predictions

# Compute test error as a function of number of trees
spam.ada.err <- sapply(seq_len(B), FUN = function(b) {
  agg.pred <- apply(spam.ada.preds[, seq_len(b), drop = FALSE], 
                    MARGIN = 1, FUN = function(x) sign(sum(x)))
  err(agg.pred, obs = spam.tst$type)
})

## spam.ada.preds <- sapply(seq_len(B), FUN = function(i) {
##   class.labels <- predict(spam.ada[[i]], newdata = spam.tst,
##                           type = "class")
##   alpha[i] * fac2num(class.labels)
## })  # (N x B) matrix of un-aggregated predictions
## 
## # Compute test error as a function of number of trees
## spam.ada.err <- sapply(seq_len(B), FUN = function(b) {
##   agg.pred <- apply(spam.ada.preds[, seq_len(b), drop = FALSE],
##                     MARGIN = 1, FUN = function(x) sign(sum(x)))
##   err(agg.pred, obs = spam.tst$type)
## })
## min(spam.ada.err)  # minimum misclassification error

min(spam.ada.err)  # minimum misclassification error

spam.tree.10 <- rpart(type ~ ., data = spam.trn, 
                      maxdepth = 10, method = "class")
pred <- predict(spam.tree.10, newdata = spam.tst, type = "class")
pred <- as.numeric(as.character(pred))  # coerce to numeric
mean(pred != spam.tst$type)

# Plot train and test errors as a function of the number of trees
palette("Okabe-Ito")  # colorblind friendly palette
plot(spam.ada.err, col = 1, type = "l", las = 1, 
     xlab = "Number of trees", ylab = "Misclassification error")
lines(spam.bag.err, col = 2)
lines(spam.bag.sub.err, col = 3)
abline(h = min(spam.ada.err), col = 1, lty = 2)
abline(h = min(spam.bag.err), col = 2, lty = 2)
abline(h = min(spam.bag.sub.err), col = 3, lty = 2)
legend("topright", legend = c("AdaBoost.M1", "Bagging", "Bagging (N/2)"), 
       lty = 1, col = c(1, 2, 3), inset = 0.01, bty = "n")
palette("default")

ames <- as.data.frame(AmesHousing::make_ames())
ames$Sale_Price <- ames$Sale_Price / 1000  # rescale response
set.seed(2101)  # for reproducibility
id <- sample.int(nrow(ames), size = floor(0.7 * nrow(ames)))
ames.trn <- ames[id, ]  # training data/learning sample
ames.tst <- ames[-id, ]  # test data
ames.xtst <- subset(ames.tst, select = -Sale_Price)  # features only

library(randomForest)

# Fit a typical bagged tree ensemble 
system.time({
  set.seed(942)  # for reproducibility
  ames.bag <- 
    randomForest(Sale_Price ~ ., data = ames.trn, mtry = 80, 
                 ntree = 500, xtest = ames.xtst, 
                 ytest = ames.tst$Sale_Price, keep.forest = TRUE)
})

# Print results
print(ames.bag)

# Fit a bagged tree ensemble using six-node trees on 5% samples
system.time({
  set.seed(1021)
  ames.bag.6.5 <- 
    randomForest(Sale_Price ~ ., data = ames.trn, mtry = 80, 
                 ntree = 500, maxnodes = 6, 
                 sampsize = floor(0.05 * nrow(ames.trn)),
                 replace = FALSE, keep.forest = TRUE,
                 xtest = ames.xtst, ytest = ames.tst$Sale_Price)
})

# Print results
print(ames.bag.6.5)

# Test set MSE as a function of the number of trees
mse.bag <- ames.bag$test$mse
mse.bag.6.5 <- ames.bag.6.5$test$mse

## # saveRDS("../data/chap-ensembles-ames-bag.rds")

preds.trn <- predict(ames.bag, newdata = ames.trn, 
                     predict.all = TRUE)$individual
preds.tst <- predict(ames.bag, newdata = ames.tst, 
                     predict.all = TRUE)$individual

library(glmnet)

# Fit the LASSO regularization path
lasso.ames.bag <- glmnet(
  x = preds.trn,  # individual tree predictions are the predictors
  y = ames.trn$Sale_Price,  # same response variable
  lower.limits = 0,  # coefficients should be strictly positive
  standardize = FALSE,  # no need to standardize
  family = "gaussian"  # least squares regression
)

plot(lasso.ames.bag, xvar = "lambda",
     col = adjustcolor("forestgreen", alpha.f = 0.3), las = 1)

# Assess performance of fit using an independent test set
perf <- assess.glmnet(
  object = lasso.ames.bag,  # fitted LASSO model
  newx = preds.tst, # test predictions from individual trees
  newy = ames.tst$Sale_Price,  #same response variable (test set) 
  family = "gaussian"  # for MSE and MAE metrics
)
perf <- do.call(cbind, args = perf)  # bind results into matrix

# List of results
ames.bag.post <- as.data.frame(cbind(
  "ntree" = lasso.ames.bag$df, perf, 
  "lambda" = lasso.ames.bag$lambda)
)

# Sort in ascending order of number of trees
head(ames.bag.post <- ames.bag.post[order(ames.bag.post$ntree), ])

# Print results corresponding to smallest test MSE
ames.bag.post[which.min(ames.bag.post$mse), ]

library(treemisc)

# Post-process ames.bag.6.5 ensemble
preds.trn.6.5 <- predict(ames.bag.6.5, newdata = ames.trn, 
                         predict.all = TRUE)$individual
preds.tst.6.5 <- predict(ames.bag.6.5, newdata = ames.tst, 
                         predict.all = TRUE)$individual
ames.bag.6.5.post <- 
  isle_post(preds.trn.6.5, y = ames.trn$Sale_Price, 
            family = "gaussian", newX = preds.tst.6.5, 
            newy = ames.tst$Sale_Price)

## par(mar = c(4, 4, 3, 0.1))
## plot(ames.bag.6.5.post$lasso.fit, xvar = "lambda",
##      col = adjustcolor("purple", alpha.f = 0.3), las = 1)
## abline(v = log(z$lambda), lty = 2)

palette("Okabe-Ito")
plot(mse.bag, type = "l", las = 1, xlab = "Number of trees", 
     ylim = c(range(mse.bag, mse.bag.6.5)), ylab = "Test MSE")
lines(mse.bag.6.5, col = 2)
lines(ames.bag.post, lty = 2)
lines(ames.bag.6.5.post$results, col = 2, lty = 2)
legend("topright", legend = c("ames.bag", "ames.bag (post)", 
                              "ames.bag.6.5", "ames.bag.6.5 (post)"),
       col = c(1, 1, 2, 2), lty = c(1, 2, 1, 2), bty = "n")
palette("default")
