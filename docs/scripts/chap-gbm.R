library(data.table)
library(dplyr)
library(gbm)
library(ggplot2)
library(glmnet)

# Set the plotting theme for ggplot2-based figures
theme_set(theme_bw())

# Okabe-Ito color scale for ggplot2-based figures
scale_oi <- function(n = 2, fill = FALSE, alpha = 1, ...) {
  okabe.ito <- palette.colors(n, palette = "Okabe-Ito", alpha = alpha)
  if (isFALSE(fill)) {
    scale_colour_manual(values = unname(okabe.ito), ...)
  } else {
    scale_fill_manual(values = unname(okabe.ito), ...)
  }
}

# Colorblind-friendly palette
oi.cols <- unname(palette.colors(8, palette = "Okabe-Ito"))

knitr::knit_engines$set(python = reticulate::eng_python)
reticulate::use_condaenv("r-reticulate-trees", required = TRUE)

# Read in ALS data
url <- "https://web.stanford.edu/~hastie/CASI_files/DATA/ALS.txt"
als <- read.table(url, header = TRUE)

# Split into train/test sets
trn <- als[!als$testset, -1]
tst <- als[als$testset, -1]
X.trn <- subset(trn, select = -dFRS)
X.tst <- subset(tst, select = -dFRS)
y.trn <- trn$dFRS
y.tst <- tst$dFRS

library(data.table)
library(gbm)

# Read in the Ames housing data and split into train/test sets
ames <- as.data.frame(AmesHousing::make_ames())
ames$Sale_Price <- ames$Sale_Price / 1000  # rescale response
set.seed(4919)  # for reproducibility
id <- sample.int(nrow(ames), size = floor(0.7 * nrow(ames)))
ames.trn <- ames[id, ]
ames.tst <- ames[-id, ]

# Fit a GBM using 500 depth-two trees with no shrinkage
set.seed(2153)  # for reproducibility
invisible(capture.output(  # to hide stupid cat() output regarding CV results
  ames.gbm <- gbm(Sale_Price ~ ., data = ames.trn, distribution = "gaussian", 
                  interaction.depth = 2, shrinkage = 1, n.trees = 500, 
                  cv.folds = 5, n.cores = 1)
))

# Grab train and 5-fold CV errors
err <- data.table(
  "iter" = seq_along(ames.gbm$train.error),
  "Train" = ames.gbm$train.error,
  "5-fold CV" = ames.gbm$cv.error
)
err <- melt(err, id.vars = "iter")

# Plot results
ggplot(err, aes(iter, value, color = variable)) +
  geom_line() +
  geom_vline(xintercept = gbm.perf(ames.gbm, method = "cv", plot.it = FALSE), 
             linetype = "dashed", color = oi.cols[3]) +
  xlab("Number of trees") +
  ylab ("LS loss") +
  scale_colour_manual(values = oi.cols) +
  theme(legend.position = c(0.85, 0.5), legend.title = element_blank())

# Read in previously saved results
mses.shrinkage <- readRDS("../data/chap-gbm-als-mses-shrinkage.rds")

# Plot results
ggplot(mses.shrinkage, aes(ntree, value, color = variable)) +
  geom_line() +
  xlab("Number of trees") +
  ylab("Mean-squared error") +
  facet_wrap(~ model) +
  scale_colour_manual(values = oi.cols) + 
  #theme(legend.position = "top", legend.title = element_blank()) 
  theme(legend.position = c(0.15, 0.85), legend.title = element_blank()) 

# Setup DMatrix objects and watchlist
library(data.table)
library(xgboost)

# Set up DMatrix objects and watchlist
dtrn <- xgb.DMatrix(data = data.matrix(X.trn), label = y.trn)
dtst <- xgb.DMatrix(data = data.matrix(X.tst), label = y.tst)
wl <- list(train = dtrn, test = dtst)

# Run XGBoost under different sets of parameters
set.seed(2333)  # for reproducibility
xgb1 <- xgb.train(data = dtrn, nrounds = 250, watchlist = wl, verbose = 0,
                  params = list(eta = 0.3, max_depth = 3, subsample = 0.5,
                                colsample_bytree = 1))
xgb2 <- xgb.train(data = dtrn, nrounds = 250, watchlist = wl, verbose = 0,
                  params = list(eta = 0.3, max_depth = 3, subsample = 1.0,
                                colsample_bytree = 0.5))

# Grab test set MSEs as a function of number of trees
mses <- data.frame(
  "Subsample rows" = xgb1$evaluation_log$test_rmse ^ 2,
  "Subsample columns" = xgb2$evaluation_log$test_rmse ^ 2,
  "ntree" = xgb1$evaluation_log$iter,
  check.names = FALSE
)
mses <- melt(as.data.table(mses), id.vars = "ntree")

# Plot results
ggplot(mses, aes(ntree, value, color = variable)) +
  geom_line() +
  xlab("Number of trees") +
  ylab("Mean-squared error") +
  scale_colour_manual(values = oi.cols) + 
  #theme(legend.position = "top", legend.title = element_blank()) 
  theme(legend.position = c(0.85, 0.85), legend.title = element_blank()) 

## lsboost <- function(X, y, ntree = 100, shrinkage = 0.1, depth = 6,
##                     subsample = 0.5, init = mean(y)) {
##   yhat <- rep(init, times = nrow(X))  # initialize fit; f_0(x)
##   trees <- vector("list", length = ntree)  # to store each tree
##   ctrl <-  # control tree-specific parameters
##     rpart::rpart.control(cp = 0, maxdepth = depth, minbucket = 10)
##   for (tree in seq_len(ntree)) {  # Step 2) of Algorithm 8.1
##     id <- sample.int(nrow(X), size = floor(subsample * nrow(X)))
##     samp <- X[id, ]  # random subsample
##     samp$pr <- y[id] - yhat[id]  # pseudo residual
##     trees[[tree]] <-  # fit tree to current pseudo residual
##       rpart::rpart(pr ~ ., data = samp, control = ctrl)
##     yhat <- yhat + shrinkage * predict(trees[[tree]], newdata = X)
##   }
##   res <- list("trees" = trees, "shrinkage" = shrinkage,
##               "depth" = depth, "subsample" = subsample, "init" = init)
##   class(res) <- "lsboost"
##   res
## }

## # Extend R's generic predict() function to work with "lsboost" objects
## predict.lsboost <- function(object, newdata, ntree = NULL,
##                             individual = FALSE, ...) {
##   if (is.null(ntree)) {
##     ntree <- length(object[["trees"]])  # use all trees
##   }
##   shrinkage <- object[["shrinkage"]]  # extract learning rate
##   trees <- object[["trees"]][seq_len(ntree)]
##   pmat <- sapply(trees, FUN = function(tree) {  # all predictions
##     shrinkage * predict(tree, newdata = newdata)
##   })  # compute matrix of (shrunken) predictions; one for each tree
##   if (isTRUE(individual)) {
##     pmat  # return matrix of (shrunken) predictions
##   } else {
##     rowSums(pmat) + object$init  # return boosted predictions
##   }
## }

## ladboost <- function(X, y, ntree = 100, shrinkage = 0.1, depth = 6,
##                      subsample = 0.5, init = median(y)) {
##   yhat <- rep(init, times = nrow(X))  # initialize fit
##   trees <- vector("list", length = ntree)  # to store each tree
##   ctrl <-  # control tree-specific parameters
##     rpart::rpart.control(cp = 0, maxdepth = depth, minbucket = 10)
##   for (tree in seq_len(ntree)) {
##     id <- sample.int(nrow(X), size = floor(subsample * nrow(X)))
##     samp <- X[id, ]
##     samp$pr <- sign(y[id] - yhat[id])  # use signed residual
##     trees[[tree]] <-
##       rpart::rpart(pr ~ ., data = samp, control = ctrl)
##     #------------------------------------------------------------------
##     # Line search; update terminal node estimates using median
##     #------------------------------------------------------------------
##     where <- trees[[tree]]$where  # terminal node assignments
##     map <- tapply(samp$pr, INDEX = where, FUN = median)
##     trees[[tree]]$frame$yval[where] <- map[as.character(where)]
##     #
##     # Could use partykit instead:
##     #
##     # trees[[tree]] <- partykit::as.party(trees[[tree]])
##     # med <- function(y, w) median(y)  # see ?partykit::predict.party
##     # yhat <- yhat +
##     #   shrinkage * partykit::predict.party(trees[[tree]],
##     #                                       newdata = X, FUN = med)
##     #------------------------------------------------------------------
##     yhat <- yhat + shrinkage * predict(trees[[tree]], newdata = X)
##   }
##   res <- list("trees" = trees, "shrinkage" = shrinkage,
##               "depth" = depth, "subsample" = subsample, "init" = init)
##   class(res) <- "ladboost"
##   res
## }

## library(treemisc)
## 
## # Split Ames data into train/test sets using a 70/30 split
## ames <- as.data.frame(AmesHousing::make_ames())
## ames$Sale_Price <- ames$Sale_Price / 1000  # rescale response
## set.seed(4919)  # for reproducibility
## id <- sample.int(nrow(ames), size = floor(0.7 * nrow(ames)))
## ames.trn <- ames[id, ]
## ames.tst <- ames[-id, ]
## 
## # Fit a gradient tree boosted ensemble with 500 trees
## set.seed(1110)  # for reproducibility
## ames.bst <-
##   lsboost(subset(ames.trn, select = -Sale_Price),  # features only
##           y = ames.trn$Sale_Price, ntree = 500, depth = 4,
##           shrinkage = 0.1)

## set.seed(1128)  # for reproducibility
## ames.rfo <-  # fit a default RF for comparison
##   randomForest(subset(ames.trn, select = -Sale_Price),
##                y = ames.trn$Sale_Price, ntree = 500,
##                # Monitor test set performance (MSE, in this case)
##                xtest = subset(ames.tst, select = -Sale_Price),
##                ytest = ames.tst$Sale_Price)
## 
## # Helper function for computing RMSE
## rmse <- function(pred, obs, na.rm = FALSE) {
##   sqrt(mean((pred - obs)^2, na.rm = na.rm))
## }
## 
## # Compute RMSEs from both models on the test set as a function of the
## # number of trees in each ensemble (i.e., B = 1, 2, ..., 500)
## rmses <- matrix(nrow = 500, ncol = 2)  # to store results
## colnames(rmses) <- c("GBM", "RF")
## rmses[, "GBM"] <- sapply(seq_along(ames.bst$trees), FUN = function(B) {
##   pred <- predict(ames.bst, newdata = ames.tst, ntree = B)
##   rmse(pred, obs = ames.tst$Sale_Price)
## })  # add GBM results
## rmses[, "RF"] <- sqrt(ames.rfo$test$mse)  # add RF results

library(data.table)

# Read in previously saved results
rmses <- readRDS("../data/chap-gbm-lsboost-ames-rmses.rds")
colnames(rmses) <- c("ntree", "Gradient boosted trees", "Random forest")

# Convert to long format for plotting
rmses.long <- melt(as.data.table(rmses), id.vars = "ntree")

# Plot results
ggplot(rmses.long, aes(ntree, sqrt(value), color = variable)) +
  geom_line() +
  xlab("Number of trees") +
  ylab("Test RMSE") +
  scale_oi() +
  theme_bw() +
  #theme(legend.position = "top", legend.title = element_blank()) 
  theme(legend.position = c(0.8, 0.85), legend.title = element_blank()) 

library(gbm)
library(lattice)
library(pdp)

# Split spam data into train/test sets using a 70/30 split
data(spam, package = "kernlab")
spam$type <- ifelse(spam$type == "spam", 1, 0)
set.seed(852)  # for reproducibility
id <- sample.int(nrow(spam), size = floor(0.7 * nrow(spam)))
spam.trn <- spam[id, ]  # training data
spam.tst <- spam[-id, ]  # test data

# Fit a gradient tree boosted ensemble using 5-fold cross-validation
set.seed(1611)  # for reproducibility
spam.gbm <- gbm(type ~ ., data = spam.trn, distribution = "bernoulli",
                n.trees = 5000, interaction.depth = 2, shrinkage = 0.01, 
                cv.folds = 5)
best.iter <- gbm.perf(spam.gbm, method = "cv", plot.it = FALSE)

# Partial dependence via fast recursion method
pd <- partial(spam.gbm, pred.var = c("address", "receive"), chull = TRUE, 
              n.trees = best.iter, quantiles = TRUE, probs = 1:100/101)

# Plot results as a surface
lattice::wireframe(
  yhat ~ address * receive, data = pd, scales = list(arrows = FALSE), 
  par.settings = list(axis.line = list(col = 'transparent')), drape = TRUE, 
  colorkey = FALSE, #screen = list(z = -120, x = 75), 
  aspect = c(1, 0.5), panel.aspect = 0.618,
  distance = 0.05, zoom = 0.9, zlab = "", 
  col.regions = hcl.colors(100, palette = "Blue-Red 3")
)

res <- readRDS("../data/chap-gbm-monotonic-bank.rds")
ggplot(res$pd, aes(euribor3m, yhat)) +
  geom_line() +
  geom_rug(data = res$rug, aes(euribor3m), sides = "b", inherit.aes = FALSE) +
  xlab("Euribor 3 month rate") +
  ylab("Partial dependence") +
  facet_wrap( ~ group) +
  theme_bw()

library(survival)

# Prep the data a bit
pbc2 <- pbc[!is.na(pbc$trt), ]  # use randomized subjects
pbc2$id <- NULL  # remove ID column
# Consider transplant patients to be censored at day of transplant 
pbc2$status <- ifelse(pbc2$status == 2, 1, 0)  
facs <- c("sex", "spiders", "hepato", "ascites", "trt", "edema")
for (fac in facs) {  # coerce to factor
  pbc2[[fac]] <- as.factor(pbc2[[fac]])
}

library(gbm)

set.seed(1551)  # for reproducibility
pbc2.gbm <- gbm(Surv(time, status) ~ ., data = pbc2, 
                distribution = "coxph", n.trees = 3000, 
                interaction.depth = 3, shrinkage = 0.001,
                cv.folds = 5)
(best.iter <- gbm.perf(pbc2.gbm, method = "cv", plot.it = FALSE))

vi <- summary(pbc2.gbm, n.trees = best.iter, plotit = FALSE)
dotchart(vi$rel.inf, labels = vi$var, xlab = "Variable importance")

library(ggplot2)
library(pdp)

# Create list of c-ICE/PD plots for top 4 predictors
top4 <- c("bili", "copper", "age", "albumin")
pdps.top4 <- lapply(top4, FUN = function(x) {
  partial(pbc2.gbm, pred.var = x, check.class = FALSE, 
          recursive = FALSE,  n.trees = best.iter, ice = TRUE, 
          center = TRUE, plot = TRUE, plot.engine = "ggplot2", 
          rug = TRUE, alpha = 0.1) + 
    ylab("Log hazard")  # change default y-axis label
})

# Display list of plots in a grid
gridExtra::grid.arrange(grobs = pdps.top4, nrow = 2)

gbm.2way <- function(object, data, var.names = object$var.names, 
                     n.trees = object$n.trees) {
  var.pairs <- combn(var.names, m = 2, simplify = TRUE)
  h <- combn(var.names, m = 2, simplify = TRUE, FUN = function(x) {
    interact.gbm(object, data = data, i.var = x, n.trees = n.trees)
  })
  res <- as.data.frame(t(var.pairs))
  res$h <- h
  names(res) <- c("var1", "var2", "h")
  res[order(h, decreasing = TRUE), ]
}

# Compute H-statistics for all pairs of predictors
pbc2.h <- gbm.2way(pbc2.gbm, data = pbc2, n.trees = best.iter)
head(pbc2.h, n = 5)  # look at top 5

pd <- partial(pbc2.gbm, pred.var = c("bili", "age"), chull = TRUE, 
              check.class = FALSE, n.trees = best.iter)
autoplot(pd, legend.title = "PD") + 
  xlab("Serum bilirunbin (mg/dl)") +
  ylab("Age (years)")

library(fastshap)

p <- predict(pbc2.gbm, newdata = pbc2, n.trees = best.iter)
max.id <- which.max(p)  # row ID highest predicted log hazard

# Define prediction wrapper for explain
pfun <- function(object, newdata) {
  predict(object, newdata = newdata, n.trees = best.iter)
}

# Estimate feature contributions for newx using 1,000 Monte Carlo reps
X <- pbc2[, pbc2.gbm$var.names]  # feature columns only
newx <- pbc2[max.id, pbc2.gbm$var.names]
set.seed(1408)  # for reproducibility
(ex <- explain(pbc2.gbm, X = X, nsim = 1000, pred_wrapper = pfun,
               newdata = newx))

library(waterfall)

# Reshape Shapley values for plotting and include feature values
res <- data.frame("feature" = paste0(names(newx), "=", t(newx)),
                  "shapley.value" = t(ex))

# Waterfall chart of feature contributions
palette("Okabe-Ito")
waterfallchart(feature ~ shapley.value, data = res, origin = mean(p),
               summaryname = "f(x) - baseline", col = 2:3,
               xlab = "Log hazard")
mosaic::ladd(panel.abline(v = max(p), lty = 2, col = 1))
mosaic::ladd(panel.abline(v = mean(p), lty = 2, col = 1))
mosaic::ladd(panel.text(2.5, 8, labels = "f(x)", col = 1))
mosaic::ladd(panel.text(-0.55, 8, labels = "baseline", col = 1))
palette("default")

## import numpy as np

## import pandas as pd

## import scipy.stats

## from ngboost import NGBRegressor

## from ngboost.distns import Normal

## 
## 
## # Read in ALS data and split into train/test sets

## url = "https://web.stanford.edu/~hastie/CASI_files/DATA/ALS.txt"

## als = pd.read_csv(url, sep =" ")

## als_trn = als[als["testset"] == False]

## als_tst = als[als["testset"] == True]

## X_trn = als_trn.drop(["testset", "dFRS"], axis=1)  # features only

## X_tst = als_tst.drop(["testset", "dFRS"], axis=1)  # features only


## ngb = NGBRegressor(Dist=Normal, n_estimators=2000, learning_rate=0.01,

##                    verbose_eval=0, random_state=1601)


## _ = ngb.fit(X_trn, Y=als_trn["dFRS"], X_val=X_tst,

##             Y_val=als_tst["dFRS"], early_stopping_rounds=5)


## pred = ngb.predict(X_tst)

## np.mean(np.square(als_tst["dFRS"].values - pred))


## dist = ngb.pred_dist(X_tst.head(1)).params

## dist

## scipy.stats.norm(dist["loc"][0], scale=dist["scale"][0]).cdf(0)


# Read in the ALS data
url <- "https://web.stanford.edu/~hastie/CASI_files/DATA/ALS.txt"
als <- read.table(url, header = TRUE)

# Split into train/test sets
trn <- als[!als$testset, -1]  # training data w/o testset column
tst <- als[als$testset, -1]  # test data w/o testset column
X.trn <- subset(trn, select = -dFRS)
X.tst <- subset(tst, select = -dFRS)
y.trn <- trn$dFRS
y.tst <- tst$dFRS

library(treemisc)

set.seed(1122)  # for reproducibility
lsb.fit <- lsboost(X.trn, y = y.trn, shrinkage = 0.01, ntree = 1000, 
                   depth = 2, subsample = 0.5)

# Mean squared error function
mse <- function(y, yhat, na.rm = FALSE) {
  mean((y - yhat) ^ 2, na.rm = na.rm)
}

# Compute test MSE as a function of the number of trees
preds.tst <- predict(lsb.fit, newdata = X.tst, individual = TRUE)
mse.boost <- sapply(seq_len(ncol(preds.tst)), FUN = function(ntree) {
  # Only aggregate predictions from first B/ntree trees
  pred.ntree <- rowSums(preds.tst[, seq_len(ntree), drop = FALSE]) + 
    lsb.fit$init  # don't forget to add on the initial fit/mean response
  mse(y.tst, yhat = pred.ntree)
})

library(treemisc)

# Fit a LASSO model to the individual training predictions
preds.trn <- predict(lsb.fit, newdata = X.trn, individual = TRUE)
als.boost.post <- isle_post(preds.trn, y = y.trn, offset = lsb.fit$init,
                            newX = preds.tst, newy = y.tst, 
                            family = "gaussian")

# Plot the coefficient paths from the LASSO model
plot(als.boost.post$lasso.fit, xvar = "lambda", las = 1, label = TRUE,
     col = adjustcolor("forestgreen", alpha.f = 0.3), 
     cex.axis = 0.8, cex.lab = 0.8)

## library(treemisc)
## 
## # Fit a LASSO model to the individual training predictions
## preds.trn <- predict(lsb.fit, newdata = X.trn, individual = TRUE)
## als.boost.post <- isle_post(preds.trn, y = y.trn, offset = lsb.fit$init,
##                             newX = preds.tst, newy = y.tst,
##                             family = "gaussian")
## 
## # Plot the coefficient paths from the LASSO model
## plot(als.boost.post$lasso.fit, xvar = "lambda", las = 1, label = TRUE,
##      col = adjustcolor("forestgreen", alpha.f = 0.3),
##      cex.axis = 0.8, cex.lab = 0.8)

# Plot regularization path
palette("Okabe-Ito")
plot(mse.boost, type = "l", las = 1,  
     ylim = range(c(mse.boost, als.boost.post$results$mse)), 
     xlab = "Number of trees", ylab = "Test MSE")
lines(als.boost.post$results, col = 2)
abline(h = min(mse.boost), lty = 2)
abline(h = min(als.boost.post$results$mse), col = 2, lty = 2)
palette("default")

## names(bank) <- gsub("\\.", replacement = "_", x = names(bank))
## bank$y <- ifelse(bank$y == "yes", 1, 0)
## bank$contact <- ifelse(bank$contact == "telephone", 1, 0)
## bank$duration <- NULL  # remove target leakage

## bank$id <- seq_len(nrow(bank))  # need a unique row identifier
## cats <- names(which(sapply(bank, FUN = class) == "character"))
## lhs <- paste(setdiff(names(bank), cats), collapse = "+")
## fo <- as.formula(paste(lhs, "~ variable + value"))
## bank <- as.data.table(bank)  # coerce to data.table
## bank.ohe <- dcast(melt(bank, id.vars = setdiff(names(bank), cats)),
##                   formula = fo, fun = length)
## bank$id <- bank.ohe$id <- NULL

## set.seed(1056)  # for reproducibility
## trn.id <- caret::createDataPartition(bank.ohe$y, p = 0.5, list = FALSE)
## bank.trn <- data.matrix(bank.ohe[trn.id, ])  # training data
## bank.tst  <- data.matrix(bank.ohe[-trn.id, ])  # test data

## library(xgboost)
## 
## xnames <- setdiff(names(bank.ohe), "y")
## dm.trn <- xgb.DMatrix(bank.trn[, xnames], label = bank.trn[, "y"])
## dm.tst <- xgb.DMatrix(bank.tst[, xnames], label = bank.tst[, "y"])

## params <- list(
##   eta = 0.01,  # shrinkage/learning rate
##   max_depth = 3,
##   subsample = 0.5,
##   objective = "binary:logistic",  # for predicted probabilities
##   eval_metric = "rmse",  # square root of Brier score
##   nthread = 8
## )

## watch.list <- list(train = dm.trn, eval = dm.tst)
## 
## # Train an XGBoost model without early stopping
## set.seed(1100)  # for reproducibility
## bank.xgb.1 <-
##   xgb.train(params, data = dm.trn, nrounds = 3000, verbose = 0,
##             watchlist = watch.list)
## (best.iter <- which.min(bank.xgb.1$evaluation_log$eval_rmse))

library(xgboost)

bank.xgb.1.log <- readRDS("../data/chap-gbm-ex-bank.xgb-1-log.rds")
# bank.xgb.1 <- xgb.load("../data/chap-gbm-ex-bank.xgb-1")
bank.xgb.2 <- xgb.load("../data/chap-gbm-ex-bank.xgb-2")

# Print optimal number of iterations from GBM without early stopping
(best.iter <- which.min(bank.xgb.1.log$eval_rmse))

## set.seed(1100)  # for reproducibility
## (bank.xgb.2 <-
##   xgb.train(params, data = dm.trn, nrounds = 3000, verbose = 0,
##             watchlist = watch.list, early_stopping_rounds = 150))

print(bank.xgb.2)

## palette("Okabe-Ito")
## plot(bank.xgb.1$evaluation_log[, c(1, 2)], type = "l",
##      xlab = "Number of trees",
##      ylab = "RMSE (square root of Brier score)")
## lines(bank.xgb.1$evaluation_log[, c(1, 3)], type = "l", col = 2)
## abline(v = best.iter, col = 2, lty = 2)
## abline(v = bank.xgb.2$niter, col = 3, lty = 2)
## legend("topright", legend = c("Train", "Test"), inset = 0.01, bty = "n",
##        lty = 1, col = 1:2)
## palette("default")

palette("Okabe-Ito")
plot(bank.xgb.1.log[, c(1, 2)], type = "l",
     xlab = "Number of trees",
     ylab = "RMSE (square root of Brier score)")
lines(bank.xgb.1.log[, c(1, 3)], type = "l", col = 2)
abline(v = best.iter, col = 2, lty = 2)
abline(v = bank.xgb.2$niter, col = 3, lty = 2)
legend("topright", legend = c("Train", "Test"), inset = 0.01, bty = "n",
       lty = 1, col = 1:2)
palette("default")

## shap.trn <- predict(bank.xgb, newdata = dm.trn, ntreelimit = best.iter,
##                     predcontrib = TRUE, approxcontrib = FALSE)
## shap.trn <- shap.trn[, -which(colnames(shap.trn) == "BIAS")]
## 
## # Shapley-based variable importance
## shap.vi <- colMeans(abs(shap.trn))
## shap.vi <- shap.vi[order(shap.vi, decreasing = TRUE)]
## dotchart(shap.vi[1:10], xlab = "mean(|SHAP value|)", pch = 19)

shap.trn <- readRDS("../data/chap-gbm-ex-bank-shapley-trn.rds")

# Set graphical parameters for this plot
par(
  mar = c(4, 8, 0.1, 0.1), 
  cex.lab = 0.95, 
  cex.axis = 0.8,  # was 0.9
  mgp = c(2, 0.7, 0), 
  tcl = -0.3, 
  las = 1
)

# Shapley-based variable importance
shap.vi <- colMeans(abs(shap.trn))
shap.vi <- shap.vi[order(shap.vi, decreasing = TRUE)]  # sort
dotchart(shap.vi[1:10], xlab = "mean(|SHAP value|)", pch = 19)

## head(xgb.importance(model = bank.xgb, trees = 0:(best.iter - 1)))

bank.xgb.vi <- readRDS("../data/chap-gbm-ex-bank-vi.rds")
head(bank.xgb.vi)

## shap.age <- data.frame("age" = bank.trn[, "age"],
##                        "shap" = shap.trn[, "age"])
## 
## # Shapley dependence plot for age
## cols <- palette.colors(3, palette = "Okabe-Ito")
## ggplot(shap.age, aes(age, shap)) +
##   geom_point(alpha = 0.1) +
##   geom_smooth(se = FALSE, color = cols[2]) +
##   geom_hline(yintercept = 0, linetype = "dashed", color = cols[3]) +
##   xlab("Age (years)") + ylab("Shapley value")

shap.age <- readRDS("../data/chap-gbm-ex-bank-shap-age.rds")

# Shapley dependence plot for age
cols <- palette.colors(3, palette = "Okabe-Ito")
ggplot(shap.age, aes(age, shap)) +
  ggrastr::rasterise(geom_point(alpha = 0.1)) +#, dpi = 300) +
  geom_smooth(se = FALSE, color = cols[2]) +
  geom_hline(yintercept = 0, linetype = "dashed", color = cols[3]) +
  xlab("Age (years)") + ylab("Shapley value") 
