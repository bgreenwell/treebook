library(dplyr)
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
oi.cols <- unname(palette.colors(8, palette = "Okabe-Ito"))

# Ames housing data
ames <- as.data.frame(AmesHousing::make_ames())
ames$Sale_Price <- ames$Sale_Price / 1000  # rescale response
set.seed(4919)  # for reproducibility
id <- sample.int(nrow(ames), size = floor(0.7 * nrow(ames)))
ames.trn <- ames[id, ]
ames.tst <- ames[-id, ]

# Root mean square error helper function
rmse <- function(pred, obs, na.rm = FALSE) {
  sqrt(mean((pred - obs) ^ 2, na.rm = na.rm))
}

# Modified version of randomForest::MDSplot(); removed par(pty = "s")
mds.plot <- function(rf, fac, k = 2, palette = NULL, pch = 20, ...) {
  if (!inherits(rf, "randomForest")) 
    stop(deparse(substitute(rf)), " must be a randomForest object")
  if (is.null(rf$proximity)) 
    stop(deparse(substitute(rf)), " does not contain a proximity matrix")
  rf.mds <- stats::cmdscale(1 - rf$proximity, eig = TRUE, k = k)
  colnames(rf.mds$points) <- paste("Dim", 1:k)
  nlevs <- nlevels(fac)
  if (is.null(palette)) {
    palette <- if (requireNamespace("RColorBrewer", 
                                    quietly = TRUE) && nlevs < 12) 
      RColorBrewer::brewer.pal(nlevs, "Set1")
    else rainbow(nlevs)
  }
  if (k <= 2) {
    plot(rf.mds$points, col = palette[as.numeric(fac)], pch = pch, 
         ...)
  }
  else {
    pairs(rf.mds$points, col = palette[as.numeric(fac)], 
          pch = pch, ...)
  }
  invisible(rf.mds)
}

knitr::knit_engines$set(python = reticulate::eng_python)
reticulate::use_condaenv("r-reticulate-trees", required = TRUE)

corfun <- function(N = 30, rho = 0) {
  Sigma <- matrix(rho, nrow = N, ncol = N)
  diag(Sigma) <- 1
  X <- MASS::mvrnorm(N * 100, mu = rep(0, N), Sigma = Sigma)
  apply(X, MARGIN = 1, mean)
}
cors <- 0:10 / 10
res <- NULL
for (i in seq_along(cors)) {
  res <- rbind(res, cbind(cors[i], corfun(rho = cors[i])))
}
res <- as.data.frame(res)
boxplot(V2 ~ V1, data = res, xlab = "Pairwise correlation", 
        ylab = "Sample mean", col = 2)

data(spam, package = "kernlab")
set.seed(852)  # for reproducibility
id <- sample.int(nrow(spam), size = floor(0.7 * nrow(spam)))
spam.trn <- spam[id, ]  # training data
spam.tst <- spam[-id, ]  # test data

# Helper function that returns the given node and all its ancestors (i.e., a 
# vector of node numbers)
path.to.root <- function(node) {
  if(node == 1) {  # root node?
    node
  }
  else {  # recurse, %/% 2 gives the parent of node
    c(node, path.to.root(node %/% 2))
  }
}

# Plot six different bagged trees
par(mfrow = c(2, 3), mar = c(4, 4, 0.1, 0.1))
set.seed(1923)  # for reproducibility
for (i in 1:6) {
  id <- sample.int(nrow(spam.trn), replace = TRUE)
  trn <- spam.trn[id, ]
  tree <- rpart::rpart(type ~ ., data = trn, cp = 0, model = TRUE, maxdepth = 3)
  node <- 15
  nodes <- as.numeric(row.names(tree$frame))
  cols <- ifelse(nodes %in% path.to.root(node), "forestgreen", "lightgrey")
  rpart.plot::prp(tree, nn = TRUE, col = cols, branch.col = cols, 
                  split.col = cols, nn.col = cols)
}

## knitr::include_graphics("diagrams/rf-classification.png", error = FALSE)

## knitr::include_graphics("diagrams/rf-regression.png", error = FALSE)

set.seed(1251)  # for reproducibility
mease <- treemisc::gen_mease(1000, nsim = 250)
#cols <- palette.colors(2, palette = "Okabe-Ito", alpha = 0.5)
#plot(x2 ~ x1, data = mease, xlab = expression(x[1]), ylab = expression(x[2]),
#     pch = c(15, 19)[mease$yclass1 + 1], col = cols[mease$yclass1 + 1],
#     cex = 0.8)
#legend("topleft", legend = c(expression(Y==0), expression(Y==1)), 
#       pch = c(15, 19), col = cols, bg = "white", cex = 0.8)
ggplot(mease, aes(x = x1, y = x2)) +
  geom_point(aes(color = as.factor(yclass1), shape = as.factor(yclass1)),
             alpha = 0.5) +
  scale_oi(name = "", labels = c("Y = 0", "Y = 1")) +
  scale_shape(name = "", labels = c("Y = 0", "Y = 1")) +
  #theme(legend.position = "top")
  theme(legend.justification = c(-0.1, -0.1), legend.position = c(0, 0))

# Setup
prob.med <- readRDS("../data/chap-rf-simulation-mease-prob-med.rds")
prob.med$type <- ifelse(prob.med$type == "rf", "RF", "CF")  # or `toupper()`
ggplot(prob.med, aes(x = true, y = median)) +
  geom_point(aes(color = type), alpha = 0.1) +
  facet_wrap(~ nodesize) +
  scale_colour_manual(values = oi.cols[2:3], name = "", 
                      labels = c("Y = 0", "Y = 1")) +
  geom_abline(intercept = 0, slope = 1, 
              linetype = "dashed", color = oi.cols[1]) +
  xlab("True probability") +
  ylab("Median predicted probability") +
  theme(legend.position = "top")

prob.mse <- readRDS("../data/chap-rf-simulation-mease-prob-mse.rds")
prob.mse$type <- ifelse(prob.mse$type == "rf", "RF", "CF")  # or `toupper()`
ggplot(prob.mse, aes(x = type, y = mse)) +
  geom_boxplot(aes(fill = type)) +
  facet_wrap(~ nodesize, nrow = 1) +
  #scale_oi() +
  scale_oi(fill = TRUE, alpha = 0.7) +
  xlab("") +
  ylab("Mean squred error") +
  theme(legend.position = "none")

## spam.bs <- readRDS("../data/rf-spam-brier-scores.rds")
## 
## # Plot results
## palette("Okabe-Ito")
## plot(spam.bs$rf.nsim.1, ylim = range(unlist(spam.bs)), type = "l", col = 1,
##      xlab = "Number of trees", ylab = "Brier score (test data)")
## lines(spam.bs$rf.nsim.10, col = 2)
## lines(spam.bs$rf.nsim.20, col = 3)
## lines(spam.bs$rf.nsim.30, col = 4)
## abline(h = spam.bs$tree, lty = 2)
## legend("topright", legend = c(expression(n[min]==1), expression(n[min]==10),
##                               expression(n[min]==20), expression(n[min]==30),
##                               "Single tree"),
##        inset = 0.01, bty = "n", lty = c(1, 1, 1, 1, 2), col = c(1:4, 1),
##        cex = 0.8)
## palette("default")

crforest <- function(X, y, mtry = NULL, B = 5, oob = TRUE) {
  min.node.size <- if (is.factor(y)) 1 else 5
  N <- nrow(X)  # number of observations
  p <- ncol(X)  # number of features          
  train <- cbind(X, "y" = y)  # training data frame
  fo <- as.formula(paste("y ~ ", paste(names(X), collapse = "+")))
  if (is.null(mtry)) {  # use default definition
    mtry <- if (is.factor(y)) sqrt(p) else p / 3
    mtry <- floor(mtry)  # round down to nearest integer
  }
  # CTree parameters; basically force the tree to have maximum depth
  ctrl <- party::ctree_control(mtry = mtry, minbucket = min.node.size,
                               minsplit = 10, mincriterion = 0)
  forest <- vector("list", length = B)  # to store each tree
  for (b in 1:B) {  # fit trees to bootstrap samples
    boot.samp <- sample(1:N, size = N, replace = TRUE)
    forest[[b]] <- party::ctree(fo, data = train[boot.samp, ],
                                control = ctrl)
    if (isTRUE(oob)) {  # store row indices for OOB data
      attr(forest[[b]], which = "oob") <- 
        setdiff(1:N, unique(boot.samp))
    }
  }
  forest  # return the "forest" (i.e., list) of trees
}

## X <- subset(ames.trn, select = -Sale_Price)  # feature columns
## set.seed(1408)  # for reproducibility
## ames.crf <- crforest(X, y = ames.trn$Sale_Price, B = 300)

ames.crf.preds <- readRDS("../data/chap-rf-crforest-ames-preds.rds")
preds.tst <- ames.crf.preds$preds.tst
preds.oob <- ames.crf.preds$preds.oob
B <- ncol(preds.tst)  # number of trees in the forest

## B <- length(ames.crf)  # number of trees in forest
## preds.tst <- matrix(nrow = nrow(ames.tst), ncol = B)
## for (b in 1:B) {  # store predictions from each tree in a matrix
##   preds.tst[, b] <- predict(ames.crf[[b]], newdata = ames.tst)
## }
## pred.tst <- rowMeans(preds.tst)  # average predictions across trees
## 
## # Root-mean-square error function
## rmse <- function(pred, obs, na.rm = FALSE) {
##   sqrt(mean((pred - obs) ^ 2, na.rm = na.rm))
## }
## 
## # Root mean square error on test data
## rmse(pred.tst, obs = ames.tst$Sale_Price)

pred.tst <- rowMeans(preds.tst)  # average predictions across trees
rmse(pred.tst, obs = ames.tst$Sale_Price)

rmse.tst <- numeric(B)  # to store RMSEs
for (b in 1:B) {
  pred <- rowMeans(preds.tst[, 1:b, drop = FALSE], na.rm = TRUE)
  rmse.tst[b] <- rmse(pred, obs = ames.tst$Sale_Price, na.rm = TRUE)
}

set.seed(1226)  # for reproducibility
N <- 100  # sample size
obs <- 1:N  # original observations
res <- replicate(10000, sample(obs, size = N, replace = TRUE))
inbag <- apply(res, MARGIN = 2, FUN = function(boot.sample) {
  mean(obs %in% boot.sample)  # proportion in bootstrap sample
})
mean(inbag)

## preds.oob <- matrix(nrow = nrow(ames.trn), ncol = B)  # OOB predictions
## for (b in 1:B) {  # WARNING: Might take a minute or two!
##   oob.rows <- attr(ames.crf[[b]], which = "oob")  # OOB row IDs
##   preds.oob[oob.rows, b] <-
##     predict(ames.crf[[b]], newdata = ames.trn[oob.rows, ])
## }
## pred.oob <- rowMeans(preds.oob)  # average OOB predictions across trees
## 
## # Peek at results
## preds.oob[1:3, 1:6]

preds.oob[1:3, 1:6]

pred.oob <- rowMeans(preds.oob, na.rm = TRUE)
rmse(pred.oob, obs = ames.trn$Sale_Price, na.rm = TRUE)

rmse.oob <- numeric(B)  # to store RMSEs
for (b in 1:B) {
  pred <- rowMeans(preds.oob[, 1:b, drop = FALSE], na.rm = TRUE)
  rmse.oob[b] <- rmse(pred, obs = ames.trn$Sale_Price, na.rm = TRUE)
}

# Include test error for a single tree for comparison
ames.ctree <- party::ctree(Sale_Price ~ ., data = ames.trn)
rmse.tst.ctree <- rmse(predict(ames.ctree, newdata = ames.tst), 
                       obs = ames.tst$Sale_Price)

# Plot OOB and test RMSE
palette("Okabe-Ito")
plot(rmse.tst, type = "l", col = 1, xlab = "Number of trees", 
     ylab = "RMSE", las = 1)
lines(rmse.oob, col = 2)
abline(h = rmse.tst.ctree, lty = 2, col = 3)
legend("topright", legend = c("CRF (test)", "CRF (OOB)", "Ctree (test)"), 
       lty = c(1, 1, 2), col = 1:3, inset = 0.01, bty = "n")
palette("default")

res <- readRDS("../data/chap-rf-tuning-mtry.rds")
pal <- palette.colors(3, palette = "Okabe-Ito", alpha = 0.7)[2:3]
boxplot(error ~ method * i, data = res, col = pal, xlab = expression(m[try]),
        ylab = "Mean squared error", names = NA, xaxt = "n")
axis(1, at = seq(from = 1.5, to = 19.5, by = 2), labels = 1:10)
legend("topright", inset = .05, legend = c("OOB", "TEST"), fill = pal)
abline(v = floor(10 / 3) + 2.5, lty = 2)

# Read in results, wrangle, and plot
levs <- c("RF (Gini)", "RF (permutation)", "RF (Gini-corrected)")
temp <- readRDS("../scripts/bias_rf_vi.rds")
temp$method <- factor(temp$method, levels = levs)  # to reorder panels
temp %>%
  group_by(Variable, method) %>%
  summarize(avg = mean(Importance)) %>%
  ggplot(aes(Variable, avg)) +
    geom_col() +
    geom_hline(yintercept = 4, linetype = "dashed") +
    xlab("") +
    ylab("Average rank") +
    facet_wrap(~ method) +
    theme_bw()

## library(Rcpp)
## 
## cppFunction(
##   "
##   NumericMatrix proximity(IntegerMatrix x) {
##     int nrow = x.nrow();
##     NumericMatrix out(nrow, nrow);
##     for (int i = 0; i < nrow; i++) {
##       for (int j = i + 1; j < nrow; j++) {
##         out(i, j) = sum(x(i, _) == x(j, _));
##       }
##     }
##     return out / x.ncol();
##   }
##   "
## )

library(randomForest)

# Read in Swiss banknote data and mislabel one of the cases
bn2 <- treemisc::banknote
bn2$y[101] <- 0  # mislabeled case

# Fit a default RF
set.seed(2117)  # for reproducibility
bn2.rfo <- randomForest(as.factor(y) ~ ., data = bn2, proximity = TRUE)

# Compute proximity-based outlyingness measure
out <- outlier(bn2.rfo)

# Plot results
palette("Okabe-Ito")
plot(outlier(bn2.rfo), col = bn2$y + 1, las = 1, 
     xlab = "Case number", ylab = "Proximity-based outlier score")
top2 <- order(out, decreasing = TRUE)[1:2]
text(top2, out[top2], labels = top2, pos = 1)
legend("topright", legend = c("Labeled as genuine", "Labeled as counterfeit"), 
       col = 1:2, pch = 1, inset = 0.01, bty = "n")
palette("default")

bn <- treemisc::banknote
X.original <- subset(bn, select = -y)  # features only
X.synthetic <- X.original
set.seed(1034)
for (i in seq_len(ncol(X.original))) {
  X.synthetic[[i]] <- sample(X.synthetic[[i]], replace = TRUE)
}
X <- rbind(X.original, X.synthetic)

# Add binary indicator (doesn't)
X$y <- rep(c("original", "synthetic"), each = nrow(bn))

# Construct original and synthetic data, then combine
set.seed(1327)  # for reproducibility
bn.urfo <- randomForest(as.factor(y) ~ ., data = X, ntree = 1000,
                        proximity = TRUE)

# Compute scaling coordinates from proximities of the original observations
bn.mds <- cmdscale(1 - bn.urfo$proximity[1:200, 1:200])

# Plot scaling coordinates (include outlying case corresponding to row 70)
palette("Okabe-Ito")
plot(bn.mds, col = bn$y + 1, las = 1,
     xlab = "Scaling coordinate 1", ylab = "Scaling coordinate 2")
# points(bn.mds[70, 1], bn.mds[70, 2], pch = 19, col = 3)
legend("topleft", legend = c("Labeled as genuine", "Labeled as counterfeit"), 
       col = 1:2, pch = 1, inset = 0.01, bty = "n")
palette("default")

## library(ranger)
## 
## # Read in Ames housing data and split into train/test sets
## ames <- as.data.frame(AmesHousing::make_ames())
## ames$Sale_Price <- ames$Sale_Price / 1000  # rescale response
## set.seed(2101)  # for reproducibility
## trn.id <- sample.int(nrow(ames), size = floor(0.7 * nrow(ames)))
## ames.trn <- ames[trn.id, ]
## ames.tst <- ames[-trn.id, ]
## 
## # Compute test set predictions from an RF and CSRF
## set.seed(1645)  # for reproducibility
## rfo <- ranger(Sale_Price ~ ., data = ames.trn)
## rfo.pred <- predict(rfo, data = ames.tst)$predictions
## # csrfo.pred <- csrf(Sale_Price ~ ., training_data = ames.trn,
## #                    test_data = ames.tst, verbose = TRUE)
## 
## # Read in previously generate results
## csrfo.pred <- readRDS("../data/rf-csrf-ames.rds")
## 
## # Mean squared error
## mean((rfo.pred - ames.tst$Sale_Price) ^ 2)
## mean((csrfo.pred - ames.tst$Sale_Price) ^ 2)
## 
## # Mean absolute error
## mean(abs(rfo.pred - ames.tst$Sale_Price))
## mean(abs(csrfo.pred - ames.tst$Sale_Price))

spam.rfos.preds <- readRDS("../data/chap-rf-se-spam-preds.rds")
mtry <- c(5, 19, 57)

# Load the email spam data and split into train/test sets using a 70/30 split
data(spam, package = "kernlab")
set.seed(1258)  # for reproducibility
trn.id <- sample(nrow(spam), size = 0.7 * nrow(spam), replace = FALSE)
spam.trn <- spam[trn.id, ]
spam.tst <- spam[-trn.id, ]

par(mar = c(4, 4, 2, 0.1), cex.lab = 0.95, cex.axis = 0.8, 
    mgp = c(2, 0.7, 0), tcl = -0.3, las = 1, mfrow = c(1, 3))

# Plot RF test set prediction standard errors, colored by whether or not the 
# observations were misclassified
ylim <- range(sapply(spam.rfos.preds, FUN = function(x) {
  range(x$se[, "spam"])
}))
palette("Okabe-Ito")
for (i in seq_along(spam.rfos.preds)) {
  pred <- spam.rfos.preds[[i]]
  classes <- ifelse(pred$predictions[, "spam"] > 0.5, "spam", "nonspam")
  id <- (classes == spam.tst$type) + 1
  plot(pred$predictions[, "spam"], pred$se[, "spam"], col = id, 
       pch = c(19, 1)[id], main = paste("mtry =", mtry[i]), 
       ylim = c(0, max(ylim)), xlab = "Predicted probablitiy", 
       ylab = "Standard error")
  if (i == 1) {
    legend("topleft", legend = c("Misclassified"), pch = 19, col = 1,
           inset = 0.01, bty = "n")
  }
}
palette("default")

library(ranger)

# Read in Ames housing data and split into train/test sets using a 70/30 split
ames <- as.data.frame(AmesHousing::make_ames())
ames$Sale_Price <- ames$Sale_Price / 1000
set.seed(2101)  # for reproducibility
trn.id <- sample.int(nrow(ames), size = floor(0.7 * nrow(ames)))
ames.trn <- ames[trn.id, ]
ames.tst <- ames[-trn.id, ]

# Fit a quantile regression forest
set.seed(1235)  # for reproducibility
rfo <- ranger(Sale_Price ~ ., data = ames.trn)
qrfo <- ranger(Sale_Price ~ ., data = ames.trn, quantreg = TRUE)

# Find most expensive house in test set
id <- which.max(ames.tst$Sale_Price)
pred.rfo <- predict(rfo, data = ames.tst[id, ])$predictions
pred.qrfo <- predict(qrfo, data = ames.tst[id, ], type = "quantiles",
                     quantiles = c(0.025, 0.5, 0.975))$predictions

# Compute predictions for different quantiles
pred <- predict(qrfo, data = ames.tst, type = "quantiles", 
                quantiles = c(0.025, 0.5, 0.975))

# Sort, center, and add other relevant columns
pred <- pred$predictions
pred <- cbind(pred, "length" = pred[, 3] - pred[, 1])
pred <- cbind(pred, "y" = ames.tst$Sale_Price)
ord <- order(pred[, "length"], decreasing = FALSE)
pred <- pred[ord, ]
means <- (pred[, 1] + pred[, 3]) / 2
pred[, 1] <- pred[, 1] - means
pred[, 2] <- pred[, 2] - means
pred[, 3] <- pred[, 3] - means
pred[, "y"] <- pred[, "y"] - means

# Plot results (similar to Figure 3 of Meinshausen (2006))
res <- as.data.frame(pred)
res$index <- seq_len(nrow(res))
ylim <- range(res[, 1:3])
palette("Okabe-Ito")
plot(y ~ index, data = res, xlab = "Ordered observation number", 
     ylab = "Sale price / 1000 (centered)", col = 2, ylim = ylim)
points(res$index, res$`quantile= 0.5`, col = 3)
lines(res$index, res$`quantile= 0.025`, col = 1)
lines(res$index, res$`quantile= 0.975`, col = 1)
legend("topleft", inset = 0.01, bty = "n", pch = c(1, 1, NA), 
       lty = c(NA, NA, 1), col = c(2, 3, 1),
       legend = c("Observed", "Predicted median", "Prediction internal"))
palette("default")
# cols <- palette.colors(3, palette = "Okabe-Ito")
# ggplot(res, aes(x = index, y = y)) + 
#   geom_point(color = cols[3], alpha = 0.5) +
#   geom_point(aes(x = index, y = `quantile= 0.5`), 
#              color = cols[2], alpha = 0.3) +
#   geom_line(aes(x = index, y = `quantile= 0.025`), color = cols[1]) +
#   geom_line(aes(x = index, y = `quantile= 0.975`), color = cols[1]) +
#   xlab("Ordered observation number") +
#   ylab("Sale price (centered)") +
#   theme_bw()

# Generate data
set.seed(1038)  # for reproducibility
x1 <- runif(100, min = -5, max = 5)
x2 <- x1 + rnorm(length(x1))
X <- cbind(x1, x2)
#R <- loadings(princomp(X, cor = FALSE, fix_sign = FALSE))  # rotation matrix from PCA
R <- eigen(cov(X))$vectors  # same as above
XR <- X %*% R
colnames(XR) <- colnames(X)

# Wrangle data for plotting
d1 <- rbind(
  cbind(as.data.frame(X), "label" = "Original axes"),
  cbind(as.data.frame(X), "label" = "Rotated axes"),
  cbind(as.data.frame(XR), "label" = "Rotated points")
)
d2 <- data.frame(
  "label" = c("Original axes", "Rotated axes", "Rotated points"),
  intercept = c(0, 0, 0),
  slope = c(0, R[2, 1] / R[1, 1], 0)
)
d3 <- data.frame(
  "label" = c("Original axes", "Rotated axes", "Rotated points"),
  intercept = c(0, 0, 0),
  slope = c(1e+06, R[2, 2] / R[1, 2], 1e+06)
)

# Color palette
cols <- unname(palette.colors(3, palette = "Okabe-Ito"))

# Plot data
ggplot(d1, aes(x1, x2)) +
  geom_point(color = cols[1], alpha = 0.3) +
  geom_abline(data = d2, aes(intercept = intercept, slope = slope), 
              linetype = 2, color = cols[2]) +
  geom_abline(data = d3, aes(intercept = intercept, slope = slope), 
              linetype = 2, color = cols[3]) +
  xlab(expression(X[1])) +
  ylab(expression(X[2])) +
  coord_fixed() +
  facet_wrap(~ label, nrow = 1) +
  theme_bw()
# set.seed(1038)  # for reproducibility
# x <- runif(100, min = -5, max = 5)
# y <- x + rnorm(length(x))
# X <- cbind(x, y)
# R <- loadings(princomp(X, cor = FALSE))  # rotation matrix from PCA
# par(mfrow = c(1, 3), las = 1)
# pt.col <- adjustcolor("purple2", alpha.f = 0.5)
# plot(X, xlim = c(-8, 8), ylim = c(-8, 8), 
#      col = pt.col, asp = 1,
#      xlab = expression(X[1]), ylab = expression(X[2]),
#      main = "Original axes")
# abline(h = 0, v = 0, lty = 2)
# plot(X, xlim = c(-8, 8), ylim = c(-8, 8), 
#      col = pt.col, asp = 1,
#      xlab = expression(X[1]), ylab = expression(X[2]),
#      main = "Rotation of axes")
# abline(a = 0, b = R[2, 1] / R[1, 1], lty = 2)
# abline(a = 0, b = R[2, 2] / R[1, 2], lty = 2)
# plot(X %*% R, xlim = c(-8, 8), ylim = c(-8, 8), 
#      col = pt.col, asp = 1,
#      xlab = expression(X[1]), ylab = expression(X[2]),
#      main = "Rotation of points")
# abline(h = 0, v = 0, lty = 2)

treemisc::rrm

set.seed(1038)
X1 <- runif(100, min = -5, max = 5)
X2 <- X1 + rnorm(length(X1))
X <- cbind(X1, X2)
palette("Okabe-Ito")  # colorblind-friendly color palette
plot(X, xlim = c(-8, 8), ylim = c(-8, 8), col = 1, las = 1,
     xlab = expression(x[1]), ylab = expression(x[2]))
pcR <- loadings(princomp(X, cor = FALSE, fix_sign = FALSE))  # PCA
points(X %*% pcR, col = 2)  # plot PCA rotation
abline(0, 1, lty = 2, col = 1)  # original axis
abline(h = 0, lty = 2, col = 2)  # axis after PCA rotation
for (i in 3:5) {  # plot random rotations
  R <- treemisc::rrm(2)  # generate a random 2x2 rotation matrix 
  points(X %*% R, col = adjustcolor(i, alpha.f = 0.5))
}
legend("topleft", legend = "Original sample", pch = 1, col = 1, 
       inset = 0.01, bty = "n")
palette("default")

library(treemisc)

eslmix <- load_eslmix()
class(eslmix)  # should be a list
names(eslmix)  # names of components

x <- as.data.frame(eslmix$x)  # training data
xnew <- as.data.frame(eslmix$xnew)  # evenly spaced grid of points
x$y <- as.factor(eslmix$y)  # coerce to factor for plotting 
xnew$prob <- eslmix$prob  # Pr(Y = 1 | xnew) 

# Colorblind-friendly palette
oi.cols <- unname(palette.colors(8, palette = "Okabe-Ito"))

# Construct scatterplot of training points
p <- ggplot(x, aes(x = x1, y = x2, color = y)) +
  geom_point(alpha = 1, show.legend = FALSE) + 
  scale_colour_manual(values = oi.cols) +
  theme_bw()

# Add optimal (i.e., Bayes) decision boundary
p + geom_contour(data = xnew, aes(x = x1, y = x2, z = prob), 
                 breaks = 0.5, color = oi.cols[4], 
                 inherit.aes = FALSE, linetype = 2)

library(rotationForest)
library(rpart)

# Fit an RF with and without random rotations
set.seed(2200)  # for reproducibility
rfo1 <- rforest(eslmix$x, y = eslmix$y, ntree = 1000, nodesize = 10)
rfo2 <- rotationForest(eslmix$x, y = eslmix$y, L = 1000)  # rotation forest
rfo3 <- rforest(eslmix$x, y = eslmix$y, ntree = 1000, nodesize = 10, 
                rotate = TRUE)  # random rotation forest

# Compute predicted probabilities (i.e., Pr(Y = 1)) for each method and stack
# into a single data frame
res <- lapply(1:3, FUN = function(x) {  # one copy for each model
  as.data.frame(eslmix$xnew)
})
res[[1]]$prob <- predict(rfo1, newX = eslmix$xnew) 
res[[2]]$prob <- predict(rfo2, newdata = as.data.frame(eslmix$xnew)) 
res[[3]]$prob <- predict(rfo3, newX = eslmix$xnew)  
res[[1]]$method <- "Random forest"
res[[2]]$method <- "Rotation forest"
res[[3]]$method <- "Random rotation forest"
res <- do.call(rbind, args = res)

# Plot estimated decision boundary from each model
ggplot(x, aes(x = x1, y = x2, color = y)) +
  geom_point(alpha = 0.7, show.legend = FALSE) + 
  scale_colour_manual(values = oi.cols) +
  theme_bw() +
  geom_contour(data = res, aes(x = x1, y = x2, z = prob), breaks = 0.5, 
               color = oi.cols[3], inherit.aes = FALSE) +
  facet_wrap(~ method) + 
  geom_contour(data = xnew, aes(x = x1, y = x2, z = prob), breaks = 0.5,
               color = oi.cols[4], inherit.aes = FALSE, linetype = 2)

plotPNG = function(img, add = FALSE) {
  res = dim(img)[2:1] # get the resolution, [x, y]
  plot(1, 1, xlim = c(1, res[1]), ylim = c(1, res[2]), type = "n", 
       xaxs = "i", yaxs = "i", xaxt = "n", yaxt = "n", xlab = "", ylab = "", 
       bty = "n")
  rasterImage(img, 1, 1, res[1], res[2])
}

# Load example data from NumPy array
np <- reticulate::import("numpy")
X <- np$load("../data/rf-itree-example.npy")

# Load IsoTree example
img <- png::readPNG("../diagrams/itree-path.png", native = TRUE)

# Display plots side by side
par(mfrow = c(1, 2))
plot(X, pch = 19, col = adjustcolor(1, alpha.f = 0.1), xlab = expression(x[1]),
     ylab = expression(x[2]))
points(X[1, , drop = FALSE], pch = 19, col = "purple")
plotPNG(img)

## # ccfraud <- data.table::fread("some/path/to/ccfraud.csv")
## 
## # Randomly permute rows
## set.seed(2117)  # for reproducibility
## ccfraud <- ccfraud[sample(nrow(ccfraud)), ]
## 
## # Split data into train/test sets
## set.seed(2013)  # for reproducibility
## trn.id <- sample(nrow(ccfraud), size = 10000, replace = FALSE)
## ccfraud.trn <- ccfraud[trn.id, ]
## ccfraud.tst <- ccfraud[-trn.id, ]
## 
## # Check class distribution in each
## proportions(table(ccfraud.trn$Class))
## proportions(table(ccfraud.tst$Class))

ccfraud <- data.table::fread("../data/ccfraud.csv")

# Randomly permute rows
set.seed(2117)  # for reproducibility
ccfraud <- ccfraud[sample(nrow(ccfraud)), ]

# Split data into train/test sets
set.seed(2013)  # for reproducibility
trn.id <- sample(nrow(ccfraud), size = 10000, replace = FALSE)
ccfraud.trn <- ccfraud[trn.id, ]
ccfraud.tst <- ccfraud[-trn.id, ]

# Check class distribution in each
proportions(table(ccfraud.trn$Class))
proportions(table(ccfraud.tst$Class))

## library(isotree)
## 
## # Fit a default isolation forest
## ccfraud.ifo <- isolation.forest(ccfraud.trn[, -31], nthreads = 1,
##                                 seed = 2223)
## 
## # Compute anomaly scores for the test observations
## head(scores <- predict(ccfraud.ifo, newdata = ccfraud.tst))

library(isotree)

# Read in previously stored results
ccfraud.ifo <- readRDS("../data/rf-ccfraud-ifo.rds")
scores <- readRDS("../data/rf-ccfraud-scores-test.rds")

# Compute anomaly scores for the test observations
head(scores)

#cutoff <- sort(unique(scores))
# Compute precision and recall across various cutoffs
cutoff <- seq(from = min(scores), to = max(scores), length = 999)
cutoff <- c(0, cutoff)
precision <- recall <- numeric(length(cutoff))
for (i in seq_along(cutoff)) {
  yhat <- ifelse(scores >= cutoff[i], 1, 0)
  tp <- sum(yhat == 1 & ccfraud.tst$Class == 1)  # true positives
  tn <- sum(yhat == 0 & ccfraud.tst$Class == 0)  # true negatives
  fp <- sum(yhat == 1 & ccfraud.tst$Class == 0)  # false positives
  fn <- sum(yhat == 0 & ccfraud.tst$Class == 1)  # false negatives
  precision[i] <- tp / (tp + fp)  # precision (or PPV)
  recall[i] <- tp / (tp + fn)  # recall (or sensitivity)
}
precision <- c(precision, 0)
recall <- c(recall, 0)
head(cbind(recall, precision))

# Compute data for lift chart
ord <- order(scores, decreasing = TRUE)
y <- ccfraud.tst$Class[ord]  # order according to sorted scores
prop <- seq_along(y) / length(y)
lift <- cumsum(y) / sum(ccfraud.tst$Class)  # convert to proportion
head(cbind(prop, lift))

# Precision-recall curve
ccfraud.pr <- data.frame(recall, precision)
p1 <- ggplot(ccfraud.pr, aes(recall, precision)) +
  geom_line(color = oi.cols[2]) +
  geom_hline(yintercept = mean(ccfraud.tst$Class), linetype = 2, color = oi.cols[1]) +
  #xlim(0, 1) +
  #ylim(0, 1) +
  scale_x_continuous(breaks = 0:10 / 10) +
  scale_y_continuous(breaks = 0:10 / 10) +
  xlab("Recall") +
  ylab("Precision")

# Lift curve
ccfraud.lift <- data.frame(prop, lift)
p2 <- ggplot(ccfraud.lift, aes(prop, lift)) +
  geom_line(color = oi.cols[2]) +
  geom_abline(intercept = 0, slope = 1, linetype = 2, color = oi.cols[1]) +
  #xlim(0, 1) +
  #ylim(0, 1) +
  scale_x_continuous(breaks = 0:10 / 10) +
  scale_y_continuous(breaks = 0:10 / 10) +
  xlab("Proportion of sample inspected") +
  ylab("Proportion of anomalies identified")

# Display plots side by side
gridExtra::grid.arrange(p1, p2, nrow = 1)

scores.trn <- predict(ccfraud.ifo, newdata = ccfraud.trn)
to.explain <- max(scores) - mean(scores.trn)

max.id <- which.max(scores)  # row ID for max anomaly score
(max.x <- ccfraud.tst[max.id, ])
max(scores)

library(fastshap)

X <- ccfraud.trn[, 1:30]  # feature columns only
max.x <- max.x[, 1:30]  # feature columns only!
pfun <- function(object, newdata) {  # prediction wrapper
  predict(object, newdata = newdata)
}

# Generate feature contributions
set.seed(1351)  # for reproducibility
ex <- explain(ccfraud.ifo, X = X, newdata = max.x, 
              pred_wrapper = pfun, adjust = TRUE, 
              nsim = 1000)
sum(ex)  # should sum to f(x) - baseline whenever `adjust = TRUE` 

# Transpose feature contributions
res <- data.frame(
  "feature" = paste0(names(ex), "=", round(max.x, digits = 2)),
  "shapley.value" = as.numeric(as.vector(ex[1,]))
)

# Plot feature contributions
ggplot(res, aes(x = shapley.value, y = reorder(feature, shapley.value))) +
  geom_point() +
  geom_vline(xintercept = 0, linetype = "dashed") +
  xlab("Shapley-based feature contribution") +
  ylab("") +
  theme(axis.text.y = element_text(size = rel(0.8)))

library(ranger)
library(treemisc)  # for isle_post() function

# Load the Ames housing data and split into train/test sets
ames <- as.data.frame(AmesHousing::make_ames())
ames$Sale_Price <- ames$Sale_Price / 1000  # rescale response
set.seed(2101)  # for reproducibility
trn.id <- sample.int(nrow(ames), size = floor(0.7 * nrow(ames)))
ames.trn <- ames[trn.id, ]  # training data/learning sample
ames.tst <- ames[-trn.id, ]  # test data
xtst <- subset(ames.tst, select = -Sale_Price)  # test features only

# Function to compute RMSE as a function of number of trees
rmse <- function(object, X, y) {  # only works with "ranger" objects
  p <- predict(object, data = X, predict.all = TRUE)$predictions
  sapply(seq_len(ncol(p)), FUN = function(i) {
    pred <- rowMeans(p[, seq_len(i), drop = FALSE])
    sqrt(mean((pred - y) ^ 2))
  })
}

# Fit a default RF with 1,000 maximal depth trees
set.seed(942)  # for reproducibility
system.time({ 
  rfo <- ranger(Sale_Price ~ ., data = ames.trn, num.trees = 1000)
})

# Fit an RF with 1,000 shallow (depth-4) trees on 5% bootstrap samples 
set.seed(1021)  # for reproducibility
system.time({
  rfo.4.5 <- ranger(Sale_Price ~ ., data = ames.trn, num.trees = 1000, 
                    max.depth = 4, sample.fraction = 0.05)
})

# Test set MSE as a function of the number of trees
rmse.rfo <- rmse(rfo, X = xtst, y = ames.tst$Sale_Price)
rmse.rfo.4.5 <- rmse(rfo.4.5, X = xtst, y = ames.tst$Sale_Price)
c("Test RMSE (RFO)" = rmse.rfo[1000], 
  "Test RMSE (RFO.4.5)" = rmse.rfo.4.5[1000])

treepreds <- function(object, newdata) {
  p <- predict(object, data = newdata, predict.all = TRUE)
  p$predictions  # return predictions component
}

# Post-process RFO ensemble using an independent test set
preds.trn <- treepreds(rfo, newdata = ames.trn)
preds.tst <- treepreds(rfo, newdata = ames.tst)
rfo.post <- treemisc::isle_post(
  X = preds.trn, 
  y = ames.trn$Sale_Price, 
  newX = preds.tst, 
  newy = ames.tst$Sale_Price,
  family = "gaussian"
)

# Post-process RFO.4.5 ensemble using an independent test set
preds.trn.4.5 <- treepreds(rfo.4.5, newdata = ames.trn)
preds.tst.4.5 <- treepreds(rfo.4.5, newdata = ames.tst)
rfo.4.5.post <- treemisc::isle_post(
  X = preds.trn.4.5, 
  y = ames.trn$Sale_Price,
  newX = preds.tst.4.5, 
  newy = ames.tst$Sale_Price,
  family = "gaussian"
)

palette("Okabe-Ito")
plot(rmse.rfo, type = "l", ylim = c(20, 50),
     las = 1, xlab = "Number of trees", ylab = "Test RMSE")
lines(rmse.rfo.4.5, col = 2)
lines(sqrt(rfo.post$results$mse), col = 1, lty = 2)
lines(sqrt(rfo.4.5.post$results$mse), col = 2, lty = 2)
legend("topright", col = c(1, 2, 1, 2), lty = c(1, 1, 2, 2),
       legend = c("RFO", "RFO.4.5","RFO (post)", "RFO.4.5 (post)"),
       inset = 0.01, bty = "n")
palette("default")

res <- rfo.4.5.post$results  # post-processing results on test set
lambda <- res[which.min(res$mse), "lambda"]   # optimal penalty parameter 
coefs <- coef(rfo.4.5.post$lasso.fit, s = lambda)[, 1L]
int <- coefs[1L]  # intercept
tree.coefs <- coefs[-1L]  # no intercept
trees <- which(tree.coefs == 0)  # trees to remove

# Remove trees corresponding to zeroed-out coefficients
rfo.4.5.def <- deforest(rfo.4.5, which.trees = trees)

# Check size of each object
c(
  "RFO.4.5" = format(object.size(rfo.4.5), units = "MB"),
  "RFO.4.5 (deforested)" = format(object.size(rfo.4.5.def), units = "MB")
)

ames.big <-  # stack data on top of itself 100 times
  do.call("rbind", args = replicate(100, ames.trn, simplify = FALSE))

# Compute reweighted predictions from a ``deforested'' ranger object
predict.def <- function(rf.def, weights, newdata, intercept = TRUE) {
  preds <- predict(rf.def, data = newdata, 
                   predict.all = TRUE)$predictions
  res <- if (isTRUE(intercept)) {  # returns a one-column matrix
    cbind(1, preds) %*% weights
  } else {
    preds %*% weights
  }
  res[, 1, drop = TRUE]  # coerce to atomic vector
}

# Scoring time for original RFO.4.5 fit
system.time({  # full random forest
  preds <- predict(rfo.4.5, data = ames.big)
})

# Scoring time for post-processed RFO.4.5 fit using updated weights
weights <- coefs[coefs != 0]  # LASSO-based weights for remaining trees
system.time({ 
  preds.post <- predict.def(rfo.4.5.def, weights = weights, 
                            newdata = ames.big)
})

final.test.rmse <- round(caret::RMSE(
  pred = predict.def(rfo.4.5.def, weights = weights, newdata = ames.tst), 
  obs = ames.tst$Sale_Price
), digits = 2)

t3 <- read.csv("https://hbiostat.org/data/repo/titanic3.csv",
               stringsAsFactors = TRUE)
keep <- c("survived", "pclass", "age", "sex", "sibsp", "parch")
t3 <- t3[, keep]  # only retain key variables

sapply(t3, FUN = function(x) mean(is.na(x)))

library(partykit)

# Fit a conditional inference tree using missingness as response
temp <- t3  # temporary copy
temp$age <- as.factor(ifelse(is.na(temp$age), "y", "n"))
(t3.ctree  <- ctree(age ~ ., data = temp))
# plot(t3.ctree)  # plot omitted

library(mice)

set.seed(1125)  # for reproducibility
imp <- mice(t3, method = "cart", m = 21, minbucket = 5,
            printFlag = FALSE)

# Display nonparametric densities
densityplot(imp)

t3.mice <- complete(
  data = imp,      # "mids" object (multiply imputed data set)
  action = "all",  # return list of all imputed data sets
  include = FALSE  # don't include original data (i.e., data with NAs)
)
length(t3.mice)  # returns a list of completed data sets

# Generate completed data set using RF's proximity-based imputation
set.seed(2121)  # for reproducibility
t3.rfimpute <- 
  randomForest::rfImpute(as.factor(survived) ~ ., data = t3, 
                         iter = 5, ntree = 500)

# Construct matrix of imputed values 
m <- imp$m  # number of MICE-based imputation runs
na.id <- which(is.na(t3$age))
x <- matrix(NA, nrow = length(na.id), ncol = m + 1)
for (i in 1:m) x[, i] <- t3.mice[[i]]$age[na.id]
x[, m + 1] <- t3.rfimpute$age[na.id]

# Plot results
palette("Okabe-Ito")
plot(x[, 1], type = "n", xlim = c(1, length(na.id)), ylim = c(0, 100),
     las = 1, ylab = "Imputed value")
for (i in 1:m) {
  lines(x[, i], col = adjustcolor(1, alpha.f = 0.1))
}
lines(rowMeans(x[, 1:m]), col = 1, lwd = 2)
lines(x[, m + 1], lwd = 2, col = 2)
legend("topright", legend = c("MICE: CART", "RF: proximity"), lty = 1,
       col = 1:2, bty = "n")
palette("default")

t3[c(956, 959), ]

library(ranger)

# Obtain a list of probability forests, one for each imputed data set
set.seed(2147)  # for reproducibility
rfos <- lapply(t3.mice, FUN = function(x) {
  ranger(as.factor(survived) ~ ., data = x, probability = TRUE, 
         importance = "permutation")
})

# Check OOB errors (Brier-score, in this case)
sapply(rfos, FUN = function(forest) forest$prediction.error)

# Compute list of VI scores, one for each model. Note: can use 
#`FUN = ranger::importance` to be safe
vis <- lapply(rfos, FUN = importance)

# Stack into a data frame
head(vis <- as.data.frame(do.call(rbind, args = vis)))

# Display boxplots of results
boxplot(vis, las = 1)

pfun <- function(object, newdata) {  # mean(prob(survived=1|x))
  mean(predict(object, data = newdata)$predictions[, "1"])
}

library(pdp)

# Construct PD plots for each model
pdps <- lapply(1:m, FUN = function(i) {
  partial(rfos[[i]], pred.var = c("age", "pclass", "sex"), 
          pred.fun = pfun, train = t3.mice[[i]], cats = "pclass",
          quantiles = TRUE, probs = 1:19/20)
})

# Stack into a single data frame for plotting
for (i in seq_along(pdps)) {
  pdps[[i]]$m <- i
}
head(pdps <- do.call(rbind, args = pdps))

library(ggplot2)

# Plot results
deciles <- quantile(t3$age, prob = 1:9/10, na.rm = TRUE)
ggplot(pdps, aes(age, yhat, color = sex,
               group = interaction(m, sex))) +
  geom_line(alpha = 0.3) +
  geom_rug(aes(age), data = data.frame("age" = deciles), 
           sides = "b", inherit.aes = FALSE) +
  labs(x = "Age (years)", y = "Surival probability") +
  facet_wrap(~ pclass) +
  scale_colour_manual(values = c("black", "orange")) +  # Okabe-Ito
  theme_bw() +
  theme(legend.title = element_blank(),
        legend.position = "top")

jack.dawson <- data.frame(
  #survived = 0L,  # in case you haven't seen the movie
  pclass = 3L,  # using `3L` instead of `3` to treat as integer
  age = 20.0,
  sex = factor("male", levels = c("female", "male")),
  sibsp = 0L,  
  parch = 0L  
)

library(fastshap)

# Prediction wrapper for `fastshap::explain()`; has to return a single
# (atomic) vector of predictions
pfun <- function(object, newdata) {  # compute prob(survived=1|x)
  predict(object, data = newdata)$predictions[, 2]
}

# Estimate feature contributions for each imputed training set
set.seed(754)
ex.jack <- lapply(1:21, FUN = function(i) {
  X <- subset(t3.mice[[i]], select = -survived)
  explain(rfos[[i]], X = X, newdata = jack.dawson, nsim = 1000, 
          adjust = TRUE, pred_wrapper = pfun)
})

# Bind together into one data frame
ex.jack <- do.call(rbind, args = ex.jack)

# Add feature values to column names
names(ex.jack) <- paste0(names(ex.jack), "=", t(jack.dawson))
print(ex.jack)

# Jack's predicted probability of survival across all imputed
# data sets
pred.jack <- data.frame("pred" = sapply(rfos, FUN = function(rfo) {
  pfun(rfo, jack.dawson)
}))

# Plot setup (e.g., side-by-side plots)
par(mfrow = c(1, 2),  mar = c(4, 4, 2, 0.1), 
    las = 1, cex.axis = 0.7) 

# Construct boxplots of results
boxplot(pred.jack, col = adjustcolor(2, alpha.f = 0.5))
mtext("Predicted probability of surviving", line = 1)
boxplot(ex.jack, col = adjustcolor(3, alpha.f = 0.5), horizontal = TRUE)
mtext("Feature contribution", line = 1)
abline(v = 0, lty = "dashed")

gen_binary <- function(...) {
  d <- treemisc::gen_friedman1(...)  # regression data
  d$y <- d$y - 23  # shift intercept
  d$prob <- plogis(d$y)  # inverse logit to obtain class probabilities
  #d$prob <- exp(d$y) / (1 + exp(d$y))  # same as above
  d$y <- rbinom(nrow(d), size = 1, prob = d$prob)  # 0/1 outcomes
  d
}

# Generate samples
set.seed(1921)  # for reproducibility
trn <- gen_binary(100000)  # training data
tst <- gen_binary(100000)  # test data

(pi1 <- proportions(table(gen_binary(1000000)$y))["1"])

isocal <- function(prob, y) {  # isotonic calibration function
  ord <- order(prob)
  prob <- prob[ord]  # put probabilities in increasing order
  y <- y[ord]  
  prob.cal <- isoreg(prob, y)$yf  # fitted values
  data.frame("original" = prob, "calibrated" = prob.cal)
}

library(ranger)

# Fit a probability forest (omitting the prob column)
set.seed(1446)  # for reproducibility
(rfo1 <- ranger(y ~ . - prob, data = trn, probability = TRUE,
                verbose = FALSE))

prob1 <- predict(rfo1, data = tst)$predictions[, 2]

mean((prob1 - tst$y) ^ 2)  # Brier score
mean((prob1 - tst$prob) ^ 2)  # MSE between predicted and true probs

prob.adjust <- function(p, observed.ratio, true.ratio) {
  f.ratio <- (1 / p - 1) * (1 / observed.ratio)
  1 / (1 + true.ratio * f.ratio)
}

trn.1 <- trn[trn$y == 1, ]
trn.0 <- trn[trn$y == 0, ]
trn.down <- rbind(trn.0[seq_len(nrow(trn.1)), ], trn.1)
table(trn.down$y)

set.seed(1146)  # for reproducibility
rfo2 <- ranger(y ~ . - prob, data = trn.down, probability = TRUE)

# Predicted probabilities for the positive class: P(Y=1|x)
prob2 <- predict(rfo2, data = tst)$predictions[, 2]
mean((prob2 - tst$y) ^ 2)  # Brier score
mean((prob2 - tst$prob) ^ 2)  # MSE

prob3 <- prob.adjust(prob2, observed.ratio = 1, 
                     true.ratio = (1 - pi1) / pi1)
mean((prob3 - tst$y) ^ 2)  # Brier score
mean((prob3 - tst$prob) ^ 2)  # MSE between predicted and true probs

# Colors
cols <- palette.colors(4, palette = "Okabe-Ito")

# Compute calibration curves
cal1 <- isocal(prob1, tst$y)
cal2 <- isocal(prob2, tst$y)
cal3 <- isocal(prob3, tst$y)

# Add additional columns and bind results together
cal1$true <- tst$prob[order(prob1)]
cal2$true <- tst$prob[order(prob2)]
cal3$true <- tst$prob[order(prob3)]
cal1$method <- "The good"
cal2$method <- "The ugly"
cal3$method <- "The bad"
cal <- rbind(cal1, cal2, cal3)

# Reorder factor levels to that the facets are in the right order
cal$method <- factor(cal$method, levels = c("The good", "The bad", "The ugly"))

# Construct plot
ggplot(cal, aes(original, true)) +
  ggrastr::rasterise(geom_point(alpha = 0.03)) +#, dpi = 300) +
  # geom_point(alpha = 0.03) +
  geom_abline(intercept = 0, slope = 1, color = cols[3]) +
  geom_line(aes(original, calibrated), color = cols[2]) +
  facet_wrap(~ method) +
  xlab("Predicted probability") +
  ylab("Actual probability")

pred <- readRDS("../data/chap-rf-bank-pred.rds")
rfo.summary <- readRDS("../data/chap-rf-bank-rfo-summary.rds")
vi <- readRDS("../data/chap-rf-bank-vi.rds")
euribor3m.grid <- readRDS("../data/chap-rf-bank-quantiles.rds")
pd <- readRDS("../data/chap-rf-bank-pd.rds")

url <- paste0("https://archive.ics.uci.edu/ml/machine-learning",
              "-databases/00222/bank-additional.zip")
temp <- tempfile(fileext = ".zip")  # to store zipped file
download.file(url, destfile = temp)
bank <- read.csv(unz(temp, "bank-additional/bank-additional-full.csv"), 
                 sep = ";", stringsAsFactors = FALSE)
unlink(temp)  # delete temporary file

names(bank) <- gsub("\\.", replacement = "_", x = names(bank))
bank$y <- ifelse(bank$y == "yes", 1, 0)
bank$duration <- NULL  # remove target leakage

# Split data into train/test sets using a 50/50 split
set.seed(1056)
trn.id <- caret::createDataPartition(bank$y, p = 0.5, list = FALSE)
bank.trn <- bank[trn.id, ]  # training data
bank.tst  <- bank[-trn.id, ]  # test data

## library(SparkR, lib.loc = "C:\\spark")
## library(ggplot2)
## 
## # Start a local connection to Spark using all available cores
## sparkR.session(master = "local[*]")

## bank.trn.sdf <- createDataFrame(bank.trn)
## bank.tst.sdf <- createDataFrame(bank.tst)
## 
## # Fit a regression/probability forest
## bank.rfo <- spark.randomForest(
##   bank.trn.sdf, y ~ ., type = "regression",
##   numTrees = 500, maxDepth = 10, seed = 1205
## )

## p <- predict(bank.rfo, newData = bank.tst.sdf)  # Pr(Y=yes|x)
## head(summarize(p, brier_score = mean((p$prediction - p$y)^2)))
## 
## #>   brier_score
## #> 1  0.07815544

cal <- treemisc::calibrate(pred$prediction, y = pred$y, 
                           method = "iso", pos.class = 1)
cg <- treemisc::lift(pred$prediction, y = pred$y, pos.class = 1,
                     cumulative = TRUE)

# Plot calibration curve and cumulative gains chart
par(
  mfrow = c(1, 2),
  mar = c(4, 4, 0.1, 0.1), 
  cex.lab = 0.95, 
  cex.axis = 0.8,  # was 0.9
  mgp = c(2, 0.7, 0), 
  tcl = -0.3, 
  las = 1
)
palette("Okabe-Ito")
plot(cal, refline = FALSE)
abline(0, 1, lty = 2, col = 2)
legend("topleft", legend = "Perfectly calibrated", lty = 2, col = 2,
       bty = "n", inset = 0.01, cex = 0.6)
plot(cg$prop, cg$lift / 100, type = "l",
     xlab = "% Contacted", ylab = "# subscribed (in hundreds)")
abline(0, sum(bank.tst$y == 1) / 100, lty = 2, col = 2)
grid()
legend("topleft", legend = "Baseline", lty = 2, col = 2,
       bty = "n", inset = 0.01, cex = 0.6)
palette("default")

## rfo.summary <- summary(rfo)  #  extract summary information
## (vi <- rfo.summary$featureImportances)  # gross...

vi

vi <- substr(vi, start = regexpr(",", text = vi)[1] + 1, 
             stop = nchar(vi) - 1)
vi <- gsub("\\[", replacement = "c(", x = vi)
vi <- gsub("\\]", replacement = ")", x = vi)
vi <- paste0("cbind(", vi, ")")
vi <- as.data.frame(eval(parse(text = vi)))
names(vi) <- c("feature.id", "importance")
vi$feature.name <- rfo.summary$features[vi[, 1] + 1]
head(vi[order(vi$importance, decreasing = TRUE), ], n = 10)

## euribor3m.grid <- as.DataFrame(unique(  # DataFrame of unique quantiles
##   approxQuantile(bank.trn.sdf, cols = "euribor3m",
##                  probabilities = 1:29 / 30, relativeError = 0)
## ))
## names(euribor3m.grid) <- "euribor3m"
## 
## # Training data without euribor3m
## trn.wo.euribor3m <- bank.trn.sdf  # copy of training data
## trn.wo.euribor3m$euribor3m <- NULL  # remove euribor3m
## 
## # Create a Cartesian product
## pd <- crossJoin(euribor3m.grid, trn.wo.euribor3m)
## dim(pd)  # nrow(euribor3m.grid) * nrow(trn.wo.euribor3m)
## 
## #> [1] 514850     20

## ggplot(pd, aes(x = euribor3m, y = yhat)) +
##   geom_line() +
##   geom_rug(data = as.data.frame(euribor3m.grid),
##            aes(x = euribor3m), inherit.aes = FALSE) +
##   xlab("Euribor 3 month rate") +
##   ylab("Partial dependence") +
##   theme_bw()
## 
## sparkR.stop()  # stop the Spark session

ggplot(pd, aes(x = euribor3m, y = yhat)) + 
  geom_line() +
  geom_rug(data = as.data.frame(euribor3m.grid), 
           aes(x = euribor3m), inherit.aes = FALSE) +
  xlab("Euribor 3 month rate") +
  ylab("Partial dependence") +
  theme_bw()
