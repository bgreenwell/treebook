library(dplyr)
library(ggplot2)

# Read in results, wrangle, and plot
sim <- readRDS("../scripts/bias_splitter.rds") %>%
  as.data.frame() %>%
  tibble::rowid_to_column("nsim")
names(sim)[names(sim) == "RPART"] <- "CART"
sim %>%
  tidyr::pivot_longer(CART:CTree, names_to = "method",
                      values_to = "splitter") %>%
  ggplot(aes(splitter)) +
    geom_bar(aes(y = ..prop.., group = 1)) +
    geom_hline(yintercept = 1/7, linetype = "dashed", color = 2) +
    scale_y_continuous(labels = scales::percent) +
    xlab("") +
    ylab("Selection percentage") +
    facet_wrap(~ method) +
    theme_light()

aq <- airquality[!is.na(airquality$Ozone), ]
N <- nrow(aq)   # sample size
gX <- aq$Temp   # g(X)
gY <- aq$Ozone  # h(Y)
Tstat <- sum(gX * gY)  # linear statistic
mu <- sum(gX) * mean(gY)
Sigma <- var(gY) * sum(gX ^ 2) - var(gY) * sum(gX) ^ 2 / N

# Quadratic test statistic (1.3)
(cq <- ((Tstat - mu) / sqrt(Sigma)) ^ 2) 
1 - pchisq(cq, df = 1)  # p-value

library(coin)
independence_test(Ozone ~ Temp, data = aq, teststat = "quadratic")

mushroom <- treemisc::mushroom
(ctab <- xtabs(~ odor + Edibility, data = mushroom))
(Tstat <- as.vector(ctab))  # multivariate linear statistic

gX <- model.matrix(~ odor - 1, data = mushroom)  # g(X)
hY <- model.matrix(~ Edibility - 1, data = mushroom)  # h(Y)
mu <- as.vector(colSums(gX) %*% t(colMeans(hY)))
Sigma <- var(hY) %x% (t(gX) %*% gX) - 
  var(hY) %x% (colSums(gX) %x% t(colSums(gX))) / nrow(hY)

# Quadratic test statistic (1.3)
(cq <- t(Tstat - mu) %*% MASS::ginv(Sigma) %*% (Tstat - mu))
1 - pchisq(cq, df = qr(Sigma)$rank)  # p-value

independence_test(Edibility ~ odor, data = mushroom, 
                  teststat = "quadratic")

gi.test <- function(x, y, g = identity, h = identity) {
  xy <- na.omit(cbind(x, y))  # only retain complete cases
  gx <- g(xy[, 1L])  # transformation function applied to x
  hy <- h(xy[, 2L])  # influence function applied to y
  lin <- sum(gx * hy)  # linear statistic
  mu <- sum(gx) * mean(hy)  # conditional expectation
  sigma <- var(hy) * sum(gx ^ 2) -  # conditional covariance
    var(hy) * sum(gx) ^ 2 / length(hy)
  c.quad <- ((lin - mu) / sqrt(sigma)) ^ 2  # quadratic test statistic
  pval <- 1 - pchisq(c.quad, df = 1)  # p-value
  c("chisq" = c.quad, "pval" = pval)  # return results
}

xnames <- setdiff(names(aq), "Ozone")  # feature names
set.seed(1938)  # for reproducibility
res <- sapply(xnames, FUN = function(x) {  # test each feature
  gi.test(airquality[[x]], y = airquality[["Ozone"]])
})
t(res)  # print transpose of results (nicer printing)

# Bonferroni adjusted p-values (same as 5 * pval in this case)
p.adjust(res["pval", ], method = "bonferroni")

bn <- treemisc::banknote  # start with the root node
xnames <- setdiff(names(bn), "y")  # feature names
res <- sapply(xnames, FUN = function(x) {  # test each feature
  gi.test(bn[[x]], y = bn[["y"]])
})
t(res) # print transpose of results (nicer printing)

# Bonferroni adjusted p-values (same as 6 * pval in this case)
p.adjust(res["pval", ], method = "bonferroni")

res <- sapply(xnames, FUN = function(x) {
  it <- independence_test(bn[["y"]] ~ bn[[x]], teststat = "quadratic")
  c("chisq" = statistic(it), "pval" = pvalue(it))
})
t(res)  # print transpose of results (nicer printing)

# Bonferroni adjusted p-values (same as 6 * pval in this case)
p.adjust(res["pval", ], method = "bonferroni")

set.seed(912)  # for reproducibility
xvals <- sort(unique(aq$Temp))  # potential cut points
splits <- matrix(0, nrow = length(xvals), ncol = 2)
colnames(splits) <- c("cutoff", "chisq")
for (i in seq_along(xvals)) {
  x <- ifelse(aq$Temp <= xvals[i], 0, 1)  # binary indicator
  y <- aq$Ozone
  # Ignore pathological splits or splits that are too small
  if (length(table(x)) < 2 || any(table(x) < 7)) {
    res <- NA
  } else {
    res <- gi.test(x, y)["chisq"]
  }
  splits[i, ] <- c(xvals[i], res)  # store cutpoint and test statistic
}
splits <- na.omit(splits)
splits[which.max(splits[, "chisq"]), ]

# Plot the test statistic for each cutoff (Figure 3.2)
plot(splits, type = "b", pch = 19, col = 2, las = 1,
     xlab = "Temperature split value (degrees Fahrenheit)", 
     ylab = "Test statistic")
abline(v = 82, lty = "dashed")

set.seed(2213)  # for reproducibility
n <- 100
x <- rnorm(n, mean = 0)
y <- x + rnorm(length(x), mean = 0, sd = 0.1)
# `ylim` was found by calling `par()$usr` after the last to `plot()`
par(mfrow =c(1, 2), mar =c(4, 4, 1, 1) + 0.1)
plot(x, y, las = 1, xlab = expression(X), ylab = expression(Y), col = 6,
     ylim = c(-3.783863, 10.530149))
y[1:3] <- 10
plot(x, y, las = 1, xlab = expression(X), ylab = expression(Y), col = 6,
     ylim = c(-3.783863, 10.530149))

set.seed(2142)  # for reproducibility
N <- seq(from = 5, to = 100, by = 5)  # range of sample sizes
res <- sapply(N, FUN = function(n) {
  pvals <- replicate(1000, expr = {  # simulate 1,000 p-values 
    x <- rnorm(n, mean = 0)          # from each test
    y <- x + rnorm(length(x), mean = 0, sd = 0.1)
    y[1:3] <- 10  # insert outliers
    test1 <- gi.test(x, y)  # no transformations
    test2 <- gi.test(x, y, g = rank, h = rank)  # convert to ranks
    c(test1["pval"], test2["pval"])  # extract p-values
  })
  apply(pvals, MARGIN = 1, FUN = function(x) mean(x < 0.05))
})

# Plot the results (Figure 3.4)
plot(N, res[2L, ], xlab = "Sample size", ylab = "Power", type = "l", 
     ylim = c(0, 1), las = 1)
lines(N, res[1L, ], col = 2, lty = 2)
legend("bottomright", 
       legend = c("Rank transformation", "No transformation"),
       lty = c(1, 2), col = c(1, 2), inset = 0.01, 
       box.col = "transparent")

library(partykit)

# Fit a default CTree using Bonferroni adjusted p-values
aq <- airquality[!is.na(airquality$Ozone), ]
(aq.cit <- ctree(Ozone ~ ., data = aq))
plot(aq.cit)  # Figure 3.5

strucchange::sctest(aq.cit, 1)

aq.cit2 <- party::ctree(Ozone ~ ., data = aq)  # refit the same tree
root <- party::nodes(aq.cit2, where = 1)[[1L]]  # extract root node
split.stats <- root$psplit$splitstatistic  # split statistics
cutpoints <- aq[[root$psplit$variableName]][split.stats > 0]
cq <- split.stats[split.stats > 0] ^ 2

# Plot split statistics (Figure 3.6; compare to Figure 3.2)
plot(cutpoints[order(cutpoints)], cq[order(cutpoints)], col = 4, 
     pch = 19, type = "b", las = 1, 
     xlab = "Temperature split value (degrees Fahrenheit)", 
     ylab = "Test statistic")
abline(v = root$psplit$splitpoint, lty = "dashed")

set.seed(1525)  # for reproducibility
aq.cart <- rpart::rpart(Ozone ~ ., data = aq, cp = 0)
aq.cart.pruned <- treemisc::prune_se(aq.cart, se = 1)  # 1-SE rule
plot(partykit::as.party(aq.cart.pruned))

## # Cross-validation
## make.folds <- function(df, k = 5) {	
##   folds <- cut(seq_len(nrow(df)), breaks = k, labels = FALSE)	
##   split(sample(nrow(df)), f = folds)  # return list of row numbers
## }	
## 
## set.seed(913)  # for reproducibility
## res <- replicate(100, expr = {
##   folds <- make.folds(aq, k = 5)
##   rmses <- sapply(folds, FUN = function(fold) {
##     trn <- aq[-fold, ]
##     tst <- aq[fold, ]  # fold to omit
##     ct <- party::ctree(Ozone ~ ., data = trn)
##     sqrt(mean((predict(ct, newdata = tst) - tst$Ozone) ^ 2))  # RMSE
##   })
##   c("mean" = mean(rmses), "sd" = sd(rmses))
## })
## apply(res, MARGIN = 1, mean)

## library(caret)
## 
## set.seed(932)  # for reproducibility
## ctrl <- trainControl("repeatedcv", number = 5, repeats = 10)
## tune <- train(Ozone ~ ., data = aq, method = "ctree", trControl = ctrl,
##               na.action = na.omit,
##               tuneGrid = data.frame(mincriterion = 1:99/100))
## tune$results
## 
## set.seed(1000)  # for reproducibility
## out <- NULL
## alphas <- 1:99/100  # grid of alpha values
## for (i in seq_along(alphas)) {
##   res <- replicate(10, expr = {
##     folds <- make.folds(aq, k = 5)  # assign rows to folds
##     rmses <- sapply(folds, FUN = function(fold) {
##       trn <- aq[-fold, ]  # train folds
##       tst <- aq[fold, ]   # test fold
##       ct <- party::ctree(Ozone ~ ., data = trn,
##                          control = party::ctree_control(mincriterion = 1 - alphas[i]))
##       sqrt(mean((predict(ct, newdata = tst) - tst$Ozone) ^ 2))  # RMSE
##     })
##     c("alpha" = alphas[i], "rmse" = mean(rmses))
##   })
##   out <- cbind(out, res)
## }
## plot(t(out), las = 1, type = "p", col = 6)
## lines(1:99 / 100, tapply(out["rmse", ], INDEX = out["alpha", ], FUN = mean),
##       lwd = 2)

wine <- treemisc::wine
reds <- wine[wine$type == "red", ]  # reds only
rm(wine)  # remove from global environment
reds$type <- NULL  # remove column
reds$quality <- as.ordered(reds$quality)  # coerce to ordinal
head(reds$quality)  # print first few quality scores

(reds.cit <- ctree(quality ~ ., data = reds))

p <- predict(reds.cit, newdata = reds)  # fitted values
table(predicted = p, observed = reds$quality)  # contingency table

## set.seed(2023)  # for reproducibility
## (vi <- varimp(reds.cit, nperm = 100))  # variable importance scores
## dotchart(vi, pch = 19, xlab = "Variable importance")  # Figure 3.8

par(
  mar = c(4, 8, 0.1, 0.1), 
  cex.lab = 0.95, 
  cex.axis = 0.8,  # was 0.9
  mgp = c(2, 0.7, 0), 
  tcl = -0.3, 
  las = 1
)
set.seed(2023)  # for reproducibility
(vi <- varimp(reds.cit, nperm = 100))  # variable importance scores
dotchart(vi, pch = 19, xlab = "Variable importance")  # Figure 3.8

library(ggplot2)

pfun <- function(object, newdata) {  # prediction wrapper
  mean(as.integer(predict(object, newdata = newdata)))
}
xvars <- names(vi[1L:3L])
pds <- lapply(xvars, FUN = function(xvar) {
  pd <- pdp::partial(reds.cit, pred.var = xvar, pred.fun = pfun)
  pd$yhat.id <- NULL  # not needed
  pd
})
pdps <- lapply(pds, FUN = function(pd) {
  ggplot(pd, aes(.data[[names(pd)[1L]]], .data[[names(pd)[2L]]])) +
    geom_line() +
    theme_bw() +
    ylab("Partial dependence")
})
gridExtra::grid.arrange(grobs = pdps, nrow = 1)

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

library(coin)

independence_test(Surv(time, status) ~ bili, data = pbc2,
                  teststat = "quadratic")

# Our `gi.test()` function from earlier should also work
lr.scores <- coin::logrank_trafo(Surv(pbc2$time, pbc2$status))
gi.test(pbc2$bili, y = lr.scores)

(pbc2.cit <- partykit::ctree(Surv(time, status) ~ ., data = pbc2))

plot(pbc2.cit)  # Figure 3.10
