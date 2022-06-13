library(ggplot2)
library(randomForest)

# Set the plotting theme
theme_set(theme_bw())

# Load the Ames housing data
ames <- as.data.frame(AmesHousing::make_ames())
ames$Sale_Price <- ames$Sale_Price / 1000  # rescale response
set.seed(4919)
id <- sample.int(nrow(ames), size = floor(0.7 * nrow(ames)))
ames.trn <- ames[id, ]
ames.tst <- ames[-id, ]

# Reproduce bagged tree ensemble from previous chapter
ames.bag <- readRDS("../data/chap-ensembles-ames-bag.rds")

rmse <- function(predicted, actual, na.rm = TRUE) {
  sqrt(mean((predicted - actual) ^ 2, na.rm = na.rm))
}
(baseline.rmse <- rmse(predict(ames.bag, newdata = ames.trn), 
                       actual = ames.trn$Sale_Price))

nperm <- 30  # number of permutation to use per feature
xnames <- names(subset(ames.trn, select = -Sale_Price))
vi <- matrix(nrow = nperm, ncol = length(xnames))
colnames(vi) <- xnames
for (j in colnames(vi)) {
  for (i in seq_len(nrow(vi))) {
    temp <- ames.trn  # temporary copy of training data
    temp[[j]] <- sample(temp[[j]])  # permute feature values
    pred <- predict(ames.bag, newdata = temp)  # score permuted data
    permuted.rmse <- rmse(pred, actual = temp$Sale_Price) ^ 2
    vi[i, j] <- permuted.rmse - baseline.rmse  # smaller is better 
  }
}

# Average VI scores across all permutations
head(vi.avg <- sort(colMeans(vi), decreasing = TRUE))

top10 <- names(vi.avg)[1L:10L]
vi <- as.data.frame(vi)  # coerce to data frame
vi <- reshape(vi, times = names(vi), timevar = "feature",
              varying = list(names(vi)), direction = "long",
              v.names = "importance")  # wide to long conversion
vi.top10 <- vi[vi$feature %in% top10, ]

# Display raw permutation importance scores (Figure 6.1)
par(mar = c(4.1, 6, 0.1, 0.1))
boxplot(importance ~ feature, data = vi.top10, las = 1, col = 2,
        cex.axis = 0.7, horizontal = TRUE,
        xlab = "Permutation importance", ylab = "")

x.grid <- quantile(ames.trn$Gr_Liv_Area, prob = 1:30 / 31)
pd <- numeric(length(x.grid))
for (i in seq_along(x.grid)) {
  temp <- ames.trn  # temporary copy of data 
  temp[["Gr_Liv_Area"]] <- x.grid[i]
  pd[i] <- mean(predict(ames.bag, newdata = temp))
}

# PD plot for above grade square footage (Figure 6.2)
plot(x.grid, pd, type = "l", xlab = "Above ground square footage",
     ylab = "Partial dependence", las = 1)
rug(x.grid)  # add rug plot to x-axis 

x1.grid <- quantile(ames.trn$Gr_Liv_Area, prob = 1:30 / 31)
x2.grid <- quantile(ames.trn$First_Flr_SF, prob = 1:30 / 31)
df1 <- expand.grid("Gr_Liv_Area" = x1.grid, 
                   "First_Flr_SF" = x2.grid)  # Cartesian product

df2 <- subset(ames.trn, select = -c(Gr_Liv_Area, First_Flr_SF))

# Perform a cross-join between the two data sets
pd <- merge(df1, df2, all = TRUE)  # Cartesian product
dim(pd)  # print dimensions

pd$yhat <- predict(ames.bag, newdata = pd)  # might take a few minutes!
pd <- aggregate(yhat ~ Gr_Liv_Area + First_Flr_SF, data = pd, 
                FUN = mean)

library(lattice)

# PD plot for above grade and first floor square footage
levelplot(yhat ~ Gr_Liv_Area * First_Flr_SF, data = pd, 
          contour = TRUE, col = "white", scales = list(tck = c(1, 0)),
          col.regions = hcl.colors(100, palette = "viridis"))

ice <- partial(ames.bag, pred.var = "Gr_Liv_Area", ice = TRUE, 
               center = TRUE, quantiles = TRUE, probs = 1:30 / 31)
set.seed(1123)  # for reproducibility
samp <- sample.int(nrow(ames.trn), size = 500)  # sample 500 homes
autoplot(ice[ice$yhat.id %in% samp, ], alpha = 0.1) + 
  ylab("Conditional expectation")

library(randomForest)

# Fit a bagged tree ensemble
set.seed(1452)  # for reproducibility
(iris.bag <- randomForest(Species ~ ., data = iris, mtry = 4))

library(pdp)
library(ggplot2)

# Prediction wrapper that returns average prediction for each class
pfun <- function(object, newdata) {
  colMeans(predict(object, newdata = newdata, type = "prob"))
}

# Partial dependence of probability for each class on petal width
p <- partial(iris.bag, pred.var = "Petal.Width", pred.fun = pfun)
ggplot(p, aes(Petal.Width, yhat, color = as.factor(yhat.id))) +
  geom_line() +
  theme(legend.title = element_blank(),
        legend.position = "top")

sample.shap <- function(f, obj, R, x, feature, X) {
  phi <- numeric(R)  # to store Shapley values
  N <- nrow(X)  # sample size
  p <- ncol(X)  # number of features
  b1 <- b2 <- x
  for (m in seq_len(R)) {
    w <- X[sample(N, size = 1), ]
    ord <- sample(names(w))  # random permutation of features
    swap <- ord[seq_len(which(ord == feature) - 1)]
    b1[swap] <- w[swap]
    b2[c(swap, feature)] <- w[c(swap, feature)]
    phi[m] <- f(obj, newdata = b1) - f(obj, newdata = b2)
  }
  mean(phi)  # return approximate feature contribution
}

X <- subset(ames.trn, select = -Sale_Price)  # features only
set.seed(2207)  # for reproducibility
sample.shap(predict, obj = ames.bag, R = 100, x = X[1, ], 
            feature = "Gr_Liv_Area", X = X)

library(fastshap)
library(ggplot2)

# Find observation with highest predicted sale price
pred <- predict(ames.bag, newdata = ames.tst)
highest <- which.max(pred)
pred[highest]

# fastshap needs to know how to compute predictions from your model
pfun <- function(object, newdata) predict(object, newdata = newdata)

# Need to supply feature columns only in fastshap::explain()
X <- subset(ames.trn, select = -Sale_Price)  # feature columns only
newx <- ames.tst[highest, names(X)]

# Compute feature contributions for observation with highest prediction
set.seed(1434)  # for reproducibility
ex <- explain(ames.bag, X = X, nsim = 100, newdata = newx, 
              pred_wrapper = pfun, adjust = TRUE)
ex[1, 1:5]  # peek at a few
autoplot(ex, type = "contribution", num_features = 10, 
         feature_values = newx)

ex <- explain(ames.bag, feature_names = "Gr_Liv_Area", X = X, 
              nsim = 50, pred_wrapper = pfun)

# Shapley dependence plot
autoplot(ex, type = "dependence", X = X, alpha = 0.3)
