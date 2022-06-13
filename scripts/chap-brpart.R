library(ggplot2)
library(rpart)
library(rpart.plot)
library(treemisc)

# Set the plotting theme for ggplot2-based figures
theme_set(theme_bw())

# Colorblind-friendly palette
oi.cols <- unname(palette.colors(8, palette = "Okabe-Ito"))

# Okabe-Ito color scale for ggplot2-based figures
scale_oi <- function(n = 2, fill = FALSE, alpha = 1, ...) {
  okabe.ito <- palette.colors(n, palette = "Okabe-Ito", alpha = alpha)
  if (isFALSE(fill)) {
    scale_colour_manual(values = unname(okabe.ito), ...)
  } else {
    scale_fill_manual(values = unname(okabe.ito), ...)
  }
}

# Classification example
banknote <- treemisc::banknote
p1 <- ggplot(banknote, aes(x = top, y = bottom, 
                           shape = as.factor(y), color = as.factor(y))) +
  geom_point(alpha = 0.7, size = 2) +
  geom_segment(aes(x = 11.15, y = 9.55, xend = 11.15, yend = 6.63625), alpha = 0.5, color = "black") +
  geom_segment(aes(x = 7, y = 9.55, xend = 13, yend = 9.55), alpha = 0.5, color = "black") +
  # geom_hline(yintercept = 9.55, color = "black") +
  annotate("text", x = 9, y = 11.5, label = "Region 1", size = 3) +
  annotate("text", x = 9, y = 7.5, label = "Region 2", size = 3) +
  annotate("text", x = 11.8, y = 8.5, label = "Region 3", size = 3) +
  xlab("Length of top edge (mm)") +
  ylab("Length of bottom edge (mm)") +
  ggtitle("Classification problem") +
  scale_color_viridis_d(name = "", breaks = c(0, 1), 
                        labels = c("Genuine", "Counterfeit")) +
  scale_shape_discrete(name = "", breaks = c(0, 1), 
                       labels = c("Genuine", "Counterfeit")) +
  theme(legend.position = "none") +
  # theme(legend.justification = c(-0.1, 1.1), legend.position = c(0, 1),
  #       legend.key = element_rect(colour = "transparent", fill = "transparent"),
  #       legend.key.size = unit(0.4, "cm")) +
  coord_cartesian(xlim = c(7.47, 12.53), ylim = c(6.63625, 12.98875), 
                  expand = FALSE)

# Regression example
# p2 <- ggplot(airquality, aes(Temp, Wind, z = Ozone)) + 
#   stat_summary_hex(bins = 25) +
#   scale_fill_viridis_c(name = "Ozone (ppb)") +
p2 <- ggplot(airquality, aes(Temp, Wind, color = Ozone)) + 
  geom_point(size = 2) +
  scale_color_viridis_c(name = "Ozone (ppb)") +
  # geom_vline(xintercept = 82.5) +
  geom_segment(aes(x = 82.5, xend = 82.5, y = -1, yend = 25), alpha = 0.5, color = "black") +
  geom_segment(aes(x = 50, xend = 82.5, y = 7.15, yend = 7.15), alpha = 0.5, color = "black") +
  annotate("text", x = 67, y = 19, label = "Region 1", size = 3) +
  annotate("text", x = 67, y = 4, label = "Region 2", size = 3) +
  annotate("text", x = 92, y = 17, label = "Region 3", size = 3) +
  xlab("Temperature (degrees F)") +
  ylab("Wind speed (mph)") +
  ggtitle("Regression problem") +
  theme(legend.position = "none") +
  # theme(legend.position = c(0.8, 0.8), 
  #       legend.key = element_rect(colour = "transparent", fill = "transparent"),
  #       legend.key.size = unit(0.4, "cm")) +
  coord_cartesian(xlim = c(55.391, 99.589), ylim = c(1.358548, 21.630471), 
                  expand = FALSE)

# Display plots side by side
gridExtra::grid.arrange(p1, p2, nrow = 1)

head(bn <- treemisc::banknote)  # load and peek at data

bn2 <- treemisc::banknote
bn2$y <- ifelse(bn2$y == 0, "Genuine", "Counterfeit")
tree <- rpart(y ~ bottom + top, data = bn2, method = "class", cp = 0)
tree_diagram(tree)

ggplot(data.frame(x = c(0, 1)), aes(x)) +
  # stat_function(fun = function(x) 1 - pmax(x, 1 - x), color = set1[1L]) +
  stat_function(fun = function(x) 2 * x * (1 - x), color = oi.cols[1]) +
  stat_function(fun = function(x) ifelse(x %in% c(0, 1), 0, 
                                         -x * log(x) - (1 - x) * log(1 - x)),
                color = oi.cols[2]) +
  labs(x = "Expected proportion of successes (p)", y = "Node impurity") +
  # annotate("text", x = 0.5, y = 0.3, label = "Misclassification error",
  #          color = set1[1L], size = 4) +
  annotate("text", x = 0.5, y = 0.55, label = "Gini index",
           color = oi.cols[1], size = 4) +
  annotate("text", x = 0.73, y = 0.7, label = "Cross entropy",
           color = oi.cols[2], size = 4)

gini <- function(y) {  # y should be coded as 0/1
  p <- mean(y)  # proportion of successes (or 1s)
  2 * p * (1 - p)  # Gini index
}

splits <- function(node, x, y, n) {  # y should be coded as 0/1
  xvals <- sort(unique(node[[x]]))  # sorted, unique values
  xvals <- xvals[-length(xvals)] + diff(xvals) / 2  # midpoints
  res <- matrix(nrow = length(xvals), ncol = 2)  # to store results
  colnames(res) <- c("cutpoint", "gain")
  for (i in seq_along(xvals)) {  # loop through each midpoint
    left <- node[node[[x]] >= xvals[i], y, drop = TRUE]  # left child
    right <- node[node[[x]] < xvals[i], y, drop = TRUE]  # right child
    p <- c(nrow(node), length(left), length(right)) / n  # proportions
    gain <- p[1L] * gini(node[[y]]) -  # Equation (2.3)
      p[2L] * gini(left) - p[3L] * gini(right)
    res[i, ] <- c(xvals[i], gain)  # store split point and gain
  }
  res  # return matrix of results
}

res <- splits(bn, x = "bottom", y = "y", n = nrow(bn))
head(res, n = 5)  # peek at first five rows
plot(res, type = "b", col = 2, las = 1,  
     xlab = "Split value for bottom edge length (mm)", 
     ylab = "Gain")  # Figure 2.5

res[which.max(res[, "gain"]), ]  # extract row with maximum gain

find_best_split <- function(node, x, y, n) {
  res <- matrix(nrow = length(x), ncol = 2)  # to store output
  rownames(res) <- x  # set row names to feature names
  colnames(res) <- c("cutpoint", "gain")  # column names
  for (xname in x) {  # loop through each feature
    # Compute optimal split
    cutpoints <- splits(node, x = xname, y = y, n = n)
    res[xname, ] <- cutpoints[which.max(cutpoints[, "gain"]), ]
  }
  res[which.max(res[, "gain"]), , drop = FALSE]
}

features <- c("top", "bottom")  # feature names
find_best_split(bn, x = features, y = "y", n = nrow(bn))

left <- bn[bn$bottom >= 9.55, ]  # left child node
right <- bn[bn$bottom < 9.55, ]  # right child node

table(left$y)  # class distribution in left child node
table(right$y)  # class distribution in right child node

find_best_split(right, x = features, y = "y", n = nrow(bn))

data(attrition, package = "modeldata")

# Fit trees with different priors
set.seed(904)  # for reproducibility
tree1 <- rpart(Attrition ~ OverTime + MonthlyIncome, data = attrition, 
               maxdepth = 2, cp = 0)
# tree2 <- rpart(Attrition ~ OverTime + MonthlyIncome, data = attrition, 
#                maxdepth = 2, cp = 0, parms = list("prior" = c(0.4, 0.6)))
# loss <- matrix(c(0, 8, 1, 0), nrow = 2)
# tree2 <- rpart(Attrition ~ OverTime + MonthlyIncome, data = attrition, 
#                maxdepth = 2, cp = 0, parms = list("loss" = loss))
tree2 <- rpart(Attrition ~ OverTime + MonthlyIncome, data = attrition,
               maxdepth = 2, cp = 0,
               parms = list("prior" = c(1 - 0.6059444, 0.6059444)))

# Plot tree diagrams
par(mfrow = c(1, 2))
tree_diagram(tree1)
tree_diagram(tree2)

left <- attrition[attrition$OverTime == "No", ]  # left child
right <- attrition[attrition$OverTime == "Yes", ]  # right child
table(attrition$Attrition)  # class frequencies
table(left$Attrition)  # class frequencies (left node)
table(right$Attrition)  # class frequencies (right node)

par(mfrow = c(1, 2))
tree_diagram(rpart(Ozone ~ ., data = airquality, maxdepth = 1))
plot(Ozone ~ Temp, data = airquality, las = 1, col = adjustcolor("black", alpha.f = 0.5))
segments(40, 26.54430, 82.5, 26.54430, lwd = 2, col = 3)
segments(82.5, 75.40541, 110, 75.40541, lwd = 2, col = 3)
abline(v = 82.5, lty = "dashed")

sse <- function(y, na.rm = TRUE) {
  sum((y - mean(y, na.rm = na.rm)) ^ 2, na.rm = na.rm)
}

splits.sse <- function(node, x, y) {
  xvals <- sort(unique(node[[x]]))  # sorted, unique values
  xvals<- xvals[-length(xvals)] + diff(xvals) / 2  # midpoints
  res <- matrix(nrow = length(xvals), ncol = 2)
  colnames(res) <- c("cutpoint", "gain")
  for (i in seq_along(xvals)) {  # loop through each feature
    left <- node[node[[x]] >= xvals[i], y, drop = TRUE]  # left
    right <- node[node[[x]] < xvals[i], y, drop = TRUE]  # right
    gain <- sse(node[[y]]) - sse(left) - sse(right)  # Equation (2.6)
    res[i, ] <- c(xvals[i], gain)  # store cutpoint and associated gain
  }
  res  # return matrix of results
}

# Find optimal split for `Temp`
aq <- airquality[!is.na(airquality$Ozone), ]
res <- splits.sse(aq, x = "Temp", y = "Ozone")
res[which.max(res[, "gain"]), ]

# Plot results
res[, "gain"] <- res[, "gain"] / 1000  # rescale for plotting
plot(res, type = "b", col = 2, las = 1, 
     xlab = "Temperature split value (degrees Fahrenheit)", 
     ylab = "Gain/1000")
abline(v = 82.5, lty = 2, col = 2)

features <- c("Solar.R", "Wind", "Temp", "Month", "Day")
sapply(features, FUN = function(xname) {
  res <- splits.sse(aq, x = xname, y = "Ozone")
  res[which.max(res[, "gain"]), ]
})

set.seed(1143)  # for reproducibility
tree <- rpart(Ozone ~ Temp + Wind, data = airquality, cp = 0)
tree2 <- treemisc::prune_se(tree, se = 1)
x <- seq(from = 56, to = 97, length = 51)
y <- seq(from = 1.7, to = 20.7, length = 51)
z <- outer(x, y, FUN = function(x, y) {
  predict(tree2, newdata = data.frame(Temp = x, Wind = y))
})

# Display plots side by side
par(mfrow = c(1, 2), mar = c(0, 2, 0, 0) + 0.2)
par(fig= c (0, 1/3, 0, 1))
tree_diagram(tree2)
par(fig = c(1/3, 1, 0, 1), new = TRUE)
plot3D::scatter3D(airquality$Temp, airquality$Wind, airquality$Ozone, pch = 19, 
                  cex = 1, theta = 210, phi = 20, bty = "b2", alpha = 1,
                  col = viridisLite::viridis(100, option = "D"), colkey = FALSE,
                  xlab = "Temp", ylab = "Wind", zlab = "Ozone",  
                  ticktype = "detailed", cex.axis = 0.8, cex.lab = 0.8,
                  surf = list(x = x, y = y, z = z, shade = FALSE, facets = NA, 
                              col = grey(0.6)))

## split.cat <- function(x) {
##   lev <- levels(x[drop = TRUE])
##   if(length(lev) == 2) {
##     splitpoint <- lev[1]
##   } else {
##     comb <- do.call("c", lapply(1:(length(lev) - 2),
##       function(x) combn(lev, x, simplify = FALSE)))
##     comb
##   }
## }

tree_diagram(rpart(Edibility ~ ., data = mushroom, maxdepth = 2))

m <- treemisc::mushroom  # load mushroom data
m$veil.type <- NULL  # remove useless feature
m$Edibility <- ifelse(m$Edibility == "Poison", 1, 0)
m2 <- m  # make a copy of the original data

table(m2$veil.color)

ordinalize <- function(x, y) {  # convert nominal to ordered
  map <- tapply(y, INDEX = x, FUN = mean)
  list("mapping" = map, "encoded" = map[x])
}

# Check which numeric values `veil.color` gets mapped to
ordinalize(m2$veil.color, m2$Edibility)$map

xnames <- setdiff(names(m2), "Edibility")
for (xname in xnames) {  # mean/target encode each feature
  m2[[xname]] <- ordinalize(m2[[xname]], y = m2[["Edibility"]])$encoded
}

# Take a peek at the re-encoded data
m2[1L:8L, 1L:5L]

find_best_split(m2, x = xnames, y = "Edibility", n = nrow(m2))

# Summarize split
left <- m[m2$odor >= 0.5170068, ]
right <- m[m2$odor < 0.5170068, ]

table(left$Edibility)  # non-pure node
table(right$Edibility)  # pure node

# Ordinalize left child node and find next best split
right.ord <- right
for (xname in xnames) {  # mean/target encode each feature
  right.ord[[xname]] <- 
    ordinalize(right.ord[[xname]], 
               y = right.ord[["Edibility"]])$encoded
}

# Find best split in newly "ordinalized" predictors
find_best_split(right.ord, x = xnames, y = "Edibility", n = nrow(m2))

sort(ordinalize(m$odor, m$Edibility)$map)
sort(ordinalize(right[["spore.print.color"]], 
                y = right[["Edibility"]])$map)

aq <- airquality
set.seed(2053) 
for (i in 1:10) {
  aq[[paste0("cat", i)]] <- sample(letters, size = nrow(aq), replace = TRUE)
}
set.seed(2056)  # for reproducibility
tree <- rpart(Ozone ~ ., data = aq)
# tree <- treemisc::prune_se(rpart(Ozone ~ ., data = aq, cp = 0))
tree_diagram(tree)

# Generate training data
n <- 500
set.seed(1503)  # for reproducibility
trn <- data.frame(x = runif(n, min = 0, max = 2 * pi))
trn$y <- sin(trn$x) + rnorm(n, sd = 0.3)

# Generate test data
set.seed(1943)  # for reproducibility
tst <- data.frame(x = runif(n, min = 0, max = 2 * pi))
tst$y <- sin(tst$x) + rnorm(n, sd = 0.3)

# Scatterplot of training data with true response function
ggplot(trn, aes(x, y)) +
  geom_point(shape = 1, alpha = 0.3) +
  stat_function(fun = sin, colour = "black")

# Generate grid for prediction
xx <- seq(from = 0, to = 2 * pi, length = 500)

# Plot predictions from a saturated tree
tree0 <- rpart(y ~ x, data = trn, cp = 0, minbucket = 2)
yhat0 <- predict(tree0, newdata = data.frame(x = xx))
p0 <- ggplot(trn, aes(x, y)) +
  geom_point(shape = 1, alpha = 0.3) +
  stat_function(fun = sin, colour = "black") +
  geom_line(data = data.frame(x = xx, y = yhat0), color = 2) +
  ggtitle("Overgrown decision tree")

# Plot predicttions from a decision stump
tree1 <- rpart(y ~ x, data = trn, maxdepth = 1)
yhat1 <- predict(tree1, newdata = data.frame(x = xx))
p1 <- ggplot(trn, aes(x, y)) +
  geom_point(shape = 1, alpha = 0.3) +
  stat_function(fun = sin, colour = "black") +
  geom_line(data = data.frame(x = xx, y = yhat1), color = 2) +
  ggtitle("Undergrown decision tree")

# Side by side plots
gridExtra::grid.arrange(p0, p1, nrow = 1)

# Choose optimal subtree based on internal 10-fold cross-validation
cp <- tree0$cptable[, "CP"]
# cp_cv <- tree0$cptable[which.min(tree0$cptable[, "xerror"]), "CP"]
cp_cv <- prune_se(tree0, prune = FALSE)
# tree_cv <- prune(tree0, cp = cp_cv)
tree_cv <- prune_se(tree0)

# Sequence of pruned trees
trees <- lapply(cp, FUN = function(x) {
  # rpart(y ~ x, data = d, cp = x)
  prune(tree0, cp = x)
})

# Plot sequence of prune trees (first 15 splits + most complex tree)
par(mfrow = c(4, 4), mar = c(0, 0, 0, 0))
for (i in c(length(cp), 16L:2L)) {
  if (i == 12) {
    # if (trees[[i]]$control$cp == cp_cv) {
    plot_tree(trees[[i]], uniform = TRUE, compress = TRUE, 
              branch.col = 3, branch.lwd = 2)
  } else {
    plot_tree(trees[[i]], uniform = TRUE, compress = TRUE,
              branch.col = "grey35")
  }
}

# Compute relative error for the test set
tst_err <- t(sapply(trees, FUN = function(x) {
  pred <- predict(x, newdata = tst)
  # rmse <- sqrt(mean((pred - tst$y) ^ 2))
  err <- if (is_root(x)) {
    1
  } else {
    1 - cor(pred, tst$y) ^ 2
  }
  c("nsplit" = nsplits(x), "err" = err)
}))

# Plot test set and 10-fold cross-validation results
palette("Okabe-Ito")
plot(tst_err, type = "l", col = 1, ylim = c(0, 1),
     xlab = "Number of splits", ylab = "Relative error", las = 1)
lines(tree0$cptable[, c("nsplit", "xerror")], type = "l", col = 2)
legend("topright", legend = c("Test error", "10-fold CV"),
       inset = 0.02, lty = 1, col = 1:2)
abline(v = tst_err[which.min(tst_err[, "err"]), "nsplit"], lty = 2, col = 1)
abline(v = tree0$cptable[tree0$cptable[, "CP"] == cp_cv, "nsplit"], lty = 2, 
       col = 2)
palette("default")

# Fit a decision tree to the mushroom with 3 splits
set.seed(841)  # for reproducibility
tree <- rpart(Edibility ~ ., data = mushroom, cp = 0.005)
tree_diagram(tree, prob = FALSE)

tree2 <- prune(tree, cp = 0.01)
tree3 <- prune(tree, cp = 0.02)
# tree4 <- prune(tree, cp = 1)
par(mfrow = c(1, 2))
tree_diagram(tree2, prob = FALSE, faclen = 3, cex = 0.8)
tree_diagram(tree3, prob = FALSE, faclen = 3, cex = 0.8)

tree_diagram(rpart(y ~ ., data = banknote, method = "class"))

bn2 <- treemisc::banknote  # load Swiss banknote data
bn2$y <- ifelse(bn2$diagonal >= 140.65, 1, 0)  # new target
bn2$diagonal <- NULL  # remove column
features <- c("length", "left", "right", "bottom", "top")
res <- sapply(features, FUN = function(feature) {
  find_best_split(bn2, x = feature, y = "y", n = nrow(bn2))
})
rownames(res) <- c("cutpoint", "gain")
res[, order(res["gain", ], decreasing = TRUE)]

library(rpart)

# Load the Swiss banknote data and re-encode the response
bn <- banknote
bn$y <- ifelse(bn$y == 0, "Genuine", "Counterfeit")

# Fit a CART-like tree using top and bottom as the only features
(bn.tree <- rpart(y ~ top + bottom, data = bn))

summary(bn.tree)  # print more verbose tree summary

summary(rpart(y ~ ., data = bn, method = "class"))

## treemisc::tree_diagram(bn.tree)

tree <- rpart(y ~ ., data = bn, maxdepth = 1, method = "class")
bn2 <- bn
bn2$y <- ifelse(bn2$diagonal >= 140.65, "Counterfeit", "Genuine")
bn2$diagonal <- NULL
surr <- rpart(y ~ right, data = bn2, method = "class", maxdepth = 1)
par(mfrow = c(1, 2))
tree_diagram(tree, prob = FALSE)
tree_diagram(surr, prob = FALSE)

mushroom <- treemisc::mushroom

# Fit a default tree with zero penalty on tree size
set.seed(1054)  # for reproducibility
(mushroom.tree <- rpart(Edibility ~ ., data = mushroom, cp = 0))

tree_diagram(mushroom.tree)  # Figure 2.20

unlist(mushroom.tree$control)

## ctrl <- rpart.control(cp = 0, xval = 5)  # can also be a names list
## tree <- rpart(Edibility ~ ., data = mushroom, control = ctrl)
## tree <- rpart(Edibility ~ ., data = mushroom, cp = 0, xval = 5)

mushroom.tree$parms

proportions(table(mushroom$Edibility))  # observed class proportions

## parms <- list("split" = "information")  # use cross-entropy split rule
## rpart(Edibility ~ ., data = mushroom, parms = parms)

## levels(mushroom$Edibility)  # inspect order of levels
## (loss <- matrix(c(0, 5, 1, 0), nrow = 2))  # loss matrix
## rpart(Edibility ~ ., data = mushroom, parms = list("loss" = loss))

mushroom.tree$variable.importance

plotcp(mushroom.tree, upper = "splits", las = 1)  # Figure 2.21
mushroom.tree$cptable  # print cross-validation results

mushroom.tree$cptable[1L:3L, "CP"] / (8124 / 3916)

tree_diagram(prune(mushroom.tree, cp = 0.1))  # Figure 2.22

ames <- as.data.frame(AmesHousing::make_ames())
ames$Sale_Price <- ames$Sale_Price / 1000  # rescale response
set.seed(2101)  # for reproducibility
trn.id <- sample.int(nrow(ames), size = floor(0.7 * nrow(ames)))
ames.trn <- ames[trn.id, ]  # training data/learning sample
ames.tst <- ames[-trn.id, ]  # test data

library(rpart)
library(treemisc)  # for prune_se() function

# Fit a regression tree with no penalty on complexity
set.seed(1547)  # for reproducibility
ames.tree <- rpart(Sale_Price ~ ., data = ames.trn, cp = 0)

rmse <- function(pred, obs) {  # computes RMSE
  sqrt(mean((pred - obs) ^ 2))
}

# Compute train RMSE
rmse(predict(ames.tree, newdata = ames.trn), obs = ames.trn$Sale_Price)

# Compute test RMSE
rmse(predict(ames.tree, newdata = ames.tst), obs = ames.tst$Sale_Price)

ames.tree.1se <- prune_se(ames.tree, se = 1)  # prune using 1-SE rule

# Train RMSE on pruned tree
rmse(predict(ames.tree.1se, newdata = ames.trn), 
     obs = ames.trn$Sale_Price)

# Test RMSE on pruned tree
rmse(predict(ames.tree.1se, newdata = ames.tst), 
     obs = ames.tst$Sale_Price)

par(mar = c(0.1, 0.1, 1, 0.1), cex.lab = 0.95,  cex.axis = 0.8, 
    mgp = c(2, 0.7, 0), tcl = -0.3, las = 1, mfrow = c(1, 2))
plot(ames.tree, main = "No pruning", uniform = TRUE)
plot(ames.tree.1se, main = "1-SE rule", uniform = TRUE)

## vi <- sort(ames.tree.1se$variable.importance, decreasing = TRUE)
## vi <- vi / sum(vi)  # scale to sum to 1
## dotchart(vi[1:10], xlab = "Variable importance", pch = 19)

par(
  mar = c(4, 8, 0.1, 0.1), 
  cex.lab = 0.95, 
  cex.axis = 0.8,  # was 0.9
  mgp = c(2, 0.7, 0), 
  tcl = -0.3, 
  las = 1
)
vi <- sort(ames.tree.1se$variable.importance, decreasing = TRUE)
vi <- vi / sum(vi)  # scale to sum to 1
dotchart(vi[1:10], xlab = "Variable importance", pch = 19)

library(ggplot2)
library(pdp)

# Compute partial dependence of predicted Sale_Price on Gr_Liv_Area
pd <- partial(ames.tree.1se, pred.var = "Gr_Liv_Area")
autoplot(pd, rug = TRUE, train = ames.trn) +  # Figure 2.25
  ylab("Partial dependence")

data(attrition, package = "modeldata")

# Fit classification trees with default priors and costs
set.seed(904)  # for reproducibility
tree1 <- rpart(Attrition ~ OverTime + MonthlyIncome, data = attrition,
               maxdepth = 2, cp = 0)

# Specify unequal misclassification costs
loss <- matrix(c(0, 8, 1, 0), nrow = 2)
tree2 <- rpart(Attrition ~ OverTime + MonthlyIncome, data = attrition,
               maxdepth = 2, cp = 0, parms = list("loss" = loss))

# Equivalent approach using altered priors
tree3 <- rpart(Attrition ~ OverTime + MonthlyIncome, data = attrition,
               maxdepth = 2, cp = 0,
               parms = list("prior" = c(1 - 0.6059444, 0.6059444)))

# Display trees side by side (Figure 2.26)
par(mfrow = c(1, 3))
tree_diagram(tree1)  # default costs and priors
tree_diagram(tree2)  # unequal costs
tree_diagram(tree3)  # altered priors

levels(attrition$Attrition)  # can be changed with relevel()
matrix(c(0, 8, 1, 0), nrow = 2)

library(rpart)

# Saturated tree with altered priors using all predictors
att.cart <- 
  rpart(Attrition ~ ., data = attrition, cp = 0, minsplit = 2,
        parms = list("prior" = c(1 - 0.6059444, 0.6059444)))

# Plot pruning results (Figure 2.27)
par(mfrow = c(2, 1))
plotcp(att.cart, upper = "splits")
(att.cart.1se <- prune_se(att.cart, se = 1))
tree_diagram(att.cart.1se, tweak = 0.8)

library(treemisc)  # for prune_se()

data(LetterRecognition, package = "mlbench")
lr <- LetterRecognition  # shorter name for brevity
set.seed(1051)  # for reproducibility
trn.ids <- sample(nrow(lr), size = 14000, replace = FALSE)
lr.trn <- lr[trn.ids, ]  # training data
lr.tst <- lr[-trn.ids, ]  # test data

set.seed(1703)  # for reproducibility
lr.cart <- rpart(lettr ~ ., data = lr.trn, cp = 0, xval = 10)
lr.cart <- prune_se(lr.cart, se = 1)  # prune using 1-SE rule

# Compute accuracy on test set
pred <- predict(lr.cart, newdata = lr.tst, type = "class")
sum(diag(table(pred, lr.tst$lettr))) / length(pred)

table(lr$lettr)

data(ltrfreqs, package = "regtools")

# Compute correct class priors
priors <- ltrfreqs$percent
priors <- priors / sum(priors)  # class priors should sum to 1
names(priors) <- ltrfreqs$ltr
priors <- priors[order(ltrfreqs$ltr)]

# Refit tree using correct priors
set.seed(1718)  # for reproducibility
lr.cart.priors <- rpart(lettr ~ ., data = lr.trn, cp = 0, 
                        parms = list(prior = priors))
lr.cart.priors <- prune_se(lr.cart.priors, se = 1)

# Sample test set to reflect correct class frequencies
ltrfreqs2 <- ltrfreqs
names(ltrfreqs2) <- c("lettr", "prior")
ltrfreqs2$prior <- ltrfreqs2$prior / sum(ltrfreqs2$prior)
temp <- merge(lr.tst, ltrfreqs2)  # merge the two data sets
set.seed(1107)  # for reproducibility
lr.tst2 <- temp[sample(nrow(temp), replace = TRUE,
                       prob = temp$prior), ]

pred2 <- predict(lr.cart, newdata = lr.tst2, type = "class")
pred3 <- predict(lr.cart.priors, newdata = lr.tst2, type = "class")
sum(diag(table(pred2, lr.tst2$lettr))) / length(pred2)
sum(diag(table(pred3, lr.tst2$lettr))) / length(pred3)

library(mlbench)
library(rpart)
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

# Fit a decision tree
set.seed(2028)  # for reproducibility
tree <- rpart(classes ~ ., data = trn, cp = 0, minsplit = 2)
tree.1se <- prune_se(tree, se = 1)

# Add decision boundary for pruned decision tree
pfun <- function(object, newdata) {
  predict(object, newdata = newdata, type = "class")
}
# decision_boundary(tree, train = trn, y = "classes", x1 = "x.1", x2 = "x.2", 
#                   pfun = pfun, lty = 2)
decision_boundary(tree.1se, train = trn, y = "classes", x1 = "x.1", x2 = "x.2", 
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
legend("topright", legend = c("CART", "Bayes"), 
       lty = 1:2, inset = 0.01, bty = "n")

library(rpart)
library(rpart.plot)

# Read in email spam data
data(spam, package = "kernlab")

set.seed(1108)  # for reproducibility
trees <- lapply(1:6, FUN = function(i) {
  id <- sample.int(nrow(spam), size = floor(0.7 * nrow(spam)))
  tree <- rpart(type ~ ., data = spam[id, ], cp = 0, model = TRUE)
  cp <- tree$cptable
  cp <- cp[cp[, "nsplit"] == 4, "CP"]
  prune(tree, cp = cp)
})

par(mfrow = c(2, 3), mar = c(4, 4, 0.1, 0.1))
for (tree in trees) {
  prp(tree)
}
