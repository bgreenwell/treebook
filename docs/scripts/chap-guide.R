library(ggplot2)
library(partykit)
library(rpart)
library(treemisc)

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

# Load the credit card default data
credit <- read.csv("../data/credit.csv", stringsAsFactors = FALSE)

guide.chisq.test <- function(x, y) {
  y <- as.factor(sign(y - mean(y)))  # discretize response
  if (is.numeric(x)) {  # discretize numeric features 
    bins <- quantile(x, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)
    bins <- c(-Inf, bins, Inf)
    x <- as.factor(findInterval(x, vec = bins))  # quartiles
  }
  tab <- table(y, x)  # form contingency table
  if (any(row.sums <- rowSums(tab) == 0)) {  # check rows
    tab <- tab[-which(row.sums == 0), ]  # omit zero margin totals
  }
  if (any(col.sums <- colSums(tab) == 0)) {  # check columns
    tab <- tab[, -which(col.sums == 0)]  # omit zero margin totals
  }
  chisq.test(tab)$p.value  # p-value from chi-squared test
}

aq <- airquality[!is.na(airquality$Ozone), ]
pvals <- sapply(setdiff(names(aq), "Ozone"), FUN = function(x) {
  guide.chisq.test(aq[[x]], y = aq[["Ozone"]])
})
p.adjust(pvals, method = "bonferroni")  # Bonferroni adjusted p-values

ames <- as.data.frame(AmesHousing::make_ames())
set.seed(2101)  # for reproducibility
trn.id <- sample.int(nrow(ames), size = floor(0.7 * nrow(ames)))
ames.trn <- ames[trn.id, ]
ames.tst <- ames[-trn.id, ]
ames.cit <- party::ctree(Sale_Price ~ ., data = ames.trn)
# sqrt(mean((predict(ames.cit, newdata = ames.tst) - ames.tst$Sale_Price) ^ 2))

# Load the Palmer penguins data
penguins <- as.data.frame(palmerpenguins::penguins)
penguins <- subset(penguins, select = species:bill_depth_mm)
penguins <- subset(penguins, select = -island)
penguins <- na.omit(penguins)

# Plot raw data
cols <- palette.colors(palette = "Okabe-Ito", alpha = 0.5)[2:4][penguins$species]
plot(bill_length_mm ~ bill_depth_mm, data = penguins, col = cols, las = 1,
     pch = c(15, 17, 19)[penguins$species])
legend("topleft", legend = c("Adelie", "Chinstrap", "Gentoo"), 
       col = palette.colors(palette = "Okabe-Ito", alpha = 0.5)[2:4],
       pch = c(15, 17, 19), bty = "n", cex = 0.7)

library(gbm)
library(randomForest)

# Set up plotting window
par(mar = c(4, 4, 2, 0.1), cex.lab = 0.95, cex.axis = 0.8, 
    mgp = c(2, 0.7, 0), tcl = -0.3, las = 1, mfrow = c(3, 2))

# GUIDE
plot(bill_length_mm ~ bill_depth_mm, data = penguins, col = cols, las = 1,
     pch = c(15, 17, 19)[penguins$species], main = "GUIDE (linear splits)")
abline(0.98814248, b = 2.3814716, lty = 1)
x <- (0.98814248 + 6.8259055 / 0.20643636) / (1 / 0.20643636 - 2.3814716)
xx <- seq(from = x, to = 23, length = 100)
lines(xx, y = -6.8259055 / 0.20643636 + xx / 0.20643636)
legend("topleft", legend = c("Adelie", "Chinstrap", "Gentoo"), 
       col = palette.colors(palette = "Okabe-Ito", alpha = 0.5)[2:4],
       pch = c(15, 17, 19), bty = "n", cex = 0.7)

# LDA
penguins_lda <- MASS::lda(species ~ ., data = penguins)
plot(bill_length_mm ~ bill_depth_mm, data = penguins, col = cols, las = 1,
     pch = c(15, 17, 19)[penguins$species], main = "LDA")
legend("topleft", legend = c("Adelie", "Chinstrap", "Gentoo"), 
       col = palette.colors(palette = "Okabe-Ito", alpha = 0.5)[2:4],
       pch = c(15, 17, 19), bty = "n", cex = 0.7)
decision_boundary(penguins_lda, train = penguins, y = "species", 
                  x1 = "bill_depth_mm", x2 = "bill_length_mm", 
                  grid.resolution = 500)

# CART
set.seed(928)  # for reproducibility
penguins_rpart <- prune_se(rpart(species ~ ., data = penguins, cp = 0), se = 1)
plot(bill_length_mm ~ bill_depth_mm, data = penguins, col = cols, las = 1,
     pch = c(15, 17, 19)[penguins$species], main = "CART")
legend("topleft", legend = c("Adelie", "Chinstrap", "Gentoo"), 
       col = palette.colors(palette = "Okabe-Ito", alpha = 0.5)[2:4],
       pch = c(15, 17, 19), bty = "n", cex = 0.7)
decision_boundary(penguins_rpart, train = penguins, y = "species", 
                  x1 = "bill_depth_mm", x2 = "bill_length_mm", 
                  grid.resolution = 500)

# CTree
penguins_ctree <- ctree(species ~ ., data = penguins)
plot(bill_length_mm ~ bill_depth_mm, data = penguins, col = cols, las = 1,
     pch = c(15, 17, 19)[penguins$species], main = "CTree")
legend("topleft", legend = c("Adelie", "Chinstrap", "Gentoo"), 
       col = palette.colors(palette = "Okabe-Ito", alpha = 0.5)[2:4],
       pch = c(15, 17, 19), bty = "n", cex = 0.7)
decision_boundary(penguins_ctree, train = penguins, y = "species", 
                  x1 = "bill_depth_mm", x2 = "bill_length_mm", 
                  grid.resolution = 500)

# RF
set.seed(1434)  # for reproducibility
penguins_rf <- randomForest(species ~ ., data = penguins, ntree = 1000)
plot(bill_length_mm ~ bill_depth_mm, data = penguins, col = cols, las = 1,
     pch = c(15, 17, 19)[penguins$species], main = "Random forest")
legend("topleft", legend = c("Adelie", "Chinstrap", "Gentoo"), 
       col = palette.colors(palette = "Okabe-Ito", alpha = 0.5)[2:4],
       pch = c(15, 17, 19), bty = "n", cex = 0.7)
pfun <- function(object, newdata) {
  predict(object, newdata = newdata, type = "class")
}
decision_boundary(penguins_rf, train = penguins, y = "species", 
                  x1 = "bill_depth_mm", x2 = "bill_length_mm", 
                  pfun = pfun, grid.resolution = 500)

# GBM
set.seed(1439)  # for reproducibility
penguins_gbm <- gbm(species ~ ., data = penguins, n.trees = 999,
                    distribution = "multinomial",
                    interaction.depth = 2, shrinkage = 0.01, cv.folds = 5)
best.iter <- gbm.perf(penguins_gbm, plot.it = FALSE, method = "cv")
plot(bill_length_mm ~ bill_depth_mm, data = penguins, col = cols, las = 1,
     pch = c(15, 17, 19)[penguins$species], main = "Gradient boosted trees")
legend("topleft", legend = c("Adelie", "Chinstrap", "Gentoo"), 
       col = palette.colors(palette = "Okabe-Ito", alpha = 0.5)[2:4],
       pch = c(15, 17, 19), bty = "n", cex = 0.7)
pfun <- function(object, newdata) {
  z <- c("Adelie", "Chinstrap", "Gentoo")
  p <- predict(object, newdata = newdata, type = "response", 
               n.trees = best.iter)[, , 1]
  apply(p, MARGIN = 1, FUN = function(x) z[which.max(x)])
}
decision_boundary(penguins_gbm, train = penguins, y = "species", 
                  x1 = "bill_depth_mm", x2 = "bill_length_mm", 
                  pfun = pfun, grid.resolution = 500)

ames.vi <- read.table("../guide-v38.0/ames_vi/ames_vi.txt", header = TRUE)
ggplot(ames.vi, aes(x = Score, y = reorder(Variable, Score), color = Type)) +
  geom_point(size = 1) +
  ylab("") +
  scale_oi(3) +
  theme(text = element_text(size = 10), legend.position = c(0.9, 0.1))

# Download and read in the credit default data from the UCI ML repo
tf <- tempfile(fileext = ".xls")
url <- paste0("https://archive.ics.uci.edu/ml/",  # sigh, long URLs...
              "machine-learning-databases/",
              "00350/default%20of%20credit%20card%20clients.xls")
download.file(url, destfile = tf)
credit <- as.data.frame(readxl::read_xls(tf, skip = 1))

# Clean up column names a bit
names(credit) <- tolower(names(credit))
names(credit)[names(credit) == "default payment next month"] <- 
  "default"

str(credit)  # compactly display structure of the data frame

# Remove ID column
credit$id <- NULL

# Clean up categorical features
credit$sex <- ifelse(credit$sex == 1, yes = "male", no = "female")
credit$education <- ifelse(
  test = credit$education == 1, 
  yes = "graduate school", 
  no = ifelse(
    test = credit$education == 2, 
    yes = "university",
    no = ifelse(
      test = credit$education == 3,
      yes = "high school",
      no = "other"
    )
  )
)
credit$marriage <- ifelse(
  test = credit$marriage == 1, 
  yes = "married",
  no = ifelse (
    test = credit$marriage == 2,
    yes = "single",
    no = "other"
  )
)
credit$default <- ifelse(credit$default == 1, yes = "yes", no = "no")

# Coerce character columns to factors
for (i in seq_len(ncol(credit))) {
  if (is.character(credit[[i]])) {
    credit[[i]] <- as.factor(credit[[i]])
  }
}

## set.seed(1342)  # for reproducibility
## trn.ids <- sample(nrow(credit), size = 0.7 * nrow(credit),
##                   replace = FALSE)
## credit.trn <- credit[trn.ids, ]
## credit.tst <- credit[-trn.ids, ]

## treemisc::guide_setup(credit.trn, path = "guide-v38.0/credit",
##                       dv = "default", file.name = "credit",
##                       verbose = TRUE)
## 
## #> Writing data file to guide-v38.0/credit/credit.txt...
## #> Writing description file to guide-v38.0/credit/credit_desc.txt...

cat(readLines("../guide-v38.0/credit/credit_desc.txt"), sep = "\n")

cat(readLines("../guide-v38.0/credit/credit_in.txt"), sep = "\n")

cat(readLines("../guide-v38.0/credit_loss/credit_loss.txt"), sep = "\n")
