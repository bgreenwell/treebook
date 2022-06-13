import numpy as np
import pandas as pd
import scipy.stats

from ngboost import NGBRegressor
from ngboost.distns import Normal


# Read in ALS data
url = "https://web.stanford.edu/~hastie/CASI_files/DATA/ALS.txt"
als = pd.read_csv(url, sep =" ")

# Split into train/test sets
als_trn = als[als["testset"] == False]
als_tst = als[als["testset"] == True]
X_trn = als_trn.drop(["testset", "dFRS"], axis=1)  # features only
X_tst = als_tst.drop(["testset", "dFRS"], axis=1)  # features only

# Fit an NGBoost model
ngb = NGBRegressor(Dist=Normal, n_estimators=2000, learning_rate=0.01,
                   verbose_eval=0, random_state=1601)
_ = ngb.fit(X_trn, Y=als_trn["dFRS"], X_val=X_tst,
            Y_val=als_tst["dFRS"], early_stopping_rounds=5)

# Compute predictions on test set
pred = ngb.predict(X_tst)

# Mean square error (test set)
np.mean(np.square(als_tst["dFRS"].values - pred))

# Get estimated distribution parameters for first test observation
dist = ngb.pred_dist(X_tst.head(1)).params
dist

scipy.stats.norm(dist["loc"][0], scale=dist["scale"][0]).cdf(0)
