__author__ = 'dudevil'


import time
import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from utils import *

np.random.seed(26081988)

joined = mergeFeaturesResponses()
test_features = pd.read_csv('data/features/kmeans_test_features_1000c.csv', header=None, index_col=0)

X = joined.iloc[:, :4000]
Y = joined.iloc[:, 4000:]

# train ridge regression
logWithTimestamp('Fitting Ridge')
ridge = Ridge(alpha=1.0)
ridge.fit(X, Y)
logWithTimestamp('Ridge fitted')

# prepare ridge output for the random forest
y_linear = ridge.predict(X)
test_linear = ridge.predict(test_features)

# train RandomForest regressor
logWithTimestamp('Starting RF')
rf = RandomForestRegressor(n_estimators=100, max_features='sqrt', n_jobs=-1)
rf.fit(y_linear, Y)
logWithTimestamp('RF fitted')

# get galaxy id's
galaxies = test_features.index.values
# append galaxy id's to prediction
submit = np.append(galaxies.reshape(galaxies.shape[0], 1), rf.predict(test_linear), axis=1)

saveSubmission(submit, 'data/submit/testsub.csv')
logWithTimestamp('Exiting')
