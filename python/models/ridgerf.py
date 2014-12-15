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

np.random.seed(15122014)

joined = mergeFeaturesResponses('data/features/kmeans_features_1000c_20k.csv')
#test_features = pd.read_csv('data/features/kmeans_test_features_1000c.csv', header=None, index_col=0)

X = joined.iloc[:, :4000]
Y = joined.iloc[:, 4000:]

n_alphas = 10

alphas = np.logspace(-5, -1, n_alphas)
n_trees = np.logspace(1, 3, 50, dtype='int')
n_folds = 6
kf = KFold(len(joined), n_folds=n_folds)


results = np.zeros((len(n_trees), n_folds+1), dtype='float32')
alpha = 0.1
ridge = Ridge(alpha=alpha)
logWithTimestamp('Training ridge with alpha %f' % alpha)
ridge.fit(X, Y)
ridge_predictions = ridge.predict(X)
logWithTimestamp('Ridge regressor trained')

for i, n_trees in enumerate(n_trees):
    logWithTimestamp('Cross-validation with n_trees %d' % n_trees)
    results[i, 0] = n_trees
    rf = RandomForestRegressor(n_estimators=n_trees, max_features='sqrt', n_jobs=-1)
    j = 1
    for train, test in kf:
        logWithTimestamp('\tCV loop #%d' % j)
        rf.fit(ridge_predictions[train, :], Y.iloc[train])
        results[i, j] = np.sqrt(mean_squared_error(rf.predict(ridge_predictions[test, :]),
                                                    Y.iloc[test]))
        j += 1

savetxt('data/tidy/cross_val_ntrees.csv', results, delimiter=',', fmt='%.10f')

# results = np.zeros((n_alphas, 7), dtype='float64')
# for i, alpha in enumerate(alphas):
#     logWithTimestamp('Cross-validation with alpha %.10f' % alpha)
#     ridge = Ridge(alpha=alpha)
#     rf = RandomForestRegressor(n_estimators=100, max_features='sqrt', n_jobs=-1)
#     j = 1
#     results[i, 0] = alpha
#     for train, test in kf:
#         logWithTimestamp('\tCV loop #%d' % j)
#         ridge.fit(X.iloc[train], Y.iloc[train])
#         y_linear = ridge.predict(X.iloc[train])
#         y_test_linear = ridge.predict(X.iloc[test])
#         results[i, j] = np.sqrt(mean_squared_error(y_test_linear, Y.iloc[test]))
#         rf.fit(y_linear, Y.iloc[train])
#         results[i, j + 3] = np.sqrt(mean_squared_error(rf.predict(y_test_linear), Y.iloc[test]))
#         j += 1
#
# savetxt('data/tidy/cross_val_alpha.csv', results, delimiter=',', fmt='%.10f')

#
# # train RandomForest regressor
# logWithTimestamp('Starting RF')
# logWithTimestamp('RF fitted')
#
# # get galaxy id's
# galaxies = test_features.index.values
# # append galaxy id's to prediction
# submit = np.append(galaxies.reshape(galaxies.shape[0], 1), rf.predict(test_linear), axis=1)
#
# saveSubmission(submit, 'data/submit/testsub.csv')
logWithTimestamp('Exiting')
