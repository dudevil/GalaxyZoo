__author__ = 'dudevil'

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator
from utils import *


class ElasticRF(BaseEstimator):

    def __init__(self,
                 alpha=10.0,
                 l1_ratio=0.00001,
                 max_iter=5000,
                 n_trees=686,
                 max_features='auto',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_leaf_nodes=None,
                 n_jobs=-1,
                 random_state=16122014):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.n_estimators = n_trees
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _check_fitted(self):
        if not hasattr(self, 'elnet') or not hasattr(self, 'rf'):
            raise AttributeError("Model has not been trained yet.")

    def fit(self, X, y):
        self.elnet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter)
        self.rf = RandomForestRegressor(n_estimators=self.n_estimators, max_features=self.max_features,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        max_leaf_nodes=self.max_leaf_nodes,
                                        n_jobs=self.n_jobs,
                                        random_state=self.random_state)
        logWithTimestamp('Fitting ElasticNet')
        self.elnet.fit(X, y)
        logWithTimestamp('Fitting RandomForest')
        self.rf.fit(self.elnet.predict(X), y)

    def predict(self, X):
        self._check_fitted()
        return self.rf.predict(self.elnet.predict(X))


joined = mergeFeaturesResponses('data/features/kmeans_features_1000c_20k.csv', n_obs=5000)
#test_features = pd.read_csv('data/features/kmeans_test_features_1000c.csv', header=None, index_col=0)

X = joined.iloc[:, :4000]
Y = joined.iloc[:, 4000:]

logWithTimestamp('Filtering by variance')
vsel = VarianceThreshold()
X = vsel.fit_transform(X)

logWithTimestamp('Cross-validating ElasticRF')
erf = ElasticRF()
erf_cv = np.sqrt(cross_val_score(erf, X, Y, n_jobs=-1, pre_dispatch=3))

logWithTimestamp('ElasticRF cv-score: %f with sd: %f' % (enet_cv.mean(), enet_cv.std()))
