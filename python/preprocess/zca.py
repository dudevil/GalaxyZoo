__author__ = 'dudevil'
# mostly taken from: https://gist.github.com/duschendestroyer/5170087
# but assumes patches are in rows

import numpy as np
from scipy import linalg
from sklearn.utils import array2d, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator


class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=10**-2, copy=False, n_components=None):
        self.regularization = regularization
        self.copy = copy
        self.n_components = n_components

    def fit(self, X, y=None):
        X = array2d(X)
        X = as_float_array(X, copy = self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        sigma = np.dot(X.T, X) / X.shape[1]
        U, S, V = linalg.svd(sigma)
        if self.n_components:
            U = U[:self.n_components]
        self.explained_variance_ = (S ** 2) / X.shape[0]
        self.explained_variance_ratio_ = (self.explained_variance_ /
                                     self.explained_variance_.sum())

        tmp = np.dot(U, np.diag(1/np.sqrt(S+self.regularization)))
        self.components_ = np.dot(tmp, U.T)
        #if self.n_components:
        #    self.components_ = self.components_[:self.n_components]
        return self

    def transform(self, X):
        X = array2d(X)
        if self.mean_ is not None:
            X -= self.mean_
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed

    def inverse_transform(self, X):
        X = array2d(X)
        X_original = np.dot(X, self.components_)
        return X_original + self.mean_