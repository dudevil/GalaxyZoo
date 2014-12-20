__author__ = 'dudevil'


import pandas as pd
from utils import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

features = pd.read_csv('data/features/kmeans_features_1000c_20k_nonem.csv', index_col=0, header=None)

centroids = pd.read_csv('data/models/centroids_61578_1000.csv', header=None, index_col=False)

def show_centroid(c):
    plt.imshow(c.reshape(8,8), cmap=cm.Greys, interpolation='none')
    plt.show()