__author__ = 'dudevil'

from numpy import savetxt
from pandas import read_csv
import time

_start_time = time.time()

def logWithTimestamp(msg):
    print('[%d] %s' % (int(time.time()-_start_time), msg))


def saveSubmission(array, filename='data/submit/submission_%d.csv' % time.time()):
    """
    Save array in the GalaxyZoo submission format
    :param array: array to be saved
    :param filename: destination file
    """

    header = "GalaxyID,Class1.1,Class1.2,Class1.3," \
             "Class2.1,Class2.2," \
             "Class3.1,Class3.2," \
             "Class4.1,Class4.2," \
             "Class5.1,Class5.2,Class5.3,Class5.4," \
             "Class6.1,Class6.2," \
             "Class7.1,Class7.2,Class7.3," \
             "Class8.1,Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,Class8.7," \
             "Class9.1,Class9.2,Class9.3," \
             "Class10.1,Class10.2,Class10.3," \
             "Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,Class11.6"

    if array.shape != (79975, 38):
        logWithTimestamp('WARNING: Array dimensions do not match submission dimentions')

    # format strings as 9-digit floats
    fmt = ['%.9f'] * 38
    # GalaxyID should be integer
    fmt[0] = '%d'
    savetxt(filename, array, delimiter=',', header=header, fmt=fmt, comments='')

def mergeFeaturesResponses(features='data/features/kmeans_features_1000c.csv', n_obs=None):
    """
    If features is a string, reads features dataframe from the file features, and joins
    them with the responses provided by Kaggle. Only galaxies contained in both sets are returned.

    :param features: if string csv filnemae to be read, if array - a 2D array of features,
                        first column of the array should be GalaxyID
    :param n_obs: number of features to parse from the file
    :return: DataFrame of shape [n_galaxies, n_features + n_responses]
    """
    if isinstance(features, basestring):
        features = read_csv(features, header=None, index_col=0, nrows=n_obs)
    response = read_csv('data/training_solutions_rev1.csv', header=0, index_col=0)
    logWithTimestamp('Input read')
    return features.merge(response, left_index=True, right_index=True, how='inner')