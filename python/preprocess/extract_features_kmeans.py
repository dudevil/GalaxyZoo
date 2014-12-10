#!/home/dudevil/prog/dmlabs/GalaxyZoo/python/.env/bin/python

"""
 Main module for feature extraction using K-means
"""

import os
import time
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_extraction.image import extract_patches_2d
import sklearn.cluster as cluster
from sklearn.externals import joblib
from multiprocessing import Pool, cpu_count
from memory_profiler import profile
__author__ = 'dudevil'

# The current working dir should be the project top-level directory

seed = 211114
np.random.seed(seed)
start_time = 0


def _logWithTimestamp(msg):
    print('[%d] %s' % (int(time.time()) - start_time, msg))


def crop_and_resize(image, crop_offset=137, resolution=25):
    img = image[crop_offset:-crop_offset, crop_offset:-crop_offset]
    if not isinstance(resolution, tuple):
        resolution = (resolution, resolution)
    return misc.imresize(img, resolution)


def plotImageGrid(images, image_size=(), nrow=None, ncol=None):
    """
    Helper function to plot a grid of images

    :param images: an array of images to be plotted
    :param image_size: if specified each image will be reshaped to image size, prior to plotting
    :param nrow: number of rows in grid
    :param ncol: number of collumns in grid
    """
    if image_size:
        x, y = image_size
    else:
        x, y = images.shape[1:]
    if not (nrow or ncol):
        # if not specified claculate the grid size
        nrow = ncol = np.ceil(np.sqrt(len(images)))
    plt.figure(figsize=(x * ncol, y * nrow))
    for i, image in enumerate(images):
        plt.subplot(nrow, ncol, i + 1)
        if image_size:
            plt.imshow(image.reshape(image_size), cmap=cm.Greys_r)
        else:
            plt.imshow(image, cmap=cm.Greys_r)
        plt.axis('off')


def readImages(n_images=None, images_path='data/raw/images_training_rev1',
               grey=True, crop_offset=137, resize=(25, 25)):
    galaxy_files = [os.path.join(images_path, f)
                    for f in os.listdir(images_path)
                    if f.endswith('.jpg')]
    if n_images:
        galaxy_files = np.random.choice(galaxy_files, n_images)
    else:
        n_images = len(galaxy_files)
    if grey:
        shape = (n_images, resize[0], resize[1])
    else:
        shape = (n_images, resize[0], resize[1], 3)
    images = np.zeros(shape, dtype=np.float)
    for i, img_file in enumerate(galaxy_files):
        images[i, ...] = crop_and_resize(misc.imread(img_file, flatten=grey),
                                         crop_offset=crop_offset, resolution=resize)
    galaxies = [os.path.splitext(os.path.basename(g_file))[0] for g_file in galaxy_files]
    return (images, galaxies)


def _fileNameSuffix(base, n_patches, patch_size, n_samples='', n_centroids='', n_components=''):
    """
     Just a utility function to build a filename from input parameters
    """
    base = "%s_n_%d_np_%d_ps_%d" % (base, n_samples, n_patches, patch_size[0])
    if n_centroids:
        base += '_nc_%d' % n_centroids
    if n_components:
        base += '_ncomp_%d' % n_components
    base += '.png'
    return base

@profile
def buildFeatureDictionary(images, n_patches=10, patch_size=(8, 8), n_centroids=100, save_pics=True):
    # set up the array and split images to patches
    X = np.zeros((n_patches * len(images), patch_size[0] * patch_size[1]))
    n_images = len(images)
    for i, image in enumerate(images):
        patches = extract_patches_2d(image, patch_size=patch_size, max_patches=n_patches)
        X[i * n_patches:i * n_patches + n_patches, ...] = patches.reshape((n_patches, -1))

    # substract mean and rescale pixel values
    X = (X - np.mean(X, axis=1, keepdims=True)) / 255

    if save_pics:
        # get some random patches to plot
        sample_patches_ind = np.random.choice(X.shape[0], 100)
        plotImageGrid(X[sample_patches_ind, ...],
                      image_size=patch_size, nrow=10, ncol=10)
        plt.savefig(_fileNameSuffix('pictures/patches', n_patches, patch_size, n_images))

    # PCA\Whitening
    _logWithTimestamp("==== Starting PCA ====")
    # perform whitening
    # 42 components = 99% explained varience
    n_components = 42
    pca = RandomizedPCA(n_components=n_components, whiten=True, random_state=seed)
    X = pca.fit_transform(X)

    _logWithTimestamp("variance explained: %f" % np.sum(pca.explained_variance_ratio_[:n_components]))
    _logWithTimestamp("==== PCA fitted =====")

    if save_pics:
        # plot whitened patches after inverse transform
        plotImageGrid(pca.inverse_transform(X[sample_patches_ind, ...]),
                      image_size=patch_size, nrow=10, ncol=10)
        plt.savefig(_fileNameSuffix('pictures/patches', n_patches, patch_size,
                                    n_images, n_components=n_components))

    # ##KMEANS
    _logWithTimestamp("==== Starting K-Means====")
    k_means = cluster.MiniBatchKMeans(n_clusters=n_centroids)
    k_means.fit(X)
    _logWithTimestamp("==== K-Means fitted ====")

    # get centroids and transform them to original space
    D = pca.inverse_transform(k_means.cluster_centers_)
    if save_pics:
        # plot 100 random centroids
        plotImageGrid(D, image_size=patch_size)
        plt.savefig(_fileNameSuffix('pictures/centroids', n_patches, patch_size,
                                    n_images, n_centroids=n_centroids))

    # save the results of calculations for future use
    np.savetxt('data/models/centroids_%d_%d.csv' % (n_images, n_centroids), D, delimiter=',')
    joblib.dump(pca, 'data/models/pca_%d.pkl' % n_images)
    return D


def loadFeatureDict(dict_file='data/models/centroids_10.csv'):
    return np.loadtxt(dict_file, delimiter=',')


def loadPCAtransform(pca_file='data/models/pca_10.pkl'):
    return joblib.load(pca_file)


def extractFeatures(patch, D, soft=True):
    # normalize patch
    patch = (patch - np.mean(patch, axis=1, keepdims=True)) / 255
    # compute Euclidian distance from patch to centroid
    z = np.sqrt(np.sum(np.square(D - patch.reshape(1, -1)), axis=1))
    if soft:
        # f_k = max{0, mu - z_k}
        features = np.clip(np.mean(z) - z, a_min=0, a_max=np.inf)
    else:
        # 1-of-K hard assignment
        features = np.zeros(len(D))
        features[z.argmin()] = 1
    return features


def mapFeatures(image, D, patch_size=(8, 8), stride=1):
    n_features = len(D)
    p_x, p_y = patch_size
    x, y = image.shape
    # initialize array to store features
    features = np.zeros(
        ((y - p_y) / stride + 1, (x - p_x) / stride + 1, n_features),
        dtype=np.float)
    # extract features from each patch
    for i in xrange(features.shape[0]):
        for j in xrange(features.shape[1]):
            y_coor = i * stride
            x_coor = j * stride
            features[i, j, :] = extractFeatures(image[y_coor: y_coor + p_y,
                                                x_coor: x_coor + p_x], D)
    return features


def pool(features, pool_split=2, type='max'):
    feat_x, feat_y, feat_z = features.shape
    pooled_features = np.zeros((pool_split, pool_split, feat_z), dtype=np.float32)
    # split the features array and pool features in the subarray
    for i, x in enumerate(np.vsplit(features, pool_split)):
        for j, y in enumerate(np.hsplit(x, pool_split)):
            if type == 'max':
                res = np.max(y, axis=(0, 1))
            # mean pooling
            if type == 'mean':
                res = np.mean(y, axis=(0, 1))
            pooled_features[i, j, ...] = res
    return pooled_features


def featuresFromImages(images, D):
    features = np.zeros((len(images), len(D)*4))
    for i, image in enumerate(images):
        features[i, ...] = pool(mapFeatures(image, D)).reshape((1, -1))
    return features

if __name__ == '__main__':
    start_time = time.time()
    images, galaxies = readImages()
    _logWithTimestamp('Images read')
    D = buildFeatureDictionary(images, n_centroids=1000, save_pics=False)
    _logWithTimestamp('Feature Dictionary ready')
    pool = Pool()
    res = []
    def apply_callback(result):
        res.append(result)

    for chunk in np.array_split(images, cpu_count()):
        _logWithTimestamp('Starting worker to chunk len %d' % len(chunk))
        result = pool.apply_async(featuresFromImages, args=[chunk, D,], callback=apply_callback)
    pool.close()
    pool.join()
    features = np.vstack(tuple(res))

    #features = featuresFromImages(images, D)
    # resize the array to add galaxies id's in place
    features.resize((features.shape[0], features.shape[1]+1), refcheck=False)
    # append galaxy ids
    features[:, 0] = np.array(galaxies)
    _logWithTimestamp('Features mapped to images')
    np.savetxt('data/tidy/pooled_features_test.csv', features, delimiter=',')
    _logWithTimestamp('Features saved, exiting')