"""
 Main module for feature extraction using K-means
"""

import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_extraction.image import extract_patches_2d
import sklearn.cluster as cluster
from sklearn.externals import joblib

__author__ = 'dudevil'

# The current working dir should be the project top-level directory

seed = 211114
np.random.seed(seed)


def crop_and_resize(image, crop_offset=108, resolution=64):
    img = image[crop_offset:-crop_offset, crop_offset:-crop_offset]
    if not isinstance(resolution, tuple):
        resolution = (resolution, resolution)
    return misc.imresize(img, resolution)


def plotImageGrid(images, image_size=(), nrow=4, ncol=4):
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
    plt.figure(figsize=(x * ncol, y * nrow))
    for i, image in enumerate(images):
        plt.subplot(nrow, ncol, i + 1)
        if image_size:
            plt.imshow(image.reshape(image_size), cmap=cm.Greys_r)
        else:
            plt.imshow(image, cmap=cm.Greys_r)
        plt.axis('off')


def readImages(n_images=None, images_path='data/raw/images_training_rev1',
               grey=True, crop_offset=108, resize=(64, 64)):
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
    return images


def buildFeatureDictionary(images, n_patches=32, patch_size=(16, 16), save_pics=True):
    # set up the array and split images to patches
    X = np.zeros((n_patches * len(images), patch_size[0] * patch_size[1]))
    for i, image in enumerate(images):
        patches = extract_patches_2d(image, patch_size=patch_size, max_patches=n_patches)
        X[i * n_patches:i * n_patches + n_patches, ...] = patches.reshape((n_patches, -1))

    # substract mean and rescale pixel values
    X = (X - np.mean(X, axis=1, keepdims=True)) / 255

    if save_pics:
        # get some random patches to plot
        sample_patches_ind = np.random.choice(X.shape[0], 100)
        plotImageGrid(X[sample_patches_ind, ...],
                      image_size=(patch_size, patch_size), nrow=10, ncol=10)
        plt.savefig('pictures/patches16_g_10.png')

    # PCA\Whitening
    print("==== Starting PCA ====")
    # perform whitening
    # 42 components = 99% explained varience
    n_components = 42
    pca = RandomizedPCA(n_components=n_components, whiten=True, random_state=seed)
    w_X = pca.fit_transform(X)

    print("variance explained: %f" % np.sum(pca.explained_variance_ratio_[:n_components]))
    print("==== PCA fitted =====")

    if save_pics:
        # plot whitened patches after inverse transform
        orig_X = pca.inverse_transform(w_X[sample_patches_ind, ...])
        plotImageGrid(orig_X, image_size=(patch_size, patch_size), nrow=10, ncol=10)
        plt.savefig('pictures/patches_whitened16_g_10.png')

    # ##KMEANS
    print("==== Starting K-Means====")
    k_means = cluster.KMeans(n_clusters=10, n_jobs=8)
    k_means.fit(w_X)
    print("==== K-Means fitted ====")

    # get centroids and transform them to original space
    D = pca.inverse_transform(k_means.cluster_centers_)
    if save_pics:
        # plot 100 random centroids
        plotImageGrid(D, image_size=(16, 16), nrow=1, ncol=10)
        plt.savefig('pictures/centroids_16_g_10.png')

    # save the results of calculations for future use
    np.savetxt('data/models/centroids_10.csv', D, delimiter=',')
    joblib.dump(pca, 'data/models/pca_10.pkl')
    return pca, D


def loadFeatureDict(dict_file='data/models/centroids_10.csv'):
    return np.loadtxt(dict_file, delimiter=',')


def loadPCAtransform(pca_file='data/models/pca_10.pkl'):
    return joblib.load(pca_file)


def extractFeatures(patch, D, soft=True):
    # compute Euclidian distance from patch to centroid
    z = np.sqrt(np.sum(np.square(D - patch.reshape(1, -1)), axis=1))
    print(z)
    if soft:
        # f_k = max{0, mu - z_k}
        features = np.clip(np.mean(z) - z, a_min=0, a_max=np.inf)
    else:
        # 1-of-K hard assignment
        features = np.zeros(len(D))
        features[z.argmin()] = 1
    return features


def mapFeatures(image, D, patch_size=(16, 16), stride=1):
    n_features = len(D)
    p_x, p_y = patch_size
    x, y = image.shape
    # initialize array to store features
    features = np.zeros(
        ((y - p_y) / stride + 1, (x - p_x) / stride + 1, n_features),
        dtype=np.float)
    # extract features from each patch
    for i in xrange(0, y - p_y + 1, stride):
        for j in xrange(0, x - p_x + 1, stride):
            features[i, j, :] = extractFeatures(image[i: i + p_y, j: j + p_x], D)
    return features


def pool(features, pool_size=(2, 2), type='max'):
    feat_x, feat_y, feat_z = features.shape
    pool_x, pool_y = pool_size
    pooled_y = feat_y / pool_y
    pooled_x = feat_y / pool_y
    # allocate the pooled features array
    pooled_features = np.zeros((pooled_y, pooled_x, feat_z), dtype=np.float32)
    # iterate over image and select subregions for pooling
    for i in xrange(0, pooled_y):
        for j in xrange(0, pooled_x):
            # select pool_x x pool_y pooling region
            pool_region = features[i * pool_y:i * pool_y + pool_y,
                          j * pool_x:j * pool_x + pool_x, ...]
            # max pooling
            if type == 'max':
                pooled_features[i, j, ...] = np.max(pool_region, axis=(0, 1))
            # mean pooling
            if type == 'mean':
                pooled_features[i, j, ...] = np.mean(pool_region, axis=(0, 1))
    return pooled_features


if __name__ == '__main__':
    image = readImages(1)[0]
    D = loadFeatureDict()
    mapped_features = mapFeatures(image, D)
    features = pool(mapped_features)
