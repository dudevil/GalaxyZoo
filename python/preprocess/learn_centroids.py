##
##
##
##
##

import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_extraction.image import extract_patches_2d
import sklearn.cluster as cluster
from sklearn.externals import joblib

seed = 211114
np.random.seed(seed)

train_images_path = 'data/raw/images_training_rev1'
n_images = 3000
patch_size = 16

galaxy_files = [os.path.join(train_images_path, f)
                for f in os.listdir(train_images_path)
                if f.endswith('.jpg')]

# choose a subset of the training images
galaxy_files = np.random.choice(galaxy_files, n_images)
n_images = len(galaxy_files)

def crop_and_resize(image, crop_offset=108, resolution=64):
    img = image[crop_offset:-crop_offset, crop_offset:-crop_offset]
    if not isinstance(resolution, tuple):
        resolution = (resolution, resolution)
    return misc.imresize(img, resolution)


def loadFeatureMapping(dict_file):
    pass


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
    plt.figure(figsize=(x*ncol, y*nrow))
    for i, image in enumerate(images):
        plt.subplot(nrow, ncol, i+1)
        if image_size:
            plt.imshow(image.reshape(image_size), cmap=cm.Greys_r)
        else:
            plt.imshow(image, cmap=cm.Greys_r)
        plt.axis('off')

X = np.zeros((32*n_images, patch_size*patch_size))

for i, file in enumerate(galaxy_files):
    image = crop_and_resize(misc.imread(file, flatten=True))
    patches = extract_patches_2d(image, patch_size=(patch_size, patch_size), max_patches=32)
    for j, patch in enumerate(patches):
        mean = patch.mean()
        #var = patch.var()
        patches[j] = (patch - mean) #/(var + 5)
    X[i*32:i*32+32, ...] = patches.reshape((32, -1))

X /= 255.0

def buildFeatureDictionary(images, ,patch_size, save_pics=True ):
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
    print("variance explained: %d" % np.sum(pca.explained_variance_ratio_[:n_components]))
    print("==== PCA fitted =====")

    if save_pics:
        # plot whitened patches after inverse transform
        orig_X = pca.inverse_transform(w_X[sample_patches_ind, ...])
        plotImageGrid(orig_X, image_size=(patch_size, patch_size), nrow=10, ncol=10)
        plt.savefig('pictures/patches_whitened16_g_10.png')

    ###KMEANS
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

def extractFeatures(patch, D, soft=True):
    # compute Euclidian distance from patch to centroid
    z = np.sqrt(np.sum(np.square(D - patch.reshape(1, -1)), axis=1))
    print(z)
    if soft:
        # f_k = max{0, mu - z_k}
        features = np.clip(np.mean(z) - z, a_min=0, a_max=np.inf)
    else:
        # 1-ok-K hard assignment
        features = np.zeros(len(D))
        features[z.argmin()] = 1
    return features


def mapFeatures(image, D, patch_size=(16, 16), stride=1):
    n_features = len(D)
    p_x, p_y = patch_size
    x, y = image.shape
    features = np.zeros(
        ((y - p_y) / stride + 1, (x - p_x) / stride + 1, n_features),
        dtype=np.float)
    for i in xrange(0, y - p_y + 1, stride):
        for j in xrange(0, x - p_x + 1, stride):
            features[i, j, :] = extractFeatures(image[i: i + p_y, j: j + p_x], D)
