##
##
##
##
##

import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import partial
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_extraction.image import PatchExtractor, extract_patches_2d

train_images_path = '/home/dudevil/prog/dmlabs/GalaxyZoo/data/raw/images_training_rev1'
seed = 171114
n_images = 1000

galaxy_files = [os.path.join(train_images_path, f)
                for f in os.listdir(train_images_path)
                if f.endswith('.jpg')][:n_images]

def crop_and_resize(image, crop_offset=108, resolution=64):
    img = image[crop_offset:-crop_offset, crop_offset:-crop_offset]
    if not isinstance(resolution, tuple):
        resolution = (resolution, resolution)
    return misc.imresize(img, resolution)

def splitPatches(image, size=16):
    patch1 = image[:size, :size]
    patch2 = image[-size:, :size]
    patch3 = image[:size, -size:]
    patch4 = image[-size:, -size:]
    return np.array([patch1, patch2, patch3, patch4])

def normalize(image):
    """
    Performs image normalization by substracting the mean of the patch and dividing by variance + constant
    see: http://www.cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf

    :param image:
    :return: noralized image
    """
    # i won't rescale varience yet because the image looks like bullshit
    # I will also subtract means by color channel
    image[:,:,0] = image[:, :, 0].mean()
    image[:,:,1] = image[:, :, 1].mean()
    image[:,:,2] = image[:, :, 2].mean()
    return image

def plotImageGrid(images, x, y):
    for i, image in enumerate(images):
        plt.subplot(x, y, i+1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()

X = np.zeros((16*n_images, 16*16*3))

for i, file in enumerate(galaxy_files):
    image = crop_and_resize(misc.imread(file))
    patches = extract_patches_2d(image, patch_size=(16,16), max_patches=16)
    for patch in patches:
        patch = patch - np.mean(patch, axis=(0,1), keepdims=True)
    X[i*16:i*16+16,...] = patches.reshape((16, -1))

import sklearn.cluster as cluster

pca = RandomizedPCA(n_components=200, whiten=True, random_state=seed)
nX = pca.fit_transform(X)

print("==== PCA fitted =====")
print("variance explained")
print(pca.explained_variance_ratio_)


###KMEANS
k_means = cluster.KMeans(n_clusters=256, n_jobs=7)
k_means.fit(nX)
print("==== K-Means fitted ====")


tmp = k_means.cluster_centers_.copy()
tmp = tmp.reshape((256,16,16,3))
plotImageGrid(tmp,16,16)