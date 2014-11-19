##
##
##
##
##

import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_extraction.image import extract_patches_2d
import sklearn.cluster as cluster


seed = 191114
np.random.seed(seed)

train_images_path = '/home/dudevil/prog/dmlabs/GalaxyZoo/data/raw/images_training_rev1'
n_images = 200
patch_size = 32

galaxy_files = [os.path.join(train_images_path, f)
                for f in os.listdir(train_images_path)
                if f.endswith('.jpg')]
galaxy_files = np.random.choice(galaxy_files, n_images)

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

def plotImageGrid(images, image_size=(), nrow=4, ncol=4):
    """
    Helper function to plot a grid of images

    :param images: an array of images to be plotted
    :param image_size: if specified each image will be reshaped to image size, prior to plotting
    :param nrow: number of rows in grid
    :param ncol: number of collumns in grid
    """
    if image_size:
        x, y, c = image_size
    else:
        x, y, c = images.shape[1:]
    plt.figure(figsize=(x*ncol, y*nrow))
    for i, image in enumerate(images):
        plt.subplot(x, y, i+1)
        if image_size:
            plt.imshow(image.reshape(image_size))
        else:
            plt.imshow(image)
        plt.axis('off')

X = np.zeros((16*n_images, patch_size*patch_size*3))
var = np.zeros((16*n_images, 3))

for i, file in enumerate(galaxy_files):
    image = crop_and_resize(misc.imread(file))
    patches = extract_patches_2d(image, patch_size=(32, 32), max_patches=16)
    for j, patch in enumerate(patches):
        mean = np.mean(patch, axis=(0, 1), keepdims=True)
        var = np.var(patch, axis=(0,1), keepdims=True)
        patches[j] = (patch - mn)/(var + 5)
    X[i*16:i*16+16, ...] = patches.reshape((16, -1))

X /= 255.0
#get some random patches to plot
sample_patches_ind = np.random.choice(X.shape[0], 36)
plotImageGrid(X[sample_patches_ind, ...],
              image_size=(patch_size, patch_size, 3), nrow=6, ncol=6)
plt.savefig('patches32.png')

#perform whitening
pca = RandomizedPCA(n_components=200, whiten=True, random_state=seed)
w_X = pca.fit_transform(X)

print("==== PCA fitted =====")
print("variance explained:")
print(pca.explained_variance_ratio_)

#plot whitened patches after inverse transform
orig_X = pca.inverse_transform(w_X[sample_patches_ind, ...])
plotImageGrid(orig_X, image_size=(patch_size, patch_size, 3), nrow=6, ncol=6)
plt.savefig('patches_whitened32.png')

###KMEANS
k_means = cluster.KMeans(n_clusters=100, n_jobs=3)
k_means.fit(X)
print("==== K-Means fitted ====")


# get centroids and transform them to original space
#tmp = pca.inverse_transform(k_means.cluster_centers_.copy())
tmp = k_means.cluster_centers_.copy()
plotImageGrid(tmp, image_size=(16, 16, 3), nrow=10, ncol=10 )
plt.savefig('centroids_nw.png')
