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


seed = 191114
np.random.seed(seed)

train_images_path = '/home/dudevil/prog/dmlabs/GalaxyZoo/data/raw/images_training_rev1'
n_images = 200
patch_size = 16

galaxy_files = [os.path.join(train_images_path, f)
                for f in os.listdir(train_images_path)
                if f.endswith('.jpg')]
galaxy_files = np.random.choice(galaxy_files, n_images)

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
#get some random patches to plot
sample_patches_ind = np.random.choice(X.shape[0], 36)
plotImageGrid(X[sample_patches_ind, ...],
              image_size=(patch_size, patch_size), nrow=6, ncol=6)
plt.savefig('patches16_g.png')
#plt.show()

#perform whitening
# 42 components = 99% explained varience
pca = RandomizedPCA(n_components=42, whiten=True, random_state=seed)
w_X = pca.fit_transform(X)

print("==== PCA fitted =====")
print("variance explained: %d" % np.sum(pca.explained_variance_ratio_))

#plot whitened patches after inverse transform
orig_X = pca.inverse_transform(w_X[sample_patches_ind, ...])
plotImageGrid(orig_X, image_size=(patch_size, patch_size), nrow=6, ncol=6)
plt.savefig('patches_whitened16_g.png')
#plt.show()

###KMEANS
k_means = cluster.KMeans(n_clusters=100, n_jobs=4)
k_means.fit(w_X)
print("==== K-Means fitted ====")


# get centroids and transform them to original space
tmp = pca.inverse_transform(k_means.cluster_centers_.copy())
#tmp = k_means.cluster_centers_.copy()
plotImageGrid(tmp, image_size=(16, 16), nrow=10, ncol=10 )
plt.savefig('centroids_16_g.png')

