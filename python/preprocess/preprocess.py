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

galaxy_files = [os.path.join(train_images_path, f)
                for f in os.listdir(train_images_path)
                if f.endswith('.jpg')]

def crop_and_resize(image, crop_offset=108, resolution=60):
    img = image[crop_offset:-crop_offset, crop_offset:-crop_offset]
    if not isinstance(resolution, tuple):
        resolution = (resolution, resolution)
    return misc.imresize(img, resolution)

def splitPatches(image, size=32):
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
    return (image - np.mean(image))/np.sqrt(np.var(image) + 10)


def plotImageGrid(images, x, y):
    for i, image in enumerate(images):
        plt.subplot(x, y, i+1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()


image1 = misc.imread(galaxy_files[0])
image1 = crop_and_resize(image1)
plotImageGrid(splitPatches(image1), 2, 2)

image2 = misc.imread(galaxy_files[1])
image2 = crop_and_resize(image2)
plotImageGrid(splitPatches(image2), 2, 2)

