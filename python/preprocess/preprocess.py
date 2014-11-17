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

train_images_path = '/home/dudevil/prog/dmlabs/GalaxyZoo/data/raw/images_training_rev1'

galaxy_files = [os.path.join(train_images_path, f)
                for f in os.listdir(train_images_path)
                if f.endswith('.jpg')]

def crop_and_resize(image, crop_offset=0, resolution=64):
    img = image[crop_offset:-crop_offset, crop_offset:-crop_offset]
    if not isinstance(resolution, tuple):
        resolution = (resolution, resolution)
    return misc.imresize(img, resolution)

def split_patches(image, size=16):
    x, y, d = image.shape
    x_chunks = x/size
    y_chunks = y/size
    return [patch
            for vsplt in np.vsplit(image, x_chunks)
            for patch in np.hsplit(vsplt, y_chunks)]

def normalize(image):
    """
    Performs image normalization by substracting the mean of the patch and dividing by variance + constant
    see: http://www.cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf

    :param image:
    :return: noralized image
    """
    return (image - np.mean(image))/np.sqrt(np.var(image) + 10)

def whiten(image):
    pca = RandomizedPCA(whiten=True).fit(image)




image_file = galaxy_files[0]
image = misc.imread(image_file, flatten=False)

splt = split_patches(crop_and_resize(image, crop_offset=108))
splt = map(normalize, splt)

for i, patch in enumerate(splt):
    plt.subplot(4, 4, i+1)
    plt.imshow(patch)
    plt.axis('off')

plt.show()