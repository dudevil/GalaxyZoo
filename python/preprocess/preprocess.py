##
##
##
##
##

import os
from scipy import misc


def read_files(n=None):
    train_images_path = os.path.join(os.getcwd(), 'data/images_training_rev1')
    for fl in os.listdir(train_images_path)[0:n]:
        if fl.endswith('.jpg'):
            yield misc.imread(os.path.join(train_images_path, fl))


def crop_and_resize(image, crop_offset=0, resolution=69):
    img = image[crop_offset:-crop_offset, crop_offset:-crop_offset]
    if not isinstance(resolution, tuple):
        resolution = (resolution, resolution)
    return misc.imresize(img, resolution)

files = [f for f in read_files(n=10)]