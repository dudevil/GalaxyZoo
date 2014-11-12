##
##
##
##
##

import os
from scipy import misc
import matplotlib.pyplot as plt
from functools import partial

train_images_path = '/home/dudevil/prog/R/GalaxyZoo/data/images_test_rev1'

galaxy_files = [os.path.join(train_images_path, f)
                for f in os.listdir(train_images_path)
                if f.endswith('.jpg')]

def crop_and_resize(image, crop_offset=0, resolution=69):
    img = image[crop_offset:-crop_offset, crop_offset:-crop_offset]
    if not isinstance(resolution, tuple):
        resolution = (resolution, resolution)
    return misc.imresize(img, resolution)

galaxy_files1 = galaxy_files[1:1000]
galaxy_images = map(misc.imread, galaxy_files)
mapf = partial(crop_and_resize, crop_offset=108, resolution=69)
galaxy_images_resized = map(mapf, galaxy_images)

# file1 = galaxy_files[0]
# image = misc.imread(file1)
# image2 =
# plt.subplot(121)
# plt.imshow(image)
# plt.axis('off')
# plt.subplot(122)
# plt.imshow()
# plt.axis('off')
# plt.show()