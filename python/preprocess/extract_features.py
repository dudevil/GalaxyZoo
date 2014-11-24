__author__ = 'dudevil'

import numpy as np
import learn_centroids

train_images_path = 'data/raw/images_training_rev1'
seed = 211114
np.random.seed(seed)
patch_size = 16
n_images = 1000

galaxy_files = [os.path.join(train_images_path, f)
                for f in os.listdir(train_images_path)
                if f.endswith('.jpg')]

galaxy_files = np.random.choice(galaxy_files, n_images)

images = np.zeros(n_images, dtype=np.float)

def