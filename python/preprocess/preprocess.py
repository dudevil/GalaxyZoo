##
##
##
##
##




from os import listdir
from scipy import misc

images_path = '/home/dudevil/prog/R/GalaxyZoo/data/images_test_rev1/'

galaxy_files = [f for f in listdir(images_path) if f.endswith('.jpg')]

file1 = galaxy_files[0]
image = misc.imread()

