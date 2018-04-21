
import os
from os.path import join, getsize, isfile
import numpy as np
import cv2


# images_dir = 'tiny_valid'
# images_dir = 'other/valid_ori'
images_dir = '/data/cbl/clic/test_data/test/'


image_list = os.listdir(images_dir)

pixel_num = 0

for image in image_list:
  image_path = join(images_dir, image)
  image_data = cv2.imread(image_path)

  height, width, channel = image_data.shape

  pixel_num += height * width

print('Pixel_num: {}'.format(pixel_num))
