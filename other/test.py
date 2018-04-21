
import os
from os.path import join, getsize, isfile
import numpy as np
from PIL import Image

def mse(image0, image1):
    return np.sum(np.square(image1 - image0))

def mse2psnr(mse):
    return 20. * np.log10(255.) - 10. * np.log10(mse)

def evaluate(image_files):
    num_dims = 0
    sqerror_values = []

    for image_file, image_file_rec in image_files:
        image = np.asarray(Image.open(image_file), dtype=np.float32)
        image_rec = np.asarray(Image.open(image_file_rec), dtype=np.float32)

        num_dims += image.size

        sqerror_values.append(mse(image, image_rec))

    return mse2psnr(np.sum(sqerror_values) / num_dims)

def get_images_size(folder):
  total_size = 0

  for f in os.listdir(folder):
    if isfile(join(folder, f)) and ('.png' not in f):
      total_size += getsize(join(folder, f))

  return total_size


# -----

decoder_size = getsize('./decode.py')
print('File size: {}'.format(decoder_size))

images_size = get_images_size('./valid_encoded')
print('Images size: {}'.format(images_size))

# -----

# image_files = []
# ori_folder = './valid_ori'
# encoded_folder = './valid_encoded'
# ori_images = os.listdir('valid_ori')

# for f in ori_images:
#   image_files.append((join(ori_folder, f), join(encoded_folder, f)))

# psnr = evaluate(image_files)

# print('PSNR: {}'.format(psnr))

# -----