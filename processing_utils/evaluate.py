#!/usr/bin/env python3

import os
from os.path import join, getsize, isfile
import argparse
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

    print('mse: ', np.sum(sqerror_values) / 26034944)

    return mse2psnr(np.sum(sqerror_values) / num_dims)


def get_code_size(folder):
  total_size = 0

  for f in os.listdir(folder):
    if isfile(join(folder, f)) and ('.png' not in f):
      total_size += getsize(join(folder, f))

  return total_size


def calc_bpp(code_dir, pixel_num):
  code_size = get_code_size(code_dir)

  bpp = code_size * 8.0 / pixel_num

  return bpp


def calc_psnr(ori_images_dir, recons_images_dir):
  image_files = []

  ori_images = os.listdir(ori_images_dir)

  for f in ori_images:
    image_files.append((join(ori_images_dir, f), join(recons_images_dir, f)))

  psnr = evaluate(image_files)

  return psnr

def my_parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '-m',
    '--model_num',
    help='Determine which model to use',
    type=str,
    choices=['0', '1', '2', '3'],
    required=True
  )

  args = parser.parse_args()

  return args



if __name__ == '__main__':
  args = my_parse_args()
  model_num = args.model_num


  # ori_images_dir = 'other/valid_ori'
  ori_images_dir = 'tiny_valid'


  # full_valid_pixel_num = 251858137
  tiny_valid_pixel_num = 26034944

  # pixel_num = full_valid_pixel_num
  pixel_num = tiny_valid_pixel_num

  if model_num != -1:
    model_dir = 'model_{}'.format(model_num)
  else:
    model_dir = 'other'

  recons_images_dir = join(model_dir, 'recons_data')
  code_dir = join(model_dir, 'encoded_data')

  bpp = calc_bpp(code_dir, pixel_num)
  print('Bpp: {:6f}'.format(bpp))

  psnr = calc_psnr(ori_images_dir, recons_images_dir)
  print('PSNR: {}'.format(psnr))

