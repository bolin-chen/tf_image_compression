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


def get_code_size(encoded_dir, encoded_files):
  total_size = 0

  for f in encoded_files:
    # if isfile(join(folder, f)) and ('.png' not in f):
    total_size += getsize(join(encoded_dir, f))

  return total_size


def calc_bpp(encoded_dir, encoded_files, pixel_num):
  code_size = get_code_size(encoded_dir, encoded_files)

  bpp = code_size * 8.0 / pixel_num

  return bpp


def calc_psnr(encoded_dir, recons_files):
  image_files = []

  ori_dir = 'other/valid_ori'

  for f in recons_files:
    image_files.append((join(encoded_dir, f), join(ori_dir, f)))

  psnr = evaluate(image_files)

  return psnr

def my_parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '-s',
    '--submit_num',
    help='Submit number',
    type=str,
    required=True
  )

  args = parser.parse_args()

  return args



if __name__ == '__main__':
  args = my_parse_args()

  # full_valid_pixel_num = 251858137
  full_valid_pixel_num = 710839849

  pixel_num = full_valid_pixel_num

  # encoded_dir = 'submit/{}/encoded/'.format(args.submit_num)
  encoded_dir = 'submit/{}/test_encoded/'.format(args.submit_num)
  encoded_files = [file for file in os.listdir(encoded_dir) if file.endswith('.encoded')]

  recons_files = [file for file in os.listdir(encoded_dir) if file.endswith('.png')]
  # print(encoded_files)

  bpp = calc_bpp(encoded_dir, encoded_files, pixel_num)
  print('Bpp: {:6f}'.format(bpp))

  psnr = calc_psnr(encoded_dir, recons_files)
  print('PSNR: {}'.format(psnr))

