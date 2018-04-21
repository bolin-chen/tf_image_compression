#!/usr/bin/env python3

import argparse
import os
from os.path import join
from skimage import io
from pathlib import Path


offset = 64
crop_size = 128
overlap = 0.0


def my_parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '-t',
    '--crop_type',
    help='Which type of images to crop',
    type=str,
    choices=['train', 'valid'],
    required=True
  )


  args = parser.parse_args()

  return args


input_dir_train_ori = '/data/cbl/clic/ori_train_data'
input_dir_train_recons = '/data/cbl/clic/recons_train_data'

output_dir_train_ori = '/data/cbl/clic/crop_ori_train_data'
output_dir_train_recons = '/data/cbl/clic/crop_recons_train_data'


input_dir_valid_ori = '/data/cbl/clic/ori_valid_data'
input_dir_valid_recons = '/data/cbl/clic/recons_valid_data'

output_dir_valid_ori = '/data/cbl/clic/crop_ori_valid_data'
output_dir_valid_recons = '/data/cbl/clic/crop_recons_valid_data'


def crop_and_save(input_dir, output_dir):
  for filename in os.listdir(input_dir):
    image_data = io.imread(join(input_dir, filename))

    image_data = image_data[offset :, offset:, :]

    width, length, channel = image_data.shape

    width_patch_num = (width - crop_size) // (crop_size * (1 - overlap)) + 1
    length_patch_num = (length - crop_size) // (crop_size * (1 - overlap)) + 1

    width_patch_num = int(width_patch_num)
    length_patch_num = int(length_patch_num)

    # print('width_patch_num: {}, length_patch_num: {}'.format(width_patch_num, length_patch_num))

    for i in range(width_patch_num):
      for j in range(length_patch_num):
        patch_name = filename.replace('.png', '_{}_{}.png'.format(i, j))

        start_i = crop_size * (1 - overlap) * i
        start_j = crop_size * (1 - overlap) * j

        start_i = int(start_i)
        start_j = int(start_j)

        # print('start_i: {}, start_j: {}'.format(start_i, start_j))

        patch_data = image_data[start_i : start_i + crop_size, start_j : start_j + crop_size, :]
        # patch_data = image_data[i * crop_size: (i + 1) * crop_size, j * crop_size : (j + 1) * crop_size, :]

        io.imsave(join(output_dir, patch_name), patch_data)

    #     break
    #   break
    # break


if __name__ == '__main__':
  args = my_parse_args()

  if args.crop_type == 'train':

    Path(output_dir_train_ori).mkdir(parents=True, exist_ok=True)
    Path(output_dir_train_recons).mkdir(parents=True, exist_ok=True)


    crop_and_save(input_dir_train_recons, output_dir_train_recons)
    print('Images in {} crop complete'.format(input_dir_train_recons))

    crop_and_save(input_dir_train_ori, output_dir_train_ori)
    print('Images in {} crop complete'.format(input_dir_train_ori))

  elif args.crop_type == 'valid':

    Path(output_dir_valid_ori).mkdir(parents=True, exist_ok=True)
    Path(output_dir_valid_recons).mkdir(parents=True, exist_ok=True)

    crop_and_save(input_dir_valid_recons, output_dir_valid_recons)
    print('Images in {} crop complete'.format(input_dir_valid_recons))

    crop_and_save(input_dir_valid_ori, output_dir_valid_ori)
    print('Images in {} crop complete'.format(output_dir_valid_ori))
