#!/usr/bin/env python3


import os
from os.path import join
from skimage import io
from pathlib import Path


# offset = 64
crop_size = 128
overlap = 0.0


# crop_size = 256
# overlap = 0.5


# crop_size = 512
# overlap = 0.75


# input_prof_dir_train = '/data/cbl/clic/professional/train'
# input_mobi_dir_train = '/data/cbl/clic/mobile/train'

# output_prof_dir_train = '/data/cbl/clic/crop_{}/professional/'.format(crop_size)
# output_mobi_dir_train = '/data/cbl/clic/crop_{}/mobile/'.format(crop_size)


# input_prof_dir_valid = '/data/cbl/clic/professional/valid'
# input_mobi_dir_valid = '/data/cbl/clic/mobile/valid'

# output_prof_dir_valid = '/data/cbl/clic/valid_crop_{}/professional/'.format(crop_size)
# output_mobi_dir_valid = '/data/cbl/clic/valid_crop_{}/mobile/'.format(crop_size)



# crop_size = 128
# overlap = 0.5


input_prof_dir_train = '/data/cbl/clic/professional/train'
input_mobi_dir_train = '/data/cbl/clic/mobile/train'

output_prof_dir_train = '/data/cbl/clic/large_crop_{}/professional/'.format(crop_size)
output_mobi_dir_train = '/data/cbl/clic/large_crop_{}/mobile/'.format(crop_size)


input_prof_dir_valid = '/data/cbl/clic/professional/valid'
input_mobi_dir_valid = '/data/cbl/clic/mobile/valid'

output_prof_dir_valid = '/data/cbl/clic/large_valid_crop_{}/professional/'.format(crop_size)
output_mobi_dir_valid = '/data/cbl/clic/large_valid_crop_{}/mobile/'.format(crop_size)

def crop_and_save(input_dir, output_dir):
  for filename in os.listdir(input_dir):
    image_data = io.imread(join(input_dir, filename))

    # image_data = [offset :, offset:, :]

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



Path(output_prof_dir_train).mkdir(parents=True, exist_ok=True)
Path(output_mobi_dir_train).mkdir(parents=True, exist_ok=True)
Path(output_prof_dir_valid).mkdir(parents=True, exist_ok=True)
Path(output_mobi_dir_valid).mkdir(parents=True, exist_ok=True)

crop_and_save(input_prof_dir_train, output_prof_dir_train)
print('Images in {} crop complete'.format(input_prof_dir_train))

crop_and_save(input_mobi_dir_train, output_mobi_dir_train)
print('Images in {} crop complete'.format(input_mobi_dir_train))


crop_and_save(input_prof_dir_valid, output_prof_dir_valid)
print('Images in {} crop complete'.format(input_prof_dir_valid))

crop_and_save(input_mobi_dir_valid, output_mobi_dir_valid)
print('Images in {} crop complete'.format(input_mobi_dir_valid))
