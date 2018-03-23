
import os
from os.path import join
import cv2

crop_size = 128

input_professional_dir = '/data/cbl/clic/professional/train'
input_mobile_dir = '/data/cbl/clic/mobile/train'

output_professional_dir = '/data/cbl/clic/crop_{}/professional/'.format(crop_size)
output_mobile_dir = '/data/cbl/clic/crop_{}/mobile/'.format(crop_size)


# input_professional_dir = '/data/cbl/clic/professional/valid'
# input_mobile_dir = '/data/cbl/clic/mobile/valid'

# output_professional_dir = '/data/cbl/clic/valid_crop_{}/professional/'.format(crop_size)
# output_mobile_dir = '/data/cbl/clic/valid_crop_{}/mobile/'.format(crop_size)

def crop_and_save(input_dir, output_dir):
  for filename in os.listdir(input_dir):
    image_data = cv2.imread(join(input_dir, filename))
    width, length, channel = image_data.shape

    width_patch_num = width // crop_size
    length_patch_num = length // crop_size

    for i in range(width_patch_num):
      for j in range(length_patch_num):
        patch_name = filename.replace('.png', '_{}_{}.png'.format(i, j))
        patch_data = image_data[i * crop_size: (i + 1) * crop_size, j * crop_size : (j + 1) * crop_size, :]
        cv2.imwrite(join(output_dir, patch_name), patch_data)

    #     break
    #   break
    # break


crop_and_save(input_professional_dir, output_professional_dir)
print('Images in {} crop complete'.format(input_professional_dir))

crop_and_save(input_mobile_dir, output_mobile_dir)
print('Images in {} crop complete'.format(input_mobile_dir))
