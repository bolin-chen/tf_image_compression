#!/usr/bin/env python3

import cv2

def read_image_list(data_list):
  f = open(data_list, 'r')
  image_paths = []
  for line in f:
    image = line.strip("\n")
    image_paths.append(image)

  return image_paths


# data_list = 'data_info/train_data_list.txt'
data_list = 'data_info/valid_data_list.txt'


max_image_height = 0
min_image_height = 1e4

max_image_width = 0
min_image_width = 1e4

image_path_list = read_image_list(data_list)

for image_path in image_path_list:
  image = cv2.imread(image_path)
  height, width, channel = image.shape

  if (height > max_image_height):
    max_image_height = height

  if (height < min_image_height):
    min_image_height = height

  if (width > max_image_width):
    max_image_width = width

  if (width < min_image_width):
    min_image_width = width

print('data_list: {}'.format(data_list))

print('max_image_height: {}'.format(max_image_height))
print('min_image_height: {}'.format(min_image_height))

print('max_image_width: {}'.format(max_image_width))
print('min_image_width: {}'.format(min_image_width))



# data_list: data_info/valid_data_list.txt
# max_image_height: 2048
# min_image_height: 384
# max_image_width: 2048
# min_image_width: 512

