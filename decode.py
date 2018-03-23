#!/usr/bin/env python3

import argparse
import os
import tensorflow as tf
from pathlib import Path
import numpy as np
# import cv2
import scipy
# import matplotlib.image as mpimg
import range_coder
import json
import sys

from utils import utils
from data_loader import data_loader


np.random.seed(1234)
tf.set_random_seed(1234)

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

  parser.add_argument(
    '-g',
    '--gpu_num',
    help='Determine which gpu to use',
    type=str,
    choices=['-1', '0', '1', '2', '3'],
    required=True
  )

  parser.add_argument(
    '-d',
    '--debug_mode',
    help='Debug mode',
    type=str,
    choices=['on', 'off'],
    default='off'
  )

  parser.add_argument(
    '-p',
    '--params_file',
    help='File for model parameters',
    type=str,
    default=''
  )

  parser.add_argument(
    '-i',
    '--input_dir',
    help='Directory for input compressed data',
    type=str,
    default='encoded_data'
  )

  parser.add_argument(
    '-o',
    '--output_dir',
    help='Directory for output uncompressed data',
    type=str,
    default='recons_data'
  )

  args = parser.parse_args()

  return args


def apply_range_decoder(seq_data_len, decodepath):
  resolution = 4096
  prob = np.load('data_info/distribution_info.npy' )

  # Avoid zero prob
  modified_freq = prob * resolution + 1
  modified_prob = modified_freq / np.sum(modified_freq)

  # print(modified_prob)

  cum_freq = range_coder.prob_to_cum_freq(modified_prob, resolution=resolution)

  # print('-----')
  # print(cum_freq)

  # cum_freq = [0] + [i for i in range(1, 256 + 1)]

  range_decoder = range_coder.RangeDecoder(decodepath)

  # Whether cum_freq resolution influences performance ?
  seq_data = range_decoder.decode(seq_data_len, cum_freq)

  return seq_data


def get_img_info(filename, config):
  name_sep = config['name_sep']

  path_without_extension = filename.replace('.encoded', '')

  img_info = path_without_extension.split(name_sep)[1]
  [seq_data_len, height, width] = img_info.split('_')
  seq_data_len = int(seq_data_len)
  height = int(height)
  width = int(width)

  return seq_data_len, height, width


def get_recons_image_path(filename, args, config):
  name_sep = config['name_sep']

  output_dir = args.output_dir
  filename_without_extension = filename.replace('.encoded', '')
  recons_filename = filename_without_extension.split(name_sep)[0] + '.png'
  recons_image_path = str(Path(output_dir) / recons_filename)

  return recons_image_path


def uncompress(sess, model, args):

  print(args)

  config_path = 'model_{}/config.json'.format(args.model_num)
  with open(config_path, 'r') as f:
    config = json.load(f)

  print(config)

  batch_size = 64
  encoded_patch_size = 32
  encoded_patch_channel = 32
  patches_placeholder = tf.placeholder(tf.float32, shape=[None, encoded_patch_size, encoded_patch_size, encoded_patch_channel])
  patch_batch, iterator = data_loader.get_patch_batch(batch_size, patches_placeholder)

  decoder_output_op = model.decoder(patch_batch)

  utils.restore_params(sess, args)

  input_dir = args.input_dir
  for filename in os.listdir(input_dir):

    filepath = str(Path(input_dir) / filename)

    seq_data_len, height, width = get_img_info(filename, config)

    # print('filepath: {}'.format(filepath))
    # print('seq_data_len: {}'.format(seq_data_len))
    # print('height: {}'.format(height))
    # print('width: {}'.format(width))

    seq_data = apply_range_decoder(seq_data_len, filepath)

    # print(seq_data[200 : 210])
    # print('-----')
    # sys.stdout.flush()

    # print('len(seq_data): {}'.format(len(seq_data)))

    seq_data = np.asarray(seq_data).astype(np.float32)
    encoded_patches = seq_data.reshape(-1, 32, 32, 32)

    # print('encoded_patches.shape: {}'.format(encoded_patches.shape))

    sess.run(iterator.initializer, feed_dict={patches_placeholder: encoded_patches})

    decoded_patches_list = []
    while True:
      try:
        decoded_output = sess.run(decoder_output_op)
        decoded_patches_list.append(decoded_output)
      except tf.errors.OutOfRangeError:
        break


    decoded_patches = np.concatenate(decoded_patches_list, axis=0)

    # print(decoded_patches[100, 60 : 70, 20, 0])
    # print('-----')
    # sys.stdout.flush()

    # print('decoded_patches.shape: {}'.format(decoded_patches.shape))

    recons_image = utils.concat_patches(decoded_patches, height, width)

    recons_image_path = get_recons_image_path(filename, args, config)

    print('recons_image_path: {}'.format(recons_image_path))
    # print('recons_image.shape: {}'.format(recons_image.shape))

    # print(recons_image[200 : 210, 500, 0])
    # sys.stdout.flush()

    # break

    # cv2.imwrite(recons_image_path, recons_image)
    scipy.misc.imsave(recons_image_path, recons_image)
    # mpimg.imsave(recons_image_path, recons_image)

    # print(recons_image[100 : 110, 500, 0])
    # sys.stdout.flush()
    # print('-----')

    # test_array_path = recons_image_path.replace('recons_data/', 'other/test_array/').replace('.png', '')
    # np.save(test_array_path, recons_image)
    # print('test_array_path: {}'.format(test_array_path))

    # print('Uncompress complete')

    # break




if __name__ == '__main__':
  args = my_parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num


  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  sess = tf.Session(config=config)

  if args.model_num == '0':
    from model_0 import model
  elif args.model_num == '1':
    from model_1 import model
  elif args.model_num == '2':
    from model_2 import model
  elif args.model_num == '3':
    from model_3 import model

  uncompress(sess, model, args)


