#!/usr/bin/env python3

import argparse
import os
import tensorflow as tf
from pathlib import Path
import numpy as np
from skimage import io
import range_coder
import json

from utils import utils
from data_loader import data_loader



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
    choices=['0', '1', '2', '3'],
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
    '-v',
    '--data_list',
    help='File for data_list',
    type=str,
    default='data_info/tiny_valid_data_list.txt'
  )

  parser.add_argument(
    '-o',
    '--output_dir',
    help='Directory for output compressed data',
    type=str,
    default='model_{}/encoded_data'
  )

  args = parser.parse_args()

  return args


def apply_range_encoder(seq_data, encodepath, args, config):
  resolution = config['resolution']
  prob = np.load('data_info/distribution_info_{}.npy'.format(args.model_num))

  # Avoid zero prob
  modified_freq = prob * resolution + 1
  modified_prob = modified_freq / np.sum(modified_freq)

  # print(modified_prob)

  cum_freq = range_coder.prob_to_cum_freq(modified_prob, resolution=resolution)

  # print('-----')
  # print(cum_freq)

  # cum_freq = [0] + [i for i in range(1, 256 + 1)]


  range_encoder = range_coder.RangeEncoder(encodepath)
  # Whether cum_freq resolution influences performance ?
  range_encoder.encode(seq_data, cum_freq)
  range_encoder.close()




def get_encodepath(image_path, image, seq_data, args, config, encoded_patches_shape):
    encoded_save_dir = args.output_dir.format(args.model_num)
    new_extension = '.encoded'
    filename_without_extension = image_path.split('/')[-1].replace('.png', '')

    # Add size, height, width info.
    seq_data_len = len(seq_data)
    height, width, channel = image.shape

    encoded_height, encoded_width, encoded_channel = encoded_patches_shape

    name_sep = config['name_sep']
    img_info = ''

    img_info +=  name_sep + '{}_{}_{}'.format(encoded_height, encoded_width, encoded_channel)

    img_info +=  name_sep + '{}_{}_{}'.format(seq_data_len, height, width)

    encodepath = str(Path(encoded_save_dir) / filename_without_extension) + img_info + new_extension

    return encodepath


def compress(sess, model, args):

  print(args)

  config_path = 'model_{}/config.json'.format(args.model_num)
  with open(config_path, 'r') as f:
    config = json.load(f)

  print(config)

  # bottleneck_channel = config['bottleneck_channel']

  data_list = args.data_list
  image_path_list = utils.read_image_list(data_list)

  batch_size = 64
  patch_size = config['patch_size']
  patches_placeholder = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3])
  patch_batch, iterator = data_loader.get_patch_batch(batch_size, patches_placeholder)

  quan_scale = config['quan_scale']

  encoder_output_op = model.encoder(patch_batch, patch_size, quan_scale)

  utils.restore_params(sess, args)


  # To be paralleled
  for image_path in image_path_list:
    image = io.imread(image_path)
    image_patches = utils.crop_image_input_patches(image, patch_size)

    sess.run(iterator.initializer, feed_dict={patches_placeholder: image_patches})

    encoded_patches = []
    while True:
      try:
        encoded_output = sess.run(encoder_output_op)
        encoded_patches.append(encoded_output)
      except tf.errors.OutOfRangeError:
        break

    # print('len(encoded_patches): {}'.format(len(encoded_patches)))
    # print(encoded_patches[0].shape)
    # print(encoded_patches[-1].shape)

    encoded_patches_shape = encoded_patches[0][0].shape

    # seq_data = np.concatenate(encoded_patches).reshape(-1).astype(int).tolist()

    seq_data = np.concatenate(encoded_patches).reshape(-1, np.prod(encoded_patches_shape))

    # print(seq_data.reshape(-1).shape[0])

    # encoded_order = [int(i) for i in np.load('data_info/order_info_{}.npy'.format(args.model_num))]
    # seq_data = [seq_data_item[encoded_order] for seq_data_item in seq_data]

    seq_data = np.asarray(seq_data).reshape(-1).astype(int).tolist()

    # print(len(seq_data))

    encodepath = get_encodepath(image_path, image, seq_data, args, config, encoded_patches_shape)

    # print('np.concatenate(encoded_patches).shape: {}'.format(np.concatenate(encoded_patches).shape))
    # print('len(seq_data): {}'.format(len(seq_data)))
    # print('type(seq_data): {}'.format(type(seq_data)))
    # print('np.max(np.asarray(seq_data)): {}'.format(np.max(np.asarray(seq_data))))
    # print('np.min(np.asarray(seq_data)): {}'.format(np.min(np.asarray(seq_data))))
    # print(seq_data)

    # break


    encoded_save_dir = args.output_dir.format(args.model_num)
    if not os.path.exists(encoded_save_dir):
      os.makedirs(encoded_save_dir)

    apply_range_encoder(seq_data, encodepath, args, config)

    print('encodepath: {}'.format(encodepath))
    # print('Range coder encoded complete')


    # decoded_seq_data = apply_range_decoder(seq_data_len, encodepath)
    # print('len(decoded_seq_data): {}'.format(len(decoded_seq_data)))
    # print('Range coder decoded complete')

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

  compress(sess, model, args)


