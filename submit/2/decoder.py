#!/usr/bin/env python3

import argparse
import os
import tensorflow as tf
from pathlib import Path
import numpy as np
from skimage import io
import range_coder
import json
import sys

from utils import utils
from data_loader import data_loader
from rmbe import rmbe



np.random.seed(1234)
tf.set_random_seed(1234)



def apply_range_decoder(seq_data_len, decodepath, config):
  resolution = config['resolution']
  prob = np.load('data_info/distribution_info.npy')


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

  img_info = path_without_extension.split(name_sep)[-1]
  [seq_data_len, height, width] = img_info.split('_')
  seq_data_len = int(seq_data_len)
  height = int(height)
  width = int(width)

  return seq_data_len, height, width


def get_recons_image_path(filename, config):
  name_sep = config['name_sep']

  output_dir = 'encoded/'
  filename_without_extension = filename.replace('.encoded', '')
  recons_filename = filename_without_extension.split(name_sep)[0] + '.png'
  recons_image_path = str(Path(output_dir) / recons_filename)

  return recons_image_path



def get_encoded_shape(input_dir, config):
  name_sep = config['name_sep']

  sample_filename = os.listdir(input_dir)[0]
  encoded_shape = sample_filename.split(name_sep)[1]
  [encoded_height, encoded_width, encoded_channel] = encoded_shape.split('_')
  encoded_height = int(encoded_height)
  encoded_width = int(encoded_width)
  encoded_channel = int(encoded_channel)

  return encoded_height, encoded_width, encoded_channel


def uncompress(model):


  config_path = 'config.json'
  with open(config_path, 'r') as f:
    config = json.load(f)

  print(config)

  patch_size = config['patch_size']

  input_dir = 'encoded'


  g1 = tf.Graph()
  sess = tf.Session(graph=g1)

  with sess.as_default():
    with g1.as_default():
      encoded_height, encoded_width, encoded_channel = get_encoded_shape(input_dir, config)
      patches_placeholder = tf.placeholder(tf.float32, shape=[None, encoded_height, encoded_width, encoded_channel])


      batch_size = 64
      patch_batch, iterator = data_loader.get_patch_batch(batch_size, patches_placeholder)

      quan_scale = config['quan_scale']

      decoder_output_op = model.decoder(patch_batch, quan_scale)

      params_file = 'params/params'
      saver = tf.train.Saver()
      saver.restore(sess, params_file)

  for filename in os.listdir(input_dir):

    filepath = str(Path(input_dir) / filename)

    seq_data_len, height, width = get_img_info(filename, config)

    # print('filepath: {}'.format(filepath))
    # print('seq_data_len: {}'.format(seq_data_len))
    # print('height: {}'.format(height))
    # print('width: {}'.format(width))

    seq_data = apply_range_decoder(seq_data_len, filepath, config)

    # encoded_order = np.load('data_info/order_info_{}.npy'.format(args.model_num))
    # decoded_order = [int(i) for i in sorted(range(len(encoded_order)), key=lambda k: encoded_order[k])]

    # print(filepath)
    # print(encoded_order.shape)
    # print(len(seq_data))

    # seq_data = np.asarray(seq_data).reshape(-1, encoded_height * encoded_width * encoded_channel)

    # print('shape_1', seq_data.shape)
    # print('shape_order', len(decoded_order))

    # seq_data = [seq_data_item[decoded_order] for seq_data_item in seq_data]

    # print(seq_data[200 : 210])
    # print('-----')
    # sys.stdout.flush()

    # print('len(seq_data): {}'.format(len(seq_data)))

    seq_data = np.asarray(seq_data).astype(np.float32)

    # print('shape_2', seq_data.shape)

    encoded_patches = seq_data.reshape(-1, encoded_height, encoded_width, encoded_channel)

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

    recons_image = utils.concat_patches(decoded_patches, height, width, patch_size)
    recons_image = rmbe.rmbe(recons_image)

    recons_image_path = get_recons_image_path(filename, config)

    print('recons_image_path: {}'.format(recons_image_path))
    # print('recons_image.shape: {}'.format(recons_image.shape))

    # print(recons_image[200 : 210, 500, 0])
    # sys.stdout.flush()

    # break

    # cv2.imwrite(recons_image_path, recons_image)

    recons_image = np.around(recons_image).astype(np.uint8)

    io.imsave(recons_image_path, recons_image)
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
  # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

  # config = tf.ConfigProto()
  # config.gpu_options.allow_growth = True

  # sess = tf.Session(config=config)

  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  import model

  uncompress(model)

