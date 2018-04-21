#!/usr/bin/env python3

import argparse
import os
import sys

cwd = os.getcwd()
# print('cwd', cwd)
sys.path.append(cwd)

import tensorflow as tf
from pathlib import Path
import numpy as np
import range_coder
import json
from skimage import io
from PIL import Image
import importlib


from utils import utils
from data_loader import data_loader


def my_parse_args():
  parser = argparse.ArgumentParser()


  parser.add_argument(
    '-g',
    '--gpu_num',
    help='Determine which gpu to use',
    type=str,
    choices=['0', '1', '2', '3'],
    required=True
  )


  args = parser.parse_args()

  return args



def compress_and_uncompress(sess, model, args):

  print(args)

  config_path = 'rm_block_effect/recons_model/config.json'
  with open(config_path, 'r') as f:
    config = json.load(f)

  print(config)

  # data_list = 'data_info/train_data_list.txt'
  data_list = 'data_info/ori_valid_data_list.txt'
  image_path_list = utils.read_image_list(data_list)

  batch_size = 64
  patch_size = config['patch_size']
  patches_placeholder = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3])
  patch_batch, iterator = data_loader.get_patch_batch(batch_size, patches_placeholder)

  quan_scale = config['quan_scale']

  encoder_output_op = model.encoder(patch_batch, patch_size, quan_scale)
  decoder_output_op = model.decoder(encoder_output_op, quan_scale)

  params_file = 'rm_block_effect/recons_params/params'
  saver = tf.train.Saver()
  saver.restore(sess, params_file)

  for image_path in image_path_list:
    image = io.imread(image_path)
    image_patches = utils.crop_image_input_patches(image, patch_size)

    sess.run(iterator.initializer, feed_dict={patches_placeholder: image_patches})

    # decoded_patches = image_patches


    decoded_patches_list = []
    while True:
      try:
        decoded_output = sess.run(decoder_output_op)
        decoded_patches_list.append(decoded_output)
      except tf.errors.OutOfRangeError:
        break

    # print('len(decoded_patches): {}'.format(len(decoded_patches)))

    decoded_patches = np.concatenate(decoded_patches_list, axis=0)

    height, width, channel = image.shape
    recons_image = utils.concat_patches(decoded_patches, height, width, patch_size)

    recons_image = np.around(recons_image).astype(np.uint8)


    recons_image_path = image_path.replace('ori', 'recons')


    print('recons_image_path: {}'.format(recons_image_path))

    io.imsave(recons_image_path, recons_image)

    # image_read = np.asarray(Image.open(recons_image_path), dtype=np.float32)

    # residual = image_read - recons_image
    # print(residual)

    # print(recons_image)
    # print('~~~~~')
    # print(image_read)
    # print('-----')

    # print('Compress and uncompress complete')

    # break




if __name__ == '__main__':
  args = my_parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  sess = tf.Session(config=config)

  module_name = 'rm_block_effect/recons_model/model.py'.replace('/', '.').replace('.py', '')


  model = importlib.import_module(module_name)

  compress_and_uncompress(sess, model, args)


