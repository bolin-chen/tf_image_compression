#!/usr/bin/env python3

import argparse
import os
import tensorflow as tf
from pathlib import Path
import numpy as np
import cv2
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
    help='Directory for output uncompressed data',
    type=str,
    default='other/test_img'
  )

  args = parser.parse_args()

  return args


def compress_and_uncompress(sess, model, args):

  print(args)

  config_path = 'model_{}/config.json'.format(args.model_num)
  with open(config_path, 'r') as f:
    config = json.load(f)

  print(config)

  data_list = args.data_list
  image_path_list = utils.read_image_list(data_list)

  batch_size = 64
  patch_size = config['patch_size']
  patches_placeholder = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3])
  patch_batch, iterator = data_loader.get_patch_batch(batch_size, patches_placeholder)

  encoder_output_op = model.encoder(patch_batch, patch_size)
  decoder_output_op = model.decoder(encoder_output_op)

  utils.restore_params(sess, args)

  for image_path in image_path_list:
    image = cv2.imread(image_path)
    image_patches = utils.crop_image_input_patches(image)

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
    recons_image = utils.concat_patches(decoded_patches, height, width)

    output_dir = args.output_dir
    recons_filename = image_path.split('/')[-1]
    recon_image_path = str(Path(output_dir) / recons_filename)

    print('recon_image_path: {}'.format(recon_image_path))

    cv2.imwrite(recon_image_path, recons_image)

    # print('Compress and uncompress complete')

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

  compress_and_uncompress(sess, model, args)


