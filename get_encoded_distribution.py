#!/usr/bin/env python3

import argparse
import os
import tensorflow as tf
from pathlib import Path
import numpy as np
import cv2
import range_coder
import json
import importlib

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
    default='data_info/train_data_patch_list_{}.txt'
  )


  parser.add_argument(
    '-f',
    '--model_file',
    help='The model file used to calculate distribution',
    type=str,
    default='model_{}/params_for_test/model.py'
  )

  parser.add_argument(
    '-o',
    '--output_file',
    help='File to keep encoded distribution info',
    type=str,
    default='data_info/distribution_info_{}.npy'
  )

  args = parser.parse_args()

  return args


def get_distribution(sess, model, args):

  print(args)

  config_path = 'model_{}/config.json'.format(args.model_num)
  with open(config_path, 'r') as f:
    config = json.load(f)

  print(config)

  patch_size = config['patch_size']
  quan_scale = config['quan_scale']
  # bottleneck_channel = config['bottleneck_channel']

  data_list = args.data_list.format(patch_size)

  batch_size = 64

  patch_batch = data_loader.get_data_batch(data_list, batch_size)



  encoder_output_op = model.encoder(patch_batch, patch_size, quan_scale)

  utils.restore_params(sess, args)



  freq = np.zeros(quan_scale)
  bins = [i for i in range(quan_scale + 1)]

  # print(bins)

  while True:
    try:
      encoded_output = sess.run(encoder_output_op)
      encoded_hist = np.histogram(encoded_output, bins)

      # print(encoded_hist[0])
      # print(np.sum(encoded_hist[0]))

      freq += encoded_hist[0]

      # break

    except tf.errors.OutOfRangeError:
      break


  prob = freq / sum(freq)

  print(prob)

  output_file = args.output_file.format(args.model_num)

  np.save(output_file, prob)

  # i = 0
  # while True:
  #   if i == 0:
  #     output_file = output_file[:-4] + '-{}.npy'.format(i)
  #   else:
  #     output_file = output_file[:-6] + '-{}.npy'.format(i)

  #   if Path(output_file).is_file():
  #     i += 1
  #   else:
  #     np.save(output_file, prob)
  #     break

  print('Prob disttribution saved to {} complete'.format(output_file))


if __name__ == '__main__':
  args = my_parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  sess = tf.Session(config=config)

  model_file = args.model_file.format(args.model_num)
  module_name = model_file.replace('/', '.').replace('.py', '')

  # print(Path(model_file).is_file())

  model = importlib.import_module(module_name)

  # if args.model_num == '0':
  #   from model_0 import model
  # elif args.model_num == '1':
  #   from model_1 import model
  # elif args.model_num == '2':
  #   from model_2 import model
  # elif args.model_num == '3':
  #   from model_3 import model

  get_distribution(sess, model, args)


