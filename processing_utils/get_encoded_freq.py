#!/usr/bin/env python3

import argparse
import os
import tensorflow as tf


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
    default='data_info/train_data_list.txt'
  )

  args = parser.parse_args()

  return args


def cal_encoded_freq(sess, model, args):
  encoded_freq = np.zeros(256)

  restore()

  get_batch()

  while True:
    try:
      encoded_output = sess.run(encoder_output_op)


    except tf.errors.OutOfRangeError:
      break

  break


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

  cal_encoded_freq(sess, model, args)

