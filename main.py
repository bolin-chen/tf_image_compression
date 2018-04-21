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
    '-p',
    '--param_save',
    help='Save model parameters',
    type=str,
    choices=['on', 'off'],
    default='on'
  )

  parser.add_argument(
    '-s',
    '--summary_save',
    help='Save summaries for tensorboard',
    type=str,
    choices=['on', 'off'],
    default='on'
  )

  parser.add_argument(
    '-t',
    '--timeline_save',
    help='Save timeline information',
    type=str,
    choices=['on', 'off'],
    default='off'
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
    '-o',
    '--overload_log',
    help='Whether to overload the previous log',
    type=str,
    choices=['on', 'off'],
    default='on'
  )

  parser.add_argument(
    '-a',
    '--additional_param',
    help='Additional parameter for searching hyperparameters',
    type=str,
    default='None'
  )

  parser.add_argument(
    '-l',
    '--load_ckpt',
    help='Load previous ckpt',
    type=str,
    choices=['on', 'off'],
    default='off'
  )

  parser.add_argument(
    '-r',
    '--reset_step',
    help='Reset global step',
    type=str,
    choices=['on', 'off'],
    default='off'
  )

  parser.add_argument(
    '-x',
    '--max_step',
    help='Max step',
    type=str,
    default='None'
  )

  parser.add_argument(
    '-b',
    '--lr_and_bound',
    help='Learning rate',
    type=str,
    default='None'
  )

  parser.add_argument(
    '-f',
    '--fine_tune',
    help='Load params and fine tune',
    type=str,
    choices=['off', 'btnk', 'input'],
    default='off'
  )

  args = parser.parse_args()

  return args

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

  model.train(sess, args)


