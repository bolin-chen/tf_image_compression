#!/usr/bin/env python3

import argparse
import os
import tensorflow as tf

from model_0 import model as model_0
from model_1 import model as model_1
from model_2 import model as model_2
from model_3 import model as model_3


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

  args = parser.parse_args()

  return args

if __name__ == '__main__':
  args = my_parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  sess = tf.Session(config=config)

  if args.model_num == '0':
    model_0.train(sess, args)
  elif args.model_num == '1':
    model_1.train(sess, args)
  elif args.model_num == '2':
    model_2.train(sess, args)
  elif args.model_num == '3':
    model_3.train(sess, args)


