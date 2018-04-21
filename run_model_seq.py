#!/usr/bin/python3

import argparse
import subprocess

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
    '-a',
    '--additional_parameter',
    help='Additional parameter for searching hyperparameters',
    type=str,
    default='None'
  )

  parser.add_argument(
    '-n',
    '--search_num',
    help='The number of the hyperparameters searching',
    type=str,
    default='None'
  )

  args = parser.parse_args()

  return args


def main(args):
  model_num = args.model_num
  gpu_num = args.gpu_num

  search_num = args.search_num

  command = './main.py'

  for i in range(int(search_num)):
    # subprocess.call([command, '-m', model_num, '-g', gpu_num, '-a', str(i)])
    subprocess.call([command, '-o', 'off', '-m', model_num, '-g', gpu_num, '-a', str(i)])



if __name__ == '__main__':
  args = my_parse_args()

  main(args)


