#!/usr/bin/env python3

import argparse
import os
import sys


cwd = os.getcwd()
# print('cwd', cwd)
sys.path.append(cwd)

import timeit
import numpy as np
import logging
from pathlib import Path
import json
from skimage import io

from basic_block import basic_block
from utils import utils

import tensorflow as tf


patch_size = 128


def read_image_list(data_list):
  f = open(data_list, 'r')
  recons_image_list = []
  ori_image_list = []
  for line in f:
    recons_image = line.strip("\n")
    ori_image = recons_image.replace('recons_train', 'ori_train')
    ori_image = ori_image.replace('recons_valid', 'ori_valid')
    # print 'list'
    # print image
    # print label
    ori_image_list.append(ori_image)
    recons_image_list.append(recons_image)


  return recons_image_list, ori_image_list


def parse_data(recons_path, ori_path):
  recons_image_string = tf.read_file(recons_path)
  recons_image = tf.image.decode_image(recons_image_string, channels=3)
  recons_image = tf.to_float(recons_image)

  ori_image_string = tf.read_file(ori_path)
  ori_image = tf.image.decode_image(ori_image_string, channels=3)
  ori_image = tf.to_float(ori_image)

  # image = tf.image.convert_image_dtype(image, tf.float32)

  return recons_image, ori_image


def get_train_and_valid_data_batch(sess, train_data_list, valid_data_list, batch_size):
  num_threads = 4
  batch_prefetch = 1


  # train_data = tf.data.TextLineDataset(train_data_list)

  train_recons_image_list, train_ori_image_list = read_image_list(train_data_list)
  train_data = tf.data.Dataset.from_tensor_slices((train_recons_image_list, train_ori_image_list))


  train_data = train_data.repeat()

  train_data = train_data.map(parse_data,  num_parallel_calls=num_threads)



  train_data = train_data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  # train_data = train_data.batch(batch_size)
  train_data = train_data.prefetch(batch_prefetch)


  # -----
  # valid_data = tf.data.TextLineDataset(valid_data_list)

  valid_recons_image_list, valid_ori_image_list = read_image_list(train_data_list)
  valid_data = tf.data.Dataset.from_tensor_slices((valid_recons_image_list, valid_ori_image_list))

  valid_data = valid_data.map(parse_data,  num_parallel_calls=num_threads)
  valid_data = valid_data.batch(batch_size)
  valid_data = valid_data.prefetch(batch_prefetch)


  # -----
  handle_placeholder = tf.placeholder(tf.string, shape=[])
  iterator = tf.data.Iterator.from_string_handle(handle_placeholder, train_data.output_types, train_data.output_shapes)
  data_batch = iterator.get_next()

  train_iterator = train_data.make_one_shot_iterator()
  valid_iterator = valid_data.make_initializable_iterator()

  train_handle = sess.run(train_iterator.string_handle())
  valid_handle = sess.run(valid_iterator.string_handle())

  return data_batch, handle_placeholder, train_handle, valid_handle, valid_iterator


def model(input, train_data_mean, train_data_std):
  with tf.variable_scope('normalize'):
    output = tf.reshape(input, [-1, patch_size, patch_size, 3])
    output = (output - train_data_mean) / train_data_std

  output = basic_block.my_conv2d(
      inputs=output,
      filters=32,
      strides=[2, 2],
      kernel_size=[3, 3],
      padding="SAME",
      activation=tf.nn.relu,
      kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
      bias_initializer=tf.constant_initializer(0.0),
      name='conv_1'
    )

  output = basic_block.my_conv2d(
      inputs=output,
      filters=64,
      strides=[2, 2],
      kernel_size=[3, 3],
      padding="SAME",
      activation=tf.nn.relu,
      kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
      bias_initializer=tf.constant_initializer(0.0),
      name='conv_2'
    )

  output = basic_block.my_conv2d(
      inputs=output,
      filters=64,
      strides=[1, 1],
      kernel_size=[3, 3],
      padding="SAME",
      activation=tf.nn.relu,
      kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
      bias_initializer=tf.constant_initializer(0.0),
      name='conv_3'
    )


  output = basic_block.my_conv2d(
      inputs=output,
      filters=64,
      strides=[1, 1],
      kernel_size=[3, 3],
      padding="SAME",
      activation=tf.nn.relu,
      kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
      bias_initializer=tf.constant_initializer(0.0),
      name='conv_4'
    )

  output = basic_block.my_conv2d_transpose(
      inputs=output,
      filters=32,
      strides=[2, 2],
      kernel_size=[3, 3],
      padding="SAME",
      activation=tf.nn.relu,
      kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
      bias_initializer=tf.constant_initializer(0.0),
      name='conv_5'
    )

  output = basic_block.my_conv2d_transpose(
      inputs=output,
      filters=3,
      strides=[2, 2],
      kernel_size=[3, 3],
      padding="SAME",
      activation=tf.identity,
      kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
      bias_initializer=tf.constant_initializer(0.0),
      name='conv6'
    )

  with tf.variable_scope('denormalize'):
    output = output * train_data_std + train_data_mean

    output = tf.clip_by_value(output, 0.0, 255.0)


  return output




def get_loss(x, y):
  loss_op = tf.reduce_mean(tf.squared_difference(x, y))

  return loss_op


def optimize(loss, boundaries, lr_values):
  global_step_op = tf.Variable(0, name='global_step', trainable=False)


  learning_rate_op = tf.train.piecewise_constant(global_step_op, boundaries, lr_values)

  optimizer = tf.train.AdamOptimizer(learning_rate_op)

  grads_and_vars = optimizer.compute_gradients(loss)
  # capped_grads_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads_and_vars]
  train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step_op)
  # train_op = optimizer.apply_gradients(capped_grads_and_vars, global_step=global_step_op)


  return train_op, global_step_op, learning_rate_op



def train(sess, args):
  load_ckpt = args.load_ckpt

  batch_size = 64
  num_steps = 300000
  boundaries = [200000, 300000]
  lr_values = [1e-4, 1e-5, 1e-6]


  cur_file_dir = Path(__file__).parent
  log_path = str(cur_file_dir / 'train.log')
  model_param_dir = str(cur_file_dir / 'params')
  Path(model_param_dir).mkdir(parents=True, exist_ok=True)
  model_param_path = model_param_dir + '/params'

  train_data_params_file = 'rm_block_effect/channel_normalization_params.npz'
  train_data_params = np.load(train_data_params_file)
  train_data_mean = train_data_params['mean']
  train_data_std = train_data_params['std']


  if load_ckpt == 'on':
    utils.set_logger(log_path)

    logging.info('Not overload_log')

  else:
    utils.set_logger(log_path, mode='w')

    logging.info('Overload_log')



  train_data_list = 'data_info/recons_train_data_patch_list.txt'
  valid_data_list = 'data_info/recons_valid_data_patch_list.txt'

  # train_data_list = 'data_info/train_data_patch_list_128.txt'
  # valid_data_list = 'data_info/valid_data_patch_list_128.txt'


  data_batch, handle_placeholder, train_handle, valid_handle, valid_iterator = get_train_and_valid_data_batch(sess, train_data_list, valid_data_list, batch_size)

  # print(sess.run(data_batch))
  # return

  # Avoid summary info
  logging.getLogger().setLevel(logging.WARNING)


  recons_data, ori_data = data_batch


# # -----
#   variable_init = tf.global_variables_initializer()
#   sess.run(variable_init)

#   ori_data_value = sess.run(data_batch, feed_dict={handle_placeholder: train_handle})

#   # print(type(recons_data_value))
#   print(type(ori_data_value))
#   # print(recons_data_value.shape)
#   print(ori_data_value[0].shape, ori_data_value[1].shape)

#   return

  output = model(recons_data, train_data_mean, train_data_std)

  loss_op = get_loss(ori_data, output)


  train_op, global_step_op, learning_rate_op = optimize(loss_op, boundaries, lr_values)


  saver = tf.train.Saver()
  if load_ckpt == 'on':
    saver.restore(sess, model_param_path)

    logging.info('Load previous params')
  else:
    variable_init = tf.global_variables_initializer()
    sess.run(variable_init)




  # saver.save(sess, model_param_path)
  # logging.info('Model paremeters saved to: {}'.format(model_param_path))
  # return

  # Avoid summary info
  logging.getLogger().setLevel(logging.INFO)

  logging.info('-----')
  logging.info(args)
  logging.info('-----')


  train_loss_display_step = 200
  valid_loss_display_step = 20000


  global_step = sess.run(global_step_op)

  # normal train
  for step in range(global_step + 1, num_steps + 1):
    _, loss, global_step, learning_rate_value = sess.run([train_op, loss_op, global_step_op, learning_rate_op], feed_dict={handle_placeholder: train_handle})

    if step % train_loss_display_step == 0:
      logging.info('Step: {:d}, loss: {:.8f}, lr: {:.8f}'.format(global_step, loss, learning_rate_value))

    if step % valid_loss_display_step == 0:
      sess.run(valid_iterator.initializer)

      [valid_loss] = sess.run([loss_op], feed_dict={handle_placeholder: valid_handle})
      logging.info('Valid loss: {:.8f}'.format(valid_loss))

      saver.save(sess, model_param_path)



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

  parser.add_argument(
    '-l',
    '--load_ckpt',
    help='Load previous ckpt',
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

  train(sess, args)

