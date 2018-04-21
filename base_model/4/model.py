
import os

import timeit
import numpy as np
import logging
from pathlib import Path
import json

from data_loader import data_loader
from basic_block import basic_block
from utils import utils

import tensorflow as tf


train_data_list = 'data_info/train_data_patch_list.txt'
valid_data_list = 'data_info/valid_data_patch_list.txt'
train_data_params_file = 'data_info/channel_normalization_params.npz'

cur_file_dir = Path(__file__).parent
summary_path = str(cur_file_dir / 'summary/')
model_param_path = str(cur_file_dir / 'params/params')
timeline_path = str(cur_file_dir / 'timeline/timeline_merged.json')
log_path = str(cur_file_dir / 'log/train.log')

train_data_params = np.load(train_data_params_file)
train_data_mean = train_data_params['mean']
train_data_std = train_data_params['std']


global_var = None


def encoder(input, patch_size, quan_scale, is_training=False, reuse=None, getter=None):
  with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):

    # input = tf.Print(input, [tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2], tf.shape(input)[3]], 'Input tensor shape:')
    # input = tf.Print(input, [input], 'Input tensor:')

    with tf.variable_scope('normalize'):
      output = tf.reshape(input, [-1, patch_size, patch_size, 3])

      tf.summary.image("ori_image", output, max_outputs=4)
      tf.summary.histogram('ori_input', output)

      output = (output - train_data_mean) / train_data_std

      tf.summary.histogram('normalized_input', output)

    # output = tf.Print(output, [output], 'Normalized tensor:')

    output = basic_block.my_conv2d(
        inputs=output,
        filters=32,
        strides=[2, 2],
        kernel_size=[3, 3],
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0),
        name='encode_1'
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
        name='encode_2'
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
        name='encode_3'
      )

    output = basic_block.my_conv2d(
        inputs=output,
        filters=64,
        strides=[1, 1],
        kernel_size=[3, 3],
        padding="SAME",
        activation=tf.identity,
        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0),
        name='encode_4'
      )

    with tf.variable_scope('sigmoid_scale_quantize'):
      output = tf.nn.sigmoid(output) * (quan_scale - 1)
      output = tf.stop_gradient(tf.round(output) - output) + output

      tf.add_to_collection('bitrate_reg_var', output)

      tf.summary.histogram('sigmoid_scale_quantize_result', output)

    return output


def decoder(input, quan_scale, is_training=False, reuse=None, getter=None):
  with tf.variable_scope('decoder', reuse=reuse, custom_getter=getter):

    with tf.variable_scope('reverse_sigmoid_scale'):

      # output = basic_block.reverse_sigmoid(input / 255.0)

      # stable reverse sigmoid
      output = basic_block.reverse_sigmoid((input + 1e-6) / (quan_scale - 1 + 1e-5))

      tf.summary.histogram('reverse_sigmoid_scale_result', output)

    # output = tf.Print(output, [output[5, 20:30, 20, 0]])

    output = basic_block.my_conv2d(
        inputs=output,
        filters=64,
        strides=[1, 1],
        kernel_size=[3, 3],
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0),
        name='decode_4'
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
        name='decode_3'
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
        name='decode_2'
      )

    # output = tf.Print(output, [output[5, 20:30, 20, 0]])

    output = basic_block.my_conv2d_transpose(
        inputs=output,
        filters=3,
        strides=[2, 2],
        kernel_size=[3, 3],
        padding="SAME",
        activation=tf.identity,
        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0),
        name='decode_1'
      )

    # output = tf.Print(output, [output[5, 20:30, 20, 0]])

    with tf.variable_scope('denormalize'):
      output = output * train_data_std + train_data_mean

      tf.summary.histogram('denormalized_input', output)
      tf.summary.image("recons_image", output, max_outputs=4)

      # To be modified
      # output = tf.clip_by_value(output, 0.0, 1.0)

      output = tf.clip_by_value(output, 0.0, 255.0)

      # output = tf.Print(output, [output[5, 20:30, 20, 0]])

    return output


def get_loss(x, y):
  loss_op = tf.reduce_mean(tf.squared_difference(x, y))
  tf.summary.scalar('loss', loss_op)

  l2_decay = 0.0
  trainable_variables   = tf.trainable_variables()
  kernels = [v for v in trainable_variables if v.name.endswith('kernel:0')]
  l2_loss_op = tf.add_n([ tf.nn.l2_loss(k) for k in kernels]) * l2_decay

  loss_op += l2_loss_op
  tf.summary.scalar('l2_loss', l2_loss_op)
  tf.summary.scalar('loss_add_l2', loss_op)

  bitrate_reg_decay = 0.0
  bitrate_reg_var = tf.get_collection('bitrate_reg_var')
  bitrate_loss_op = tf.reduce_mean(bitrate_reg_var) * bitrate_reg_decay

  # loss_op = tf.Print(loss_op, [loss_op], 'Loss: ')
  # loss_op = tf.Print(loss_op, [tf.shape(bitrate_reg_var)], 'bitrate_reg_var: ')

  loss_op += bitrate_loss_op
  tf.summary.scalar('bitrate_loss', bitrate_loss_op)
  tf.summary.scalar('loss_add_l2_bitrate', loss_op)

  return loss_op


def optimize(loss, config):
  global_step_op = tf.Variable(0, name='global_step', trainable=False)

  boundaries = config['boundaries']
  lr_values = config['lr_values']

  learning_rate_op = tf.train.piecewise_constant(global_step_op, boundaries, lr_values)

  optimizer = tf.train.AdamOptimizer(learning_rate_op)

  grads_and_vars = optimizer.compute_gradients(loss)
  # capped_grads_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads_and_vars]
  train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step_op)
  # train_op = optimizer.apply_gradients(capped_grads_and_vars, global_step=global_step_op)

  for grad, var in grads_and_vars:
    tf.summary.histogram(var.name + '/gradient', grad)

  return train_op, global_step_op, learning_rate_op


def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter


def train(sess, args):

  config_path = 'model_{}/config.json'.format(args.model_num)
  with open(config_path, 'r') as f:
    config = json.load(f)

  patch_size = config['patch_size']
  batch_size = config['batch_size']
  num_steps = config['num_steps']
  quan_scale = config['quan_scale']

  overload_log = args.overload_log
  load_ckpt = args.load_ckpt


  if load_ckpt == 'on' or overload_log == 'off':
    utils.set_logger(log_path)

    logging.info('Not overload_log')

  else:
    utils.set_logger(log_path, mode='w')

    logging.info('Overload_log')


  global global_var
  additional_param = args.additional_param
  if additional_param == '0':
    pass
  elif additional_param == '1':
    pass
  elif additional_param == '2':
    pass
  elif additional_param == '3':
    pass


  data_batch, handle_placeholder, train_handle, valid_handle, valid_iterator = data_loader.get_train_and_valid_data_batch(sess, train_data_list, valid_data_list, batch_size, flip_ud=False, flip_lr=False, rot_90=False)

  # print(sess.run(data_batch))
  # return

  # Avoid summary info
  logging.getLogger().setLevel(logging.WARNING)

  output_train = encoder(data_batch, patch_size, quan_scale, is_training=True)
  output_train = decoder(output_train, quan_scale, is_training=True)

  loss_op_train = get_loss(data_batch, output_train)
  train_op, global_step_op, learning_rate_op = optimize(loss_op_train, config)


  # -----
  ema_decay = 0.9
  ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
  var_class = tf.trainable_variables(scope='class')
  ema_op = ema.apply(var_class)

  grouped_train_op = tf.group(train_op, ema_op)

  output_eval = encoder(data_batch, patch_size, quan_scale, is_training=False, reuse=True, getter=get_getter(ema))
  output_eval = decoder(output_eval, quan_scale, is_training=False, reuse=True, getter=get_getter(ema))

  loss_op_eval = get_loss(data_batch, output_eval)
  #-----

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


  utils.add_trainable_variables_to_summary()
  if args.summary_save == 'on':
    summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
  merged_summaries = tf.summary.merge_all()

  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  many_runs_timeline = utils.TimeLiner()

  # Avoid summary info
  logging.getLogger().setLevel(logging.INFO)

  logging.info('-----')
  logging.info(args)
  logging.info(config)
  logging.info('-----')

  # debug mode
  if args.debug_mode == 'on':
    logging.info('-----')
    logging.info('Debug mode')
    logging.info('-----')

    return


  train_loss_display_step = 200
  valid_loss_display_step = 20000

  global_step = sess.run(global_step_op)

  # normal train
  for step in range(global_step + 1, num_steps + 1):
    if step % train_loss_display_step == 0:
      if args.summary_save == 'on':
        _, loss, global_step, learning_rate_value, summary_value = sess.run([grouped_train_op, loss_op_train, global_step_op, learning_rate_op, merged_summaries], feed_dict={handle_placeholder: train_handle}, options=options, run_metadata=run_metadata)
      else:
        _, loss, global_step, learning_rate_value = sess.run([grouped_train_op, loss_op_train, global_step_op, learning_rate_op], feed_dict={handle_placeholder: train_handle}, options=options, run_metadata=run_metadata)

      logging.info('Step: {:d}, loss: {:.8f}, lr: {:.8f}'.format(global_step, loss, learning_rate_value))

      if step % valid_loss_display_step == 0:
        sess.run(valid_iterator.initializer)

        [valid_loss] = sess.run([loss_op_eval], feed_dict={handle_placeholder: valid_handle}, options=options, run_metadata=run_metadata)
        logging.info('Valid loss: {:.8f}'.format(valid_loss))

      if args.param_save == 'on':
        saver.save(sess, model_param_path)
        # logging.info('Model paremeters saved to: {}'.format(model_param_path))

      if args.summary_save == 'on':
        summary_writer.add_summary(summary_value, global_step=global_step)
        # logging.info('Summaries saved to: {}'.format(summary_path))

    else:
      _, loss, global_step = sess.run([grouped_train_op, loss_op_train, global_step_op], feed_dict={handle_placeholder: train_handle}, options=options, run_metadata=run_metadata)

    if args.timeline_save == 'on':
      many_runs_timeline.update_timeline(run_metadata.step_stats)

    # logging.info('{}_{}'.format(step, global_step))

  if args.timeline_save == 'on':
    many_runs_timeline.save(timeline_path)
    logging.info('Timeline saved to: {}'.format(timeline_path))


