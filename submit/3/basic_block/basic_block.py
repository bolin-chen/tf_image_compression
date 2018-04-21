
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average


def batch_norm(x, is_training, eps=1e-05, decay=0.9, name='batch_norm'):
  with tf.variable_scope(name):
    params_shape = x.shape[-1:]

    moving_mean = tf.get_variable('mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
    moving_variance = tf.get_variable('variance', params_shape, initializer=tf.ones_initializer, trainable=False)

    beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
    gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)

    if is_training:
      mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
      with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                    assign_moving_average(moving_variance, variance, decay)]):

        return tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
    else:
      return tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, eps)



def my_conv2d(inputs, filters, strides, kernel_size, padding, activation, kernel_initializer, bias_initializer, name, is_training=True, use_bn=False, bn_decay=0.6):
  with tf.variable_scope(name):
    input_shape = inputs.get_shape().as_list()
    weight_shape = kernel_size + [input_shape[-1], filters]

    weight = tf.get_variable('kernel', shape=weight_shape, initializer=kernel_initializer)
    output = tf.nn.conv2d(inputs, weight, strides=[1, strides[0], strides[1], 1], padding=padding)

    bias = tf.get_variable('bias', shape=[filters],initializer=bias_initializer)
    output = tf.nn.bias_add(output, bias)

    if use_bn:
      output = batch_norm(output, is_training=is_training, eps=1e-3, decay=bn_decay)

    tf.summary.histogram('pre_activation', output)

    output = activation(output)

    tf.summary.histogram('activation', output)

    return output


def my_conv2d_transpose(inputs, filters, strides, kernel_size, padding, activation, kernel_initializer, bias_initializer, name, is_training=True, use_bn=False, bn_decay=0.6):
  with tf.variable_scope(name):
    input_shape = inputs.get_shape().as_list()
    weight_shape = kernel_size + [filters, input_shape[-1]]
    output_shape = tf.stack([tf.shape(inputs)[0], input_shape[1] * 2, input_shape[2] * 2, filters])

    weight = tf.get_variable('kernel', shape=weight_shape, initializer=kernel_initializer)
    output = tf.nn.conv2d_transpose(inputs, weight, output_shape=output_shape, strides=[1, strides[0], strides[1], 1], padding=padding)

    bias = tf.get_variable('bias', shape=[filters],initializer=bias_initializer)
    output = tf.nn.bias_add(output, bias)

    if use_bn:
      output = batch_norm(output, is_training=is_training, eps=1e-3, decay=bn_decay)

    tf.summary.histogram('pre_activation', output)

    output = activation(output)

    tf.summary.histogram('activation', output)

    return output


def res_block(inputs, filters, strides, kernel_size, padding, activation, kernel_initializer, bias_initializer, name, layer_num):
  with tf.variable_scope(name):
    output = inputs

    for i in range(layer_num):
      output = my_conv2d(
        inputs=output,
        filters=filters,
        strides=strides,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='conv_{}'.format(i)
      )

    output = inputs + output

    return output


def res_block_2(inputs, filters, strides, kernel_size, padding, activation, kernel_initializer, bias_initializer, name, layer_num):
  with tf.variable_scope(name):
    for i in range(layer_num):
      output = my_conv2d(
        inputs=inputs,
        filters=filters,
        strides=strides,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='conv_{}'.format(i)
      )

      inputs = inputs + output

    return output


def dense_block(inputs, filters, strides, kernel_size, padding, activation, kernel_initializer, bias_initializer, name, layer_num):
  with tf.variable_scope(name):
    input_shape = inputs.get_shape().as_list()

    output = my_conv2d(
      inputs=inputs,
      filters=input_shape[-1] / 2,
      strides=[1, 1],
      kernel_size=[1, 1],
      padding=padding,
      activation=activation,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      name='conv_transition'
    )

    inputs = output

    for i in range(layer_num):
      output = my_conv2d(
        inputs=inputs,
        filters=filters,
        strides=strides,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='conv_{}'.format(i)
      )

      inputs = tf.concat((inputs, output), axis=3)

    return output


def reverse_sigmoid(input):
  output = tf.log(input / (1 - input))

  return output
