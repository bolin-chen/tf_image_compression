
import tensorflow as tf


def my_conv2d(inputs, filters, strides, kernel_size, padding, activation, kernel_initializer, bias_initializer, name):
  with tf.variable_scope(name):
    input_shape = inputs.get_shape().as_list()
    weight_shape = kernel_size + [input_shape[-1], filters]

    weight = tf.get_variable('kernel', shape=weight_shape, initializer=kernel_initializer)
    output = tf.nn.conv2d(inputs, weight, strides=[1, strides[0], strides[1], 1], padding=padding)

    bias = tf.get_variable('bias', shape=[filters],initializer=bias_initializer)
    output = tf.nn.bias_add(output, bias)

    tf.summary.histogram('pre_activation', output)

    output = activation(output)

    tf.summary.histogram('activation', output)

    return output


def my_conv2d_transpose(inputs, filters, strides, kernel_size, padding, activation, kernel_initializer, bias_initializer, name):
  with tf.variable_scope(name):
    input_shape = inputs.get_shape().as_list()
    weight_shape = kernel_size + [filters, input_shape[-1]]
    output_shape = tf.stack([tf.shape(inputs)[0], input_shape[1] * 2, input_shape[2] * 2, filters])

    weight = tf.get_variable('kernel', shape=weight_shape, initializer=kernel_initializer)
    output = tf.nn.conv2d_transpose(inputs, weight, output_shape=output_shape, strides=[1, strides[0], strides[1], 1], padding=padding)

    bias = tf.get_variable('bias', shape=[filters],initializer=bias_initializer)
    output = tf.nn.bias_add(output, bias)

    tf.summary.histogram('pre_activation', output)

    output = activation(output)

    tf.summary.histogram('activation', output)

    return output


def reverse_sigmoid(input):
  output = tf.log(input / (1 - input))
  return output
