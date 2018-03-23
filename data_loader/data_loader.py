
import tensorflow as tf
import random
import logging

from utils import utils


def parse_data(image_path):
  image_string = tf.read_file(image_path)
  image = tf.image.decode_image(image_string, channels=3)

  image = tf.image.convert_image_dtype(image, tf.float32)

  return image


def flip_up_down(image):
  seed = random.random()
  image = tf.image.random_flip_up_down(image, seed=seed)

  return image


def flip_left_right(image):
    seed = random.random()
    image = tf.image.random_flip_left_right(image, seed=seed)

    return image


def rotate90(image):
    k = tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32)[0]
    image = tf.image.rot90(image, k)

    return image

def get_train_and_valid_data_batch(sess, train_data_list, valid_data_list, batch_size, shuffle=True, repeat=True, flip_ud=False, flip_lr=False, rot_90=False):
  num_threads = 4
  batch_prefetch = 2


  train_data = tf.data.TextLineDataset(train_data_list)

  if shuffle:
    train_data = train_data.shuffle(buffer_size=10000)
  if repeat:
    train_data = train_data.repeat()

  train_data = train_data.map(parse_data,  num_parallel_calls=num_threads)

  if flip_ud:
    logging.info('-----')
    logging.info('Data augment: flip_ud')
    logging.info('-----')
    train_data = train_data.map(flip_up_down, num_parallel_calls=num_threads)
  if flip_lr:
    logging.info('-----')
    logging.info('Data augment: flip_lr')
    logging.info('-----')
    train_data = train_data.map(flip_left_right, num_parallel_calls=num_threads)
  if rot_90:
    logging.info('-----')
    logging.info('Data augment: rot_90')
    logging.info('-----')
    train_data = train_data.map(rotate90, num_parallel_calls=num_threads)

  train_data = train_data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  # train_data = train_data.batch(batch_size)
  train_data = train_data.prefetch(batch_prefetch)


  # -----
  valid_data = tf.data.TextLineDataset(valid_data_list)
  valid_data = valid_data.map(parse_data,  num_parallel_calls=num_threads)
  valid_data = valid_data.batch(batch_size)
  valid_data = valid_data.prefetch(batch_prefetch)


  # -----
  handle_placeholder = tf.placeholder(tf.string, shape=[])
  iterator = tf.data.Iterator.from_string_handle(handle_placeholder, train_data.output_types, train_data.output_shapes)
  data_batch = iterator.get_next()

  train_iterator = train_data.make_one_shot_iterator()
  valid_iterator = valid_data.make_one_shot_iterator()

  train_handle = sess.run(train_iterator.string_handle())
  valid_handle = sess.run(valid_iterator.string_handle())

  return data_batch, handle_placeholder, train_handle, valid_handle


def get_patch_batch(batch_size, patches_placeholder):
  num_threads = 4
  batch_prefetch = 2

  data = tf.data.Dataset.from_tensor_slices(patches_placeholder)
  data = data.batch(batch_size)
  data = data.prefetch(batch_prefetch)

  iterator = data.make_initializable_iterator()
  patch_batch = iterator.get_next()

  return patch_batch, iterator


def get_data_batch(data_list, batch_size):
  num_threads = 4
  batch_prefetch = 2

  data = tf.data.TextLineDataset(data_list)
  data = data.map(parse_data,  num_parallel_calls=num_threads)
  data = data.batch(batch_size)
  data = data.prefetch(batch_prefetch)

  iterator = data.make_one_shot_iterator()
  data_batch = iterator.get_next()

  return data_batch
