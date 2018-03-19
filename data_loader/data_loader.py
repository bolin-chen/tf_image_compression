
import tensorflow as tf
import random
import logging

# https://github.com/LokLu/Tensorflow-data-loader/blob/master/data_loader.py

def read_image_list(data_list):
  f = open(data_list, 'r')
  image_paths = []
  for line in f:
    image = line.strip("\n")
    image_paths.append(image)

  return image_paths


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

def get_data_batch(data_list, batch_size, flip_ud=False, flip_lr=False, rot_90=False):
  num_threads = 4
  batch_prefetch = 2

  image_paths = read_image_list(data_list)
  images_name_tensor = tf.constant(image_paths)
  data = tf.data.Dataset.from_tensor_slices(images_name_tensor)

  data = data.shuffle(buffer_size=10000)
  data = data.repeat()

  data = data.map(parse_data,  num_parallel_calls=num_threads)

  if flip_ud:
    logging.info('-----')
    logging.info('Data augment: flip_ud')
    logging.info('-----')
    data = data.map(flip_up_down, num_parallel_calls=num_threads)
  if flip_lr:
    logging.info('-----')
    logging.info('Data augment: flip_lr')
    logging.info('-----')
    data = data.map(flip_left_right, num_parallel_calls=num_threads)
  if rot_90:
    logging.info('-----')
    logging.info('Data augment: rot_90')
    logging.info('-----')
    data = data.map(rotate90, num_parallel_calls=num_threads)

  data = data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  data = data.prefetch(batch_prefetch)

  iterator = data.make_one_shot_iterator()
  data_batch = iterator.get_next()

  dataset_init = iterator.make_initializer(data)

  return data_batch, dataset_init
