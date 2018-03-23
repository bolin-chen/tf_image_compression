
import tensorflow as tf
from tensorflow.python.client import timeline
import logging
import json
import numpy as np
from pathlib import Path

patch_size = 128

# https://github.com/ikhlestov/tensorflow_profiling/blob/master/03_merged_timeline_example.py
class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, step_stats):
        fetched_timeline = timeline.Timeline(step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()

        # convert chrome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


def compute_time(start, end):
  seconds = end - start
  m, s = divmod(seconds, 60)
  h, m = divmod(m, 60)
  d, h = divmod(h, 24)
  print('Run time: %d:%02d:%02d:%02d' % (d, h, m, s))


def set_logger(log_path, mode='a'):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


def show_variables(sess, variables):
  variables_names = [v.name for v in variables]
  values = sess.run(variables_names)
  for k, v in zip(variables_names, values):
      print('Variable: ', k)
      print('Shape: ', v.shape)


def add_trainable_variables_to_summary():
  for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)


def read_image_list(data_list):
  f = open(data_list, 'r')
  image_paths = []
  for line in f:
    image = line.strip("\n")
    image_paths.append(image)

  return image_paths


def restore_params(sess, args):
  if args.params_file == '':
    params_file = str(Path('.') / 'model_{}'.format(args.model_num) / 'params_for_test' / 'params')
  else:
    params_file = args.params_file

  saver = tf.train.Saver()
  saver.restore(sess, params_file)

  print('Params in {} restored complete'.format(params_file))


def crop_image_input_patches(image):
  height, width, channel = image.shape

  if height % patch_size != 0:
    padding_height = patch_size - (height % patch_size)
  else:
    padding_height = 0

  if width % patch_size != 0:
    padding_width = patch_size - (width % patch_size)
  else:
    padding_width = int(0)

  padded_image = np.pad(image, ((0, padding_height), (0, padding_width), (0, 0)), 'reflect')

  padded_height, padded_width, channel = padded_image.shape
  height_patch_num = padded_height // patch_size
  width_patch_num = padded_width // patch_size

  image_patches = []
  for i in range(height_patch_num):
    for j in range(width_patch_num):
      image_patches.append(padded_image[i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size])


  # print('image.shape: {}'.format(image.shape))
  # print('padding_height: {}'.format(padding_height))
  # print('padding_width: {}'.format(padding_width))
  # print('padded_image.shape: {}'.format(padded_image.shape))
  # print('height_patch_num: {}'.format(height_patch_num))
  # print('width_patch_num: {}'.format(width_patch_num))
  # print('len(image_patches): {}'.format(len(image_patches)))


  return image_patches


def concat_patches(patches, height, width):
  if height % patch_size != 0:
    height_patch_num = height // patch_size + 1
  else:
    height_patch_num = height // patch_size

  if width % patch_size != 0:
    width_patch_num = width // patch_size + 1
  else:
    width_patch_num = width // patch_size


  # print('height_patch_num: {}'.format(height_patch_num))
  # print('width_patch_num: {}'.format(width_patch_num))
  # print('len(patches): {}'.format(len(patches)))

  width_concated_patch_list = []
  for i in range(height_patch_num):
    width_concated_patch = np.concatenate(patches[i * width_patch_num : (i + 1) * width_patch_num], axis=1)
    width_concated_patch_list.append(width_concated_patch)

  # print('len(width_concated_patch_list): {}'.format(len(width_concated_patch_list)))

  recons_image = np.concatenate(width_concated_patch_list, axis=0)

  # print('recons_image.shape before crop: {}'.format(recons_image.shape))

  recons_image = recons_image[: height, : width]

  # print('recons_image.shape: {}'.format(recons_image.shape))

  return recons_image

