
import os
import numpy as np
from skimage import io
import tensorflow as tf
from rmbe import model as rmbe_model

from data_loader import data_loader



patch_size = 128


def rmbe(image):
  offset = 64

  height, width, channel = image.shape

  new_image = rmbe_height(image, height, width, offset)

  new_image = rmbe_width(new_image, height, width, offset)


  return new_image


def run_rmbe_model(patch_list):
  g2 = tf.Graph()
  sess = tf.Session(graph=g2)

  with sess.as_default():
    with g2.as_default():
      patches_placeholder = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3])


      batch_size = 64
      patch_batch, iterator = data_loader.get_patch_batch(batch_size, patches_placeholder)

      output_op = rmbe_model.model(patch_batch)

      params_file = 'rmbe/rmbe_params/params'
      saver = tf.train.Saver()
      saver.restore(sess, params_file)

  sess.run(iterator.initializer, feed_dict={patches_placeholder: patch_list})

  new_patch_list = []
  while True:
    try:
      output = sess.run(output_op)
      new_patch_list.append(output)
    except tf.errors.OutOfRangeError:
      break

  new_patches = np.concatenate(new_patch_list, axis=0)

  # print(len(new_patch_list))
  # print(new_patches.shape)
  # print('-----')

  # print(dir(rmbe_model))
  # print('-----')

  # new_patches = patch_list

  return new_patches


def rmbe_height(image, height, width, offset):
  height_patch_num = height // patch_size
  width_patch_num = (width - offset) // patch_size

  patch_list = []
  for i in range(height_patch_num):
    for j in range(width_patch_num):
      patch = image[i * patch_size : (i + 1) * patch_size, offset + j * patch_size : offset + (j + 1) * patch_size, :]
      patch_list.append(patch)


  new_patch_list = run_rmbe_model(patch_list)
  new_image = image[:, :, :]

  for i in range(height_patch_num):
    for j in range(width_patch_num):
      new_image[i * patch_size : (i + 1) * patch_size, offset + j * patch_size : offset + (j + 1) * patch_size, :] = new_patch_list[i * width_patch_num + j]


  return new_image


def rmbe_width(image, height, width, offset):
  height_patch_num = (height - offset) // patch_size
  width_patch_num = width // patch_size

  patch_list = []
  for i in range(height_patch_num):
    for j in range(width_patch_num):
      patch = image[offset + i * patch_size : offset + (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size, :]
      patch_list.append(patch)


  new_patch_list = run_rmbe_model(patch_list)
  new_image = image[:, :, :]

  for i in range(height_patch_num):
    for j in range(width_patch_num):
      new_image[offset + i * patch_size : offset + (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size, :] = new_patch_list[i * width_patch_num + j]


  return new_image


def test():
  file_list = ['0067.png', '2017-07-27 16.21.36.png', '2017-08-28 10.51.27.png']

  for image_file in file_list:
    input_dir = 'input'
    image_path = os.path.join(input_dir, image_file)
    image = io.imread(image_path)
    new_image = rmbe(image)

    output_dir = 'output'
    save_path = os.path.join(output_dir, image_file)
    io.imsave(save_path, new_image)


# if __name__ == '__main__':
#   test()

  # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

  # config = tf.ConfigProto()
  # config.gpu_options.allow_growth = True

  # sess = tf.Session(config=config)

  # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  # sess = tf.Session()
  # import model

  # main()

