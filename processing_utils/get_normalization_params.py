
import cv2
import numpy as np

train_data_list = 'data_info/train_data_list.txt'

def read_image_list(data_list):
  f = open(data_list, 'r')
  image_paths = []
  for line in f:
    image = line.strip("\n")
    image_paths.append(image)

  return image_paths


# # https://stackoverflow.com/questions/15638612/calculating-mean-and-standard-deviation-of-the-data-which-does-not-fit-in-memory
# def online_mean_and_std(image_list):
#   n = 0
#   mean = 0
#   var = 0

#   count = 0

#   for index, image_path in enumerate(image_list):
#     x = cv2.imread(image_path)
#     x = np.asarray(x, dtype=np.float32)
#     x = x / 255

#     # print(x.shape)
#     # print(x[0, 0].shape)
#     # break

#     n = n + 1
#     delta = x - mean
#     mean = mean + delta / n
#     var = var + delta * (x - mean)

#     if index % 1000 == 0:
#       print(index)

#     # count += 1
#     # if count == 3:
#     #   print('x.shape: ', x.shape)
#     #   print('x: ', x)
#     #   print('n: ', n)

#     #   print('delta.shape: ', delta.shape)
#     #   print('delta: ', delta)

#     #   print('mean.shape: ', mean.shape)
#     #   print('mean: ', mean)

#     #   print('var.shape: ', var.shape)
#     #   print('var: ', var)

#     #   break


#   std = np.sqrt(var)


#   return mean, std


def online_mean_and_std_channel(image_list):
  n = 0
  mean = 0
  square_mean = 0

  # count = 0

  for index, image_path in enumerate(image_list):
    x = cv2.imread(image_path)
    x = np.asarray(x, dtype=np.float32)
    x = x / 255

    # print(x.shape)
    # print(x[0, 0].shape)
    # break

    prev_n = n
    n += x.shape[0] * x.shape[1]

    x = x.reshape([-1, 3])
    square_x = np.square(x)

    square_mean = square_mean * (1.0 *  prev_n / n) + np.sum(square_x, axis=0) / n
    mean = mean * (1.0 * prev_n / n) + np.sum(x, axis=0) / n

    if index % 10000 == 0:
      print(index)

    # print(1.0 * prev_n / n)
    # print(index)
    # print(x.shape)
    # print(square_x.shape)
    # print(np.sum(x, axis=0) / (128 * 128))
    # print(np.sum(square_x, axis=0) / (128 * 128))
    # print(mean)
    # print(square_mean)

    # if index == 10:
    #   break

  var = square_mean - np.square(mean)
  std = np.sqrt(var)

  return mean, std


def save_params(mean, std):
  # filename ='normalization_params.npz'

  filename ='data_info/channel_normalization_params.npz'

  np.savez(filename, mean=mean, std=std)
  print('Normalization parameters saved to {}'.format(filename))

  print('-----')
  print(mean)
  print('Mean max: {}'.format(np.max(mean)))
  print('Mean min: {}'.format(np.min(mean)))
  print('-----')
  print(std)
  print('Std max: {}'.format(np.max(std)))
  print('Std min: {}'.format(np.min(std)))
  print('-----')


image_list = read_image_list(train_data_list)
# mean, std = online_mean_and_std(image_list)

mean, std = online_mean_and_std_channel(image_list)
save_params(mean, std)
