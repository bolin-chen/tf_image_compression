#!/usr/bin/env python3


import numpy as np
import time
import line_profiler

def rlc_encoder(seq_data):
  max_count = 3

  symbol_list = []
  count_list = []

  count = 0
  pre_ele = None

  for ele in seq_data:
    if (ele == pre_ele) and (count < max_count):
      count += 1
    else:
      if pre_ele != None:
        symbol_list.append(pre_ele)
        count_list.append(count)


      pre_ele = ele
      count = 1

  symbol_list.append(pre_ele)
  count_list.append(count)

  # print('symbol_list', symbol_list)
  # print('count_list', count_list)
  # print('-----')

  encoded_data = symbol_list + count_list

  return encoded_data


@profile
def rlc_decoder(encoded_data):
  symbol_list = encoded_data[: len(encoded_data) // 2]
  count_list = encoded_data[len(encoded_data) // 2:]

  seq_data = []

  for (symbol, count) in zip(symbol_list, count_list):
    seq_data += [symbol] * count

  # print('symbol_list', symbol_list)
  # print('count_list', count_list)

  return seq_data


# @profile
# def rlc_encoder_2(seq_data):
#   max_count = 3


#   seq_data = np.asarray(seq_data)
#   residual = seq_data[1 :] - seq_data[: -1]

#   position = np.where(residual != 0)
#   position = np.insert(position, 0, -1)
#   position = np.append(position, seq_data.shape[0] - 1)

#   # print('position: {}'.format(position))

#   count_list = position[1 :] - position[: -1]

#   # print('count_list: {}'.format(count_list))

#   big_count_index = np.where(count_list > max_count)[0]

#   # print('big_count_index: {}'.format(big_count_index))

#   big_count_list = count_list[big_count_index]

#   # print('big_count_list: {}'.format(big_count_list))

#   m = big_count_list // max_count
#   n = big_count_list % max_count

#   # print('m: {}'.format(m))
#   # print('n: {}'.format(n))

#   # print('count_list: {}'.format(count_list))
#   # print('count_list > max_count: {}'.format(count_list > max_count))

#   np.place(count_list, count_list > max_count, [max_count])

#   # print('count_list: {}'.format(count_list))

#   count_list = np.append(count_list, [0])

#   # print('count_list: {}'.format(count_list))
#   # print('big_count_index + 1: {}'.format(big_count_index + 1))

#   count_list = np.insert(count_list, big_count_index + 1, n)
#   count_list = count_list[: -1]

#   # print('count_list: {}'.format(count_list))
#   # print('big_count_index + np.arange(big_count_index.shape[0]): {}'.format(big_count_index + np.arange(big_count_index.shape[0])))

#   repeat_index = np.ones(count_list.shape, dtype=np.int8)
#   repeat_index[big_count_index + np.arange(big_count_index.shape[0])] = m

#   # print('repeat_index: {}'.format(repeat_index))

#   count_list = np.repeat(count_list, repeat_index)

#   # print('count_list: {}'.format(count_list))

#   count_list = count_list[count_list > 0]

#   # print('count_list: {}'.format(count_list))

#   new_position = np.insert(np.cumsum(count_list), 0, 0)[: -1]

#   # print('new_position: {}'.format(new_position))

#   symbol_list = seq_data[new_position]

#   # print('symbol_list: {}'.format(symbol_list))

#   encoded_data = symbol_list.tolist() + count_list.tolist()

#   return encoded_data


@profile
def rlc_encoder_2(seq_data):
  max_count = 3


  seq_data = np.asarray(seq_data)
  residual = seq_data[1 :] - seq_data[: -1]

  position = np.where(residual != 0)
  position = np.insert(position, 0, -1)
  position = np.append(position, seq_data.shape[0] - 1)

  count_list = position[1 :] - position[: -1]

  big_count_index = np.where(count_list > max_count)[0]

  big_count_list = count_list[big_count_index]

  m = big_count_list // max_count
  n = big_count_list % max_count

  np.place(count_list, count_list > max_count, [max_count])

  count_list = np.append(count_list, [0])

  count_list = np.insert(count_list, big_count_index + 1, n)
  count_list = count_list[: -1]

  repeat_index = np.ones(count_list.shape, dtype=np.int8)
  repeat_index[big_count_index + np.arange(big_count_index.shape[0])] = m

  count_list = np.repeat(count_list, repeat_index)

  count_list = count_list[count_list > 0]

  new_position = np.insert(np.cumsum(count_list), 0, 0)[: -1]

  symbol_list = seq_data[new_position]

  encoded_data = symbol_list.tolist() + count_list.tolist()

  return encoded_data


@profile
def rlc_encoder_3(seq_data):
  max_count = 3


  seq_data = np.asarray(seq_data)
  residual = seq_data[1 :] - seq_data[: -1]

  position = np.where(residual != 0)
  position = np.insert(position, 0, -1)
  position = np.append(position, seq_data.shape[0] - 1)

  count_list = position[1 :] - position[: -1]

  big_count_index = np.where(count_list > max_count)[0]

  big_count_list = count_list[big_count_index]

  m = big_count_list // max_count
  n = big_count_list % max_count

  np.place(count_list, count_list > max_count, [max_count])

  count_list = np.append(count_list, [0])

  count_list = np.insert(count_list, big_count_index + 1, n)
  count_list = count_list[: -1]

  repeat_index = np.ones(count_list.shape, dtype=np.int8)
  repeat_index[big_count_index + np.arange(big_count_index.shape[0])] = m

  count_list = np.repeat(count_list, repeat_index)

  count_list = count_list[count_list > 0]

  new_position = np.insert(np.cumsum(count_list), 0, 0)[: -1]

  symbol_list = seq_data[new_position]

  encoded_data = np.concatenate([symbol_list, count_list]).tolist()

  return encoded_data


def rlc_decoder_2(encoded_data):
  symbol_list = encoded_data[: len(encoded_data) // 2]
  count_list = encoded_data[len(encoded_data) // 2 :]

  seq_data = np.repeat(symbol_list, count_list)

  seq_data = seq_data.tolist()

  return seq_data


if __name__ == '__main__':

  seq_data_0 = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
  seq_data_1 = [1, 0, 1, 1, 0, 0, 1]
  seq_data_2 = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
  seq_data_3 = [1, 1, 1, 0, 0, 0, 0, 0, 0]
  seq_data_4 = [1, 0, 1,1,1, 0,0,0,0,0, 1,1,1,1,1,1, 0,0,0,0, 1, 0,0,0,0,0,0]

  seq_data_5 = np.around(np.random.rand(64 * 32 * 32 * 64))


  # start = time.time()
  # rlc_encoder(seq_data_5)
  # print('run time: {}'.format(time.time() - start))


  start = time.time()
  rlc_encoder_2(seq_data_5)
  print('run time: {}'.format(time.time() - start))

  start = time.time()
  rlc_encoder_3(seq_data_5)
  print('run time: {}'.format(time.time() - start))

  # rlc_encoder_2(seq_data_4)


  # print(rlc_decoder_2(rlc_encoder_2(seq_data_4)))
  # print(rlc_decoder_2(rlc_encoder_2(seq_data_4)) == seq_data_4)


  # rlc_encoder(seq_data_0)
  # rlc_encoder(seq_data_1)
  # rlc_encoder(seq_data_2)

  # print(rlc_decoder(rlc_encoder(seq_data_0)) == seq_data_0)
  # print(rlc_decoder(rlc_encoder(seq_data_1)) == seq_data_1)
  # print(rlc_decoder(rlc_encoder(seq_data_2)) == seq_data_2)
