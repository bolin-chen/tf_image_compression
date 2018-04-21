

import os, sys
from os.path import join

patch_size = 256

def save_data_list(list_path, data_path):
  dirs_0 = [join(data_path[0], filename) for filename in os.listdir(data_path[0])]
  dirs_1 = [join(data_path[1], filename) for filename in os.listdir(data_path[1])]
  dirs = dirs_0 + dirs_1

  f = open(list_path, 'w')

  for item in dirs:
     f.write('{}\n'.format(item))


# list_path = 'data_info/valid_data_list.txt'
# data_path = ['/data/cbl/clic/professional/valid', '/data/cbl/clic/mobile/valid']


# list_path = 'data_info/tiny_valid_data_list.txt'
# data_path = ['/data/cbl/clic/tiny_valid/professional', '/data/cbl/clic/tiny_valid/mobile']




# train_patch_list_path = 'data_info/train_data_patch_list_{}.txt'.format(patch_size)
# train_patch_data_path = ['/data/cbl/clic/crop_{}/professional'.format(patch_size), '/data/cbl/clic/crop_{}/mobile'.format(patch_size)]

# valid_patch_list_path = 'data_info/valid_data_patch_list_{}.txt'.format(patch_size)
# valid_patch_data_path = ['/data/cbl/clic/valid_crop_{}/professional'.format(patch_size), '/data/cbl/clic/valid_crop_{}/mobile'.format(patch_size)]

# save_data_list(train_patch_list_path, train_patch_data_path)

# save_data_list(valid_patch_list_path, valid_patch_data_path)



# list_path = 'data_info/train_data_list.txt'
# data_path = '/data/cbl/clic/ori_train_data'

# dirs = [join(data_path, filename) for filename in os.listdir(data_path)]

# f = open(list_path, 'w')

# for item in dirs:
#    f.write('{}\n'.format(item))


# list_path = 'data_info/ori_valid_data_list.txt'
# data_path = '/data/cbl/clic/ori_valid_data'

# dirs = [join(data_path, filename) for filename in os.listdir(data_path)]

# f = open(list_path, 'w')

# for item in dirs:
#    f.write('{}\n'.format(item))



# list_path = 'data_info/recons_train_data_patch_list.txt'
# data_path = '/data/cbl/clic/crop_recons_train_data'

# dirs = [join(data_path, filename) for filename in os.listdir(data_path)]

# f = open(list_path, 'w')

# for item in dirs:
#    f.write('{}\n'.format(item))


# list_path = 'data_info/ori_train_data_patch_list.txt'
# data_path = '/data/cbl/clic/crop_ori_train_data'

# dirs = [join(data_path, filename) for filename in os.listdir(data_path)]

# f = open(list_path, 'w')

# for item in dirs:
#    f.write('{}\n'.format(item))


# list_path = 'data_info/recons_valid_data_patch_list.txt'
# data_path = '/data/cbl/clic/crop_recons_valid_data'

# dirs = [join(data_path, filename) for filename in os.listdir(data_path)]

# f = open(list_path, 'w')

# for item in dirs:
#    f.write('{}\n'.format(item))


# list_path = 'data_info/ori_valid_data_patch_list.txt'
# data_path = '/data/cbl/clic/crop_ori_valid_data'

# dirs = [join(data_path, filename) for filename in os.listdir(data_path)]

# f = open(list_path, 'w')

# for item in dirs:
#    f.write('{}\n'.format(item))


list_path = 'data_info/test_data_list.txt'
data_path = '/data/cbl/clic/test_data/test'

dirs = [join(data_path, filename) for filename in os.listdir(data_path)]

f = open(list_path, 'w')

for item in dirs:
   f.write('{}\n'.format(item))

