

import os, sys
from os.path import join

# list_path = 'data_info/train_data_list.txt'
# data_path = ['/data/cbl/clic/crop_128/professional', '/data/cbl/clic/crop_128/mobile']


# list_path = 'data_info/valid_data_list.txt'
# data_path = ['/data/cbl/clic/professional/valid', '/data/cbl/clic/mobile/valid']


list_path = 'data_info/tiny_valid_data_list.txt'
data_path = ['/data/cbl/clic/tiny_valid/professional', '/data/cbl/clic/tiny_valid/mobile']


dirs_0 = [join(data_path[0], filename) for filename in os.listdir(data_path[0])]
dirs_1 = [join(data_path[1], filename) for filename in os.listdir(data_path[1])]
dirs = dirs_0 + dirs_1

f = open(list_path, 'w')

for item in dirs:
   f.write('{}\n'.format(item))
