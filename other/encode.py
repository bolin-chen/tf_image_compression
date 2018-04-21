#!/usr/bin/env python3

from PIL import Image
from glob import glob

for image_file in glob('valid_ori/*.png'):
    Image.open(image_file).save(image_file[:-3].replace('ori', 'encoded') + 'jpg', quality=5, optimize=True)
