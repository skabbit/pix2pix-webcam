#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: memo

loads bunch of images from a folder (and recursively from subfolders)
preprocesses (resize or crop, canny edge detection) and saves into a new folder
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import os
import cv2
import PIL.Image
import random
from tqdm import tqdm

from skimage import filters
from skimage.exposure import rescale_intensity


dim = 512  # target dimensions,
do_crop = True  # if true, resizes shortest edge to target dimensions and crops other edge. If false, does non-uniform resize

canny_thresh1 = 10
canny_thresh2 = 100

root_path = './'  # os.path.join(root_path, 'Горы/ПРИРОДА')
in_path = "/Users/skabbit/Projects/DVFU/Workshop #10 - Neural Networks #1/herbarium-clean-2/"
out_path = os.path.join(root_path, 'herbarium')


#########################################
out_path += '_' + str(dim) + '_100-200'
if do_crop:
    out_path += '_crop'

out_shape = (dim, dim)

if os.path.exists(out_path) == False:
    os.makedirs(out_path)

# eCryptfs file system has filename length limit of around 143 chars!
# https://unix.stackexchange.com/questions/32795/what-is-the-maximum-allowed-filename-and-folder-size-with-ecryptfs
max_fname_len = 140  # leave room for extension


def get_file_list(path, extensions=['jpg', 'jpeg', 'png']):
    '''returns a (flat) list of paths of all files of (certain types) recursively under a path'''
    paths = [os.path.join(root, name)
             for root, dirs, files in os.walk(path)
             for name in files
             if name.lower().endswith(tuple(extensions))]
    return paths


paths = get_file_list(in_path)
print('{} files found'.format(len(paths)))

random.shuffle(paths)

for path in paths[:10]:
    path_d, path_f = os.path.split(path)

    # combine path and filename to create unique new filename
    out_fname = path_d.split('/')[-1] + '_' + path_f

    # take last n characters so doesn't go over filename length limit
    out_fname = os.path.splitext(out_fname)[0][-max_fname_len+4:] + '.jpg'

    # print('File {} of {}, {}'.format(i, len(paths), out_fname))
    im = PIL.Image.open(path)
    im = im.convert('RGB')
    if do_crop:
        resize_shape = list(out_shape)
        if im.width < im.height:
            resize_shape[1] = int(round(float(im.height) / im.width * dim))
        else:
            resize_shape[0] = int(round(float(im.width) / im.height * dim))
        im = im.resize(resize_shape, PIL.Image.BICUBIC)
        hw = int(im.width / 2)
        hh = int(im.height / 2)
        hd = int(dim/2)
        area = (hw-hd, hh-hd, hw+hd, hh+hd)
        im = im.crop(area)

    else:
        im = im.resize(out_shape, PIL.Image.BICUBIC)

    a1 = np.array(im)
    a2 = a1.copy()
    a2 = cv2.cvtColor(a2, cv2.COLOR_RGB2GRAY)

    # a2 = cv2.Canny(a2, canny_thresh1, canny_thresh2)

    size = 11
    a2 = cv2.GaussianBlur(a2, (size,size), 0)


    threshold = filters.threshold_isodata(a2) 
    binary = a2 <= threshold
    a2 = rescale_intensity(a2 * 255, (0, threshold), (0, 255))
    a2 = a2.astype('uint8')

    a3 = cv2.equalizeHist(a2)

    a2 = cv2.cvtColor(a2, cv2.COLOR_GRAY2RGB)
    a3 = cv2.cvtColor(a3, cv2.COLOR_GRAY2RGB)
    a5 = cv2.cvtColor(binary.astype('uint8') * 255, cv2.COLOR_GRAY2RGB)

    # print(a2.shape)

    # if a2 all black, then skip it
    if np.count_nonzero(a2) > 3000:
        a3 = np.concatenate((a1, a2, a3, a5), axis=1)
        im = PIL.Image.fromarray(a3)
        # im.save(os.path.join(out_path, out_fname))
        im.show()
