#!/usr/bin/env python
__author__ = 'solivr'

from gtiff_crop import crop_gtiff
from scipy.misc import imread
from itertools import product
from collections import namedtuple
from tqdm import tqdm
import os


def compute_patches_coords(step, overlap, shape_img):
    x_ind = [0, step]
    y_ind = [0, step]
    i = 0
    while x_ind[-1] + step < shape_img[0]:
        if i % 2 == 0:
            x_ind.append(x_ind[-1] - overlap)
        else:
            x_ind.append(x_ind[-1] + step)
        i += 1
    if i % 2 == 0:
        x_ind = x_ind + [x_ind[-1] - overlap, shape_img[0]]
    else:
        x_ind.append(shape_img[0])

    i = 0
    while y_ind[-1] + step < shape_img[1]:
        if i % 2 == 0:
            y_ind.append(y_ind[-1] - overlap)
        else:
            y_ind.append(y_ind[-1] + step)
        i += 1
    if i % 2 == 0:
        y_ind = y_ind + [y_ind[-1] - overlap, shape_img[1]]
    else:
        y_ind.append(shape_img[1])

    y_coords = [(y_ind[i - 1], y_ind[i]) for i in range(1, len(y_ind), 2)]
    x_coords = [(x_ind[i - 1], x_ind[i]) for i in range(1, len(x_ind), 2)]

    return list(product(x_coords, y_coords))


# filename = '/home/soliveir/Cadasters_all/cadasters_georeferenced/Fg_19_centered.tif'
# output_dir = '/home/soliveir/Cadasters_all/cadasters_georeferenced/patches/fg19'

def make_patches_from_gtif(gtif_filename, step, overlap, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(gtif_filename).split('.')

    img = imread(gtif_filename)
    shape = img.shape[:2]

    full_coords = compute_patches_coords(step, overlap, shape)
    CropParams = namedtuple('CropParams', 'x y w h')

    for i, (cx, cy) in tqdm(enumerate(full_coords), total=len(full_coords)):
        crop_info = CropParams(cx[0], cy[0], cx[1]-cx[0], cy[1]-cy[0])
        crop_gtiff(gtif_filename, crop_info,
                   os.path.join(output_dir, '{}_patch{}.{}'.format(basename[0], i, basename[1])))
