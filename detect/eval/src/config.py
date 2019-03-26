import os
import torch
import json
from torchvision import transforms

opj = os.path.join
ROOT = '/home/penggao/projects/pose/kp6d/detect'


def prepare_weight(ckpt):
    """
    Args
    - ckpt: (str) Relative path to weight file
    """
    weightfile = opj(ROOT, 'darknet/backup', ckpt)
    return weightfile


def prepare_cfg(name):
    """
    Prepare configuration file path

    Args
    - name: (str) Dataset name

    Return
    - cfgfile: (str) Configuration file path
    """
    if name == 'linemod-single':
        return opj(ROOT, 'darknet/cfg/single.cfg')
    elif name == 'ycb':
        return opj(ROOT, 'darknet/cfg/ycb.cfg')
    else:
        raise NotImplementedError


def class_name(name, idx):
    """
    Args
    - name: (str) Dataset name
    - idx: (int or string) Class index
    """
    LINEMOD = ('ape', 'bvise', 'bowl', 'camera', 'can', 'cat', 'cup', 'driller',
               'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone')
    YCB = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can'
           '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box',
           '010_potted_meat_can', '011_banana', '019_pitcher_base', '021_bleach_cleanser',
           '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors',
           '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
    if 'linemod' in name:
        return LINEMOD[int(idx) - 1]
    elif 'ycb' in name:
        return YCB[int(idx) - 1]
    else:
        raise NotImplementedError
