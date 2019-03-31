"""Generate train.txt and test.txt for object."""

import os
import sys
import h5py
import argparse
import numpy as np
from PIL import Image
import scipy.io as scio
from tqdm import tqdm

opj = os.path.join

YCBROOT = '/media/data_1/home/chengkun/pose_6D_methods/kp6d/datasets/ycb'
YCB_DATA_ROOT = opj(YCBROOT, 'YCB_Video_Dataset')
DATA = opj(YCBROOT, 'YCB_Video_Dataset/data')
DATA_SYN = opj(YCBROOT, 'YCB_Video_Dataset/data_syn')


def parse_arg():
    parser = argparse.ArgumentParser(description='Split for objects.')
    parser.add_argument('--outroot',
                        default='/media/data_1/home/chengkun/pose_6D_methods/kp6d/datasets/ycb',
                        type=str, help="Output dir")
    parser.add_argument('--object', type=str, help="Object", default='002_master_chef_can')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()

    with open(opj(YCBROOT, 'train_data_list.txt')) as f:
        TRAINS = [x.strip() for x in f.readlines()]

    with open(opj(YCBROOT, 'test_data_list.txt')) as f:
        TESTS = [x.strip() for x in f.readlines()]

    with open(opj(YCBROOT, 'classes.txt')) as f:
        NAMES = [x.strip() for x in f.readlines()]
        CLASSES = {}
        for i in range(len(NAMES)):
            CLASSES[NAMES[i]] = i + 1

    if not os.path.isfile(opj(args.outroot, args.object, 'train.txt')):
        TRAINS_OBJ = []
        for i in range(len(TRAINS)):
            label = np.array(Image.open('{0}/{1}-label.png'.format(YCB_DATA_ROOT, TRAINS[i])))

            if i % 100 == 0 or i == len(TRAINS_OBJ) - 1:
                print('{0}/{1} in train set contains this object.'.format(len(TRAINS_OBJ), i))

            if CLASSES[args.object] in np.unique(label):
                TRAINS_OBJ.append(TRAINS[i])

        with open(opj(args.outroot, args.object, 'train.txt'), 'w') as f:
            for item in TRAINS_OBJ:
                f.write("%s\n" % item)

    if not os.path.isfile(opj(args.outroot, args.object, 'test.txt')):
        TESTS_OBJ = []
        for i in range(len(TESTS)):
            label = np.array(Image.open('{0}/{1}-label.png'.format(YCB_DATA_ROOT, TESTS[i])))

            if i % 100 == 0 or i == len(TESTS) - 1:
                print('{0}/{1} in test set contains this object.'.format(len(TESTS_OBJ), i))

            if CLASSES[args.object] in np.unique(label):
                TESTS_OBJ.append(TESTS[i])

        with open(opj(args.outroot, args.object, 'test.txt'), 'w') as f:
            for item in TESTS_OBJ:
                f.write("%s\n" % item)
