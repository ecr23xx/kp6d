import os
import sys
import numpy as np
from tqdm import tqdm
opj = os.path.join

YCB = '/home/penggao/data/ycb/'
DARKNET = '/home/penggao/projects/kp6d/detect/darknet/data/ycb'
DATA = opj(YCB, 'data')
DATA_SYN = opj(YCB, 'data_syn')

if __name__ == '__main__':
    if os.path.exists(DARKNET):
        print("[WARNING] Delete existing darknet folder")
        print("[WARNING] Proceed (y/[n])?", end=' ')
        choice = input()
        if choice == 'y':
            print("[LOG] Start all anew")
            os.system('rm -r %s' % DARKNET)
            os.makedirs(DARKNET)
        else:
            print("[LOG] Terminated by user, exit.")
            sys.exit()
    else:
        print("[LOG] Start all anew")
        os.makedirs(DARKNET)

    print("[LOG] Loading dataset information")

    with open(opj(YCB, 'classes.txt')) as f:
        NAMES = [x.strip() for x in f.readlines()]

    with open(opj(YCB, 'image_sets/train_data_list.txt')) as f:
        TRAINS = [opj(YCB, x.strip() + '-color.png') for x in f.readlines()]

    with open(opj(YCB, 'image_sets/fat_train.txt')) as f:
        TRAINS.extend([opj(YCB, x.strip()) for x in f.readlines()])

    with open(opj(YCB, 'image_sets/test_data_list.txt')) as f:
        TESTS = [opj(YCB, x.strip() + '-color.png') for x in f.readlines()]

    print("[LOG] Preparing training data ...")

    for imgpath in tqdm(TRAINS):
        srcpath = imgpath
        srcbbox = srcpath.replace('.png', '.txt').replace('.jpg', '.txt')

        shortpath = imgpath.split(YCB)[-1]
        shortdir = '/'.join(shortpath.split('/')[:-1])

        os.makedirs(opj(DARKNET, 'images', shortdir), exist_ok=True)
        dstpath = opj(DARKNET, 'images', shortpath)
        dstbbox = dstpath.replace('.png', '.txt').replace('.jpg', '.txt')

        os.symlink(srcpath, dstpath)
        os.symlink(srcbbox, dstbbox)

    print("[LOG] Preparing test data ...")

    for imgpath in tqdm(TESTS):
        srcpath = imgpath
        srcbbox = srcpath.replace('.png', '.txt')

        shortpath = imgpath.split(YCB)[-1]
        shortdir = '/'.join(shortpath.split('/')[:-1])

        os.makedirs(opj(DARKNET, 'images', shortdir), exist_ok=True)
        dstpath = opj(DARKNET, 'images', shortpath)
        dstbbox = dstpath.replace('.png', '.txt')

        os.symlink(srcpath, dstpath)
        os.symlink(srcbbox, dstbbox)

    # train.txt
    print("[LOG] Preparing train.txt ...")
    with open(opj(DARKNET, 'train.txt'), 'w') as f:
        f.writelines([opj('data/ycb/images', x.split(YCB)[-1] + '\n') for x in TRAINS])

    # test.txt
    print("[LOG] Preparing test.txt ...")
    with open(opj(DARKNET, 'test.txt'), 'w') as f:
        f.writelines([opj('data/ycb/images', x.split(YCB)[-1] + '\n') for x in TESTS])

    # ycb.names
    print("[LOG] Preparing ycb.names ...")
    os.symlink(opj(YCB, 'classes.txt'), opj(DARKNET, 'ycb.names'))

    # ycb.data
    print("[LOG] Preparing ycb.data ...")
    with open(opj(DARKNET, 'ycb.data'), 'w') as f:
        f.write('classes = %d\n' % len(NAMES))
        f.write('train = %s\n' % opj('data/ycb/train.txt'))
        f.write('valid = %s\n' % opj('data/ycb/test.txt'))
        f.write('names = %s\n' % opj('data/ycb/ycb.names'))
        f.write('backup = %s\n' % opj('backup/ycb/'))

    print("[LOG] Number of training data", len(TRAINS))
    print("[LOG] Number of test data", len(TESTS))
    print("[LOG] Done!")
