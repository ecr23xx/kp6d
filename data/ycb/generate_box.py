import os
import json
import argparse
import imagesize
from tqdm import tqdm
from scipy.io import loadmat

from utils import *
opj = os.path.join


def parse_arg():
    parser = argparse.ArgumentParser(description='YOLO v3 training')
    parser.add_argument('--overwrite', action='store_true',
                        help="Not to overwrite existing bbox file")
    return parser.parse_args()


if __name__ == "__main__":
    print("[LOG] Start generating bbox")

    YCB = '/home/penggao/data/ycb/'
    DATA = opj(YCB, 'data/')
    DATA_SYN = opj(YCB, 'data_syn/')
    FAT = opj(YCB, 'fat/')

    print("[LOG] Load names")

    with open(opj(YCB, 'classes.txt')) as f:
        NAMES = [x.strip() for x in f.readlines()]

    print("[LOG] Generate for data/")

    # for dirname in tqdm(os.listdir(DATA)):
    #     if not os.path.isdir(opj(DATA, dirname)):
    #         continue

    #     for imgname in tqdm(os.listdir(opj(DATA, dirname))):
    #         if '-color.png' not in imgname:
    #             continue

    #         fullpath = opj(DATA, dirname, imgname)
    #         ycb_bbox = fullpath.replace('-color.png', '-box.txt')  # ycb format
    #         voc_bbox = fullpath.replace('.png', '.txt')  # voc format
    #         voc_annos = []

    #         if os.path.exists(voc_bbox):
    #             continue

    #         with open(ycb_bbox) as f:
    #             content = [x.strip() for x in f.readlines()]

    #         W, H = imagesize.get(fullpath)
    #         for x in content:
    #             class_name, x1, y1, x2, y2 = x.split(' ')
    #             idx = NAMES.index(class_name)
    #             x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    #             if x1 < 0 or x2 > W or y1 < 0 or y2 > H or x1 >= x2 or y1 >= y2:
    #                 continue
    #             xc = (x1 + x2) / 2
    #             yc = (y1 + y2) / 2
    #             w = x2 - x1
    #             h = y2 - y1
    #             voc_annos.append('%d %f %f %f %f\n' %
    #                              (idx, xc/W, yc/H, w/W, h/H))

    #         with open(voc_bbox, 'w') as f:
    #             f.writelines(voc_annos)

    print("\n[LOG] Generate for data_syn/")

    # for imgname in tqdm(os.listdir(DATA_SYN)):
    #     if '-color.png' not in imgname:
    #         continue

    #     fullpath = opj(DATA_SYN, imgname)
    #     ycb_bbox = fullpath.replace('-color.png', '-box.txt')  # ycb format
    #     voc_bbox = fullpath.replace('.png', '.txt')  # voc format
    #     ycb_annos = []
    #     voc_annos = []
    #     metapath = fullpath.replace('-color.png', '-meta.mat')

    #     if os.path.exists(ycb_bbox) and os.path.exists(voc_bbox):
    #         continue

    #     W, H = imagesize.get(fullpath)
    #     meta = loadmat(metapath)
    #     poses = meta['poses'].transpose(2, 0, 1)
    #     cam = meta['intrinsic_matrix']
    #     indexes = meta['cls_indexes'].astype(int)[0]
    #     for idx, cls_idx in enumerate(indexes):
    #         modelpath = opj(YCB, 'models/%s/points.xyz' %
    #                         NAMES[cls_idx-1])
    #         xyz = np.loadtxt(modelpath)
    #         xy = project_vertices(xyz, poses[idx], cam)
    #         x1, y1, x2, y2 = get_2d_corners(xy)
    #         if x1 < 0 or x2 > W or y1 < 0 or y2 > H or x1 >= x2 or y1 >= y2:
    #             continue
    #         ycb_annos.append('%s %f %f %f %f\n' %
    #                          (NAMES[cls_idx-1], x1, y1, x2, y2))
    #         xc, yc, w, h = corner2center(x1, y1, x2, y2)
    #         voc_annos.append('%d %f %f %f %f\n' %
    #                          (cls_idx-1, xc/W, yc/H, w/W, h/H))

    #     with open(voc_bbox, 'w') as f:
    #         f.writelines(voc_annos)

    #     with open(ycb_bbox, 'w') as f:
    #         f.writelines(ycb_annos)

    print("[LOG] Generate for fat/")

    fat_lists = []

    for dirname in tqdm(os.listdir(opj(FAT, 'mixed'))):
        for annoname in tqdm(os.listdir(opj(FAT, 'mixed', dirname))):
            if 'left.json' not in annoname:
                continue

            annopath = opj(FAT, 'mixed', dirname, annoname)
            fullpath = annopath.replace('.json', '.jpg')
            voc_bbox = fullpath.replace('.jpg', '.txt')
            voc_annos = []
            fat_lists.append(fullpath.split('ycb/')[-1] + '\n')

            if os.path.exists(voc_bbox):
                continue

            with open(annopath) as f:
                data = json.load(f)

            W, H = imagesize.get(fullpath)
            for obj in data['objects']:
                y1, x1 = obj['bounding_box']['top_left']
                y2, x2 = obj['bounding_box']['bottom_right']
                if x1 > x2 or y1 > y2 or x1 < 0 or x2 > W or y1 < 0 or y2 > H:
                    continue
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                label = NAMES.index(obj['class'][:-4])
                voc_annos.append('%d %f %f %f %f\n' %
                                 (label, xc/W, yc/H, w/W, h/H))

            with open(voc_bbox, 'w') as f:
                f.writelines(voc_annos)

    with open(opj(YCB, 'image_sets/fat_train.txt'), 'w') as f:
        f.writelines(fat_lists)

    print("\n[LOG] Done.")
