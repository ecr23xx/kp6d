import os
import sys
import h5py
import argparse
import numpy as np
import scipy.io as scio
import numpy.ma as ma
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm

opj = os.path.join

YCBROOT = '/media/data_1/home/chengkun/pose_6D_methods/kp6d/datasets/ycb'
YCB_DATA_ROOT = opj(YCBROOT, 'YCB_Video_Dataset')
DATA = opj(YCBROOT, 'YCB_Video_Dataset/data')
DATA_SYN = opj(YCBROOT, 'YCB_Video_Dataset/data_syn')

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def parse_arg():
    parser = argparse.ArgumentParser(description='Keypoints data for ycb.')
    parser.add_argument('--outroot',
                        default='/media/data_1/home/chengkun/pose_6D_methods/kp6d/datasets/ycb',
                        type=str, help="Output dir")
    parser.add_argument('--object', type=str, help="Object", default='002_master_chef_can')
    parser.add_argument('--kpnum', default=8, choices=[8],
                        type=int, help="Number of keypoints")
    parser.add_argument('--kptype', default='cluster', choices=['sift', 'cluster'],
                        type=str, help="Type of keypoints")
    parser.add_argument('--kpdroot', type=str, help="KPD data root directory",
                        default='/media/data_1/home/chengkun/pose_6D_methods/kp6d/datasets/ycb/gt')
    return parser.parse_args()


def project_vertices(cam, vertices, pose):
    """Project 3d vertices to 2d

    Args
    - vertices: (np.array) [N x 3] 3d keypoints vertices.
    - pose: (np.array) [3 x 4] pose matrix

    Returns
    - projected: (np.array) [N x 2] projected 2d points
    """
    vertices = np.concatenate(
        (vertices, np.ones((vertices.shape[0], 1))), axis=1)
    projected = np.matmul(np.matmul(cam, pose), vertices.T)
    projected /= projected[2, :]
    projected = projected[:2, :].T
    return projected


def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


if __name__ == '__main__':
    args = parse_arg()
    print("[LOG] Preparing h5 for KPD training")
    print("[LOG] Number of keypoints: %d" % args.kpnum)
    print("[LOG] Type of keypoints: %s" % args.kptype)
    print("[LOG] Object: %s" % args.object)

    KPDROOT = opj(args.kpdroot, str(args.kpnum), args.kptype, args.object)

    if os.path.exists(KPDROOT):
        print("[WARNING] Overwriting existing annotations in %s" % KPDROOT)
        print("[WARNING] Proceed (y/[n])?", end=' ')
        choice = input()
        if choice == 'y':
            os.system('rm -r %s' % KPDROOT)
        else:
            print("[LOG] Terminated by user, exit")
            sys.exit()

    os.makedirs(KPDROOT)

    print("[LOG] Preparing data")
    kp_path = os.path.join(YCBROOT, 'kps', str(args.kpnum), args.kptype, '%s_kp.ply' % args.object)
    kp_data = PlyData.read(kp_path)
    kps_xyz = np.stack((np.array(kp_data['vertex']['x']),
                        np.array(kp_data['vertex']['y']),
                        np.array(kp_data['vertex']['z'])), axis=1)
    with open(opj(YCBROOT, args.object, 'train.txt'), 'r') as f:
        trainlists = f.readlines()
    trainlists = [x.strip() for x in trainlists]

    with open(opj(YCBROOT, args.object, 'test.txt'), 'r') as f:
        testlists = f.readlines()
    testlists = [x.strip() for x in testlists]

    with open(opj(YCBROOT, 'classes.txt')) as f:
        NAMES = [x.strip() for x in f.readlines()]
        CLASSES = {}
        for i in range(len(NAMES)):
            CLASSES[NAMES[i]] = i + 1

    lists = {'train': trainlists, 'test': testlists}
    bboxes = {'train': [], 'test': []}
    kps = {'train': [], 'test': []}
    imgnames = {'train': [], 'test': []}

    for tag in ('train', 'test'):
        imglist = lists[tag]
        for i in range(len(imglist)):
            imgname = u'{0}/{1}-color.png'.format(YCB_DATA_ROOT, imglist[i])
            label = np.array(Image.open('{0}/{1}-label.png'.format(YCB_DATA_ROOT, imglist[i])))
            meta = scio.loadmat('{0}/{1}-meta.mat'.format(YCB_DATA_ROOT, imglist[i]))
            mask_label = ma.getmaskarray(ma.masked_equal(label, CLASSES[args.object]))
            rmin, rmax, cmin, cmax = get_bbox(mask_label)

            bbox = [cmin, rmin, cmax, rmax]

            index = list(meta['cls_indexes'].flatten()).index(CLASSES[args.object])
            pose = meta['poses'][:, :, index]
            cam = meta['intrinsic_matrix']
            kp = project_vertices(cam, kps_xyz, pose)

            bboxes[tag].append(bbox)
            kps[tag].append(kp)
            imgnames[tag].append(imgname)

    for tag in ('train', 'test'):
        with h5py.File(opj(KPDROOT, 'annot_%s.h5' % tag), "w") as f:
            f.create_dataset("imgname", data=np.array(imgnames[tag], dtype='S'))
            f.create_dataset("bndbox", data=np.vstack(
                bboxes[tag]).reshape(-1, 1, 4))
            f.create_dataset("part", data=np.vstack(
                kps[tag]).reshape(-1, args.kpnum, 2))

    print("[LOG] Done. H5 file has been generated in %s" % KPDROOT)
