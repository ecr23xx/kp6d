import os
import copy
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser(description='Keypoints generator')
    parser.add_argument('--dataset', default='ycb',
                        type=str, help="dataset name")
    parser.add_argument('--sixdroot', default='/media/data_1/home/chengkun/pose_6D_methods/kp6d/datasets/ycb/YCB_Video_Dataset',
                        type=str, help="Path to SIXD root")
    parser.add_argument('--outroot',
                        default='/media/data_1/home/chengkun/pose_6D_methods/kp6d/datasets',
                        type=str, help="Output dir")
    parser.add_argument('--object', type=str, help="Object", default='002_master_chef_can')
    parser.add_argument('--num', type=int, help="Number of keypoints", default=8)
    parser.add_argument('--type', choices=['sift', 'random', 'cluster', 'corner'],
                        type=str, help="Type of keypoints", default='cluster')
    return parser.parse_args()


def get_3d_corners(vertices):
    """Get vertices 3D bounding boxes

    Args
    - vertices: (np.array) [N x 3] 3d vertices

    Returns
    - corners: (np.array) [8 x 2] 2d vertices
    """
    min_x = np.min(vertices[:, 0])
    max_x = np.max(vertices[:, 0])
    min_y = np.min(vertices[:, 1])
    max_y = np.max(vertices[:, 1])
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])

    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    return corners


if __name__ == '__main__':
    np.random.seed(1)
    args = parse_args()
    assert args.type is not None, "Please specify type of keypoints"
    assert args.num is not None, "Please specify number of keypoints"
    assert args.type != 'sift', "Please go to ./pcl-sift to generate sift keypoints"

    if args.type == 'corner':
        assert args.num == 9, "Number of \"corner\" keypoints must be 9"

    print("[LOG] Number of keypoints: %d" % args.num)
    print("[LOG] Type of keypoints: %s" % args.type)

    MODEL_ROOT = os.path.join(args.sixdroot, 'models', args.object)
    KP_ROOT = os.path.join(args.outroot, args.dataset,
                           'kps', str(args.num), args.type)
    if not os.path.exists(KP_ROOT):
        os.makedirs(KP_ROOT)
    else:
        print("[WARNING] Overwrite existing files!")

    tbar = tqdm(os.listdir(MODEL_ROOT))

    for filename in tbar:
        tbar.set_description(filename)

        vertex = np.zeros(args.num,
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        if 'points.xyz' in filename:
            xyz = np.loadtxt(os.path.join(MODEL_ROOT, filename))
            if args.type == 'random':
                selected_ids = np.random.choice(
                    xyz.shape[0], args.num, replace=False)
                for i in range(args.num):
                    vertex[i][0] = xyz[selected_ids[i]][0]
                    vertex[i][1] = xyz[selected_ids[i]][1]
                    vertex[i][2] = xyz[selected_ids[i]][2]
            elif args.type == 'cluster':
                kmeans = KMeans(n_clusters=args.num, max_iter=1000, random_state=1).fit(xyz)
                dist = kmeans.transform(xyz)
                selected_ids = []
                for i in range(args.num):
                    di = dist[:, i]
                    ind = np.argsort(di)[0]
                    vertex[i][0] = xyz[ind][0]
                    vertex[i][1] = xyz[ind][1]
                    vertex[i][2] = xyz[ind][2]
            elif args.type == 'corner':
                corners = get_3d_corners(xyz)
                center = corners.mean(axis=0).reshape(1, 3)
                corners_and_center = np.concatenate((corners, center), axis=0)
                for i in range(args.num):
                    vertex[i][0] = corners_and_center[i, 0]
                    vertex[i][1] = corners_and_center[i, 1]
                    vertex[i][2] = corners_and_center[i, 2]
        else:
            continue

        data = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)

        with open(os.path.join(KP_ROOT, args.object + '_kp.ply'), mode='wb') as f:
            data.write(f)
