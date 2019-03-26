import numpy as np
from plyfile import PlyData


def project_vertices(vertices, pose, cam):
    vertices = np.concatenate(
        (vertices, np.ones((vertices.shape[0], 1))), axis=1)
    projected = np.matmul(np.matmul(cam, pose), vertices.T)
    projected /= projected[2, :]
    projected = projected[:2, :].T
    return projected


def get_2d_corners(xy):
    x_min = np.min(xy[:, 0])
    x_max = np.max(xy[:, 0])
    y_min = np.min(xy[:, 1])
    y_max = np.max(xy[:, 1])
    return x_min, y_min, x_max, y_max


def corner2center(x1, y1, x2, y2):
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return xc, yc, w, h


def load_ply(path):
    with open(path, 'rb') as f:
        content = PlyData.read(f)
    xyz = np.vstack([content['vertex']['x'],
                     content['vertex']['y'],
                     content['vertex']['z']]).T
    return xyz


def align_model(xyz, transform_mat):
    xyzw = np.concatenate((xyz, np.ones((xyz.shape[0], 1))), axis=1)
    xyzw = np.matmul(transform_mat, xyzw.T)
    xyzw /= xyzw.T[:, 3]
    xyz = xyzw.T[:, :3]
    return xyz