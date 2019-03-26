from utils import load_ply, project_vertices
import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.io import loadmat, savemat
opj = os.path.join


YCB = '/home/penggao/data/ycb/'
FAT = opj(YCB, 'fat')

if __name__ == '__main__':
    print("[LOG] Loading dataset information")

    with open(opj(YCB, 'classes.txt')) as f:
        NAMES = [x.strip() for x in f.readlines()]

    print("[LOG] Loading 3d models ...")

    POINTS = dict()
    for name in tqdm(NAMES):
        POINTS[name] = load_ply(opj(YCB, 'models2/%s/nontextured.ply' % name))

    print("[LOG] Annotating training data ...")

    P = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])

    for dirname in tqdm(os.listdir(opj(FAT, 'mixed'))):
        # load camera
        with open(opj(FAT, 'mixed', dirname, '_camera_settings.json')) as f:
            content = json.load(f)['camera_settings'][0]['intrinsic_settings']
            cam = np.array([[content['fx'], 0, content['cx']],
                            [0, content['fy'], content['cy']],
                            [0, 0, 1]])

        # load object settings
        with open(opj(FAT, 'mixed', dirname, '_object_settings.json')) as f:
            content = json.load(f)
        ALIGN = content['exported_objects']

        for annoname in tqdm(os.listdir(opj(FAT, 'mixed', dirname))):
            if 'left.json' not in annoname:
                continue

            with open(opj(FAT, 'mixed', dirname, annoname)) as f:
                meta = json.load(f)

            anno = dict()
            anno['intrinsic_matrix'] = cam
            anno['cls_indexes'] = np.zeros(
                (len(meta['objects']), 1), dtype=int)
            anno['poses'] = np.zeros((3, 4, len(meta['objects'])))

            for idx, obj in enumerate(meta['objects']):
                cls_name = obj['class'][:-4]
                cls_idx = NAMES.index(obj['class'][:-4])
                anno['cls_indexes'][idx] = cls_idx + 1

                pose_mat = np.array(obj['pose_transform_permuted']).T
                fixed_model_transform = np.array(
                    ALIGN[cls_idx]['fixed_model_transform']).T[:3, :]
                align_mat = P.dot(fixed_model_transform)
                align_mat_aug = np.concatenate(
                    (align_mat, np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0)
                anno['poses'][:, :, idx] = pose_mat.dot(align_mat_aug)[:3, :]

            matpath = opj(FAT, 'mixed', dirname,
                          annoname.replace('.json', '.mat'))
            savemat(matpath, anno)

    print("\n[LOG] Done!")
