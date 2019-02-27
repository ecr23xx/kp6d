import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

from keypoint.sppe.src.utils.img import cropBox
from keypoint.sppe.src.torchsample.torchsample.transforms import SpecialCrop, Pad


def draw_heatmap(heatmaps, save_dir):
    """Draw heatmaps of images

    Args
    - heatmap: (torch.Tensor) With size []
    - save_dir: (str)
    """
    heatmaps = heatmaps.cpu().numpy()
    for i in range(heatmaps.shape[0]):
        fig, ax = plt.subplots()
        ax.axis('off')
        plt.imshow(heatmaps[i], cmap='jet', interpolation='nearest')
        plt.savefig(os.path.join(save_dir, '%d.png' % i))


def draw_keypoints(img_path, gt_kps, pred_kps, bbox, scores, save_path):
    """Draw keypoints on cropped image

    Args
    - img_path: (str)
    - gt_kps: (np.array) [N x 2]
    - pred_kps: (np.array) [N x 2]
    - bbox: (np.array) [4]
    - scores: (np.array) [N]
    - save_path: (str)
    """
    PAD = 30

    img = Image.open(img_path)
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = x1 - PAD, y1 - PAD, x2 + PAD, y2 + PAD
    cropped_img = img.crop((x1, y1, x2, y2))

    matplotlib.use('Agg')
    fig, ax = plt.subplots()
    ax.axis('off')
    plt.imshow(cropped_img)

    max_idx = np.argsort(scores)[::-1]
    good_pred_kps = pred_kps[max_idx[:10]]
    good_gt_kps = gt_kps[max_idx[:10]]

    # Draw good points
    # for i in range(10):
    #     plt.plot((good_pred_kps[i, 0] - x1, good_gt_kps[i, 0] - x1),
    #              (good_pred_kps[i, 1] - y1, good_gt_kps[i, 1] - y1),
    #              c='aqua', linewidth=1, linestyle='--')
    plt.scatter(good_pred_kps[:, 0] - x1,
                good_pred_kps[:, 1] - y1, c='aqua', s=20, marker='x')
    plt.scatter(good_gt_kps[:, 0] - x1,
                good_gt_kps[:, 1] - y1, c='yellow', s=20)

    # Draw bad points
    # bad_pred_kps = pred_kps[max_idx[-10:]]
    # bad_gt_kps = gt_kps[max_idx[-10:]]
    # for i in range(10):
    #     plt.plot((bad_pred_kps[i, 0] - x1, bad_gt_kps[i, 0] - x1),
    #              (bad_pred_kps[i, 1] - y1, bad_gt_kps[i, 1] - y1),
    #              c='aqua', linewidth=1, linestyle='--')
    # plt.scatter(bad_pred_kps[:, 0] - x1, bad_pred_kps[:, 1] - y1, c='yellow', s=3)
    # plt.scatter(bad_gt_kps[:, 0] - x1, bad_gt_kps[:, 1] - y1, c='aqua', s=3)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    img.close()


def crop_from_dets(img, box, inputResH, inputResW):
    """
    Crop human from origin image according to Dectecion Results

    Args
    - img: (Tensor) With size [C, W, H]
    - box: (Tensor) With size [4, ]
    """
    _, imght, imgwidth = img.size()
    tmp_img = img.clone()
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)

    upLeft = torch.Tensor((float(box[0]), float(box[1])))
    bottomRight = torch.Tensor((float(box[2]), float(box[3])))

    ht = bottomRight[1] - upLeft[1]
    width = bottomRight[0] - upLeft[0]
    if width > 100:
        scaleRate = 0.2
    else:
        scaleRate = 0.3

    upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
    upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
    bottomRight[0] = max(min(imgwidth - 1, bottomRight[0] +
                             width * scaleRate / 2), upLeft[0] + 5)
    bottomRight[1] = max(min(imght - 1, bottomRight[1] +
                             ht * scaleRate / 2), upLeft[1] + 5)

    inps = cropBox(tmp_img.cpu(), upLeft, bottomRight, inputResH, inputResW)
    pt1 = upLeft
    pt2 = bottomRight

    return inps, pt1, pt2
