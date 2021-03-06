import os
import sys
import cv2
import pickle
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw
from torchvision import transforms

from utils import *
from data.linemod.sixd import SixdToolkit
from detect.eval.src.detector import Detector
from detect.eval.src.dataset import prepare_dataset
from detect.eval.src.config import prepare_cfg, prepare_weight
sys.path.append('./keypoint/train_sppe')
from keypoint.train_sppe.main_fast_inference import InferenNet_fast
from keypoint.train_sppe.utils.eval import getPrediction
from keypoint.train_sppe.utils.img import im_to_torch


def parse_arg():
    parser = argparse.ArgumentParser(description='YOLO v3 evaluation')
    # parser.add_argument('--bs', type=int, help="Batch size")
    parser.add_argument('--reso', type=int, help="Image resolution")
    parser.add_argument('--kptype', type=str, help="Keypoint type")
    parser.add_argument('--kpnum', type=int, help="Checkpoint path")
    parser.add_argument('--topk', type=int, help="Checkpoint path")
    parser.add_argument('--gpu', default='0,1,2,3', help="GPU ids")
    parser.add_argument('--name', type=str, choices=['linemod-single', 'linemod-occ'])
    parser.add_argument('--seq', type=str, help="Sequence number")
    parser.add_argument('--ckpt', type=str, help="Checkpoint path")
    parser.add_argument('-save', action='store_true', help="Save pose figure")
    return parser.parse_args()


args = parse_arg()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == '__main__':
    print(args)
    bench = SixdToolkit(dataset='hinterstoisser', kpnum=args.kpnum,
                        kptype=args.kptype, is_train=False)
    kp3d = bench.models[args.seq]

    _, val_dataloder = prepare_dataset(args.name, args.reso, 1, args.seq)
    detector = Detector(
        cfgfile=prepare_cfg(args.name),
        seq=args.seq,
        weightfile=prepare_weight(args.ckpt)
    )
    pose_model = InferenNet_fast(
        dataset=args.name,
        kernel_size=5,
        seqname=args.seq,
        kpnum=args.kpnum,
        kptype=args.kptype
    ).cuda()

    tbar = tqdm(val_dataloder)
    result = dict()
    for batch_idx, (inputs, labels, meta) in enumerate(tbar):
        img_path = meta['path'][0]
        idx = img_path.split('/')[-1].split('.')[0]
        inputs = inputs.cuda()
        with torch.no_grad():
            # object detection
            try:
                bboxes, confs = detector.detect(inputs)
            except Exception:
                # No object found
                # print("detection failed")
                continue

            # keypoint localization
            orig_img = cv2.imread(meta['path'][0])
            orig_inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            cropped_inputs, pt1, pt2 = crop_from_dets(
                orig_inp, bboxes[0], 320, 256)
            hms = pose_model(cropped_inputs.unsqueeze(0).cuda()).cpu()
            try:
                _, pred_kps, pred_kps_score = getPrediction(
                    hms, pt1.unsqueeze(0), pt2.unsqueeze(0), 320, 256, 80, 64)
            except Exception:
                # print("Jump Error frame", idx)
                continue

            # pose estimation
            K = args.topk
            best_idx = np.argsort(pred_kps_score[0, :, 0]).flip(0)
            best_k = best_idx[:K]

            pred_pose = bench.solve_pnp(
                bench.kps[args.seq][best_k], pred_kps[0][best_k].numpy())

            result[int(idx)] = {
                'bbox': bboxes[0].numpy(),
                'kps': pred_kps[0].numpy(),
                'pose': pred_pose
            }

            if args.save is True:
                f = bench.frames[args.seq][int(idx)]
                annot = f['annots'][f['obj_ids'].index(int(args.seq))]
                gt_pose = annot['pose']

                save_path = os.path.join('./results/pose/%s.png' % idx)
                draw_6d_pose(img_path, gt_pose, pred_pose,
                             kp3d, bench.cam, save_path)

    with open('./results/%s.pkl' % args.seq, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("[LOG] Done!")
