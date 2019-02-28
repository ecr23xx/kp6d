import os
import cv2
import pickle
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw
from torchvision import transforms

from utils import *
from data.sixd import SixdToolkit
from detect.eval.src.config import *
from detect.eval.src.detector import Detector
from keypoint.sppe.src.main_fast_inference import InferenNet_fast
from keypoint.sppe.src.utils.eval import getPrediction
from keypoint.sppe.src.utils.img import im_to_torch


def parse_arg():
    parser = argparse.ArgumentParser(description='YOLO v3 evaluation')
    # parser.add_argument('--bs', type=int, help="Batch size")
    parser.add_argument('--reso', type=int, help="Image resolution")
    parser.add_argument('--gpu', default='0,1,2,3', help="GPU ids")
    parser.add_argument('--name', type=str, choices=['single'])
    parser.add_argument('--seq', type=str, help="Sequence number")
    parser.add_argument('--ckpt', type=str, help="Checkpoint path")
    parser.add_argument('-save', action='store_true', help="Save pose figure")
    return parser.parse_args()


args = parse_arg()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == '__main__':
    print(args)
    bench = SixdToolkit(dataset='hinterstoisser', kpnum=17,
                        kptype='sift', is_train=False)
    kp3d = bench.models[args.seq]

    _, val_dataloder = prepare_dataset(args.name, args.reso, 1, args.seq)
    detector = Detector(
        cfgfile=prepare_cfg(args.name),
        seq=args.seq,
        weightfile=prepare_weight(args.ckpt)
    )
    pose_model = InferenNet_fast(5, args.seq, 17).cuda()

    tbar = tqdm(val_dataloder)
    result = dict()
    for batch_idx, (inputs, labels, meta) in enumerate(tbar):
        inputs = inputs.cuda()
        with torch.no_grad():
            # object detection
            bboxes, confs = detector.detect(inputs)

            # keypoint localization
            orig_img = cv2.imread(meta['path'][0])
            orig_inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            cropped_inputs, pt1, pt2 = crop_from_dets(
                orig_inp, bboxes[0], 320, 256)
            hms = pose_model(cropped_inputs.unsqueeze(0).cuda()).cpu()
            _, pred_kps, pred_kps_score = getPrediction(
                hms, pt1.unsqueeze(0), pt2.unsqueeze(0), 320, 256, 80, 64)

            # pose estimation
            pred_pose = bench.solve_pnp(
                bench.kps[args.seq], pred_kps[0].numpy())

            img_path = meta['path'][0]
            idx = img_path.split('/')[-1].split('.')[0]
            result[int(idx)] = {
                'bbox': bboxes[0].numpy(),
                'kps': pred_kps[0].numpy(),
                'pose': pred_pose
            }

            if args.save is True:
                f = bench.frames[args.seq][int(idx)]
                annot = f['annots'][f['obj_ids'].index(int(args.seq))]
                gt_pose = annot['pose']

                save_path = opj('./results/pose/%s.png' % idx)
                draw_6d_pose(img_path, gt_pose, pred_pose,
                             kp3d, bench.cam, save_path)

    with open('./results/%s.pkl' % args.seq, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("[LOG] Done!")
