import os
import cv2
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw
from torchvision import transforms

from data.linemod.sixd import SixdToolkit
from detect.eval.src.detector import Detector
from detect.eval.src.dataset import prepare_dataset
from detect.eval.src.config import prepare_weight, prepare_cfg
from utils import draw_keypoints, crop_from_dets
from keypoint.sppe.src.main_fast_inference import InferenNet_fast
from keypoint.sppe.src.utils.eval import getPrediction
from keypoint.sppe.src.utils.img import im_to_torch


def parse_arg():
    parser = argparse.ArgumentParser(description='YOLO v3 evaluation')
    parser.add_argument('--bs', type=int, help="Batch size")
    parser.add_argument('--reso', type=int, help="Image resolution")
    parser.add_argument('--gpu', default='0,1,2,3', help="GPU ids")
    parser.add_argument('--name', type=str, choices=['linemod-single'])
    parser.add_argument('--seq', type=str, help="Sequence number")
    parser.add_argument('--ckpt', type=str, help="Checkpoint path")
    return parser.parse_args()


args = parse_arg()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == '__main__':
    print(args)
    bench = SixdToolkit(dataset='hinterstoisser', kpnum=17,
                        kptype='sift', is_train=False)
    _, val_dataloder = prepare_dataset(args.name, args.reso, args.bs, args.seq)
    detector = Detector(
        cfgfile=prepare_cfg(args.name),
        seq=args.seq,
        weightfile=prepare_weight(args.ckpt)
    )
    pose_model = InferenNet_fast(
        kernel_size=5,
        name=args.seq,
        kpnum=17
    )
    pose_model = pose_model.cuda()

    tbar = tqdm(val_dataloder)
    for batch_idx, (inputs, labels, meta) in enumerate(tbar):
        inputs = inputs.cuda()
        with torch.no_grad():
            bboxes, confs = detector.detect(inputs)

            orig_img = cv2.imread(meta['path'][0])
            orig_inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            cropped_inputs, pt1, pt2 = crop_from_dets(
                orig_inp, bboxes[0], 320, 256)
            hms = pose_model(cropped_inputs.unsqueeze(0).cuda()).cpu()

            _, pred_kps, pred_kps_score = getPrediction(
                hms, pt1.unsqueeze(0), pt2.unsqueeze(0), 320, 256, 80, 64)

        img_path = meta['path'][0]
        idx = img_path.split('/')[-1].split('.')[0]
        f = bench.frames[args.seq][int(idx)]
        annot = f['annots'][f['obj_ids'].index(int(args.seq))]
        gt_kps = annot['kps']

        save_path = opj('./results/kps/%s.png' % idx)
        draw_keypoints(img_path, gt_kps, pred_kps[0].numpy(), bboxes[0].numpy(),
                       pred_kps_score[0].squeeze().numpy(), save_path)
