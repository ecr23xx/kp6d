import os
import argparse

from detect.eval.src.config import prepare_cfg, prepare_weight
from detect.eval.src.dataset import prepare_dataset
from detect.eval.src.detector import Detector


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
    detector = Detector(
        cfgfile=prepare_cfg(args.name),
        seq=args.seq,
        weightfile=prepare_weight(args.ckpt),
        reso=args.reso
    )
    _, val_dataloder = prepare_dataset(
        name=args.name,
        reso=args.reso,
        bs=args.bs,
        seq=args.seq
    )
    detector.detect_all(val_dataloder, savedir='./results/detect')
