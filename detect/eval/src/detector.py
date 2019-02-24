import os
import torch
import warnings
from tqdm import tqdm
from PIL import Image, ImageDraw

from model import YOLOv3
from utils import crop_img, IoU
from layers import NMSLayer
import config
opj = os.path.join


class Detector:
    def __init__(self, cfgfile, seq, weightfile=None, reso=416, iou_thresh=0.5, conf_thresh=0.01):
        """
        Args
        - cfgfile: (str) Configuration file
        - seq: (str) Sequence number
        - reso: (int) Image resolution
        - iou_thresh: (float) IoU threshold for NMS
        - conf_thresh: (float) Objectness score threshold for NMS
        """
        self.yolo = YOLOv3(cfgfile, reso).cuda()
        self.yolo.eval()
        self.nms = NMSLayer(conf_thresh, iou_thresh).cuda()
        if weightfile != None:
            self.yolo.load_weights(weightfile)

    def detect(self, dataloader, save=False):
        """Detect wrapper

        Args:
        - dataloader: (Dataloader)
        - save: (bool) if true, save the detection result to disk

        Returns:
        - bboxes_all: (Tensor) detected bboxes, each with size [xmin, xmax, ymin, ymax]
        """
        bboxes_list = []
        tbar = tqdm(dataloader)
        for batch_idx, (inputs, labels, meta) in enumerate(tbar):
            batch_size = inputs.size(0)
            inputs = inputs.cuda()
            bboxes = torch.zeros(batch_size, 4)
            detections = self.nms(self.yolo(inputs))

            for idx in range(batch_size):
                img_path = meta['path'][idx]
                img_name = img_path.split('/')[-1]
                img_cls = img_path.split('/')[-3]
                label = labels[idx]
                try:
                    detection = detections[detections[:, 0] == idx]
                    if detection.size(0) == 0:
                        raise Exception
                except Exception:  # no object detected
                    bbox = (0, 0, 0, 0)
                else:
                    bbox = crop_img(img_path, detection.cpu(), self.yolo.reso)
                    if save == True:
                        img = Image.open(img_path)
                        draw = ImageDraw.Draw(img)
                        draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]))
                        img.save(opj('../results/', img_name))
                bboxes[idx, 0] = bbox[0]  # xmin
                bboxes[idx, 1] = bbox[2]  # xmax
                bboxes[idx, 2] = bbox[1]  # ymin
                bboxes[idx, 3] = bbox[3]  # ymax

            bboxes_list.append(bboxes)

        return bboxes_list
