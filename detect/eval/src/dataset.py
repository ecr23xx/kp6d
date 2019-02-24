import config
import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
opj = os.path.join


class LinemodSingleDataset(torch.utils.data.dataset.Dataset):
    """Single Image dataset for SIXD"""

    def __init__(self, root, seq, listfile, transform):
        """
        Args
        - root: (str) Path to SIXD dataset test images root
        - seq: (str) LINEMOD sequence number
        - listfile: (str) Listfile
        - transform: (torchvision.transforms)
        """
        self.root = root
        self.transform = transform

        with open(listfile) as f:
            lists = f.readlines()
            lists = [x.strip() for x in lists]

        self.imgs_path = [opj(root, seq, 'rgb', idx + '.png')
                          for idx in lists]

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label_path = img_path.replace(
            'rgb', 'annots/bbox').replace('.png', '.npy')
        img_tensor = self.transform(Image.open(img_path))
        img_label = torch.Tensor(np.load(label_path))
        meta = {'path': img_path}
        return img_tensor, img_label, meta

    def __len__(self):
        return len(self.imgs_path)
