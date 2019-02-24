import os
import torch
import json
from torchvision import transforms

from dataset import LinemodSingleDataset
ROOT = '/home/penggao/projects/kp6d/detect'
opj = os.path.join


def prepare_dataset(name, reso, bs, seq=None):
    """
    Args
    - name: (str) Dataset name
    - reso: (int) Image resolution
    - bs: (int) Batch size
    - seq: (str, optional) Sequence number for linemod
    """
    LINEMOD = '/home/penggao/data/sixd/hinterstoisser/test'

    train_transform = transforms.Compose([
        transforms.Resize(size=(reso, reso), interpolation=3),
        transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.2),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=(reso, reso), interpolation=3),
        transforms.ToTensor()
    ])

    if name == 'single':
        train_datasets = LinemodSingleDataset(
            root=LINEMOD,
            seq=seq,
            listfile=opj(LINEMOD, seq, 'train.txt'),
            transform=train_transform
        )

        val_datasets = LinemodSingleDataset(
            root=LINEMOD,
            seq=seq,
            listfile=opj(LINEMOD, seq, 'val.txt'),
            transform=val_transform
        )
    else:
        raise NotImplementedError

    train_dataloder = torch.utils.data.DataLoader(
        dataset=train_datasets,
        batch_size=bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_dataloder = torch.utils.data.DataLoader(
        dataset=val_datasets,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_dataloder, val_dataloder


def prepare_weight(ckpt):
    """
    Args
    - ckpt: (str) Relative path to weight file
    """
    weightfile = opj(ROOT, 'darknet/backup', ckpt)
    return weightfile


def prepare_cfg(name):
    """
    Prepare configuration file path

    Args
    - name: (str) Dataset name

    Return
    - cfgfile: (str) Configuration file path
    """
    if name == 'single':
        return opj(ROOT, 'darknet/cfg/single.cfg')
    else:
        raise NotImplementedError


def class_name(name, idx):
    """
    Args
    - name: (str) Dataset name
    - idx: (int or string) Class index
    """
    LINEMOD = ('ape', 'bvise', 'bowl', 'camera', 'can', 'cat', 'cup', 'driller',
               'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone')
    if name == 'single' or name == 'linemod':
        return LINEMOD[int(idx) - 1]
    else:
        raise NotImplementedError
