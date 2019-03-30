import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

from keypoint.train_sppe.utils.img import flip_v, shuffleLR
from keypoint.train_sppe.utils.eval import getPrediction
from keypoint.train_sppe.models.FastPose import FastPose_SE


import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(
            storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class InferenNet_fast(nn.Module):
    def __init__(self, kernel_size, name, kpnum):
        super(InferenNet_fast, self).__init__()
        model = FastPose_SE(kpnum).cuda()
        path = '/home/penggao/projects/pose/kp6d/keypoint/exp/final_model/%s.pkl' % name
        print('Loading pose model from {}'.format(path))
        model.load_state_dict(torch.load(path))
        model.eval()
        self.pyranet = model
        self.kpnum = kpnum

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, self.kpnum)

        return out
