import numpy as np
import math
import torch

def tlbr2xywh(rois):
    """rois: (xmin, ymin, xmax, ymax)"""
    rois_ = rois.copy()
    rois_[:, 2:] -= rois_[:, :2]
    rois_[:, :2] += rois_[:, 2:]/2.0
    return rois_

def xywh2tlbr(rois):
    """rois: (cx, cy, w, h)"""
    rois_ = rois.copy()
    rois_[:, :2] -= rois_[:, 2:]/2.0
    rois_[:, 2:] += rois_[:, :2]
    return rois_

def boxtransform(rois1, rois2):
    """encode rois2 using rois1
    :param:
        rois1:          [N, 5], Variable, the rois from the first images, the first column is the image index and the
                        remainder are [cx, cy, w, h] without normalized,
        rois2:          refer to rois1
    :returns:
        delta:    the encoded rois, Variable, [N, 4]
    """
    dx = (rois2[:, 1] - rois1[:, 1])*1.0 / rois1[:, 3]  # dx
    dy = (rois2[:, 2] - rois1[:, 2])*1.0 / rois1[:, 4]  # dy
    dw = torch.log(rois2[:, 3]*1.0/rois1[:, 3])
    dh = torch.log(rois2[:, 4]*1.0/rois1[:, 4])
    return torch.stack((dx, dy, dw, dh), dim=1)

def boxtransforminv(rois1, delta):
    """decode rois2 from rois1 and delta
    :param:
        rois1:      [N, 5], Variable, [idx, cx, cy, w, h]
        delta:      [N, 4], Variable, [dx, dy, dw, dh]
    :return:
        rois2:      [N, 5], np.float, [idx, ccx, cy, w, h]
    """
    cx = rois1[:, 3] * delta[:, 0] + rois1[:, 1]
    cy = rois1[:, 4] * delta[:, 1] + rois1[:, 2]
    w = torch.exp(delta[:, 2]) * rois1[:, 3]
    h = torch.exp(delta[:, 3]) * rois1[:, 4]
    idx = rois1[:, 0]
    return torch.stack((idx, cx, cy, w, h), dim=1)

def padding(rois, p=0.1):
    """expand the rois to get more context
    rois:       FloatTensor, [N, 5], [idx, x, y, w, h]
    :return:
        rois:   FloatTensor, [N, 5], [idx, xmin, ymin, xmax, ymax]
    """
    rois = rois.clone()
    rois[:, 3] *= (1 + p)
    rois[:, 4] *= (1 + 2.7*p)
    rois[:, 1:3] -= rois[:, 3:]/2.0
    rois[:, 3:] += rois[:, 1:3]
    return rois

if __name__=='__main__':
    from torch.autograd import Variable
    rois1 = Variable(torch.rand(4, 5)).cuda()
    rois2 = Variable(torch.rand(4, 5)).cuda()
    delta = boxtransform(rois1, rois2)
    print(delta)
    print('*'*20)
    print(rois2)
    pred = boxtransforminv(rois1, delta)
    print(pred)
    rois1 = torch.rand(4, 5)
    print(rois1)
    t = padding(rois1)
    print(rois1)
    print(t)

