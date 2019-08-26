import numpy as np
import torch

def generate_target(rois1, rois2):
    """ generate label
    :param:
        rois1:   [N, 4], (xmin, ymin, xmax, ymax)
        rois2:   [N, 4], (xmin, ymin, xmax, ymax)
    :returns:
        reg_target:         N x 4, (dx, dy, dw, dh)
    """
    # regress label
    dx = ((rois2[..., 0] + rois2[..., 2])/2.0 - (rois1[..., 0] + rois1[..., 2])/2.0) / \
         (rois1[..., 2] - rois1[..., 0])  # (cx'-cx)/w
    dy = ((rois2[..., 1] + rois2[..., 3]) / 2.0 - (rois1[..., 1] + rois1[..., 3]) / 2.0) / \
         (rois1[..., 3] - rois1[..., 1])  # (cy'-cy)/h
    dw = np.log((rois2[..., 2] - rois2[..., 0]) / (rois1[..., 2] - rois1[..., 0]))  # log(w'/w)
    dh = np.log((rois2[..., 3] - rois2[..., 1]) / (rois1[..., 3] - rois1[..., 1]))  # log(h'/h)
    reg_target = np.stack((dx, dy, dw, dh), axis=1)  # N x 4
    return reg_target


def predict(rois1, pred):
    """ decode target
    :param:
        rois1:       Variable, [N, 4], (xmin, ymin, xmax, ymax)
        pred:        Variable, [N, 4], (dx, dy, dw, dh)
    :return:
        pred_tlbr:   Variable, [N, 4], (xmin, ymin, xmax, ymax)
    """
    cx_ = (rois1[..., 2] - rois1[..., 0]) * pred[..., 0] + \
          (rois1[..., 0] + rois1[..., 2])/2.0  # w*dx + cx
    cy_ = (rois1[..., 3] - rois1[..., 1]) * pred[..., 1] + \
          (rois1[..., 1] + rois1[..., 3])/2.0  # h*dy + cy
    w_ = torch.exp(pred[..., 2]) * (rois1[..., 2] - rois1[..., 0])  # exp(dw) * w
    h_ = torch.exp(pred[..., 3]) * (rois1[..., 3] - rois1[..., 1])  # exp(dh) * h

    pred = torch.stack((cx_, cy_, w_, h_), dim=1)  # (cx, cy, w, h)
    pred_tlbr = pred.data.cpu().numpy()
    pred_tlbr[:, :2] -= pred_tlbr[:, 2:]/2.0
    pred_tlbr[:, 2:] += pred_tlbr[:, :2]
    return pred_tlbr  # (xmin, ymin, xmax, ymax)

