import glob
import cv2
from os.path import join
import numpy as np
import torch
from models.CorrTrack.tracknet import tracknet
from torch.autograd import Variable
from models.CorrTrack.generate_target import predict

import logging
logger = logging.getLogger('global')

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_pretrain(model, pretrained_path):
    print('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    try:
        check_keys(model, pretrained_dict)
    except:
        logger.info('[Warning]: using pretrain as features. Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class CorrTracker():
    def __init__(self):
        self.predefine_H = 720
        self.predefine_W = 960
        self.mean = np.expand_dims(np.expand_dims(np.array([109, 120, 119]),
                                   axis=1), axis=1).astype(np.float32)
        self.net = tracknet(with_vis=False).cuda()
        weight_file = '/data/zwzhou/PycharmProjects/SiamTrack2/work/MOT/best.pth'
        print('==>  loading  checkpoint %s'%weight_file)
        self.net = load_pretrain(self.net, weight_file)
        print('==> load weights from `%s` successfully!'%weight_file)
        self.net.eval()
        self.im0 = None

    def predict(self, im0, im, rois):
        if len(rois) < 1:
            return np.empty((0, 4))
        im = im.copy()
        im0 = im0.copy()
        rois = rois.copy()
        H, W, _ = im.shape

        rois *= np.array([[self.predefine_W*1.0/W, self.predefine_H*1.0/H,
                           self.predefine_W*1.0/W, self.predefine_H*1.0/H]], dtype=np.float32)
        rois = Variable(torch.from_numpy(np.concatenate((np.zeros((len(rois), 1),
                                                                  dtype=np.float32), rois), axis=1)), volatile=True).float().cuda()
        im0 = cv2.resize(im0, dsize=(self.predefine_W, self.predefine_H))
        im0 = np.transpose(im0, (2, 0, 1)).astype(np.float32) - self.mean
        im0 = Variable(torch.from_numpy(im0), volatile=True).unsqueeze(0).cuda()

        im = cv2.resize(im, dsize=(self.predefine_W, self.predefine_H))
        im = np.transpose(im, (2, 0, 1)).astype(np.float32) - self.mean
        im = Variable(torch.from_numpy(im), volatile=True).unsqueeze(0).cuda()
        # if self.im0 is None:
        #     self.im0 = im

        reg_pred, _ = self.net(im0, im, rois)
        pred = predict(rois[:, 1:], reg_pred)
        # self.im0 = im
        pred /= np.array([[self.predefine_W*1.0/W, self.predefine_H*1.0/H,
                           self.predefine_W*1.0/W, self.predefine_H*1.0/H]], dtype=np.float32)
        return pred
