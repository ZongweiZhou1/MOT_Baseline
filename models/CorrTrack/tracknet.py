import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.correlation.modules.correlation import Correlation
from models.psroi_pooling.modules.psroi_pool import PSRoIPool
from models.backbone.sqeezenet import DilationLayer, FeatExtractorSqueezeNetx16
import collections
import h5py

class ConcatAddTable(nn.Module):
    def __init__(self, *args):
        super(ConcatAddTable, self).__init__()
        if len(args) == 1 and isinstance(args[0], collections.OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            idx = 0
            for module in args:
                self.add_module(str(idx), module)
                idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def forward(self, input):
        x_out = None
        for module in self._modules.values():
            x = module(input)
            if x_out is None:
                x_out = x
            else:
                x_out = x_out + x
        return x_out


class tracknet(nn.Module):
    """Siamese net for track with correlation"""
    def __init__(self, grid_size=7, max_disp=16, stride=2, with_vis=True):
        super(tracknet, self).__init__()
        self.pretrained_model_path = \
            '/data/zwzhou/Code/MOTDT/pretrained/squeezenet_small40_coco_mot16_ckpt_10.h5'
        featstride = 4
        self.grid_size = grid_size
        self.corr_num = int(max_disp/stride)*2+1  # max_displacement=16, stride2=2
        self.dout_base_model = 192
        self.feature_extractor = FeatExtractorSqueezeNetx16(True)

        # UpConvolution Layers
        in_channels = self.feature_extractor.n_feats[-1]
        self.stage_0 = nn.Sequential(
            nn.Dropout2d(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),)

        n_feats = self.feature_extractor.n_feats[1:]
        in_channels = 256
        out_cs = [128, 256]

        for i in range(1, len(n_feats)):
            out_channels = out_cs[-i]
            setattr(self, 'upconv_trk_{}'.format(i),
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    ))

            feat_channels = n_feats[-1 - i]
            setattr(self, 'proj_trk_{}'.format(i), nn.Sequential(
                ConcatAddTable(
                    DilationLayer(feat_channels, out_channels // 2, 3, dilation=1),
                    DilationLayer(feat_channels, out_channels // 2, 5, dilation=1),
                ),
                nn.Conv2d(out_channels // 2, out_channels // 2, 1),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True)
            ))
            in_channels = out_channels + out_channels // 2

        # correlation
        self.conv_corr1 = Correlation(pad_size=max_disp, kernel_size=1, max_displacement=max_disp,
                                      stride1=1, stride2=stride)
        self.conv_corr2 = Correlation(pad_size=max_disp, kernel_size=1, max_displacement=max_disp,
                                      stride1=1, stride2=stride)  # correlation
        in_feat = (2*int(max_disp/stride)+1)**2  # 2*in_feat+192
        self.corr_box_net = nn.Sequential(nn.Conv2d(2*(in_feat+192), in_feat, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(in_feat, 4*grid_size**2, 1))

        if with_vis:
            out_dim = 5
        else:
            out_dim = 4

        self.PSROI_pool = PSRoIPool(self.grid_size, self.grid_size, 1.0/featstride,
                                        group_size=grid_size, output_dim=out_dim)  # the last value is the visibility ratio
        # regression
        self.tracking_pred = nn.AvgPool2d((grid_size, grid_size), stride=(grid_size, grid_size))

        self.__init__module()

    def _im_to_head(self, x):
        feats = self.feature_extractor(x)
        x_trk_in = self.stage_0(feats[-1])
        n_feats = self.feature_extractor.n_feats[1:]
        outputs = []
        for i in range(1, len(n_feats)):
            x_trk_depth_out = getattr(self, 'upconv_trk_{}'.format(i))(x_trk_in)
            x_trk_depth_out = F.upsample(x_trk_depth_out,
                                         [feats[-1 - i].shape[-2], feats[-1 - i].shape[-1]],
                                         mode='bilinear')
            x_trk_project = getattr(self, 'proj_trk_{}'.format(i))(feats[-1 - i])
            x_trk_in = torch.cat((x_trk_depth_out, x_trk_project), 1)
            outputs.append(x_trk_in)
        return outputs

    def __init__module(self):
        with h5py.File(self.pretrained_model_path, mode='r') as h5f:
            for name, param in self.state_dict().items():
                if name in h5f.keys():
                    param.copy_(torch.from_numpy(np.asarray(h5f[name])))
        # fix parameters
        for name, param in self.named_parameters():
            if name.startswith('feature_extractor'):
                param.requires_grad = False

    def forward(self, ims1, ims2, rois):
        """forward function of tracknet
        :params:
            ims1:        [B, 3, H, W], the first images in the sample pairs
            ims2:        [B, 3, H, W], the second images in the sample pairs
            rois:        [nB, 5], the rois in the first images, [idx, xmin, ymin, xmax, ymax]
                         the first column is the indices of the rois from batch images
        """
        # t1 = time.time()
        B = ims1.size(0)
        legs_list = self._im_to_head(torch.cat((ims1, ims2), 0))
        # t2 = time.time()
        corr_feat1 = self.conv_corr1(legs_list[0][:B], legs_list[0][B:])  # correlation
        corr_feat2 = self.conv_corr2(legs_list[1][:B], legs_list[1][B:])
        # t3 = time.time()
        corr_feat1 = F.upsample(corr_feat1, [corr_feat2.shape[-2], corr_feat2.shape[-1]], mode='bilinear')
        corr_feat = self.corr_box_net(torch.cat((legs_list[1][:B], legs_list[1][B:], corr_feat1, corr_feat2), dim=1))
        # corr_feat = self.corr_box_net(corr_feat2)
        # t4 = time.time()
        # PSROIPooling
        roi_feats = self.PSROI_pool(corr_feat, rois)  # N x 196
        # regression
        reg_pred = self.tracking_pred(roi_feats)  # N x 5
        # t5 = time.time()
        # print(t2-t1)
        # print(t3-t2)
        # print(t4-t2)
        # print(t5-t4)
        return reg_pred.squeeze(), corr_feat2


if __name__ == '__main__':
    from torch.autograd import Variable
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    x1 = Variable(torch.rand(2, 3, 640, 480)).cuda()
    x2 = Variable(torch.rand(2, 3, 640, 480)).cuda()
    rois = Variable(torch.FloatTensor([[0, 200, 300, 300, 400],
                                      [1, 200.0, 300.0, 400.01, 420.9]])).cuda()
    net = tracknet().cuda()
    # for name, param in net.named_parameters():
    #     print(name)
    y = net(x1, x2, rois)
    print(y[0])
    print(y[0].shape)
    print(y[1].shape)

    # test correlation layer
    # corr_layer = Correlation(pad_size=1, kernel_size=1, max_displacement=1,
    #             stride1=1, stride2=1)
    # x = Variable(torch.FloatTensor([[[[1,2,3],[4,5,6],[7,8,9]]]])).cuda()
    # z = Variable(torch.FloatTensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])).cuda()
    # y = corr_layer(x, z)
    # print(y.squeeze())
    # it can be found the correlation is proceed from top-left to top-right then to bottom-right

    # # test RoI Align
    # x = Variable(torch.rand(1, 2, 5, 5)).cuda()
    # pool_layer = RoIAlignAvg(4, 4, 1)
    # rois = Variable(torch.FloatTensor([[0, 1, 1, 3, 3],[0, 0,0,2,3]])).cuda()
    # y = pool_layer(x, rois)
    # print(x.squeeze())
    # print(y)



