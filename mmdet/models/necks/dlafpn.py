import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmcv.cnn.weight_init import caffe2_xavier_init

from ..utils import ConvModule
from ..registry import NECKS


@NECKS.register_module
class DLAFPN(nn.Module):
    """DLAFPN
    Args:
        in_channels (list): number of channels for output of DLA.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 pooling_type='AVG',
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False):
        super(DLAFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels # 64?
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.dc1 = nn.ConvTranspose2d(2048,512,2,2)
        self.dc2 = nn.ConvTranspose2d(1536,256,2,2)
        self.dc3 = nn.ConvTranspose2d(768,out_channels,2,2)





    def init_weights(self):
        print("init weights in dlafpn")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        #print(len(inputs))
        #for i in range(len(inputs)):
        #    print(inputs[i].shape)
        assert len(inputs) == self.num_ins

        # x1, x2, x3, x4 -> x4, x3, x2, x1
        x1, x2, x3, x4 = inputs

        h = self.dc1(x4)
        h = self.dc2(torch.cat([x3, h],1))
        h = self.dc3(torch.cat([x2, h],1))
        

            
        
        return tuple([h])