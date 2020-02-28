# %load ~/anaconda3/envs/py36_cu10_clone/lib/python3.6/site-packages/torchvision/models/shufflenetv2.py
import torch
import torch.nn as nn

import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger

from ..registry import BACKBONES

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

@BACKBONES.register_module
class ShuffleNetV2(nn.Module):
    def __init__(self, width_mult=1., frozen_stages=-1):
        super(ShuffleNetV2, self).__init__()

        stages_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            stages_out_channels = [24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            stages_out_channels = [24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            stages_out_channels = [24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            stages_out_channels = [24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """now width_mult={} is not supported for ShuffleNetV2""".format(width_mult))
            
        self._stage_out_channels = stages_out_channels
        self.frozen_stages = frozen_stages

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                self.stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
        self._freeze_stages()
        
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')
            
    def _freeze_stages(self):
        if self.frozen_stages >= 1:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(2, self.frozen_stages + 1):
            m = getattr(self, 'stage{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.stage2(x)
        outs.append(x)
        x = self.stage3(x)
        outs.append(x)
        x = self.stage4(x)
        outs.append(x)
        #for stage_name in self.stage_names:
        #    stage = getattr(self, stage_name)
        #    x = stage(x)
        #    outs.append(x)
        
        x = self.conv5(x)
        # xubo: add pool 
        x = F.max_pool2d(x, 2, stride=2)
        outs.append(x)
        return outs