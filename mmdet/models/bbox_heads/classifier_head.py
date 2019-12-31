import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS


@HEADS.register_module
class ClassifierHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 in_channels=256,
                 num_classes=2,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0) ):
        super(ClassifierHead, self).__init__()
   
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.fp16_enabled = False

        self.loss_cls = build_loss(loss_cls)

        in_channels = self.in_channels

        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.fc_cls = nn.Linear(in_channels, num_classes)
       
        self.debug_imgs = None

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)

    @auto_fp16()
    def forward(self, x):
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)

        return cls_score

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             labels,
             label_weights,
             reduction_override=None):
        losses = dict()
        #avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        avg_factor = len(labels)
        
        losses['loss_cls'] = self.loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=avg_factor,
            reduction_override=reduction_override)
        #losses['acc'] = accuracy(cls_score, labels)
       
        return losses

