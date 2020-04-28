#coding=utf-8

import os
import mmcv
import torch
import numpy as np

from mmdet import apis
import imp
imp.reload(apis)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# -

head_config = '/home/songbai.xb/detection/mmdetection/myproject/coco/configs/centernet/centernet_r50_fpn.py'
detector = apis.init_detector(head_config, None, device='cuda:0')
detector.forward = detector.forward_dummy


# +
gt_bboxes = torch.tensor([[[10,10,100,100]]]).float().cuda()
gt_labels = torch.tensor([[1]]).cuda()
img_metas =[1]
        
train_cfg = dict(
    one_hot_smoother=0.,
    ignore_config=0.5,
    xy_use_logit=False,
    debug=False)
# -

dumpy_input = torch.randn(1,3,416,416).cuda()
dumpy_output = detector(dumpy_input)



detector.bbox_head.loss(dumpy_output[0], gt_bboxes,
             gt_labels,
             img_metas,
             train_cfg)





feat_0 = torch.randn(1, 512, 13, 13).cuda()
feat_0 = torch.randn(1, 256, 26, 26).cuda()
feat_0 = torch.randn(1, 128, 52, 52).cuda()
