import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        self.train_batch = 0

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        self.train_batch += 1
        # debug
        debug = False
        if debug:
            if self.train_batch % 100 == 0 :
                import matplotlib.pyplot as plt
                import numpy as np
                import cv2
                im = img[0].permute(1, 2, 0).cpu().squeeze().numpy()*255
                pred_mask = outs[0][0][0].clone().permute(1, 2, 0).cpu().detach().squeeze(-1).numpy()*255
                
                bbox = gt_bboxes[0][0].cpu().numpy().astype(np.int)
                cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                plt.subplot(1, 2, 1)
                plt.title('Image')
                plt.imshow(im.astype(np.int))

                plt.subplot(1, 2, 2)
                plt.title('Mask Prediction')
                plt.imshow(pred_mask)
                plt.suptitle('Score {:.3f}'.format(pred_mask.max()))
                plt.savefig(f'/home/songbai.xb/detection/mmdetection/tmp.jpg')
        return losses

    #@profile
    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_metas, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        
        imgs_per_gpu = img.size(0)
        if imgs_per_gpu==1:
            return bbox_results[0]
        else:
            return bbox_results
    

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def forward_jit(self, img, img_meta, rescale=True, nms=True):
        """
        """
        
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs, nms=nms)
        
        return bbox_list