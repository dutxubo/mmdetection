import torch
import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector

from .repkeypoints_detector import RepKeyPointsDetector
from .single_stage import SingleStageDetector
from mmdet.core import keypoint2result

from ipdb import set_trace

@DETECTORS.register_module
class SingleStageDetectorAndClassifier(RepKeyPointsDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 classifier_head=None,
                 train_cfg=None,
                 test_cfg=None,           
                 pretrained=None):
        super(SingleStageDetectorAndClassifier,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)
        #self.backbone = builder.build_backbone(backbone)
        #if neck is not None:
        #    self.neck = builder.build_neck(neck)
        #self.bbox_head = builder.build_head(bbox_head)
        self.classifier_head = builder.build_head(classifier_head)
        self.classifier_head.init_weights()
        #self.train_cfg = train_cfg
        #self.test_cfg = test_cfg
        #self.init_weights(pretrained=pretrained)


    #def init_weights(self, pretrained=None):
    #    #super(SingleStageDetectorAndClassifier, self).init_weights(pretrained)
    #    self.backbone.init_weights(pretrained=pretrained)
    #    if self.with_neck:
    #        if isinstance(self.neck, nn.Sequential):
    #            for m in self.neck:
    #                m.init_weights()
    #        else:
    #            self.neck.init_weights()
    #    self.bbox_head.init_weights()
    #    self.classifier_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        classifier_outs = self.classifier_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        #import pdb;pdb.set_trace()
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        # 最后一层的特征作为分类器特征
        classifier_outs = self.classifier_head(x[-1])
        image_labels = torch.tensor([ img_meta['image_level_label'] for img_meta in img_metas]).type_as(classifier_outs).long()
        classifier_losses = self.classifier_head.loss(classifier_outs, image_labels, None )
        
        losses['classifier_losses'] = classifier_losses['loss_cls']
        return losses

    def simple_test(self, img, img_meta, rescale=False ):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]  
        
        # 最后一层的特征作为分类器特征
        self.classifier_outs = self.classifier_head(x[-1])
        
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
