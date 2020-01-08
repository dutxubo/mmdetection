import torch

from mmdet.core import bbox2result, bbox_mapping_back
from ..registry import DETECTORS
from .reppoints_detector import RepPointsDetector

from mmdet.core import keypoint2result

@DETECTORS.register_module
class RepKeyPointsDetector(RepPointsDetector):
    """RepPoints: Point Set Representation for Object Detection.

        This detector is the implementation of:
        - RepPoints detector (https://arxiv.org/pdf/1904.11490)
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RepKeyPointsDetector,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained)

    def forward_train(self,
                      img,
                      img_metas,   #### ??? 传入img_metas
                      gt_bboxes,
                      gt_labels,
                      gt_keypoints=None,
                      gt_bboxes_ignore=None):
        #import pdb;pdb.set_trace()
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        
        #img_metas = img_meta
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_keypoints, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
    
    def simple_test(self, img, img_meta, rescale=False ):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes_keypoints(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_keypoints, det_labels in bbox_list
        ]
        
        keypoint_results = [
            keypoint2result(det_keypoints, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_keypoints, det_labels in bbox_list
        ]
        # 为了保持mmdetection评估的统一性，只返回bbox_results, 
        # keypoint_results使用类局部变量保存   权宜之计，无法进行batch测试时
        self.keypoint_results = keypoint_results[0]  
        return bbox_results[0]
    
    def forward_jit(self, img, img_meta, rescale=True, nms=True ):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes_keypoints(*bbox_inputs, nms=nms)
        
        return bbox_list