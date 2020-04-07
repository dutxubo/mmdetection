from .single_stage import SingleStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class CenterNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CenterNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                       test_cfg, pretrained)
        
        
        
    def forward_test(self, imgs, img_metas,**kwargs):
        img = imgs[0]
        img_meta = img_metas[0]
        
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg)
        #import pdb
        #pdb.set_trace()
        result_lists = self.bbox_head.get_bboxes(*bbox_inputs)
        #print("result_lists:", result_lists) #[1]
        
        return result_lists[0]