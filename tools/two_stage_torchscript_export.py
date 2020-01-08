#coding=utf-8
import sys
import argparse 
import os
import cv2
import torch
import torch.nn as nn
import pdb


from mmdet.apis import init_detector, inference_detector
from mmdet.core import bbox2roi


# TwoStageDetector 分两步固化
class TwoStageDetector(nn.Module):
    def __init__(self, torchscript_rpn, torchscript_head):
        super(TwoStageDetector, self).__init__()
        self.rpn = torchscript_rpn
        self.head = torchscript_head
    
    def forward(self, img):
        print('For two stage detector, please use forward_rpn and forward_head instead of forward.')

    @torch.jit.export
    def forward_rpn(self, img):
        return self.rpn(img)
    
    @torch.jit.export
    def forward_head(self, feature, rois):
        # type: (Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor) -> Tuple[Tensor, Tensor]
        # !!! Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] should change as needed
        return self.head(feature, rois)
    
    
class TwoStageDetectorRpn(nn.Module):
    def __init__(self, detector):
        super(TwoStageDetectorRpn, self).__init__()
        self.detector = detector.eval()
        self.fpn_level = 5
    
    def forward(self, imgs):
        x, rpn_output = self.detector.forward_rpn_jit(imgs)
        self.fpn_level = len(x)
        
        # Only tensors or tuples of tensors can be output from traced functions. So convert actual_output to jit_output.
        jit_output = tuple()
        
        jit_output += tuple(x)
        for t in rpn_output:
            jit_output += tuple(t)
        return jit_output
    
    def convert_jit_out_to_actual_out(self, jit_out):
        '''the format of jit_out is Tuple[Tensor, Tensor, Tensor, Tensor, ...]
           the format of actual network out is Tuple[Tuple[Tensor, Tensor, ...], Tuple[Tensor, Tensor, ...], ...]
        '''
        out_num = len(jit_out) // self.fpn_level
        actual_out = tuple()
        for i in range(out_num):
            actual_out += (jit_out[i*self.fpn_level: (i+1)*self.fpn_level], )
        return actual_out
    
    
class TwoStageDetectorHead(nn.Module):
    def __init__(self, detector):
        super(TwoStageDetectorHead, self).__init__()
        self.detector = detector.eval()
    
    def forward(self, feature, rois):
        output = self.detector.forward_head_jit(feature, rois) # output the cls_score and bbox_pred of rois
        return output
    
    
def export_model(model, export_file, *input_data):
    script_module = torch.jit.trace(model.eval(), (*input_data) )
    if export_file is not None:
        script_module.save(export_file)
    return script_module
    
    
def two_stage_export(detector, work_size, export_file):
    # build input tensor
    H, W, C = work_size[0], work_size[1], 3
    input_shape = (C, H, W) 
    input_tensor = torch.randn( input_shape )
    input_tensor = input_tensor.unsqueeze(0).float().cuda()
    
    # build img_metas
    img_metas = [{'img_shape': torch.tensor((H, W, C) ),
                  'ori_shape': torch.tensor((H, W, C) ),
                  'pad_shape': torch.tensor((H, W, C) ),
                  'scale_factor': torch.tensor((1.0, 1.0, 1.0, 1.0) ),
                  } for _ in range(1)]
    
    
    # build rpn net and export
    rpn_model = TwoStageDetectorRpn(detector).cuda()
    rpn_torchscript = export_model(rpn_model, None, input_tensor)
    
    
    # convert rpn_output to head_input
    jit_model_output = rpn_model(input_tensor)
    actual_out = rpn_model.convert_jit_out_to_actual_out(jit_model_output)
    feature = actual_out[0]
    rpn_outs = tuple(actual_out[1:])
    
    proposal_inputs = rpn_outs + (img_metas, detector.test_cfg.rpn)
    proposal_list = detector.rpn_head.get_bboxes(*proposal_inputs)
    proposals = proposal_list[0]
    rois = bbox2roi([proposals])
    
    # build head net and export 
    head_model = TwoStageDetectorHead(detector).cuda()
    head_torchscript = export_model(head_model, None, (feature, rois))
    
    # export twostagedetector model
    twostagedetector_scripted_module = torch.jit.script( TwoStageDetector(rpn_torchscript, head_torchscript) )
    twostagedetector_scripted_module.save(export_file)
    
    
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default='/home/songbai.xb/detection/mmdetection_merge/mmdetection/myprojects/extinguisher/configs/ga_faster_r50_caffe_fpn_1x.py', help='config_file')

    parser.add_argument('--checkpoint_file', type=str, default='/home/songbai.xb/detection/mmdetection/myprojects/extinguisher/work_dirs/all/ga_faster_rcnn_r50_caffe_mykeep_ratio_RandAugment/latest.pth', help='model weight')

    parser.add_argument('--export_file', type=str, default='/home/songbai.xb/detection/mmdetection_merge/mmdetection/myprojects/extinguisher/ga_faster_rcnn_r50.pt', help='the file to save export model')
    

    parser.add_argument('--work_size', type=int, nargs=2, metavar=('height', 'width'), help='model input size')

    parser.add_argument('--gpuid', type=str, default='0', help='visible gpu ids')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    detector = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    two_stage_export(detector, args.work_size, args.export_file)
    

if __name__=='__main__':

    
    main()
    
    # python two_stage_torchscript_export.py --work_size 1024 1024 --gpuid 4



