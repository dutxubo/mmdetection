#coding=utf-8
import sys
import argparse 
import os
import cv2
import torch
import torch.nn as nn
import pdb
from mmdet.apis import init_detector, inference_detector
import numpy as np


class SimpleDetector(nn.Module):
    def __init__(self, detector):
        super(SimpleDetector, self).__init__()
        self.detector = detector.eval()
       
    def forward(self, imgs, img_metas):
        outs = self.detector.forward_jit(imgs, img_metas)
        return outs
    
    
    
def export_model(detector, work_size, export_file):
    # build input tensor
    H, W, C = work_size[0], work_size[1], 3
    input_shape = (C, H, W) # c, h, w
    input_tensor = torch.randn( input_shape )
    input_tensor = input_tensor.unsqueeze(0).float().cuda()
    
    img_metas = [{
            'img_shape': torch.tensor( (H, W, C) ),
            'ori_shape': torch.tensor( (H, W, C) ),
            'pad_shape': torch.tensor( (H, W, C) ),
            'scale_factor': torch.tensor((1.0, 1.0, 1.0, 1.0) ),  
            } for _ in range(1)]
    
    # build jit detector
    simple_detector = SimpleDetector(detector).cuda()

    # export model
    script_module = torch.jit.trace(simple_detector.eval(), (input_tensor, img_metas ) )
    script_module.save(export_file)
    
    return script_module
    
    
def check_jit_model(detector, work_size, export_file):
    # build input tensor
    H, W, C = work_size[0], work_size[1], 3
    input_shape = (C, H, W) # c, h, w
    input_tensor = torch.randn( input_shape )
    input_tensor = input_tensor.unsqueeze(0).float().cuda()
    
    img_metas = [{
            'img_shape': torch.tensor( (H, W, C) ),
            'ori_shape': torch.tensor( (H, W, C) ),
            'pad_shape': torch.tensor( (H, W, C) ),
            'scale_factor': torch.tensor((1.0, 1.0, 1.0, 1.0) ),  
            } for _ in range(1)]
    
    # detector inference
    single_stage_detector = SimpleDetector(detector).cuda()
    output = single_stage_detector(input_tensor, img_metas)
    
    # jit_model inference
    script_module = torch.jit.load(export_file)
    script_module_output = script_module(input_tensor, img_metas)
    
    print('The difference between output and script_module_output: ', (output[0] - script_module_output[0]).sum() )
    
    
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default='/home/songbai.xb/detection/mmdetection_merge/mmdetection/myprojects/extinguisher/configs/faster_rcnn_r50_fpn_1x.py', help='config_file')

    parser.add_argument('--checkpoint_file', type=str, default='/home/songbai.xb/detection/mmdetection_merge/mmdetection/myprojects/extinguisher/work_dirs/all/faster_rcnn_r50_fpn_1x_mykeep_ratio_RandAugment/latest.pth', help='model weight')


    parser.add_argument('--export_file', type=str, default='/home/songbai.xb/detection/mmdetection_merge/mmdetection/myprojects/extinguisher/faster_rcnn_simple_version1.pt', help='the file to save export model')

    parser.add_argument('--work_size', type=int, nargs=2, metavar=('height', 'width'), help='model input size')

    parser.add_argument('--gpuid', type=str, default='0', help='visible gpu ids')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    

    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    
    # detector init
    detector = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    # export detector
    script_module = export_model(detector, args.work_size, args.export_file)
    
    # check export detector
    check_jit_model(detector, args.work_size, args.export_file )

if __name__=='__main__':

    
    main()
    
    # python single_stage_torchscript_export.py --work_size 1024 1024 --gpuid 4



