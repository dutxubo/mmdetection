#coding=utf-8
import sys
import argparse 
import os
import cv2
import torch
import torch.nn as nn
import pdb
from mmdet.apis import init_detector, inference_detector


class SingleStageDetector(nn.Module):
    def __init__(self, detector):
        super(SingleStageDetector, self).__init__()
        self.detector = detector.eval()
        self.fpn_level = 5
    
    def forward(self, imgs):
        outs = self.detector.forward_dummy(imgs)
        self.fpn_level = len(outs[0])
        
        # the jit must be a tensor or tuple(tensor) output
        jit_out = tuple()
        for t in outs:
            jit_out += tuple(t)
        return jit_out
    
    def convert_jit_out_to_actual_out(self, jit_out):
        '''the format of jit_out is Tuple[Tensor, Tensor, Tensor, Tensor, ...]
           the format of actual network out is Tuple[Tuple[Tensor, Tensor, ...], Tuple[Tensor, Tensor, ...], ...]
        '''
        out_num = len(jit_out) // self.fpn_level
        actual_out = tuple()
        for i in range(out_num):
            actual_out += (jit_out[i*self.fpn_level: (i+1)*self.fpn_level], )
        return actual_out
    
def export_model(detector, work_size, export_file):
    # build input tensor
    h, w = work_size[0], work_size[1]
    input_shape = (3, h, w) # c, h, w
    input_tensor = torch.randn( input_shape )
    input_tensor = input_tensor.unsqueeze(0).float().cuda()
    
    # build jit detector
    single_stage_detector = SingleStageDetector(detector).cuda()
    output = single_stage_detector(input_tensor)

    # export model
    script_module = torch.jit.trace(single_stage_detector.eval(), input_tensor )
    script_module.save(export_file)
    
    return script_module
    
    
def check_jit_model(detector, work_size, export_file):
    # build input tensor
    h, w = work_size[0], work_size[1]
    input_shape = (3, h, w) # c, h, w
    input_tensor = torch.randn( input_shape )
    input_tensor = input_tensor.unsqueeze(0).float().cuda()
    
    # detector inference
    single_stage_detector = SingleStageDetector(detector).cuda()
    output = single_stage_detector(input_tensor)
    
    # jit_model inference
    script_module = torch.jit.load(export_file)
    script_module_output = script_module(input_tensor)
    
    print('The difference between output and script_module_output: ', (output[0] - script_module_output[0]).sum() )
    
    
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default='/home/songbai.xb/detection/mmdetection/myprojects/extinguisher/configs/reppoints_minmax_r50_fpn_1x.py', help='config_file')

    parser.add_argument('--checkpoint_file', type=str, default='/home/songbai.xb/detection/mmdetection/myprojects/extinguisher/work_dirs/all/reppoints_minmax_r50_fpn_mykeep_ratio_RandAugment/latest.pth', help='model weight')


    parser.add_argument('--export_file', type=str, default='/home/songbai.xb/detection/mmdetection_merge/mmdetection/myprojects/extinguisher/reppoints_minmax_r50_fpn_1x.pt', help='the file to save export model')

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



