import argparse
import os

import cv2
import mmcv
import torch
import numpy as np
from mmcv import Config

from mmcv.parallel import collate, scatter
from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.apis import inference_detector, init_detector



class LoadImage(object):
    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results
    
def pre_process(model, img):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet line_profiler a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='model weight')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--gpuid',
        type=str,
        default='0',
        help='gpu id')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (args.shape[0], args.shape[0], 3)
    elif len(args.shape) == 2:
        input_shape = tuple(args.shape) + (3, ) 
    else:
        raise ValueError('invalid input shape')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    
    config = args.config
    checkpoint = args.checkpoint
    model = init_detector(config, checkpoint, device='cuda:0')
    model.cfg.data.test.pipeline[0]['img_scale'] = input_shape[:2]
    model.cfg.data.test.pipeline[1]['img_scale'] = input_shape[:2]
    
    img = cv2.imread('/home/songbai.xb/dataset/coco/val2017/000000001000.jpg', 1)
    #img = np.random.randn(*input_shape)
    
    
    data = pre_process(model, img)
    print('img:', data['img'][0].shape)
    result = model(return_loss=False, rescale=False, **data)


#@profile
#kernprof -l -v test.py
if __name__ == '__main__':
    main()
