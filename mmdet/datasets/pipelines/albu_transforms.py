# 参考https://github.com/amirassov/kaggle-imaterialist/blob/master/mmdetection/mmdet/datasets/extra_aug.py

import mmcv
import numpy as np
from ..registry import PIPELINES

import albumentations as A
from mmcv.runner import obj_from_dict

from ipdb import set_trace
@PIPELINES.register_module
class AlbuAugmentation(object):
    '''
    
    对bboxes或keypoints进行变换，需要在config文件中添加bbox_params和keypoint_params
    example:
        dict(
            type='AlbuAugmentation', 
            transforms=[
                dict(p=0.5, max_h_size=64, type='Cutout'),
                dict(brightness_limit=0.3, contrast_limit=0.3, p=0.5, type='RandomBrightnessContrast'),
                dict(p=0.5, quality_lower=80, quality_upper=99, type='JpegCompression'),
               ],
            p=1.0,
            bbox_params=dict(format='pascal_voc', label_fields=['labels']),  #format='coco' 注意
            keypoint_params=dict(format='xy')
        )
    '''

    def __init__(self, **kwargs):
        self.transform = self.transform_from_dict(**kwargs)
        

    def transform_from_dict(self, **kwargs):
        if 'transforms' in kwargs:
            kwargs['transforms'] = [obj_from_dict(transform, A) for transform in kwargs['transforms']]
        if 'bbox_params' in kwargs:
            kwargs['bbox_params'] = A.BboxParams(**kwargs['bbox_params'])
        if 'keypoint_params' in kwargs:
            kwargs['keypoint_params'] = A.KeypointParams(**kwargs['keypoint_params'])
            
        transform = A.Compose(**kwargs)
        return transform


    def __call__(self, results):
        #set_trace()
        image = results['img']
        bboxes, labels, keypoints, mask = [], [], [], None
        
        #对于albumentations来说，获取bboxes时必须同时获取label
        # albumentations使用list操作
        if 'gt_bboxes' in results.keys():
            bboxes = list(results['gt_bboxes'])
            labels = list(results['gt_labels'])
        if 'gt_keypoints' in results.keys():
            keypoints = list(results['gt_keypoints'])
        if 'gt_masks' in results.keys():
            mask = results['gt_masks']
        try :
            augmented_data = self.transform(image=image, bboxes=bboxes, labels=labels, keypoints=keypoints, mask=mask)
        except :
            return results
        
        if len(augmented_data['labels'])==0 :
            return results

        results['img'] = augmented_data['image']
        img_shape = results['img'].shape
        # mmdet使用np操作 bboxes等
        if 'gt_bboxes' in results.keys():
            results['gt_labels'] = np.array(augmented_data['labels'], dtype=np.int64)
            
            results['gt_bboxes'] = np.array(augmented_data['bboxes'], dtype=np.float32)
            
            #albumentations仿射变换等操作不包括最后一个像素，而mmdet包括最后一个像素
            results['gt_bboxes'][:, 0::2] = np.clip(results['gt_bboxes'][:, 0::2], 0, img_shape[1] - 1)
            results['gt_bboxes'][:, 1::2] = np.clip(results['gt_bboxes'][:, 1::2], 0, img_shape[0] - 1) 
        if 'gt_keypoints' in results.keys():
            results['gt_keypoints'] = np.array(augmented_data['keypoints'], dtype=np.float32)
            
            results['gt_keypoints'][:, 0::2] = np.clip(results['gt_keypoints'][:, 0::2], 0, img_shape[1] - 1)
            results['gt_keypoints'][:, 1::2] = np.clip(results['gt_keypoints'][:, 1::2], 0, img_shape[0] - 1)
        if 'gt_masks' in results.keys():
            results['gt_masks'] = augmented_data['mask']
        
        return results
    

##transforms = [
##’Identity’, ’AutoContrast’, ’Equalize’,
##’Rotate’, ’Solarize’, ’Color’, ’Posterize’,
##’Contrast’, ’Brightness’, ’Sharpness’,
##’ShearX’, ’ShearY’, ’TranslateX’, ’TranslateY’]
@PIPELINES.register_module
class RandAugment(object):
    """Applies the RandAugment policy to `image`.
       RandAugment is from the paper https://arxiv.org/abs/1909.13719,
 
    """
    def __init__(self, N=2, M=5):
        '''
        Args:
            N: Number of augmentation transformations to apply sequentially.
            M: Magnitude for all the transformations.
        '''
      
        
        self.N = N
        self.M = M
        

        magnitude = M / 10.0

        Identity = A.NoOp(p=1.0)
        
        Brightness = A.RandomBrightness(limit=0.5 * magnitude, p=1.0)
        Contrast = A.RandomContrast(limit=0.5 * magnitude, p=1.0)
        #Solarize = albu.Solarize(threshold= int(256 * magnitude), p=1.0)
        
        #HorizontalFlip = A.HorizontalFlip(p=1.0)
        Translate = A.ShiftScaleRotate(shift_limit=0.45 * magnitude, scale_limit=0, rotate_limit=0, interpolation=1, border_mode=0, p=1.0)
        Scale= A.ShiftScaleRotate( shift_limit=0, scale_limit=0.5 * magnitude, rotate_limit=0, interpolation=1, border_mode=0, p=1.0)
        #Shear = albu.IAAAffine(shear=100, p=1.0)
        #Perspective = albu.IAAPerspective(scale=(0, 0.3), keep_size=True, p=1.0)
        Rotate = A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=30 * magnitude, interpolation=1, border_mode=0, p=1.0)
        
        self.augment_list = [Identity, Translate, Scale, Rotate, Brightness, Contrast]
    
    def __call__(self, results):
        image = results['img']
        
        # albumentations使用list操作
        bboxes = list(results['gt_bboxes'])
        labels = list(results['gt_labels'])

        
        transform_list = list(np.random.choice(self.augment_list, self.N) )
        
        self.transform = A.Compose(transform_list, bbox_params=dict(format='pascal_voc', label_fields=['labels']) )
        try:
            augmented_data = self.transform(image=image, bboxes=bboxes, labels=labels)
        except :
            return results
        if len(augmented_data['labels'])==0 :
            return results
        
        results['img'] = augmented_data['image']
        # mmdet使用np操作 bboxes等
        img_shape = augmented_data['image'].shape
        results['gt_labels'] = np.array(augmented_data['labels'], dtype=np.int64)
        results['gt_bboxes'] = np.array(augmented_data['bboxes'], dtype=np.float32)
        
        #albumentations仿射变换等操作不包括最后一个像素，而mmdet包括最后一个像素
        if len(results['gt_bboxes']) > 0 :
            results['gt_bboxes'][:, 0::2] = np.clip(results['gt_bboxes'][:, 0::2], 0, img_shape[1] - 1)
            results['gt_bboxes'][:, 1::2] = np.clip(results['gt_bboxes'][:, 1::2], 0, img_shape[0] - 1)
    
        return results