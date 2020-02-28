import mmcv
import numpy as np
from numpy import random
import random


from ..registry import PIPELINES

from ipdb import set_trace

import cv2
import albumentations as albu
from .colorgram import extract

def random_erasure(img, ratio=(0.5, 0.5) , bbox=None, mode=None, fill_value=(255,255,255)):
    '''
    对img上bbox区域进行随机擦除
    
    目前只支持垂直擦除
    '''

    if bbox is not None:
        x1, y1, x2, y2 =  bbox
        x, y, w, h = x1, y1, x2 - x1 +1, y2 - y1 +1
    else:
        x, y, w, h = 0, 0, img.shape[1], img.shape[0]
    
    img_crop = img[y:y + h, x:x + w]
    
 
    if mode == 'ave':
        ave_chans = tuple(cv2.meanStdDev(img_crop)[0].reshape(-1).astype(int)) 
        img_crop = albu.CoarseDropout(max_holes=1, max_height=int(h/1.3), max_width=int(w*rate), fill_value=ave_chans, p=1.0)(image=img_crop)['image']
    elif mode == 'median':
        chann_0 = int(cv2.calcHist(img_crop, [0], None, [64], [0, 255]).argmax()*255/64 + 2)   
        chann_1 = int(cv2.calcHist(img_crop, [1], None, [64], [0, 255]).argmax()*255/64 + 2)
        chann_2 = int(cv2.calcHist(img_crop, [2], None, [64], [0, 255]).argmax()*255/64 + 2)
        ave_chans = (chann_0, chann_1, chann_2)
        img_crop = albu.Cutout(num_holes=2, max_h_size=int(h*ratio[1]), max_w_size=int(w*ratio[0]), fill_value=ave_chans, p=1)(image=img_crop)['image']
        #img_crop = albu.CoarseDropout(max_holes=1, max_height=int(h/1.3), max_width=int(w*rate), fill_value=ave_chans, p=1.0)(image=img_crop)['image']
    else:
        img_crop = albu.Cutout(num_holes=1, max_h_size=int(h*ratio[1]), max_w_size=int(w*ratio[0]), fill_value=fill_value, p=1)(image=img_crop)['image']
        #img_crop = albu.CoarseDropout(max_holes=1, max_height=int(h*ratio[1]), max_width=int(w*ratio[0]), fill_value=fill_value, p=1.0)(image=img_crop)['image']
    img[y:y + h, x:x + w] = img_crop
    return img, bbox

@PIPELINES.register_module
class Erasure(object):
    """随机擦除图像区域

 

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        
    """
    #CLASSES = ('standard', 'non-standard' , 'bk', 'fg_cnyz', 'logo_cainiao', 'fg')
    def __init__(self, erasure_labels= [4, 5], erasure_ratios=(0.3, 0.3), dst_label=0, p=0.5, background_label=None, standard_label=None):
        
        self.erasure_labels = erasure_labels
        self.erasure_ratios = erasure_ratios
        self.p = p
        self.dst_label = dst_label
        self.background_label = background_label
        self.standard_label = standard_label

    def __call__(self, results):
        if random.uniform(0,1) > self.p:
            return results
        
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        #set_trace()
        erasure_label = random.choice(self.erasure_labels)
        
        indexs =  np.where(labels == erasure_label)[0]
        if len(indexs) == 0 :
            return results
        #只选取一个bbox
        index = indexs[0]
        erasure_bbox = boxes[index,:].reshape(-1).astype(np.int32)
       
        #提取背景颜色
        if self.background_label is not None:
            background_indexs =  np.where(labels == self.background_label)[0]
            if len(background_indexs) == 0 :
                return results
            background_bbox = boxes[background_indexs[0],:].reshape(-1).astype(np.int32)
            x1, y1, x2, y2 =  background_bbox
            x, y, w, h = x1, y1, x2 - x1 +1, y2 - y1 +1
            background_img = img[y:y + h, x:x + w]
        else:
            background_img = img
        colors = extract(background_img,2)
        ave_chans = colors[0].rgb
        ave_chans = (ave_chans[2], ave_chans[1], ave_chans[0])      #rgb to bgr

        img, _ = random_erasure(img, ratio=self.erasure_ratios, bbox=erasure_bbox, fill_value=ave_chans)
        
        labels[index] = self.dst_label
        
        ##将对应的standard标签也改为背景
        if self.standard_label is not None:
            standard_indexs =  np.where(labels == self.standard_label)[0]
            if len(standard_indexs) > 0 :
                standard_index =  standard_indexs[0]
                labels[standard_index] = 0
            
        results['gt_labels'] = labels
        results['img'] = img
        
        return results


def random_paste(img, obj,  pts=None, mode=False):
    '''
    在img上pts区域，使用obj随机粘贴
    '''
    if pts is not None:
        x, y, w, h =  cv2.boundingRect(pts)
    else:
        x, y, w, h = 0, 0, img.shape[1], img.shape[0]
    background = img[y:y + h, x:x + w]
    
    # Get background and object dimensions
    bg_h, bg_w = background.shape[:2]
    obj_h, obj_w = obj.shape[:2]
    
    # Get area of background and object
    bg_area = bg_h * bg_w
    obj_area = obj_h * obj_w
    
    # Resize the object to a random percentage of the background area
    obj_resize_scale = np.random.uniform(0.1, 0.3)
    w_h_ratio = math.sqrt((obj_resize_scale * bg_area) / obj_area)
    new_obj_h, new_obj_w = obj_h * w_h_ratio, obj_w * w_h_ratio
    
    # change object height and width randomly
    new_obj_h = int(new_obj_h * np.random.normal(loc=1, scale=0.2))
    new_obj_w = int(new_obj_w * np.random.normal(loc=1, scale=0.2))
    
    new_obj_h = min(new_obj_h, bg_h-2)
    new_obj_w = min(new_obj_w, bg_w-2)
    if new_obj_h < 10 or new_obj_w < 10:
        return img, np.array([0,0,1,1])
    
    # Resize object
    obj = cv2.resize(obj, dsize=(new_obj_w, new_obj_h), interpolation = cv2.INTER_CUBIC)
    
    # Pick random spot to paste image
    pot_x, pot_y = random.randint(0, bg_w  - new_obj_w -1), random.randint(0, bg_h - new_obj_h -1)
    center = (pot_x, pot_y)
    
        
    # Seamlessly clone src into dst and put the results in output
    #gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    #thres, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU )
    mask = 255 * np.ones((obj.shape[:2]), dtype=np.uint8)
    
    #  cv2.NORMAL_CLONE or cv2.MIXED_CLONE 
    #normal_clone = cv2.seamlessClone(obj, background, mask, center, cv2.MONOCHROME_TRANSFER)
    background[pot_y:pot_y+new_obj_h, pot_x:pot_x+new_obj_w] = obj
    normal_clone = background
    img[y:y + h, x:x + w] = normal_clone
    paste_bbox = np.array([pot_x + x, pot_y + y, pot_x + new_obj_w + x, pot_y + new_obj_h + y] )
    return img, paste_bbox

import glob
import math
@PIPELINES.register_module
class Paste(object):
    """背景粘贴图像
        
    """
    #CLASSES = ('standard', 'non-standard' , 'bk', 'fg_cnyz', 'logo_cainiao', 'fg')
    def __init__(self, noise_root, noise_label=6, p=0.5, bk_label=1):
        self.noise_root = noise_root
        
        self.noise_img_paths = glob.glob(noise_root + '*')
        
        self.p = p
        self.noise_label = noise_label
        self.bk_label = bk_label
        
    def __call__(self, results):
        if random.uniform(0,1) > self.p:
            return results
        
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]

        noise_img = cv2.imread(random.choice(self.noise_img_paths) )
        
        #bk_rect = np.where(labels == self.bk_label)[0]
        h, w = img.shape[:2]
        paste_region = np.array([[w//2,0],[w-1, int(h/1.8)]] )
        
        paset_img, paste_bbox = random_paste(img, noise_img, paste_region )
        
        labels = np.append(labels, self.noise_label)
        boxes = np.append(boxes, [paste_bbox], axis=0)
        results['img'] = paset_img
        results['gt_labels'] = labels
        results['gt_bboxes'] = boxes
        
        if 'gt_keypoints' in results.keys():
            pts = results['gt_keypoints']
            
            x1, y1, x2, y2  = paste_bbox
            
            pts = np.append(pts, [[x1, y1, x2, y1, x2, y2, x1, y2]], axis=0)
           
            
            results['gt_keypoints'] = pts
        return results
    
import glob
import math
@PIPELINES.register_module
class PasteNonDetect(object):
    """背景粘贴图像
        
    """
    #CLASSES = ('standard', 'non-standard' , 'bk', 'fg_cnyz', 'logo_cainiao', 'fg')
    def __init__(self, nondetect_root, nondetect_label=0, p=0.5, standard_label=1):
        self.nondetect_root = nondetect_root
        
        self.nondetect_img_paths = glob.glob(nondetect_root + '*')
        
        self.p = p
        self.nondetect_label = nondetect_label
        self.standard_label = standard_label
        
    def __call__(self, results):
        if random.uniform(0,1) > self.p:
            return results
        
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        if isinstance(self.standard_label, list):
            curr_standard_label =  random.choice(self.standard_label)
        else:
            curr_standard_label = self.standard_label
        standard_indexs =  np.where(labels == curr_standard_label)[0]
        if len(standard_indexs) != 1:
            return results 
        
        nondetect_img = cv2.imread(random.choice(self.nondetect_img_paths) )
        
        standard_rect = boxes[standard_indexs[0],:].reshape(-1).astype(np.int32)
        
        h, w = img.shape[:2]
        paste_region = np.array([[0,standard_rect[3]],[w-1, h-1]] )
        
        paset_img, paste_bbox = random_paste(img, nondetect_img, paste_region )
        
        labels = np.append(labels, self.nondetect_label)
        boxes = np.append(boxes, [paste_bbox], axis=0)
        results['img'] = paset_img
        #results['gt_labels'] = labels
        #results['gt_bboxes'] = boxes
        
        #if 'gt_keypoints' in results.keys():
        #    pts = results['gt_keypoints']
        #    
        #    x1, y1, x2, y2  = paste_bbox
        #    
        #    pts = np.append(pts, [[x1, y1, x2, y1, x2, y2, x1, y2]], axis=0)
        #   
        #    
        #    results['gt_keypoints'] = pts
        return results


    

def crop_with_bboxes_pts(img, crop_rect, bboxes, pts, labels=None):
    '''
    crop_rect : [x1, y1, x2, y2]
    pts (array[array]): n*2k
    '''
    x1, y1, x2, y2 = crop_rect
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
    crop_img = img[y1:y2, x1:x2, :]
    crop_img_shape = crop_img.shape
    
    new_pts = pts.copy()
    new_pts[:, 0::2] = np.clip(new_pts[:, 0::2] - x1, 0, crop_img_shape[1] - 1)
    new_pts[:, 1::2] = np.clip(new_pts[:, 1::2] - y1, 0, crop_img_shape[0] - 1)
    
    new_bboxes = bboxes.copy()
    new_bboxes[:, 0::2] = np.clip(new_bboxes[:, 0::2] - x1, 0, crop_img_shape[1] - 1)
    new_bboxes[:, 1::2] = np.clip(new_bboxes[:, 1::2] - y1, 0, crop_img_shape[0] - 1)
    
    # filter out the pts that are completely cropped
    valid_inds = (new_pts[:, 0::2].max(axis=1) > new_pts[:, 0::2].min(axis=1)) & (new_pts[:, 1::2].max(axis=1) > new_pts[:, 1::2].min(axis=1))
    new_pts = new_pts[valid_inds]
    new_bboxes = new_bboxes[valid_inds]
    if labels is None:
        return crop_img, new_bboxes, new_pts
    else:
        return crop_img, new_bboxes, new_pts, labels[valid_inds]


@PIPELINES.register_module
class RandomCropBk(object):
    """基于白框的位置，对图像进行左裁剪，伪造出non-standard图像
    
    classname_to_id = {'standard' : 1, 'non-standard' : 2, 'bk' : 3, 'fg_cnyz' : 4, 'logo_cainiao' : 5, 'fg': 6}
    """

    def __init__(self, crop_ratio = 0.5, bk_label=3, standard_label=1, p=0.5):
        self.p = p
        self.crop_ratio = crop_ratio
        self.bk_label = bk_label
        self.standard_label = standard_label
        
        

    def __call__(self, results):
        if random.uniform(0,1) > self.p:
            return results
        
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        pts = results['gt_keypoints']
        img_h, img_w = img.shape[:2]
        #set_trace()
            
        bk_indexs =  np.where(labels == self.bk_label)[0]
        standard_indexs =  np.where(labels == self.standard_label)[0]
        fg_cnyzs =  np.where(labels == 2)[0]
        logo_cainiaos =  np.where(labels == 3)[0]
        
        #当图片中只有一个standard doortitle时，才对该门头的白框进行随机裁剪
        if len(bk_indexs) != 1 or len(standard_indexs)!=1 or len(fg_cnyzs)!=1 or len(logo_cainiaos)!=1:
            return results
        
        bk_index = bk_indexs[0]
        # 理论上需要判断bk框在standard内部
        standard_index =  standard_indexs[0]
        fg_cnyz =  fg_cnyzs[0]
        logo_cainiao =  logo_cainiaos[0]
        
        
        bk_bbox = boxes[bk_index,:].reshape(-1).astype(np.int32)
        
        bk_w, bk_h = bk_bbox[2] - bk_bbox[0],  bk_bbox[3] - bk_bbox[1]
        # 当图像颠倒时不进行裁剪
        if bk_h>bk_w:
            return results
        

        max_crop_w = bk_w * self.crop_ratio 
        min_crop_w = bk_w * self.crop_ratio / 500
        crop_x = random.randint( int(bk_bbox[0] + min_crop_w) , int(bk_bbox[0] + max_crop_w) )
        crop_rect = [crop_x, 0 , img_w, img_h]
        
        ## 如果裁剪超出了 fg_cnyz 和 logo_cainiao 区域，则改变对应的标签
        if crop_x > boxes[fg_cnyz][0] :
            labels[fg_cnyz] = 4
        if crop_x > boxes[logo_cainiao][0] :
            labels[logo_cainiao] = 4
            
        
        #将对应的standard标签改为 non-standard
        labels[standard_index] = 0 # 2
        
        
        crop_img, new_boxes, new_pts, new_labels = crop_with_bboxes_pts(img, crop_rect, boxes, pts, labels)
        
        
        
        results['gt_labels'] = new_labels
        results['img'] = crop_img
        results['gt_bboxes'] = new_boxes
        results['gt_keypoints'] = new_pts
        
        return results
    
    
@PIPELINES.register_module
class RandomCropCNYZ(object):
    """基于菜鸟驿站的位置，对图像进行下裁剪，伪造出non-standard图像
    
    classname_to_id = {'standard' : 1, 'non-standard' : 2, 'bk' : 3, 'fg_cnyz' : 4, 'logo_cainiao' : 5, 'fg': 6}
    """

    def __init__(self, crop_ratio = 0.5, cnyz_label=2, standard_label=5, p=0.5):
        self.p = p
        self.crop_ratio = crop_ratio
        self.cnyz_label = cnyz_label
        self.logo_label = 3
        self.standard_label = standard_label
        
        

    def __call__(self, results):
        if random.uniform(0,1) > self.p:
            return results
        
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        pts = results['gt_keypoints']
        img_h, img_w = img.shape[:2]
        #set_trace()
            
        cnyz_indexs =  np.where(labels == self.cnyz_label)[0]
        standard_indexs =  np.where(labels == self.standard_label)[0]
        #fg_cnyzs =  np.where(labels == 4)[0]
        logo_cainiao_indexs =  np.where(labels == self.logo_label)[0]
        
        #当图片中只有一个standard doortitle时，才对该门头的白框进行随机裁剪
        if len(cnyz_indexs) != 1 or len(standard_indexs)!=1  or len(logo_cainiao_indexs) != 1: #or len(fg_cnyzs)!=1 or len(logo_cainiaos)!=1:
            return results
        
        cnyz_index = cnyz_indexs[0]
        # 理论上需要判断bk框在standard内部
        standard_index =  standard_indexs[0]
        #fg_cnyz =  fg_cnyzs[0]
        logo_cainiao_index =  logo_cainiao_indexs[0]
        
        
        cnyz_bbox = boxes[cnyz_index,:].reshape(-1).astype(np.int32)
        logo_cainiao_bbox = boxes[logo_cainiao_index,:].reshape(-1).astype(np.int32)
        
        bk_w, bk_h = cnyz_bbox[2] - cnyz_bbox[0],  cnyz_bbox[3] - cnyz_bbox[1]
        

        max_crop_h = int(bk_h * self.crop_ratio)
        
        #crop_y_max = random.randint(cnyz_bbox[1], cnyz_bbox[1] + max_crop_h)

        crop_y_max = random.randint( int(cnyz_bbox[3] - max_crop_h - 1), int(cnyz_bbox[3]) )
        crop_rect = [0, 0 , img_w, crop_y_max]
        
        
        #将对应的standard标签改为 non-standard
        labels[standard_index] = 0 # 2
        
        #将对应的cnyz标签改为 fg
        labels[cnyz_index] = 4 # 2
        
        if logo_cainiao_bbox[3] > crop_y_max:
            labels[logo_cainiao_index] = 4 # 2
        
        crop_img, new_boxes, new_pts, new_labels = crop_with_bboxes_pts(img, crop_rect, boxes, pts, labels)
        
        
        
        results['gt_labels'] = new_labels
        results['img'] = crop_img
        results['gt_bboxes'] = new_boxes
        results['gt_keypoints'] = new_pts
        
        return results
    
    

@PIPELINES.register_module
class MyPreProcess(object):
    """颠倒图像进行旋转
    
    classname_to_id = {'standard' : 1, 'non-standard' : 2, 'bk' : 3, 'fg_cnyz' : 4, 'logo_cainiao' : 5, 'fg': 6}
    """

    def __init__(self):
        pass
        
        

    def __call__(self, results):
        
        
        img = results['img']
        img_h, img_w = img.shape[:2]
        
        
        if img_h < img_w :
            return results
        
        # 否则顺时针旋转90度
        img = cv2.flip(img, 0)
        img = cv2.transpose(img)
        results['img'] = img
        
        # 
        if 'gt_bboxes'  in results.keys():
            boxes = results['gt_bboxes']

            new_boxes = boxes.copy()
            new_boxes[:, 0::4], new_boxes[:, 1::4],  new_boxes[:, 2::4],  new_boxes[:, 3::4] = img_h - boxes[:, 3::4], boxes[:, 0::4],  img_h - boxes[:, 1::4],  boxes[:, 2::4]
            results['gt_bboxes'] = new_boxes
            
        if 'gt_keypoints' in results.keys():
            pts = results['gt_keypoints']
        
            # x1,y1,x2,y2,x3,y4 -> h-y4,x4,h-y1,x1,h-y2,x2,h-y3,x3
            new_pts = pts.copy()
            new_pts[:, 2::2], new_pts[:, 3::2] = pts[:, 1:7:2], pts[:, 0:6:2]
            new_pts[:, 0], new_pts[:, 1] =  pts[:, 7], pts[:, 6]
            new_pts[:, 0::2] = img_h - new_pts[:, 0::2]
            
            results['gt_keypoints'] = new_pts
        
        return results
    
    
    

@PIPELINES.register_module
class GetImageLevelLabel(object):
    """
    """

    def __init__(self, image_label=5):
        self.image_label = image_label
        
        

    def __call__(self, results):
        
        labels = results['gt_labels']
        image_label_indexs =  np.where(labels == self.image_label)[0]
        if len(image_label_indexs) > 0:
            image_level_label = 1
            #对应标签设为0，不参加检测训练
            for image_label_index in image_label_indexs:
                labels[image_label_index] = 0
            results['gt_labels'] = labels
        else:
            image_level_label = 0
        
        results['image_level_label'] = image_level_label
        return results