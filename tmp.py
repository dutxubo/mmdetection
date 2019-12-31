import time
import matplotlib
import matplotlib.pylab as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
plt.rcParams["axes.grid"] = False

import mmcv
from mmcv.runner import load_checkpoint
import mmcv.visualization.image as mmcv_image
# fix for colab


def imshow(img, win_name='', wait_time=0): 
    plt.figure(figsize=(10, 10)); 
    plt.imshow(img)


mmcv_image.imshow = imshow
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, init_detector


from mmcv import Config
from mmdet.datasets import build_dataset
config_file = 'myprojects/doortitle/configs/repkeypoints_minmax_r50_fpn_1x.py'
checkpoint_file = 'myprojects/doortitle/work_dirs/repkeypoints_minmax_r50_fpn_1x/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

cfg = Config.fromfile(config_file)
#model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
#datasets = build_dataset(cfg.data.test)

img_root = '/home/songbai.xb/projects/yunxunjian/datasets/yunxunjian/doorTitleImgUrl门头照片/'
img_root = '/mnt/disk6/songbai.xb/datasets/yunxunjian/doorTitleImg_100/'
import glob
import random
img_paths = glob.glob(img_root + '*jpg')


img_name = '20171211142728a0fd5f46-2620-4b78-895c-0bf7297b6988.jpeg'
img = img_root + img_name

img = random.choice(img_paths) #随机挑选
img = img_paths[0] #随机挑选

#result = inference_detector(model, img)
bbox_results, keypoint_results = inference_detector(model, img)
#result = result_nms(bbox_results)
score_thr = 0.3

