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




mmcv_image.imshow = imshow
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, init_detector


from mmcv import Config
from mmdet.datasets import build_dataset
config_file = '/home/songbai.xb/detection/mmdetection_merge/mmdetection/myprojects/dengxiang/configs/reppoints_minmax_res50_fpn.py'
checkpoint_file = None
model = init_detector(config_file, checkpoint_file, device='cuda:0')

cfg = Config.fromfile(config_file)
model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
#datasets = build_dataset(cfg.data.test)

img_root = '/home/songbai.xb/projects/yunxunjian/deng_xiang/dataset/20191209/images/'
import glob
import random
img_paths = glob.glob(img_root + '*jpg')


#img_name = '20171211142728a0fd5f46-2620-4b78-895c-0bf7297b6988.jpeg'
#img = img_root + img_name

#img = random.choice(img_paths) #随机挑选
img = img_paths[0] 

#result = inference_detector(model, img)
for i in range(20):
    output = inference_detector(model, img)



