

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/neck_test/retinanet_crop640_r50_fpn_50e.py  8 --validate --resume_from myproject/coco/word_dirs/neck_test/epoch_26.pth

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/neck_test/retinanet_bn_r50_fpn_1x.py  8 --validate
