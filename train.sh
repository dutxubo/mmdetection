




CUDA_VISIBLE_DEVICES=1,2,3,4 ./tools/dist_train.sh myproject/baoguoxia/configs/aug_test/reppoints_minmax_r50_fpn_2x_mt_crop.py  4 --validate

CUDA_VISIBLE_DEVICES=1,2,3,4 ./tools/dist_train.sh myproject/baoguoxia/configs/aug_test/reppoints_minmax_r50_fpn_2x_mt_erase.py  4 --validate



#CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/reg_loss_test/retinanet_r50_fpn_1x_balanced_l1.py  4 --validate

#CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/cascade_rcnn_r50_fpn_1x.py  4 --validate

#CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/reg_loss_test/retinanet_r50_fpn_1x_iou.py  4 --validate

#CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/reg_loss_test/retinanet_r50_fpn_1x_giou.py  4 --validate

#CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/reg_loss_test/retinanet_r50_fpn_1x_diou.py  4 --validate

#CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/reg_loss_test/retinanet_r50_fpn_1x_ciou.py  4 --validate
