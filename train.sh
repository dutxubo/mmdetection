

CUDA_VISIBLE_DEVICES=3,4,5,6 ./tools/dist_train.sh myproject/dengxiang/configs/lightweight/reppoints_minmax_r18_fpn.py  4 --validate

CUDA_VISIBLE_DEVICES=3,4,5,6 ./tools/dist_train.sh myproject/dengxiang/configs/lightweight/reppoints_minmax_shufflenetv2_fpn.py  4 --validate

CUDA_VISIBLE_DEVICES=3,4,5,6 ./tools/dist_train.sh myproject/dengxiang/configs/lightweight/reppoints_minmax_mobilenetv2_fpn.py  4 --validate

CUDA_VISIBLE_DEVICES=3,4,5,6 ./tools/dist_train.sh myproject/dengxiang/configs/lightweight/reppoints_minmax_efficientnetb3_fpn.py  4 --validate


#CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/reg_loss_test/retinanet_r50_fpn_1x_balanced_l1.py  4 --validate

#CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/cascade_rcnn_r50_fpn_1x.py  4 --validate

#CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/reg_loss_test/retinanet_r50_fpn_1x_iou.py  4 --validate

#CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/reg_loss_test/retinanet_r50_fpn_1x_giou.py  4 --validate

#CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/reg_loss_test/retinanet_r50_fpn_1x_diou.py  4 --validate

#CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh myproject/coco/configs/reg_loss_test/retinanet_r50_fpn_1x_ciou.py  4 --validate
