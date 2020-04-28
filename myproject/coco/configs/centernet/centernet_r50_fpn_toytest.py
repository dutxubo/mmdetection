# fp16 settings
# fp16 = dict(loss_scale=4.)

# model settings
model = dict(
    type='CenterNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='pytorch'),
    neck=dict(
        type='DLAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        ),
    
    #neck=dict(
    #    type='FPN',
    #    in_channels=[256, 512, 1024, 2048],
    #    out_channels=256,
    #    start_level=0,
    #    add_extra_convs=True,
    #    extra_convs_on_inputs=False,  # use P5
    #    num_outs=4,
    #    activation_num=1,
    #    ),
    bbox_head=dict(
        type='CenterHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=1,
        feat_channels=256,
        #strides=[4, 8, 16, 32],
        #regress_ranges=((-1,  64), (64, 128), (128, 256), (256, 1e8)),
        strides=[4],
        regress_ranges=((-1, 1e8),),
        loss_hm=dict(type='CenterFocalLoss', loss_weight=1.0),
        #loss_hm=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_wh = dict(type="L1Loss",loss_weight=0.1),
        loss_offset = dict(type="L1Loss",loss_weight=1.0))
)
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False
)
test_cfg = dict(
    a = 5
    #nms_pre=1000,
    #min_bbox_size=0,
    #score_thr=0.05,
    #nms=dict(type='nms', iou_thr=0.5),
    #max_per_img=100
)
# dataset settings
dataset_type = 'ToyDataset'

train_pipeline = [
    
    #dict(type='Resize', img_scale=(512, 512), keep_ratio=True, min_rescale=True),

    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_shape=(256, 256),
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=dataset_type,
        img_shape=(256, 256),
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        img_shape=(256, 256),
        pipeline=test_pipeline),
        test_mode=True)
# optimizer
#optimizer = dict(type='Adam', lr= 0.001)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
#optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    #step=[4]
    step=[8, 11]
)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
#total_epochs = 6
#device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './myproject/coco/work_dirs/centernet/center_fpn_r50_caffe_fpn_gn_1x_4gpu'
load_from = None
resume_from = None
workflow = [('train', 1)]

