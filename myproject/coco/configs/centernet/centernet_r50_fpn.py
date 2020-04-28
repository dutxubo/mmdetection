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
    #neck=dict(
    #    type='DLAFPN',
    #    in_channels=[256, 512, 1024, 2048],
    #    out_channels=256,
    #    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=4,
        #activation_num=1
        ),
    bbox_head=dict(
        type='CenterHead',
        num_classes=81,
        in_channels=256,
        stacked_convs=1,
        feat_channels=256,
        
        strides=[4, 8, 16, 32],
        regress_ranges=((-1,  64), (64, 128), (128, 256), (256, 1e8)),
        #strides=[4],
        #regress_ranges=((-1, 1e8), ),
        loss_hm=dict(type='CenterFocalLoss'),
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
dataset_type = 'CocoDataset'
data_root = '/home/songbai.xb/dataset/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#img_norm_cfg = dict(
#        mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    #dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    #dict(type='Resize', img_scale=(512, 512), ratio_range=(0.8, 1.2), keep_ratio=True),
    #dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size=(512, 512)),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        #img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, min_rescale=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            #dict(type='Pad', size=(512, 512)),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/home/songbai.xb/dataset/instances_train2017_part.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
#optimizer = dict(type='Adam', lr= 0.00025, betas=(0.9, 0.999), eps=1e-8)
#optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001, paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
#optimizer_config = dict(grad_clip=None)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=2000,
    warmup_ratio=1.0 / 3,
    step=[8, 11]
    #step=[60, 80]
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
#device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './myproject/coco/work_dirs/centernet/center_fpn_r50_caffe_fpn_gn_1x_4gpu'
load_from = None
resume_from = None
workflow = [('train', 1)]

