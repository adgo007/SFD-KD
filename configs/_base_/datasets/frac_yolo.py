# ========================Frequently modified parameters======================
# -----data related-----
data_root = '/home/jz207/workspace/dinglg/Data/FracAtlas/'  # Root path of data
# Path of train annotation file
train_ann_file = 'Annotations/train.json'
train_data_prefix = 'images/train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'Annotations/valid.json'
val_data_prefix = 'images/valid/'  # Prefix of val image path

test_ann_file = 'Annotations/valid.json'
test_data_prefix = 'images/valid/'
DX_METAINFO = dict(
    classes=('fractured'))

num_classes = 1  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 2
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 1
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# -----model related-----
# Basic size of multi-scale prior box

affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs
base_lr = 0.01
max_epochs = 300  # Maximum training epochs
model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (480, 480)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = 'mmyolo.YOLOv5CocoDataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 2
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 1

test_batch_size_per_gpu = 2
test_num_workers = 1
# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5)
albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]

train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline, metainfo=DX_METAINFO))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg, metainfo=DX_METAINFO))

test_dataloader = dict(
    batch_size=test_batch_size_per_gpu,
    num_workers=test_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=test_data_prefix),
        ann_file=test_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg, metainfo=DX_METAINFO))

val_evaluator = dict(
    _scope_='mmdet',
    type='CocoMetric',
    ann_file=data_root + 'Annotations/valid.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
    classwise=True)
test_evaluator = dict(
    _scope_='mmdet',
    type='CocoMetric',
    outfile_prefix=data_root,
    ann_file=data_root + 'Annotations/valid.json',
    metric='bbox',
    format_only=True,
    backend_args=None, classwise=True)
