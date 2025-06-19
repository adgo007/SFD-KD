from albumentations import RandomBrightnessContrast
from mmdet.datasets import CocoDataset
# dataset settings

dataset_type = CocoDataset
data_root = '/home/jz207/workspace/dinglg/Data/FracAtlas/'

backend_args = None

DX_METAINFO = dict(
    classes=('fractured')
)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(480, 480), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Albu',
        transforms=[
            dict(type='Blur', p=0.01),
            dict(type='MedianBlur', p=0.01),
            dict(type='ToGray', p=0.01),
            dict(type='CLAHE', p=0.01)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes',
        }),
    dict(type='YOLOXHSVRandomAug'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(480, 480), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(_scope_='mmdet', type='AspectRatioBatchSampler'),
    dataset=dict(
        _scope_='mmdet',
        type=dataset_type,
        data_root=data_root,
        ann_file='Annotations/train.json',
        data_prefix=dict(img='images/train/'),
        # filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args,
        metainfo=DX_METAINFO
    ))
val_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _scope_='mmdet',
        type=dataset_type,
        data_root=data_root,
        ann_file='Annotations/valid.json',
        data_prefix=dict(img='images/valid/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        metainfo=DX_METAINFO))

val_evaluator = dict(
    _scope_='mmdet',
    type='CocoMetric',
    ann_file=data_root + 'Annotations/valid.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
    classwise=True)

test_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _scope_='mmdet',
        type=dataset_type,
        data_root=data_root,
        ann_file='Annotations/valid.json',
        data_prefix=dict(img='images/valid/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        metainfo=DX_METAINFO))

test_evaluator = dict(
    _scope_='mmdet',
    type='CocoMetric',
    ann_file=data_root + 'Annotations/valid.json',
    metric='bbox',
    format_only=True,
    backend_args=backend_args, classwise=True)
