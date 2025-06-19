from ad_idea.configs._base_.datasets.sampler.balancesampler import BalanceSampler
data_root = '/home/jz207/workspace/dinglg/Data/DX/'   # Root path of data

img_scale = (640, 640)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
max_aspect_ratio = 100
DX_METAINFO = dict(
    classes=('boneanomaly', 'bonelesion', 'softtissue', 'fracture', 'metal',
             'periostealreaction', 'pronatorsign', 'text',))

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
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
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type=BalanceSampler, shuffle=True, annotation_file=data_root+'annotations/train.json'),
    # collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='image/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        metainfo=DX_METAINFO))

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
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        ann_file='annotations/valid.json',
        data_prefix=dict(img='image/valid/'),
        pipeline=test_pipeline,
        batch_shapes_cfg=None,metainfo=DX_METAINFO))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    outfile_prefix=data_root,
    ann_file=data_root + 'annotations/valid.json',
    metric='bbox',

    backend_args=None, classwise=True)
test_evaluator = val_evaluator
