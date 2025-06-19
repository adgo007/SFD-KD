from mmengine import read_base

with read_base():
    from .._base_.models.dx_yolo import *  # noqa
    from .._base_.datasets.dx_yolo import *  # noqa
    from .._base_.schedules.schedule_yolo_100e import *  # noqa
    from .._base_.yolo_run import *  # noqa

env_cfg = dict(cudnn_benchmark=True)
model.update(dict(
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=0.33,
        widen_factor=0.5,
        in_channels=[128, 256, 512, 1024],
        out_channels=[128, 256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        head_module=dict(
            in_channels=[128, 256, 512, 1024],
            widen_factor=0.5,
            featmap_strides=[4, 8, 16, 32]),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=[4, 8, 16, 32]),
    )
))
