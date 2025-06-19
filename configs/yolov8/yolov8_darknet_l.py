from mmengine import read_base

with read_base():
    from .._base_.models.dx_yolo import *  # noqa
    from .._base_.datasets.dxyolo import *  # noqa
    from .._base_.schedules.schedule_yolo_100e import *  # noqa
    from .._base_.yolo_run import *  # noqa

env_cfg = dict(cudnn_benchmark=True)
model.update(dict(
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=512,
        deepen_factor=1,
        widen_factor=1,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=1,
        widen_factor=1,
        in_channels=[256, 512, 512],
        out_channels=[256, 512, 512],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
), bbox_head=dict(
    head_module=dict(
        in_channels=[256, 512, 512],
        widen_factor=1,
    ), ))
