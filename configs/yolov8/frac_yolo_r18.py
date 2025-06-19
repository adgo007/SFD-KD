from mmengine import read_base

with read_base():
    from .._base_.models.frac_yolov8 import *  # noqa
    from .._base_.datasets.frac_yolo import *  # noqa
    from .._base_.schedules.schedule_yolo import *  # noqa
    from .._base_.yolo_run import *  # noqa

env_cfg = dict(cudnn_benchmark=True)
model.update(dict(
    backbone=dict(
        type='mmdet.ResNet',  # 修改为 ResNet
        depth=18,  # 指定为 ResNet18
        num_stages=4,  # 通常 ResNet18 有 4 个阶段
        out_indices=(0, 1, 2, 3),  # 输出各阶段特征用于特征融合[64, 128, 256, 512]
        frozen_stages=-1,  # 根据需要冻结网络的某些层
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')
        ),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=0.33,
        widen_factor=0.5,  # 128, 256, 512, 1024  [512, 1024, 2048, 4096]
        in_channels=[128,256,512, 1024],
        out_channels=[128,256,512, 1024],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        head_module=dict(
            in_channels=[128,256,512, 1024],
            widen_factor=0.5,
            featmap_strides=[4,8, 16, 32]),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=[4,8, 16, 32]),
    )
))