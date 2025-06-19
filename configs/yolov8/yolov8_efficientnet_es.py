from mmengine import read_base

with read_base():
    from .._base_.models.dx_yolo import *  # noqa
    from .._base_.datasets.dxyolo import *  # noqa
    from .._base_.schedules.schedule_yolo_100e import *  # noqa
    from .._base_.yolo_run import *  # noqa

env_cfg = dict(cudnn_benchmark=True)
model.update(dict(
    backbone=dict(
        type='mmdet.EfficientNet',
        arch='es',  # 使用 EfficientNet-EdgeTPU Small
        out_indices=(2, 3, 4, 5),  # 输出索引，对应于EfficientNet-ES中不同的阶段
        frozen_stages=0,  # 如果要冻结某些阶段则设置该参数
        norm_cfg=dict(type='BN', requires_grad=True),  # 使用批量归一化
        norm_eval=False,  # 训练时是否冻结BN层
        with_cp=False  # 是否使用 checkpoint (梯度检查点)
    ),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=0.33,
        widen_factor=0.5,  # 128, 256, 512, 1024  [512, 1024, 2048, 4096]
        in_channels=[64, 96, 288, 384],
        out_channels=[64, 96, 288, 384],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        head_module=dict(
            in_channels=[64, 96, 288, 384],
            widen_factor=0.5,
            featmap_strides=[4, 8, 16, 32]),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=[4, 8, 16, 32]),
    )
))
