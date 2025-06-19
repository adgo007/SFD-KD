from mmengine import read_base
with read_base():
    from .._base_.models.dx_faster import * # noqa
    from .._base_.datasets.DX import *  # noqa
    from .._base_.schedules.schedules_100e import *  # noqa
    from .._base_.default_runtime import *  # noqa

model.update(dict(
    backbone=dict(
        type='EfficientNet',
        arch='es',  # 使用 EfficientNet-EdgeTPU Small
        out_indices=(2, 3, 4, 5),  # 输出索引，对应于EfficientNet-ES中不同的阶段
        frozen_stages=0,  # 如果要冻结某些阶段则设置该参数
        norm_cfg=dict(type='BN', requires_grad=True),  # 使用批量归一化
        norm_eval=False,  # 训练时是否冻结BN层
        with_cp=False  # 是否使用 checkpoint (梯度检查点)
    ),
    neck=dict(
        type='FPN',
        in_channels=[32, 48, 144, 192],  # 根据EfficientNet-ES的特定层输出通道数设置
        out_channels=256,
        num_outs=5
    ),
))