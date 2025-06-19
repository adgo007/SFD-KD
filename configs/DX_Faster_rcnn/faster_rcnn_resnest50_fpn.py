from mmengine import read_base
with read_base():
    from .._base_.models.dx_faster import * # noqa
    from .._base_.datasets.DX import *  # noqa
    from .._base_.schedules.schedules_100e import *  # noqa
    from .._base_.default_runtime import *  # noqa
model.update(dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNeSt',
        depth=50,  # 可以选择 50, 101, 152 等
        groups=1,
        base_width=4,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        out_indices=(0, 1, 2, 3),  # 根据需要选择输出层
        frozen_stages=-1,  # 如果需要冻结某些阶段，设置这个参数
        norm_cfg=dict(type='BN', requires_grad=True)  # 使用批量归一化
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # 这需要根据选择的 ResNeSt 配置
        out_channels=256,
        num_outs=5
    ),
))