from mmengine import read_base
with read_base():
    from .._base_.models.dx_faster import * # noqa
    from .._base_.datasets.DX import *  # noqa
    from .._base_.schedules.schedules_100e import *  # noqa
    from .._base_.default_runtime import *  # noqa
model.update(dict(
    backbone=dict(
        depth=18,
        type='ResNet',
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
    ),
    neck=dict(in_channels=[64, 128, 256, 512]),

))
