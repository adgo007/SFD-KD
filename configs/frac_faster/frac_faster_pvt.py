
from mmengine import read_base

from mmengine import read_base
with read_base():
    from .._base_.models.frac_faster import * # noqa
    from .._base_.datasets.Frac import *  # noqa
    from .._base_.schedules.schedules_100e import *  # noqa
    from .._base_.default_runtime import *  # noqa

model.update(dict(
    backbone=dict(
        type='PyramidVisionTransformerV2',
        embed_dims=64,
        num_layers=[3, 6, 40, 3],
        mlp_ratios=(4, 4, 4, 4),

    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2,4,8,16],))

))
