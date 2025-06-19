from mmengine import read_base
with read_base():
    from .._base_.models.frac_faster import * # noqa
    from .._base_.datasets.Frac import *  # noqa
    from .._base_.schedules.schedules_100e import *  # noqa
    from .._base_.default_runtime import *  # noqa

model.update(dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',  # 使用批量归一化
        # dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(False,True,True,True)
    )))
