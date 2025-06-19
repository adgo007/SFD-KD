from mmengine import read_base
with read_base():
    from .._base_.models.dx_faster import * # noqa
    from .._base_.datasets.DX import *  # noqa
    from .._base_.schedules.schedules_100e import *  # noqa
    from .._base_.default_runtime import *  # noqa
model.update(dict(
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        in_channels=3,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),  # 根据实际需求选择输出层
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-6),
        with_cp=False,
        pretrained=None
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],  # 应与Swin Transformer的输出维度匹配
        out_channels=256,
        num_outs=5
    ),
)
)