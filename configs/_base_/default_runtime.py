from mmengine.visualization.vis_backend import LocalVisBackend, WandbVisBackend
from mmdet.visualization.local_visualizer import DetLocalVisualizer
from mmengine.runner.log_processor import LogProcessor

# from seg.engine.hooks import MyCheckpointHook
# from seg.engine.hooks.logger_hook import MyLoggerHook


env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type=LocalVisBackend),
    # dict(
    #     type=WandbVisBackend,
    #     init_kwargs=dict(
    #         project='mmdetection', name='exp'),
    #     define_metric_cfg=dict(mAP_50='max'))
]

visualizer = dict(
    _scope_='mmdet',
    type=DetLocalVisualizer, vis_backends=vis_backends, name='visualizer')

log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
