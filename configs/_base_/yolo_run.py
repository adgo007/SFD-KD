from mmengine.visualization.vis_backend import LocalVisBackend, WandbVisBackend
from mmdet.visualization.local_visualizer import DetLocalVisualizer
from mmengine.runner.log_processor import LogProcessor

default_scope = 'mmyolo'
randomness = dict(seed=323)
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type=LocalVisBackend),

]

visualizer = dict(
    type=DetLocalVisualizer, vis_backends=vis_backends, name='visualizer')

log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
