from mmengine import read_base
from razor.models.losses.FOSE import FOSE

from razor.models.losses.MOSE import MOSE

with read_base():
    from ..._base_.datasets.DX import *  # noqa
    from ..._base_.schedules.schedules_100e import *  # noqa
    from ..._base_.default_runtime import *  # noqa

    from ...DX_Faster_rcnn.faster_rcnn_swintransformer_fpn import model as teacher  # noqa
    from ...DX_Faster_rcnn.faster_rcnn_efficientnetes_fpn import model as student  # noqa
teacher_ckpt = 'faster_swin_DX.pth'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=student,
    teacher=teacher,
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_pre0=dict(type=MOSE, lambdas=0.5),
            loss_post0=dict(type=FOSE, lambdas=0.5),
            loss_pre1=dict(type=MOSE, lambdas=0.5),
            loss_post1=dict(type=FOSE, lambdas=0.5),
            loss_pre2=dict(type=MOSE, lambdas=0.5),
            loss_post2=dict(type=FOSE, lambdas=0.5),
            loss_pre3=dict(type=MOSE, lambdas=0.5),
            loss_post3=dict(type=FOSE, lambdas=0.5),
            loss_pre4=dict(type=MOSE, lambdas=0.5),
            loss_post4=dict(type=FOSE, lambdas=0.5),
        ),
        loss_forward_mappings=dict(
            loss_pre0=dict(s_input=dict(from_student=True, recorder='fpn', data_idx=0, ),
                           t_input=dict(from_student=False, recorder='fpn', data_idx=0, )),
            loss_post0=dict(s_input=dict(from_student=True, recorder='fpn', data_idx=0, ),
                            t_input=dict(from_student=False, recorder='fpn', data_idx=0, )),
            loss_pre1=dict(s_input=dict(from_student=True, recorder='fpn', data_idx=1, ),
                           t_input=dict(from_student=False, recorder='fpn', data_idx=1, )),
            loss_post1=dict(s_input=dict(from_student=True, recorder='fpn', data_idx=1, ),
                            t_input=dict(from_student=False, recorder='fpn', data_idx=1, )),
            loss_pre2=dict(s_input=dict(from_student=True, recorder='fpn', data_idx=2, ),
                           t_input=dict(from_student=False, recorder='fpn', data_idx=2, )),
            loss_post2=dict(s_input=dict(from_student=True, recorder='fpn', data_idx=2, ),
                            t_input=dict(from_student=False, recorder='fpn', data_idx=2, )),
            loss_pre3=dict(s_input=dict(from_student=True, recorder='fpn', data_idx=3, ),
                           t_input=dict(from_student=False, recorder='fpn', data_idx=3, )),
            loss_post3=dict(s_input=dict(from_student=True, recorder='fpn', data_idx=3, ),
                            t_input=dict(from_student=False, recorder='fpn', data_idx=3, )),
            loss_pre4=dict(s_input=dict(from_student=True, recorder='fpn', data_idx=4, ),
                           t_input=dict(from_student=False, recorder='fpn', data_idx=4, )),
            loss_post4=dict(s_input=dict(from_student=True, recorder='fpn', data_idx=4, ),
                            t_input=dict(from_student=False, recorder='fpn', data_idx=4, )),

        )))

find_unused_parameters = True

val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
