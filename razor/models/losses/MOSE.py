# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...registry import MODELS
from .utils import weighted_loss


@weighted_loss
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """A Wrapper of MSE loss.
    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: loss Tensor
    """
    return F.mse_loss(pred, target, reduction='none')


@MODELS.register_module()
class MOSE(nn.Module):
    """MSELoss.
    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 lambdas:float,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.lambdas=lambdas

    def forward(self,
                s_input: Tensor,
                t_input: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        
        s_input_channel = s_input.shape[1]  
        teacher_c = t_input.shape[1]
        adjust_channels = nn.Sequential(
            nn.Conv2d(s_input_channel, teacher_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(teacher_c),
            nn.ReLU(inplace=True)).to("cuda")

      
        teacher_h, teacher_w = t_input.shape[2], t_input.shape[3]

       
        upsample = nn.Upsample(size=(teacher_h, teacher_w), mode='bilinear', align_corners=True)
        s_input = adjust_channels(s_input)
        s_input = upsample(s_input)

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * mse_loss(
            s_input, t_input, weight, reduction=reduction, avg_factor=avg_factor)

        return loss * self.lambdas
