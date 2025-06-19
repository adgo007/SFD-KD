from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from ....registry import MODELS
from .base_connector import BaseConnector


def align(x, c, h, w):
    x.channels = x.shape[1]
    adjust_channels = nn.Sequential(
        nn.Conv2d(x.channels, c, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True))
    # 2. 调整学生特征的空间尺寸以匹配教师特征
    upsample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)

    x = adjust_channels(x)
    x = upsample(x)
    return x


@MODELS.register_module()
class FourierStudentConnector(BaseConnector):
    def __init__(self, teacher_c, teacher_h, teacher_w, init_cfg=None):
        super().__init__(init_cfg)
        self.teacher_c = teacher_c
        self.teacher_h = teacher_h
        self.teacher_w = teacher_w

    def forward_train(self, x):
        # # 先将学生特征对齐教师模型
        # # 1. 调整学生特征的通道数
        # x.channels = x.shape[1]
        # adjust_channels = nn.Sequential(
        #     nn.Conv2d(x.channels, self.teacher_c, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(self.teacher_c),
        #     nn.ReLU(inplace=True))
        # # 2. 调整学生特征的空间尺寸以匹配教师特征
        # upsample = nn.Upsample(size=(self.teacher_h, self.teacher_w), mode='bilinear', align_corners=True)
        #
        # x = adjust_channels(x)
        # x = upsample(x)
        x = align(x, self.teacher_c, self.teacher_h, self.teacher_w)
        # 进行FFT
        fft_result = torch.fft.fft2(x)

        # 创建一个低通滤波器掩码
        _, c, h, w = x.shape
        mask = torch.zeros_like(fft_result)
        cutoff_frequency_h = h // 4
        cutoff_frequency_w = w // 4
        mask[:, :, :cutoff_frequency_h, :cutoff_frequency_w] = 1
        mask[:, :, -cutoff_frequency_h:, -cutoff_frequency_w:] = 1

        # 应用掩码
        fft_result *= mask

        # 进行IFFT
        ifft_result = torch.fft.ifft2(fft_result)
        return ifft_result.real  # 取实部作为最终输出


@MODELS.register_module()
class SVDTeacherConnector(BaseConnector):
    def __init__(self, teacher_c, teacher_h, teacher_w, init_cfg=None):
        super().__init__(init_cfg)
        self.teacher_c = teacher_c
        self.teacher_h = teacher_h
        self.teacher_w = teacher_w

    def forward_train(self, x):
        x = align(x, self.teacher_c, self.teacher_h, self.teacher_w)
        # 基向量的数量(bases = H * W)
        num_bases = x.shape[2] * x.shape[3]
        # 将输入特征展平为两维(b,c,h,w-->b,c,h*W)
        original_shape = x.shape
        x = x.view(x.shape[0], x.shape[1], -1)

        # 计算SVD
        U, S, V = torch.linalg.svd(x, full_matrices=False)

        # 选择前num_bases个基向量
        U = U[:, :, :num_bases]

        # 使用基向量进行投影
        projected = torch.bmm(U, U.transpose(1, 2))
        x_projected = torch.bmm(projected, x)

        # 恢复原始的形状
        x_projected = x_projected.view(*original_shape)
        return x_projected
