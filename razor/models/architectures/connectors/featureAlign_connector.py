import torch
import torch.nn as nn
from ....registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class FeatureAlignConnector(BaseConnector):
    def __init__(self, teacher_c, teacher_h, teacher_w, init_cfg=None):
        super().__init__(init_cfg)
        self.teacher_c = teacher_c
        self.teacher_h = teacher_h
        self.teacher_w = teacher_w

    def forward_train(self, x):
        """
        对学生特征进行通道和空间尺寸对齐，使其与教师特征匹配。

        Args:
            x: student_features (torch.Tensor): 学生模型的特征 (batch, student_channels, H_s, W_s)

        Returns:
            torch.Tensor: 对齐后的学生特征，空间尺寸和通道数与教师特征相同
        """
        # # 1. 调整学生特征的通道数
        # s = x.shape
        x_channel = x.shape[1]  # 学生通道数
        adjust_channels = nn.Sequential(
            nn.Conv2d(x_channel, self.teacher_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.teacher_c),
            nn.ReLU(inplace=True))

        # # 2. 获取教师特征的空间尺寸
        # teacher_height, teacher_width = teacher_features.shape[2], teacher_features.shape[3]

        # # 3. 调整学生特征的空间尺寸以匹配教师特征
        upsample = nn.Upsample(size=(self.teacher_h, self.teacher_w), mode='bilinear', align_corners=True)
        x = adjust_channels(x)
        x = upsample(x)

        return x

