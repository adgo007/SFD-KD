import torch
import torch.nn.functional as F
import torch.nn as nn


class SVDEnhanceAttention(nn.Module):
    """ 在奇异值 S 上应用注意力机制 """

    def __init__(self, feature_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()  # 输出 0~1 之间的权重
        )

    def forward(self, S):
        attn_weights = self.attention(S)  # 计算注意力权重
        return S * attn_weights  # 重新加权 S


class FOSE(nn.Module):
    def __init__(self, lambdas=0.5, k=128):
        super().__init__()
        self.k = k
        self.lambdas = lambdas
        self.attention = None  # 关键修改：初始化注意力模块
        # 初始化 FDP 参数（复数类型）
        self.W = None
        self.b = None

    def forward(self, s_input, t_input):
        # 输入对齐
        teacher_feat = t_input
        student_feat = s_input
        b, c, h, w = teacher_feat.shape
        s_b, s_c, s_h, s_w = student_feat.shape
        # 初始化 FDP 参数（复数类型）
        self.W = nn.Parameter(
            torch.randn(1, s_c, 1, 1, dtype=torch.complex64)
        ).to('cuda')
        self.b = nn.Parameter(
            torch.zeros(1, s_c, 1, 1, dtype=torch.complex64)
        ).to('cuda')

        # **Step 1: FFT 低通滤波 (减少噪声)**
        fft_result = torch.fft.fft2(student_feat)
        fft_result_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
        fft_result_shifted = fft_result_shifted * self.W + self.b
        mask = torch.zeros_like(fft_result)
        cutoff_h, cutoff_w = s_h // 4, s_w // 4
        mask[:, :, s_h // 2 - cutoff_h:s_h // 2 + cutoff_h, s_w // 2 - cutoff_w:s_w // 2 + cutoff_w] = 1
        fft_result_shifted *= mask
        # 逆 FFT 恢复空间域
        fft_result = torch.fft.ifftshift(fft_result_shifted, dim=(-2, -1))
        student_feat = torch.fft.ifft2(fft_result).real  # 取实数部分

        # Step 2: 对教师特征进行SVD分解
        teacher_flat = teacher_feat.view(b, c, -1)  # [b, c, h*w]
        U, S, Vh = torch.linalg.svd(teacher_flat, full_matrices=False)
        self.k = min(self.k, h * w)
        # 截取前k个奇异值
        U = U[..., :self.k]  # [b, c, k]
        S = S[..., :self.k]  # [b, k]
        Vh = Vh[..., :self.k, :]  # [b, k, h*w]

        # Step 3: 应用注意力增强
        self.attention = SVDEnhanceAttention(feature_dim=self.k).to('cuda')
        S_enhanced = self.attention(S)  # [b, k]

        # Step 4: 重建教师特征
        S_diag = torch.diag_embed(S_enhanced)  # [b, k, k]
        reconstructed = U @ S_diag @ Vh  # [b, c, h*w]
        reconstructed = reconstructed.view(b, c, h, w)

        # Step 5: 计算损失
        return F.mse_loss(reconstructed, student_feat) * self.lambdas
