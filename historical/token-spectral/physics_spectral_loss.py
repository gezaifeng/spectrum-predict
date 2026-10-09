# -*- coding: utf-8 -*-
"""
更适合吸光度光谱回归的损失函数：
- 点对点误差：Weighted MSE
- 基线均值约束：DC
- 一阶平滑：TV
- 二阶曲率约束：Curvature
- 形状约束：一阶导数匹配
- 光谱角约束：SAM
- 非负软约束：negative penalty（可选）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsSpectralLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        lambda_dc: float = 0.05,
        lambda_tv: float = 0.01,
        lambda_curv: float = 0.01,
        lambda_deriv: float = 0.05,
        lambda_sam: float = 0.02,
        lambda_nonneg: float = 0.0,
    ):
        super().__init__()
        assert weight is not None
        self.register_buffer("weight", weight.float())
        self.lambda_dc = float(lambda_dc)
        self.lambda_tv = float(lambda_tv)
        self.lambda_curv = float(lambda_curv)
        self.lambda_deriv = float(lambda_deriv)
        self.lambda_sam = float(lambda_sam)
        self.lambda_nonneg = float(lambda_nonneg)

    @torch.no_grad()
    def update_weight(self, new_weight: torch.Tensor):
        self.weight.data.copy_(new_weight.to(self.weight.device).float())

    def spectral_angle_mapper(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = F.normalize(pred, p=2, dim=1, eps=1e-8)
        targ_n = F.normalize(target, p=2, dim=1, eps=1e-8)
        cos_sim = (pred_n * targ_n).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        angle = torch.acos(cos_sim)
        return angle.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 点误差
        mse = (self.weight * (pred - target) ** 2).mean()

        # 基线均值偏移
        dc = ((pred.mean(dim=1) - target.mean(dim=1)) ** 2).mean()

        # 一阶平滑
        d1 = pred[:, 1:] - pred[:, :-1]
        tv = (d1 ** 2).mean()

        # 二阶曲率
        d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        curv = (d2 ** 2).mean()

        # 导数匹配：更强调峰形和边沿位置
        gt_d1 = target[:, 1:] - target[:, :-1]
        deriv = F.l1_loss(d1, gt_d1)

        # 谱角：更看整体形状一致性
        sam = self.spectral_angle_mapper(pred, target)

        # 非负软惩罚（若你使用 linear 输出且真值基本非负时可打开）
        nonneg = F.relu(-pred).mean()

        loss = (
            mse
            + self.lambda_dc * dc
            + self.lambda_tv * tv
            + self.lambda_curv * curv
            + self.lambda_deriv * deriv
            + self.lambda_sam * sam
            + self.lambda_nonneg * nonneg
        )
        return loss
