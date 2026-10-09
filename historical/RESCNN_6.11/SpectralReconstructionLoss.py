# -*- coding: utf-8 -*-
"""
SpectralReconstructionLoss.py
适配 190 点光谱重建的组合损失。

包含：
- Weighted MSE：逐波长加权，关注峰形/关键波段。
- first-diff / second-diff：约束曲线形状与平滑性。
- cosine / SAM：约束整体光谱形状方向。
- top-k error：关注最难预测的一部分样本，避免验证集均值好看但个别样本崩掉。
- DC：约束整体基线均值。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def first_diff(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:] - x[:, :-1]


def second_diff(x: torch.Tensor) -> torch.Tensor:
    return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]


def spectral_angle_mapper(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """返回每个样本的 SAM，单位 rad，shape=(B,)。"""
    pred_n = F.normalize(pred, p=2, dim=1, eps=eps)
    target_n = F.normalize(target, p=2, dim=1, eps=eps)
    cos = (pred_n * target_n).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos)


class SpectralReconstructionLoss(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        lambda_diff1: float = 0.30,
        lambda_diff2: float = 0.05,
        lambda_cos: float = 0.10,
        lambda_sam: float = 0.10,
        lambda_topk: float = 0.20,
        lambda_dc: float = 0.05,
        topk_ratio: float = 0.10,
    ):
        super().__init__()
        if weight is None:
            weight = torch.ones(190, dtype=torch.float32)
        self.register_buffer("weight", weight.float())
        self.lambda_diff1 = float(lambda_diff1)
        self.lambda_diff2 = float(lambda_diff2)
        self.lambda_cos = float(lambda_cos)
        self.lambda_sam = float(lambda_sam)
        self.lambda_topk = float(lambda_topk)
        self.lambda_dc = float(lambda_dc)
        self.topk_ratio = float(topk_ratio)

    @torch.no_grad()
    def update_weight(self, new_weight: torch.Tensor):
        assert new_weight.shape[-1] == self.weight.shape[-1]
        self.weight.data.copy_(new_weight.to(self.weight.device).float())

    def forward_components(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        if pred.shape[1] != self.weight.numel():
            raise ValueError(f"loss weight dim={self.weight.numel()} but pred dim={pred.shape[1]}")

        w = self.weight.view(1, -1)
        err = pred - target
        mse = (w * err.pow(2)).mean()

        d1 = F.l1_loss(first_diff(pred), first_diff(target))
        d2 = F.l1_loss(second_diff(pred), second_diff(target)) if pred.shape[1] >= 3 else pred.new_tensor(0.0)

        cos = 1.0 - F.cosine_similarity(pred, target, dim=1).mean()
        sam = spectral_angle_mapper(pred, target).mean()
        dc = (pred.mean(dim=1) - target.mean(dim=1)).pow(2).mean()

        sample_mse = err.pow(2).mean(dim=1)
        k = max(1, int(sample_mse.numel() * self.topk_ratio))
        topk = torch.topk(sample_mse, k=k, largest=True).values.mean()

        total = (
            mse
            + self.lambda_diff1 * d1
            + self.lambda_diff2 * d2
            + self.lambda_cos * cos
            + self.lambda_sam * sam
            + self.lambda_topk * topk
            + self.lambda_dc * dc
        )
        return {
            "total": total,
            "mse": mse,
            "diff1": d1,
            "diff2": d2,
            "cos": cos,
            "sam": sam,
            "topk": topk,
            "dc": dc,
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.forward_components(pred, target)["total"]
