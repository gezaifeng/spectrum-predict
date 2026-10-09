# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def first_diff(x: torch.Tensor) -> torch.Tensor:
    """
    一阶差分
    x: [B, L]
    return: [B, L-1]
    """
    return x[:, 1:] - x[:, :-1]


def second_diff(x: torch.Tensor) -> torch.Tensor:
    """
    二阶差分
    x: [B, L]
    return: [B, L-2]
    """
    d1 = first_diff(x)
    return d1[:, 1:] - d1[:, :-1]


def spectral_angle_mapper(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算每个样本的光谱角（弧度）
    pred/target: [B, L]
    return: [B]
    """
    pred_norm = torch.norm(pred, dim=1).clamp_min(eps)
    target_norm = torch.norm(target, dim=1).clamp_min(eps)
    cos_sim = (pred * target).sum(dim=1) / (pred_norm * target_norm)
    cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
    return torch.acos(cos_sim)


class SpectralReconstructionLoss(nn.Module):
    """
    综合损失：
    1) weighted mse
    2) dc loss
    3) diff1 loss
    4) diff2 loss
    5) cosine loss
    6) sam loss
    7) top-k worst sample loss
    """

    def __init__(
        self,
        weight: torch.Tensor,
        lambda_dc: float = 0.1,
        lambda_diff1: float = 0.5,
        lambda_diff2: float = 0.2,
        lambda_cos: float = 0.2,
        lambda_sam: float = 0.2,
        lambda_topk: float = 0.3,
        topk_ratio: float = 0.1,
    ):
        super().__init__()
        assert weight is not None, "weight 不能为空"
        self.register_buffer("weight", weight.float())

        self.lambda_dc = float(lambda_dc)
        self.lambda_diff1 = float(lambda_diff1)
        self.lambda_diff2 = float(lambda_diff2)
        self.lambda_cos = float(lambda_cos)
        self.lambda_sam = float(lambda_sam)
        self.lambda_topk = float(lambda_topk)
        self.topk_ratio = float(topk_ratio)

    @torch.no_grad()
    def update_weight(self, new_weight: torch.Tensor):
        assert new_weight.shape[-1] == self.weight.shape[-1]
        self.weight.data.copy_(new_weight.to(self.weight.device).float())

    def forward_components(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        # 1) weighted mse
        point_err = self.weight.unsqueeze(0) * (pred - target) ** 2
        loss_mse = point_err.mean()

        # 2) dc
        loss_dc = ((pred.mean(dim=1) - target.mean(dim=1)) ** 2).mean()

        # 3) diff1
        pred_d1 = first_diff(pred)
        targ_d1 = first_diff(target)
        loss_diff1 = F.mse_loss(pred_d1, targ_d1)

        # 4) diff2
        pred_d2 = second_diff(pred)
        targ_d2 = second_diff(target)
        loss_diff2 = F.mse_loss(pred_d2, targ_d2)

        # 5) cosine
        pred_n = F.normalize(pred, p=2, dim=1, eps=1e-8)
        targ_n = F.normalize(target, p=2, dim=1, eps=1e-8)
        loss_cos = (1.0 - (pred_n * targ_n).sum(dim=1)).mean()

        # 6) sam
        loss_sam = spectral_angle_mapper(pred, target).mean()

        # 7) top-k
        sample_mse = point_err.mean(dim=1)  # [B]
        k = max(1, int(sample_mse.numel() * self.topk_ratio))
        loss_topk = torch.topk(sample_mse, k=k, largest=True).values.mean()

        total = (
            loss_mse
            + self.lambda_dc * loss_dc
            + self.lambda_diff1 * loss_diff1
            + self.lambda_diff2 * loss_diff2
            + self.lambda_cos * loss_cos
            + self.lambda_sam * loss_sam
            + self.lambda_topk * loss_topk
        )

        # 注意：同时返回 "loss" 和 "total"
        # 这样兼容你 train.py 里两种可能的写法
        return {
            "loss": total,
            "total": total,
            "mse": loss_mse,
            "dc": loss_dc,
            "diff1": loss_diff1,
            "diff2": loss_diff2,
            "cos": loss_cos,
            "sam": loss_sam,
            "topk": loss_topk,
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        comps = self.forward_components(pred, target)
        return comps["loss"]