import torch
import torch.nn as nn


class SpectralReconstructionLoss(nn.Module):
    """
    - WeightedMSE：逐波长加权提升峰形/关键段精度
    - DC：基线均值约束（抑制整体漂移）
    - TV：一阶差分平滑（稳住曲线形状）
    - base_coef L2（可选）：当模型为“双头”时，轻度惩罚基线系数，鼓励“能不用就少用”
    """
    def __init__(self, weight: torch.Tensor,
                 lambda_dc: float = 0.10,
                 lambda_tv: float = 0.02,
                 lambda_base: float = 1e-4):
        super().__init__()
        assert weight is not None, "weight 不能为空（使用 build_peak_weight 构建）"
        self.register_buffer("weight", weight.float())   # [76]
        self.lambda_dc   = float(lambda_dc)
        self.lambda_tv   = float(lambda_tv)
        self.lambda_base = float(lambda_base)

    @torch.no_grad()
    def update_weight(self, new_weight: torch.Tensor):
        """训练中动态更新逐波长权重。"""
        assert new_weight.shape[-1] == self.weight.shape[-1]
        self.weight.data.copy_(new_weight.to(self.weight.device).float())

    def forward(self, pred: torch.Tensor, target: torch.Tensor, base_coef: torch.Tensor | None = None) -> torch.Tensor:
        # pred/target: [B, 76]
        mse = (self.weight * (pred - target) ** 2).mean()
        dc  = ((pred.mean(dim=1) - target.mean(dim=1)) ** 2).mean()
        tv  = ((pred[:, 1:] - pred[:, :-1]) ** 2).mean()
        loss = mse + self.lambda_dc * dc + self.lambda_tv * tv
        if base_coef is not None:
            loss = loss + self.lambda_base * (base_coef ** 2).mean()
        return loss