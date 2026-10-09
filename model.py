# -*- coding: utf-8 -*-
# Single-head spectrum predictor baseline
# Author: YourName
# ---------------------------------------
# 用途：直接从 RGB 网格 (B,3,H,W) 回归光谱向量 (B,D)
# 特点：轻量、稳定、易于与现有训练脚本对接做消融

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- 基础组件 ---------
class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, c, hidden_scale=2, drop=0.0):
        super().__init__()
        h = int(c * hidden_scale)
        self.conv1 = ConvBNAct(c, h, k=3, s=1)
        self.drop1 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.conv2 = ConvBNAct(h, c, k=3, s=1, act=False)
        self.act   = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.drop1(self.conv1(x))
        out = self.conv2(out)
        out = self.act(out + x)
        return out


# --------- 单头光谱预测器 ---------
class SingleHeadSpectralNet(nn.Module):
    """
    直接回归光谱向量的单头基线模型。
    输入:  x ∈ R^{B×3×H×W}（色卡RGB网格）
    输出:  ŝ ∈ R^{B×D}（线性 / ReLU / Softplus 可选）
    """
    def __init__(
        self,
        out_dim: int = 76,          # D：光谱维度
        width: int = 64,            # 主干通道宽度
        depth: int = 4,             # 残差块数量
        drop: float = 0.0,          # 残差块内dropout
        out_activation: str = "linear",  # "linear" | "relu" | "softplus"
        mlp_hidden: int = 256,      # MLP隐藏维度
        global_pool: str = "avg",   # "avg" | "max" | "gem"
        gem_p: float = 3.0          # GeM pooling 的 p
    ):
        super().__init__()
        self.out_dim = out_dim
        self.global_pool = global_pool.lower()
        self.gem_p = nn.Parameter(torch.tensor(gem_p), requires_grad=True) if self.global_pool == "gem" else None

        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(3, width, k=3, s=1),
            ConvBNAct(width, width, k=3, s=1),
        )

        # Residual trunk
        blocks = []
        for _ in range(depth):
            blocks.append(ResidualBlock(width, hidden_scale=2, drop=drop))
        self.trunk = nn.Sequential(*blocks)

        # Head: GAP/GeM -> MLP -> out_dim
        self.head_pre = nn.Sequential(
            nn.Linear(width, mlp_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1 if drop > 0 else 0.0),
            nn.Linear(mlp_hidden, out_dim)
        )

        if out_activation.lower() == "linear":
            self.out_act = nn.Identity()
        elif out_activation.lower() == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif out_activation.lower() == "softplus":
            # Softplus 保证非负（适合在吸光度或强度域回归时做对比）
            self.out_act = nn.Softplus(beta=1.0, threshold=20.0)
        else:
            raise ValueError("out_activation must be in ['linear','relu','softplus']")

    def _pool(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H, W) -> (B, C)
        if self.global_pool == "avg":
            return F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
        elif self.global_pool == "max":
            return F.adaptive_max_pool2d(feat, 1).squeeze(-1).squeeze(-1)
        elif self.global_pool == "gem":
            # Generalized Mean Pooling (learnable p)
            x = torch.clamp(feat, min=1e-6)
            p = torch.clamp(self.gem_p, min=1.0, max=6.0)
            x = torch.pow(x, p.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            x = F.adaptive_avg_pool2d(x, 1)
            x = torch.pow(x, 1.0 / p.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            return x.squeeze(-1).squeeze(-1)
        else:
            raise ValueError("global_pool must be in ['avg','max','gem']")

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        feat = self.stem(x)
        feat = self.trunk(feat)
        g    = self._pool(feat)
        out  = self.head_pre(g)
        out  = self.out_act(out)
        # 兼容旧接口：如果传了 return_dict=True，就以字典形式返回
        if return_dict:
            return {"spectrum": out}
        return out


# --------- 方便与现有脚本对接的构建函数（可选）---------
def build_single_head_model(cfg=None):
    """
    可在 train.py 里统一调用：
        from model_single_head import build_single_head_model
        model = build_single_head_model(...)
    """
    if cfg is None:
        return SingleHeadSpectralNet()
    # 兼容常见配置字典
    return SingleHeadSpectralNet(
        out_dim       = getattr(cfg, "out_dim", 76) if not isinstance(cfg, dict) else cfg.get("out_dim", 76),
        width         = getattr(cfg, "width", 64)    if not isinstance(cfg, dict) else cfg.get("width", 64),
        depth         = getattr(cfg, "depth", 4)     if not isinstance(cfg, dict) else cfg.get("depth", 4),
        drop          = getattr(cfg, "drop", 0.0)    if not isinstance(cfg, dict) else cfg.get("drop", 0.0),
        out_activation= getattr(cfg, "out_activation", "linear") if not isinstance(cfg, dict) else cfg.get("out_activation", "linear"),
        mlp_hidden    = getattr(cfg, "mlp_hidden", 256) if not isinstance(cfg, dict) else cfg.get("mlp_hidden", 256),
        global_pool   = getattr(cfg, "global_pool", "avg") if not isinstance(cfg, dict) else cfg.get("global_pool", "avg"),
        gem_p         = getattr(cfg, "gem_p", 3.0)   if not isinstance(cfg, dict) else cfg.get("gem_p", 3.0),
    )


# --------- 简短使用示例（注释掉即可）---------
if __name__ == "__main__":
    B, C, H, W = 2, 3, 4, 6     # 例如 4×6 色卡网格
    D = 76
    x = torch.randn(B, C, H, W)
    model = SingleHeadSpectralNet(out_dim=D, width=64, depth=4, out_activation="linear")
    y = model(x)
    print(y.shape)  # -> torch.Size([2, 76])
