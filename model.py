# -*- coding: utf-8 -*-
"""
model.py
适配多光源 token 输入：
  Dataset 输出：x = (B, 3, L*N, K)   L=8, N=72, K=9
  也兼容：      x = (B, 3, L, N, K)

仍然使用：
- 采样点共享 MLP + K维聚合（对采样顺序不敏感）
- Transformer encoder 建模 token 关系
- attention pooling 输出 190 光谱
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SpectralTokenNet(nn.Module):
    def __init__(
        self,
        out_dim: int = 190,
        grid_rows: int = 6,
        grid_cols: int = 12,
        lights: int = 8,
        sample_points: int = 9,
        embed_dim: int = 96,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.10,
        out_activation: str = "linear",
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.lights = int(lights)
        self.sample_points = int(sample_points)
        self.embed_dim = int(embed_dim)

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        # 点级编码：3 -> D
        self.sample_encoder = MLP(3, embed_dim, embed_dim, dropout=dropout)

        # 统计项：mean/std -> D
        self.stats_encoder = MLP(6, embed_dim, embed_dim, dropout=dropout)

        # 位置编码：row/col/light -> D
        self.pos_encoder = MLP(3, embed_dim, embed_dim, dropout=0.0)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

        self.pool_score = nn.Linear(embed_dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, out_dim),
        )

        oa = str(out_activation).lower()
        if oa == "linear":
            self.out_act = nn.Identity()
        elif oa == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif oa == "softplus":
            self.out_act = nn.Softplus(beta=1.0, threshold=20.0)
        else:
            raise ValueError("out_activation must be 'linear' | 'relu' | 'softplus'")

    def _to_samples_bnk3(self, x: torch.Tensor) -> torch.Tensor:
        """
        统一为 (B, T, K, 3)
        支持：
          (B,3,T,K)
          (B,3,L,N,K)
          (B,T,K,3)
        """
        if x.dim() == 5:
            # (B,3,L,N,K) -> (B,3,T,K)
            if x.size(1) != 3:
                raise ValueError(f"5D input must be (B,3,L,N,K), got {tuple(x.shape)}")
            b, ch, L, N, K = x.shape
            x = x.reshape(b, ch, L * N, K)
            return x.permute(0, 2, 3, 1).contiguous()  # (B,T,K,3)

        if x.dim() == 4:
            if x.size(1) == 3:
                # (B,3,T,K)
                return x.permute(0, 2, 3, 1).contiguous()
            if x.size(-1) == 3:
                # (B,T,K,3)
                return x.contiguous()
            raise ValueError(f"4D input must be (B,3,T,K) or (B,T,K,3), got {tuple(x.shape)}")

        raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

    def _pos_embedding(self, T: int, device) -> torch.Tensor:
        """
        T = L*Ntokens
        位置：row/col/light
        默认 token 排列顺序为：
          t = light * N + block
        """
        N = self.grid_rows * self.grid_cols
        L = self.lights
        if T != L * N:
            # 允许你未来做 token 子集，但这里先给出明确报错，避免训练/预测形状不一致
            raise ValueError(f"token count mismatch: expected T={L*N}, got T={T}")

        idx = torch.arange(T, device=device, dtype=torch.long)
        light = torch.div(idx, N, rounding_mode="floor").float()
        block = idx % N
        row = torch.div(block, self.grid_cols, rounding_mode="floor").float()
        col = (block % self.grid_cols).float()

        r_den = max(self.grid_rows - 1, 1)
        c_den = max(self.grid_cols - 1, 1)
        l_den = max(self.lights - 1, 1)

        pos = torch.stack([row / r_den, col / c_den, light / l_den], dim=-1)  # (T,3)
        return self.pos_encoder(pos).unsqueeze(0)  # (1,T,D)

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        samples = self._to_samples_bnk3(x).float()  # (B,T,K,3)
        b, T, K, ch = samples.shape
        if ch != 3:
            raise ValueError(f"last channel must be 3, got {ch}")
        if K != self.sample_points:
            raise ValueError(f"sample_points mismatch: model K={self.sample_points}, input K={K}")

        # 点级编码
        point_feat = self.sample_encoder(samples.reshape(b * T * K, 3)).reshape(b, T, K, self.embed_dim)
        feat_mean = point_feat.mean(dim=2)
        feat_max = point_feat.max(dim=2).values

        rgb_mean = samples.mean(dim=2)
        rgb_std = samples.std(dim=2, unbiased=False)
        stats = torch.cat([rgb_mean, rgb_std], dim=-1)
        feat_stats = self.stats_encoder(stats)

        tokens = feat_mean + 0.5 * feat_max + feat_stats
        tokens = tokens + self._pos_embedding(T, samples.device)

        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        score = self.pool_score(tokens).squeeze(-1)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        pooled = (tokens * weight).sum(dim=1)

        out = self.out_act(self.head(pooled))
        if return_dict:
            return {"spectrum": out, "token_attention": weight.squeeze(-1)}
        return out


# 兼容旧 import
ResCNN = SpectralTokenNet