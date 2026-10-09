# -*- coding: utf-8 -*-
"""
model.py
面向 (3,6,12,9) 多色块随机采样 RGB 数据的光谱重建网络。

设计要点：
1. 每个色块有 K=9 个 RGB 采样点，采样点顺序本身没有物理意义。
   因此先用共享 MLP 编码每个 RGB 采样点，再在 K 维上做 mean/max 聚合，保证对采样点顺序不敏感。
2. 每个色块作为一个 token，加入其 row/col 位置编码。
3. 使用 Transformer Encoder 建模不同色块之间的联合编码关系。
4. 使用 attention pooling 得到全局表示，输出 190 维光谱。
5. 通过 block_indices 支持 72/60/48/36 等不同色块数量的消融实验。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
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
    """
    输入支持两类形状：
      1) Dataset 输出：       x = (B, 3, N, K)
      2) 原始网格直接输入：    x = (B, 3, R, C, K)

    其中：
      N = 色块数量，可为 72/60/48/36 等；
      K = 每个色块采样点数，当前默认为 9；
      输出为 (B, out_dim)，当前 out_dim=190。
    """

    def __init__(
        self,
        out_dim: int = 190,
        grid_rows: int = 6,
        grid_cols: int = 12,
        sample_points: int = 9,
        block_indices: Optional[Sequence[int] | np.ndarray] = None,
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
        self.sample_points = int(sample_points)
        self.embed_dim = int(embed_dim)

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        if block_indices is None:
            self.register_buffer("block_indices", torch.empty(0, dtype=torch.long), persistent=True)
        else:
            idx = torch.as_tensor(np.asarray(block_indices, dtype=np.int64).reshape(-1), dtype=torch.long)
            self.register_buffer("block_indices", idx, persistent=True)

        # 对每个 RGB 采样点共享编码：3 -> D
        self.sample_encoder = MLP(3, embed_dim, embed_dim, dropout=dropout)

        # 每个色块的统计信息：mean_rgb + std_rgb -> D
        self.stats_encoder = MLP(6, embed_dim, embed_dim, dropout=dropout)

        # 色块位置编码：row/col normalized -> D
        self.pos_encoder = MLP(2, embed_dim, embed_dim, dropout=0.0)

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

    def _selected_indices_for_n(self, n_tokens: int, device) -> torch.Tensor:
        """
        给 token 生成位置索引。
        - 如果模型保存了 block_indices 且长度匹配，就用它。
        - 否则默认认为输入是 row-major 的前 N 个色块。
        """
        if self.block_indices.numel() == n_tokens:
            return self.block_indices.to(device)
        return torch.arange(n_tokens, dtype=torch.long, device=device)

    def _pos_embedding(self, n_tokens: int, device) -> torch.Tensor:
        idx = self._selected_indices_for_n(n_tokens, device)
        rows = torch.div(idx, self.grid_cols, rounding_mode="floor").float()
        cols = (idx % self.grid_cols).float()

        r_den = max(self.grid_rows - 1, 1)
        c_den = max(self.grid_cols - 1, 1)
        pos = torch.stack([rows / r_den, cols / c_den], dim=-1)  # (N,2)
        return self.pos_encoder(pos).unsqueeze(0)                # (1,N,D)

    def _to_samples_bnk3(self, x: torch.Tensor) -> torch.Tensor:
        """统一为 (B,N,K,3)。"""
        if x.dim() == 5:
            # (B,3,R,C,K) -> (B,3,N,K)
            if x.size(1) != 3:
                raise ValueError(f"5D input must be (B,3,R,C,K), got {tuple(x.shape)}")
            b, ch, r, c, k = x.shape
            x = x.reshape(b, ch, r * c, k)
            if self.block_indices.numel() > 0:
                x = x[:, :, self.block_indices.to(x.device), :]
            x = x.permute(0, 2, 3, 1).contiguous()  # (B,N,K,3)
            return x

        if x.dim() == 4:
            # Dataset 默认输出：(B,3,N,K)
            if x.size(1) == 3:
                return x.permute(0, 2, 3, 1).contiguous()  # (B,N,K,3)
            # 兼容用户手工输入：(B,N,K,3)
            if x.size(-1) == 3:
                return x.contiguous()
            raise ValueError(f"4D input must be (B,3,N,K) or (B,N,K,3), got {tuple(x.shape)}")

        raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        samples = self._to_samples_bnk3(x).float()  # (B,N,K,3)
        b, n, k, ch = samples.shape
        if ch != 3:
            raise ValueError(f"last channel must be RGB=3, got {ch}")
        if k != self.sample_points:
            # 这里不强制报错也可以，但报错能尽早发现训练/预测不一致。
            raise ValueError(f"sample_points mismatch: model K={self.sample_points}, input K={k}")

        # 点级编码：对 K 个采样点共享 MLP，然后做顺序不敏感聚合。
        point_feat = self.sample_encoder(samples.reshape(b * n * k, 3)).reshape(b, n, k, self.embed_dim)
        feat_mean = point_feat.mean(dim=2)                 # (B,N,D)
        feat_max = point_feat.max(dim=2).values             # (B,N,D)

        # 显式统计项，增强对采样点离散程度/像差的表达能力。
        rgb_mean = samples.mean(dim=2)                      # (B,N,3)
        rgb_std = samples.std(dim=2, unbiased=False)         # (B,N,3)
        stats = torch.cat([rgb_mean, rgb_std], dim=-1)       # (B,N,6)
        feat_stats = self.stats_encoder(stats)               # (B,N,D)

        tokens = feat_mean + 0.5 * feat_max + feat_stats
        tokens = tokens + self._pos_embedding(n, samples.device)

        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        # attention pooling，比简单平均更容易突出关键色块。
        score = self.pool_score(tokens).squeeze(-1)          # (B,N)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)   # (B,N,1)
        pooled = (tokens * weight).sum(dim=1)                # (B,D)

        out = self.out_act(self.head(pooled))                # (B,out_dim)
        if return_dict:
            return {"spectrum": out, "block_attention": weight.squeeze(-1)}
        return out


def build_model(cfg: Optional[dict] = None) -> SpectralTokenNet:
    cfg = cfg or {}
    return SpectralTokenNet(
        out_dim=cfg.get("out_dim", 190),
        grid_rows=cfg.get("grid_rows", 6),
        grid_cols=cfg.get("grid_cols", 12),
        sample_points=cfg.get("sample_points", 9),
        block_indices=cfg.get("block_indices", None),
        embed_dim=cfg.get("embed_dim", 96),
        depth=cfg.get("depth", 3),
        num_heads=cfg.get("num_heads", 4),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        dropout=cfg.get("dropout", 0.10),
        out_activation=cfg.get("out_activation", "linear"),
    )


# 兼容原 train.py / predict.py 中 from model import ResCNN 的写法。
ResCNN = SpectralTokenNet


if __name__ == "__main__":
    model = SpectralTokenNet(out_dim=190, grid_rows=6, grid_cols=12, sample_points=9)
    x_grid = torch.randn(2, 3, 6, 12, 9)
    y = model(x_grid)
    print("grid input ->", y.shape)

    idx36 = list(range(36))
    model36 = SpectralTokenNet(out_dim=190, grid_rows=6, grid_cols=12, sample_points=9, block_indices=idx36)
    x_tokens = torch.randn(2, 3, 36, 9)
    y36 = model36(x_tokens)
    print("token input ->", y36.shape)
