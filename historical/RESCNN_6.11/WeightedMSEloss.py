# -*- coding: utf-8 -*-
"""逐波长权重工具，默认适配 190 点光谱。"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _simple_find_peaks(y: np.ndarray, height: float = 0.05) -> np.ndarray:
    """不依赖 scipy 的简单峰值检测：y[i] > y[i-1], y[i] >= y[i+1], y[i] > height。"""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < 3:
        return np.array([], dtype=np.int64)
    mid = y[1:-1]
    mask = (mid > y[:-2]) & (mid >= y[2:]) & (mid > height)
    return np.where(mask)[0].astype(np.int64) + 1


def build_peak_weight(
    dataset,
    output_dim: int = 190,
    peak_width: int = 1,
    alpha: float = 3.0,
    peak_height: float = 0.05,
) -> torch.Tensor:
    """
    根据训练集真值光谱的峰值出现频率构建逐波长权重。
    alpha=3 表示高频峰值波段权重最高约为 3。
    """
    count = np.zeros(int(output_dim), dtype=np.float32)
    for i in range(len(dataset)):
        _, spectrum = dataset[i]
        y = spectrum.detach().cpu().numpy() if torch.is_tensor(spectrum) else np.asarray(spectrum)
        y = y.astype(np.float32).reshape(-1)
        if y.shape[0] != output_dim:
            raise ValueError(f"spectrum dim mismatch: expected {output_dim}, got {y.shape[0]}")
        peaks = _simple_find_peaks(y, height=peak_height)
        for p in peaks:
            for off in range(-int(peak_width), int(peak_width) + 1):
                j = int(p) + off
                if 0 <= j < output_dim:
                    count[j] += 1.0

    weight = np.ones(int(output_dim), dtype=np.float32)
    if count.max() > 0:
        count = count / (count.max() + 1e-8)
        weight += (float(alpha) - 1.0) * count
    return torch.tensor(weight, dtype=torch.float32)


class WeightedMSELoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, output_dim: int = 190):
        super().__init__()
        if weight is None:
            weight = torch.ones(int(output_dim), dtype=torch.float32)
        self.register_buffer("weight", weight.float())

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape != target.shape:
            raise ValueError(f"input/target shape mismatch: {tuple(input.shape)} vs {tuple(target.shape)}")
        return (self.weight.view(1, -1) * (input - target).pow(2)).mean()
