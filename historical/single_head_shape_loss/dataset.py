# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def _to_chw(x: np.ndarray) -> np.ndarray:
    """(3,H,W) or (H,W,3) -> (3,H,W)"""
    if x.ndim != 3:
        raise ValueError(f"feature must be 3D, got {x.shape}")
    if x.shape[0] == 3:
        return x
    if x.shape[-1] == 3:
        return np.transpose(x, (2, 0, 1))
    raise ValueError(f"unrecognized feature shape: {x.shape}")


def sanitize_feature(
    x_chw: np.ndarray,
    clip_value: float = 8.0,
    nan_repl: float = 0.0,
    posinf_repl: float = 0.0,
    neginf_repl: float = 0.0,
) -> np.ndarray:
    """
    适用于 ratio / log_ratio 等预提取特征：
    - 先把 nan/inf 变成有限值
    - 再做裁剪，避免极端值主导训练
    """
    x = np.nan_to_num(x_chw, nan=nan_repl, posinf=posinf_repl, neginf=neginf_repl)
    if clip_value is not None:
        x = np.clip(x, -clip_value, clip_value)
    return x.astype(np.float32)


def per_image_channel_zscore(x_chw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    每个样本、每个通道在(H,W)上做 z-score。
    用于消掉不同拍摄批次的整体亮度/偏置影响。
    std 太小则不放大，避免数值爆炸。
    """
    x = x_chw.astype(np.float32)
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    std_safe = np.where(std < 1e-4, 1.0, std)
    return ((x - mean) / (std_safe + eps)).astype(np.float32)


class SpectralDatasetLogRatio(Dataset):
    """
    当前保持类名不变，兼容原训练脚本。

    data_dir:
      rgb_XXXX.npy      -> (3,H,W) 或 (H,W,3) 的 ratio / log_ratio 特征
      spectral_XXXX.npy -> (76,)
    """
    def __init__(self, data_dir: str, clip_value: float = 8.0, use_zscore: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.rgb_files = sorted([f for f in os.listdir(data_dir) if f.startswith("rgb_") and f.endswith(".npy")])
        self.sp_files = [f.replace("rgb_", "spectral_") for f in self.rgb_files]
        for sf in self.sp_files:
            if not os.path.exists(os.path.join(data_dir, sf)):
                raise FileNotFoundError(f"missing paired spectral file: {sf}")

        self.clip_value = float(clip_value) if clip_value is not None else None
        self.use_zscore = bool(use_zscore)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx: int):
        rgb_path = os.path.join(self.data_dir, self.rgb_files[idx])
        sp_path = os.path.join(self.data_dir, self.sp_files[idx])

        x = np.load(rgb_path).astype(np.float32)
        y = np.load(sp_path).astype(np.float32).reshape(-1)

        if y.shape[0] != 190:
            raise ValueError(f"spectral must be (76,), got {y.shape} @ {sp_path}")

        x = _to_chw(x)
        x = sanitize_feature(x, clip_value=self.clip_value)
        if self.use_zscore:
            x = per_image_channel_zscore(x)

        if not np.isfinite(x).all():
            raise ValueError(f"non-finite feature after sanitize @ {rgb_path}")
        if not np.isfinite(y).all():
            raise ValueError(f"non-finite spectral @ {sp_path}")

        return torch.from_numpy(x), torch.from_numpy(y)
