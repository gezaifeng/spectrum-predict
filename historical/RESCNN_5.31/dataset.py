# -*- coding: utf-8 -*-
"""
dataset.py
适配新的多色块随机采样 RGB 数据：
    rgb_xxxx.npy      -> (3,6,12,9) 或兼容格式
    spectral_xxxx.npy -> (190,)

核心设计：
1. 磁盘文件可以保持 (3, rows, cols, sample_points)。
2. Dataset 内部会把网格展平为 token：输出 (3, N, K)，其中 N 为使用的色块数，K 为每块采样点数。
3. 支持 block_indices / keep_blocks，便于做 72、60、48、36 色块消融。
4. 支持 5D 重复采样文件，例如 (10,3,6,12,9)，默认把 10 次采样拆成 10 个训练样本。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    grid_rows: int = 6
    grid_cols: int = 12
    sample_points: int = 9
    output_dim: int = 190
    rgb_prefix: str = "rgb_"
    spectral_prefix: str = "spectral_"
    rgb_scale: str = "auto"          # "auto" | "255" | "none"
    normalize_mode: str = "scale_only"  # "scale_only" | "channel_zscore" | "global_zscore"
    clip_value: Optional[float] = None
    split_repeats: bool = True


def load_block_indices(
    block_indices: Optional[Sequence[int] | np.ndarray] = None,
    block_indices_path: Optional[str] = None,
    keep_blocks: Optional[int] = None,
    total_blocks: int = 72,
) -> Optional[np.ndarray]:
    """
    生成色块选择索引，索引按 row-major：
      index = row * grid_cols + col

    优先级：block_indices > block_indices_path > keep_blocks > None
    - keep_blocks=36 表示默认取前 36 个 row-major 色块；更严谨的消融建议传入明确 block_indices_path。
    - block_indices_path 支持 .npy 或 .txt/.csv（一行或多行整数）。
    """
    if block_indices is not None:
        idx = np.asarray(block_indices, dtype=np.int64).reshape(-1)
    elif block_indices_path:
        if not os.path.exists(block_indices_path):
            raise FileNotFoundError(f"block_indices_path not found: {block_indices_path}")
        if block_indices_path.lower().endswith(".npy"):
            idx = np.load(block_indices_path).astype(np.int64).reshape(-1)
        else:
            idx = np.loadtxt(block_indices_path, delimiter="," if block_indices_path.lower().endswith(".csv") else None)
            idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    elif keep_blocks is not None:
        keep_blocks = int(keep_blocks)
        if keep_blocks <= 0 or keep_blocks > total_blocks:
            raise ValueError(f"keep_blocks must be in [1,{total_blocks}], got {keep_blocks}")
        idx = np.arange(keep_blocks, dtype=np.int64)
    else:
        return None

    if idx.size == 0:
        raise ValueError("block_indices is empty")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"block_indices contains duplicated values: {idx.tolist()}")
    if idx.min() < 0 or idx.max() >= total_blocks:
        raise ValueError(f"block_indices must be in [0,{total_blocks - 1}], got min={idx.min()}, max={idx.max()}")
    return idx.astype(np.int64)


def _replace_prefix(filename: str, old_prefix: str, new_prefix: str) -> str:
    if not filename.startswith(old_prefix):
        raise ValueError(f"filename {filename} does not start with {old_prefix}")
    return new_prefix + filename[len(old_prefix):]


def _canonical_single_rgb(arr: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    """
    将单个样本转换为 (3,R,C,K)。
    兼容：
      (3,R,C,K)
      (R,C,K,3)
      (R,C,3,K)
      (3,R,C)       -> K=1
      (R,C,3)       -> K=1
    """
    x = np.asarray(arr)

    if x.ndim == 3:
        if x.shape[0] == 3:
            x = x[:, :, :, None]              # (3,R,C,1)
        elif x.shape[-1] == 3:
            x = np.transpose(x, (2, 0, 1))[:, :, :, None]  # (R,C,3)->(3,R,C,1)
        else:
            raise ValueError(f"Unsupported 3D rgb shape: {x.shape}")

    elif x.ndim == 4:
        if x.shape[0] == 3:
            # (3,R,C,K)
            pass
        elif x.shape[-1] == 3:
            # (R,C,K,3) -> (3,R,C,K)
            x = np.transpose(x, (3, 0, 1, 2))
        elif x.shape[2] == 3:
            # (R,C,3,K) -> (3,R,C,K)
            x = np.transpose(x, (2, 0, 1, 3))
        else:
            raise ValueError(f"Unsupported 4D rgb shape: {x.shape}")
    else:
        raise ValueError(f"Unsupported rgb ndim: {x.ndim}, shape={x.shape}")

    if x.shape[0] != 3:
        raise ValueError(f"RGB channel dimension must be 3, got {x.shape}")
    if x.shape[1] != cfg.grid_rows or x.shape[2] != cfg.grid_cols:
        raise ValueError(
            f"RGB grid mismatch: expected (3,{cfg.grid_rows},{cfg.grid_cols},K), got {x.shape}"
        )
    if cfg.sample_points is not None and x.shape[3] != cfg.sample_points:
        raise ValueError(
            f"sample_points mismatch: expected K={cfg.sample_points}, got {x.shape[3]} @ shape={x.shape}"
        )

    return x.astype(np.float32, copy=False)


def _read_rgb_with_optional_repeat(path: str, repeat_idx: Optional[int], cfg: DatasetConfig) -> np.ndarray:
    arr = np.load(path)

    if arr.ndim == 5:
        # 常见格式：(T,3,R,C,K) 或 (T,R,C,K,3) 或 (T,R,C,3,K)
        if repeat_idx is None:
            # 不拆重复时，对 T 维求均值；通常不建议，但保留接口。
            singles = [_canonical_single_rgb(arr[i], cfg) for i in range(arr.shape[0])]
            x = np.mean(np.stack(singles, axis=0), axis=0)
        else:
            x = _canonical_single_rgb(arr[int(repeat_idx)], cfg)
    else:
        x = _canonical_single_rgb(arr, cfg)

    return x


def _preprocess_rgb(x: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    """
    x: (3,R,C,K)
    默认只做 0~255 -> 0~1 缩放，不做 z-score，避免把溶液整体吸光强弱抹掉。
    """
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    scale = str(cfg.rgb_scale).lower()
    if scale == "auto":
        # RGB 原始值通常在 0~255；ratio/log_ratio 则不一定。
        if float(np.nanmax(np.abs(x))) > 2.0:
            x = x / 255.0
    elif scale in ("255", "0_255", "divide255"):
        x = x / 255.0
    elif scale in ("none", "no", "false"):
        pass
    else:
        raise ValueError("rgb_scale must be 'auto' | '255' | 'none'")

    if cfg.clip_value is not None:
        c = float(cfg.clip_value)
        x = np.clip(x, -c, c)

    mode = str(cfg.normalize_mode).lower()
    if mode == "scale_only":
        pass
    elif mode == "channel_zscore":
        # 每个样本、每个 RGB 通道在全部色块和采样点上做标准化。
        # 如果你发现浓度信息被削弱，可以改回 scale_only。
        mean = x.mean(axis=(1, 2, 3), keepdims=True)
        std = x.std(axis=(1, 2, 3), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        x = (x - mean) / std
    elif mode == "global_zscore":
        mean = x.mean(keepdims=True)
        std = x.std(keepdims=True)
        x = (x - mean) / (std + 1e-6)
    else:
        raise ValueError("normalize_mode must be 'scale_only' | 'channel_zscore' | 'global_zscore'")

    if not np.isfinite(x).all():
        raise ValueError("non-finite RGB feature after preprocessing")
    return x.astype(np.float32)


def _load_spectrum(path: str, output_dim: int) -> np.ndarray:
    y = np.load(path).astype(np.float32)

    # 兼容 (190,), (1,190), (190,1)，以及 (190,2)=[wavelength,value]
    if y.ndim == 2:
        if y.shape == (output_dim, 2):
            y = y[:, 1]
        elif y.shape == (2, output_dim):
            y = y[1]
        else:
            y = y.reshape(-1)
    else:
        y = y.reshape(-1)

    if y.shape[0] != output_dim:
        raise ValueError(f"spectral must be ({output_dim},), got {y.shape} @ {path}")
    if not np.isfinite(y).all():
        raise ValueError(f"non-finite spectrum @ {path}")
    return y.astype(np.float32)


class SpectralRGBSampleDataset(Dataset):
    """
    适用于新数据格式的 Dataset。

    返回：
      x: torch.FloatTensor, shape = (3, N, K)
         N 为选中的色块数；K 为每个色块采样点数，当前默认 K=9。
      y: torch.FloatTensor, shape = (190,)
    """

    def __init__(
        self,
        data_dir: str,
        cfg: Optional[DatasetConfig] = None,
        block_indices: Optional[Sequence[int] | np.ndarray] = None,
        block_indices_path: Optional[str] = None,
        keep_blocks: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.cfg = cfg or DatasetConfig()
        total_blocks = int(self.cfg.grid_rows) * int(self.cfg.grid_cols)
        self.block_indices = load_block_indices(
            block_indices=block_indices,
            block_indices_path=block_indices_path,
            keep_blocks=keep_blocks,
            total_blocks=total_blocks,
        )

        self.rgb_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith(self.cfg.rgb_prefix) and f.endswith(".npy")
        )
        if not self.rgb_files:
            raise FileNotFoundError(f"No {self.cfg.rgb_prefix}*.npy found in {data_dir}")

        self.entries: list[tuple[str, str, Optional[int]]] = []
        for rf in self.rgb_files:
            sf = _replace_prefix(rf, self.cfg.rgb_prefix, self.cfg.spectral_prefix)
            rgb_path = os.path.join(data_dir, rf)
            sp_path = os.path.join(data_dir, sf)
            if not os.path.exists(sp_path):
                raise FileNotFoundError(f"missing paired spectral file: {sf}")

            arr = np.load(rgb_path, mmap_mode="r")
            if arr.ndim == 5 and self.cfg.split_repeats:
                for rep in range(arr.shape[0]):
                    self.entries.append((rgb_path, sp_path, rep))
            else:
                self.entries.append((rgb_path, sp_path, None))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        rgb_path, sp_path, repeat_idx = self.entries[idx]
        x = _read_rgb_with_optional_repeat(rgb_path, repeat_idx, self.cfg)  # (3,R,C,K)
        x = _preprocess_rgb(x, self.cfg)

        # (3,R,C,K) -> (3,N,K)
        c, r, col, k = x.shape
        x = x.reshape(c, r * col, k)
        if self.block_indices is not None:
            x = x[:, self.block_indices, :]

        y = _load_spectrum(sp_path, self.cfg.output_dim)
        return torch.from_numpy(x), torch.from_numpy(y)

    def get_selected_block_indices(self) -> np.ndarray:
        total = self.cfg.grid_rows * self.cfg.grid_cols
        if self.block_indices is None:
            return np.arange(total, dtype=np.int64)
        return self.block_indices.copy()

    def describe(self) -> dict:
        idx = self.get_selected_block_indices()
        return {
            "data_dir": self.data_dir,
            "num_rgb_files": len(self.rgb_files),
            "num_training_items": len(self.entries),
            "grid_rows": self.cfg.grid_rows,
            "grid_cols": self.cfg.grid_cols,
            "sample_points": self.cfg.sample_points,
            "output_dim": self.cfg.output_dim,
            "num_selected_blocks": int(len(idx)),
            "selected_block_indices": idx.tolist(),
            "rgb_scale": self.cfg.rgb_scale,
            "normalize_mode": self.cfg.normalize_mode,
            "split_repeats": self.cfg.split_repeats,
        }


# 兼容旧训练脚本中的类名。
SpectralDatasetLogRatio = SpectralRGBSampleDataset
SpectralDataset = SpectralRGBSampleDataset
