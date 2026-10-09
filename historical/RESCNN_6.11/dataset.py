# -*- coding: utf-8 -*-
"""
dataset.py
适配多光源聚合后的输入：
  特征文件: feat_xxxx.npz 或 feat_xxxx.npy
    - npz 里默认取 key="data"
    - 支持 (3, L, N, K) 或 (R,3,L,N,K)
  光谱文件: spectral_xxxx.npy -> (190,)

输出给模型：
  x: (3, L*N, K)
  y: (190,)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    # 新增：光源数
    lights: int = 8

    # 色块网格（仍然保留，主要用于位置编码/一致性检查）
    grid_rows: int = 6
    grid_cols: int = 12

    # 每块采样点数
    sample_points: int = 9

    output_dim: int = 190

    # 文件名前缀/后缀
    feat_prefix: str = "group_"          # 你也可以改成 "feat_" 等
    spectral_prefix: str = "spectral_"
    feat_ext: str = ".npz"               # ".npz" 或 ".npy"

    # 预处理
    rgb_scale: str = "auto"              # "auto" | "255" | "none"
    normalize_mode: str = "scale_only"   # "scale_only" | "channel_zscore" | "global_zscore"
    clip_value: Optional[float] = None

    split_repeats: bool = True           # 若输入是 (R,3,L,N,K)，拆成 R 个样本


def _replace_prefix(filename: str, old_prefix: str, new_prefix: str) -> str:
    if not filename.startswith(old_prefix):
        raise ValueError(f"filename {filename} does not start with {old_prefix}")
    return new_prefix + filename[len(old_prefix):]


def _load_feat(path: str) -> np.ndarray:
    if path.lower().endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        if "data" in z:
            return np.asarray(z["data"])
        if "feat" in z:
            return np.asarray(z["feat"])
        # fallback: 取第一个数组
        keys = list(z.keys())
        if not keys:
            raise ValueError(f"empty npz: {path}")
        return np.asarray(z[keys[0]])
    else:
        return np.asarray(np.load(path, allow_pickle=False))


def _canonical_single_feat(arr: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    """
    将单个样本转换为 (3, L, N, K)
    兼容：
      (3,L,N,K)
      (L,3,N,K)
      (3,N,K)  -> 视为 L=1（不建议，但兼容）
    """
    x = np.asarray(arr)

    if x.ndim == 3:
        # (3,N,K)
        if x.shape[0] != 3:
            raise ValueError(f"Unsupported 3D feat shape: {x.shape}")
        x = x[:, None, :, :]  # (3,1,N,K)

    elif x.ndim == 4:
        if x.shape[0] == 3:
            # (3,L,N,K)
            pass
        elif x.shape[1] == 3:
            # (L,3,N,K) -> (3,L,N,K)
            x = np.transpose(x, (1, 0, 2, 3))
        else:
            raise ValueError(f"Unsupported 4D feat shape: {x.shape}")
    else:
        raise ValueError(f"Unsupported feat ndim: {x.ndim}, shape={x.shape}")

    if x.shape[0] != 3:
        raise ValueError(f"channel must be 3, got {x.shape}")
    if x.shape[1] != cfg.lights:
        raise ValueError(f"lights mismatch: expected L={cfg.lights}, got {x.shape[1]} @ {x.shape}")
    total_blocks = cfg.grid_rows * cfg.grid_cols
    if x.shape[2] != total_blocks:
        raise ValueError(f"blocks mismatch: expected N={total_blocks}, got {x.shape[2]} @ {x.shape}")
    if x.shape[3] != cfg.sample_points:
        raise ValueError(f"sample_points mismatch: expected K={cfg.sample_points}, got {x.shape[3]} @ {x.shape}")
    return x.astype(np.float32, copy=False)


def _read_feat_with_optional_repeat(path: str, repeat_idx: Optional[int], cfg: DatasetConfig) -> np.ndarray:
    arr = _load_feat(path)

    # (R,3,L,N,K) or (R,L,3,N,K)
    if arr.ndim == 5:
        if repeat_idx is None:
            # 不拆重复时：对 R 维均值（保留接口，不推荐）
            singles = [_canonical_single_feat(arr[i], cfg) for i in range(arr.shape[0])]
            x = np.mean(np.stack(singles, axis=0), axis=0)
        else:
            x = _canonical_single_feat(arr[int(repeat_idx)], cfg)
    else:
        x = _canonical_single_feat(arr, cfg)

    return x


def _preprocess_feat(x: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    """
    x: (3,L,N,K)
    默认只做 scale，不做 z-score，避免抹掉吸光强度信息。
    """
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    scale = str(cfg.rgb_scale).lower()
    if scale == "auto":
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
        raise ValueError("non-finite feature after preprocessing")
    return x.astype(np.float32)


def _load_spectrum(path: str, output_dim: int) -> np.ndarray:
    y = np.load(path).astype(np.float32).reshape(-1)
    if y.shape[0] != output_dim:
        raise ValueError(f"spectral must be ({output_dim},), got {y.shape} @ {path}")
    if not np.isfinite(y).all():
        raise ValueError(f"non-finite spectrum @ {path}")
    return y.astype(np.float32)


class SpectralMultiLightDataset(Dataset):
    """
    返回：
      x: (3, L*N, K)  其中 L=8, N=72, K=9 => (3,576,9)
      y: (190,)
    """

    def __init__(self, data_dir: str, cfg: Optional[DatasetConfig] = None):
        super().__init__()
        self.data_dir = data_dir
        self.cfg = cfg or DatasetConfig()

        ext = self.cfg.feat_ext.lower()
        self.feat_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith(self.cfg.feat_prefix) and f.lower().endswith(ext)
        )
        if not self.feat_files:
            raise FileNotFoundError(f"No {self.cfg.feat_prefix}*{ext} found in {data_dir}")

        self.entries: list[tuple[str, str, Optional[int]]] = []
        for ff in self.feat_files:
            sf = _replace_prefix(ff, self.cfg.feat_prefix, self.cfg.spectral_prefix)
            feat_path = os.path.join(data_dir, ff)
            sp_path = os.path.join(data_dir, sf)
            if not os.path.exists(sp_path):
                raise FileNotFoundError(f"missing paired spectral file: {sf}")

            arr = _load_feat(feat_path)
            if arr.ndim == 5 and self.cfg.split_repeats:
                for rep in range(arr.shape[0]):
                    self.entries.append((feat_path, sp_path, rep))
            else:
                self.entries.append((feat_path, sp_path, None))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        feat_path, sp_path, repeat_idx = self.entries[idx]
        x = _read_feat_with_optional_repeat(feat_path, repeat_idx, self.cfg)  # (3,L,N,K)
        x = _preprocess_feat(x, self.cfg)

        # (3,L,N,K) -> (3, L*N, K)
        c, L, N, K = x.shape
        x = x.reshape(c, L * N, K)

        y = _load_spectrum(sp_path, self.cfg.output_dim)
        return torch.from_numpy(x), torch.from_numpy(y)

    def describe(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "num_feat_files": len(self.feat_files),
            "num_training_items": len(self.entries),
            "lights": self.cfg.lights,
            "grid_rows": self.cfg.grid_rows,
            "grid_cols": self.cfg.grid_cols,
            "tokens": self.cfg.lights * self.cfg.grid_rows * self.cfg.grid_cols,
            "sample_points": self.cfg.sample_points,
            "output_dim": self.cfg.output_dim,
            "feat_prefix": self.cfg.feat_prefix,
            "feat_ext": self.cfg.feat_ext,
            "spectral_prefix": self.cfg.spectral_prefix,
            "rgb_scale": self.cfg.rgb_scale,
            "normalize_mode": self.cfg.normalize_mode,
            "split_repeats": self.cfg.split_repeats,
        }


# 兼容旧名字（尽量不动 train.py 的 import）
SpectralDataset = SpectralMultiLightDataset