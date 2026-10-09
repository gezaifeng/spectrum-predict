# -*- coding: utf-8 -*-
"""
适用于 TokenSpectralNet 的数据集脚本
---------------------------------
默认目录结构：
    data_dir/
        rgb_0001.npy
        spectral_0001.npy
        rgb_0002.npy
        spectral_0002.npy
        ...

支持的 RGB 文件形状：
    1) (H, W, P, 3)  例如 (4, 6, 100, 3)
       表示每个色块内取了 P 个像素点的 RGB
    2) (H, W, 3)     例如 (4, 6, 3)
       表示每个色块已经聚合成一个 RGB

输出：
    image   -> torch.FloatTensor, shape [3, H, W]
    spectra -> torch.FloatTensor, shape [D]
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetStats:
    rgb_mean: Optional[np.ndarray] = None   # [3,1,1]
    rgb_std: Optional[np.ndarray] = None    # [3,1,1]
    spec_mean: Optional[np.ndarray] = None  # [D]
    spec_std: Optional[np.ndarray] = None   # [D]


class SpectralDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        normalize_rgb: bool = False,
        normalize_spectra: bool = False,
        rgb_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        spectra_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        patch_reduce: str = "mean",            # mean | median
        clamp_rgb_01: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.normalize_rgb = normalize_rgb
        self.normalize_spectra = normalize_spectra
        self.patch_reduce = patch_reduce.lower()
        self.clamp_rgb_01 = clamp_rgb_01
        self.eps = eps

        self.rgb_files = sorted(glob.glob(os.path.join(data_dir, "rgb_*.npy")))
        if len(self.rgb_files) == 0:
            raise FileNotFoundError(f"No rgb_*.npy found in {data_dir}")

        self.spec_files: List[str] = []
        for rgb_path in self.rgb_files:
            spec_path = rgb_path.replace("rgb_", "spectral_")
            if not os.path.exists(spec_path):
                raise FileNotFoundError(f"Missing matched spectral file: {spec_path}")
            self.spec_files.append(spec_path)

        self.rgb_mean = None
        self.rgb_std = None
        self.spec_mean = None
        self.spec_std = None

        if rgb_stats is not None:
            self.rgb_mean = np.asarray(rgb_stats[0], dtype=np.float32)
            self.rgb_std = np.asarray(rgb_stats[1], dtype=np.float32)
        if spectra_stats is not None:
            self.spec_mean = np.asarray(spectra_stats[0], dtype=np.float32)
            self.spec_std = np.asarray(spectra_stats[1], dtype=np.float32)

        if self.normalize_rgb and (self.rgb_mean is None or self.rgb_std is None):
            self.rgb_mean, self.rgb_std = self._compute_rgb_stats()
        if self.normalize_spectra and (self.spec_mean is None or self.spec_std is None):
            self.spec_mean, self.spec_std = self._compute_spectra_stats()

    def __len__(self):
        return len(self.rgb_files)

    def _reduce_patch_pixels(self, arr: np.ndarray) -> np.ndarray:
        """
        arr:
            (H, W, P, 3) -> (H, W, 3)
            or already (H, W, 3)
        """
        arr = arr.astype(np.float32)

        if arr.ndim == 4 and arr.shape[-1] == 3:
            # 如果是 0~255，先缩放到 0~1
            if arr.max() > 1.0:
                arr = arr / 255.0
            if self.patch_reduce == "median":
                arr = np.median(arr, axis=2)
            else:
                arr = np.mean(arr, axis=2)
            return arr.astype(np.float32)

        if arr.ndim == 3 and arr.shape[-1] == 3:
            if arr.max() > 1.0:
                arr = arr / 255.0
            return arr.astype(np.float32)

        raise ValueError(
            f"Unexpected RGB shape {arr.shape}, expected (H,W,P,3) or (H,W,3)."
        )

    def _load_rgb(self, path: str) -> np.ndarray:
        arr = np.load(path)
        arr = self._reduce_patch_pixels(arr)    # [H,W,3]
        if self.clamp_rgb_01:
            arr = np.clip(arr, 0.0, 1.0)
        arr = np.transpose(arr, (2, 0, 1))      # [3,H,W]
        return arr.astype(np.float32)

    def _load_spectrum(self, path: str) -> np.ndarray:
        spec = np.load(path).astype(np.float32).reshape(-1)
        return spec

    def _compute_rgb_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        xs = []
        for p in self.rgb_files:
            x = self._load_rgb(p)   # [3,H,W]
            xs.append(x.reshape(3, -1))
        xs = np.concatenate(xs, axis=1)  # [3, total_pixels]
        mean = xs.mean(axis=1, keepdims=True).reshape(3, 1, 1).astype(np.float32)
        std = xs.std(axis=1, keepdims=True).reshape(3, 1, 1).astype(np.float32)
        std = np.maximum(std, self.eps)
        return mean, std

    def _compute_spectra_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        ys = []
        for p in self.spec_files:
            y = self._load_spectrum(p)
            ys.append(y)
        ys = np.stack(ys, axis=0)
        mean = ys.mean(axis=0).astype(np.float32)
        std = ys.std(axis=0).astype(np.float32)
        std = np.maximum(std, self.eps)
        return mean, std

    def get_stats(self) -> DatasetStats:
        return DatasetStats(
            rgb_mean=self.rgb_mean,
            rgb_std=self.rgb_std,
            spec_mean=self.spec_mean,
            spec_std=self.spec_std,
        )

    def inverse_transform_spectrum(self, y: np.ndarray | torch.Tensor):
        if not self.normalize_spectra or self.spec_mean is None or self.spec_std is None:
            return y
        if isinstance(y, torch.Tensor):
            mean = torch.tensor(self.spec_mean, dtype=y.dtype, device=y.device)
            std = torch.tensor(self.spec_std, dtype=y.dtype, device=y.device)
            return y * std + mean
        return y * self.spec_std + self.spec_mean

    def __getitem__(self, idx: int):
        x = self._load_rgb(self.rgb_files[idx])      # [3,H,W]
        y = self._load_spectrum(self.spec_files[idx]) # [D]

        if self.normalize_rgb and self.rgb_mean is not None and self.rgb_std is not None:
            x = (x - self.rgb_mean) / self.rgb_std

        if self.normalize_spectra and self.spec_mean is not None and self.spec_std is not None:
            y = (y - self.spec_mean) / self.spec_std

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


if __name__ == "__main__":
    # 简单自检：改成你自己的路径再跑
    data_dir = r"D:\Desktop\dataset\guasnoise1-1020\train"
    if os.path.exists(data_dir):
        ds = SpectralDataset(data_dir, normalize_rgb=False, normalize_spectra=False)
        x, y = ds[0]
        print("x shape:", x.shape)   # [3,H,W]
        print("y shape:", y.shape)   # [D]
        print("len:", len(ds))
