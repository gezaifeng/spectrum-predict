import os
import numpy as np
import torch
from torch.utils.data import Dataset

def _trimmed_mean(x: np.ndarray, axis: int, q: float = 5.0) -> np.ndarray:
    """
    截断均值：丢弃两端各 q% 像素，抗眩光/阴影
    x: 任意形状数组
    axis: 按该维度做统计
    q: 百分位（0~50），默认 5%
    """
    if q <= 0:
        return x.mean(axis=axis)
    low = np.percentile(x, q, axis=axis, keepdims=True)
    high = np.percentile(x, 100 - q, axis=axis, keepdims=True)
    x_clip = np.clip(x, low, high)
    return x_clip.mean(axis=axis)

def _median(x: np.ndarray, axis: int) -> np.ndarray:
    return np.median(x, axis=axis)

def _choose_gray_block(blocks_46x3: np.ndarray) -> tuple:
    """
    在 (4,6,3) 的色块统计中自动选择最接近“中性灰”的块：
    - 最小化 |R-G| + |G-B|（色差）
    - 明度不过暗不过亮（0.15~0.85 的范围内优先）
    返回 (row, col)
    """
    # blocks_46x3: (4,6,3) in [0,1]
    r, g, b = blocks_46x3[..., 0], blocks_46x3[..., 1], blocks_46x3[..., 2]
    chroma = np.abs(r - g) + np.abs(g - b)
    lum = (r + g + b) / 3.0
    mask = (lum >= 0.15) & (lum <= 0.85)
    # 先在合格亮度内找最小色差
    if mask.any():
        idx = np.where(mask)
        flat_i = np.argmin(chroma[idx])
        row, col = idx[0][flat_i], idx[1][flat_i]
    else:
        # 若无合格亮度，退化为全局最小色差
        row, col = np.unravel_index(np.argmin(chroma), chroma.shape)
    return int(row), int(col)

def _apply_gray_balance(blocks_46x3: np.ndarray, gray_rc=None, target='avg') -> np.ndarray:
    """
    对 (4,6,3) 的块做简单“灰度/白平衡”
    - 选择一个灰块（gray_rc 指定或自动）
    - 令该灰块三通道一致：gain_c = target_lum / gray_c
    - 对所有块乘以对应 gain_c
    target: 'avg'  → 以灰块 RGB 平均作为亮度目标
            float → 直接指定目标亮度（0~1）
    """
    blocks = blocks_46x3.copy()
    if gray_rc is None:
        gr, gc = _choose_gray_block(blocks)
    else:
        gr, gc = gray_rc
    gray = blocks[gr, gc, :]  # (3,)
    # 防止除零
    eps = 1e-6
    if target == 'avg':
        tgt = float(gray.mean())
    else:
        tgt = float(target)
    gains = (tgt + eps) / (gray + eps)  # (3,)
    blocks = blocks * gains[None, None, :]
    # 限幅到 [0,1]，避免极端拉伸
    blocks = np.clip(blocks, 0.0, 1.0)
    return blocks

class SpectralDataset(Dataset):
    """
    读取 data_dir 下的成对样本：rgb_XXXX.npy (4,6,100,3) 与 spectral_XXXX.npy (76,)
    关键特性：
    - 使用“稳健统计”对每个色块 100 像素聚合（默认：5% 截断均值，与 predict.py 一致）
    - 可选灰度/白平衡归一（默认启用，自动选择最接近中性灰的色块）
    - 可选光谱归一化（不建议，一般保持 False 并确保预测端不做反归一化）
    """
    def __init__(self,
                 data_dir: str,
                 normalize_spectra: bool = False,
                 rgb_stat: str = 'trimmed_mean',  # 'trimmed_mean' | 'median' | 'mean'
                 trim_q: float = 5.0,
                 gray_balance: bool = True,
                 gray_rc: tuple = None,
                 clip_percentile: float = 0.0  # 每通道可选的像素剪裁，默认不剪裁
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.rgb_files = sorted([f for f in os.listdir(data_dir) if f.startswith("rgb") and f.endswith(".npy")])
        self.spectral_files = sorted([f.replace("rgb_", "spectral_") for f in self.rgb_files])

        # 基本一致性校验
        for s in self.spectral_files:
            if not os.path.exists(os.path.join(data_dir, s)):
                raise FileNotFoundError(f"未找到与 RGB 配对的光谱文件: {s}")

        self.normalize_spectra = normalize_spectra
        self.rgb_stat = rgb_stat
        self.trim_q = float(trim_q)
        self.gray_balance = bool(gray_balance)
        self.gray_rc = gray_rc  # e.g., (row, col)
        self.clip_percentile = float(clip_percentile)

        self.mean, self.std = None, None
        if self.normalize_spectra:
            self.mean, self.std = self.compute_spectra_stats()

    def compute_spectra_stats(self):
        all_spectra = []
        for spectral_file in self.spectral_files:
            spectral_path = os.path.join(self.data_dir, spectral_file)
            y = np.load(spectral_path).astype(np.float32)
            if y.ndim != 1:
                y = y.reshape(-1)
            all_spectra.append(y)
        all_spectra = np.stack(all_spectra, axis=0)  # (N,76)
        mean = np.mean(all_spectra, axis=0)
        std = np.std(all_spectra, axis=0)
        # 防止 std 为 0
        std = np.where(std < 1e-8, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    def __len__(self):
        return len(self.rgb_files)

    def _aggregate_blocks(self, rgb: np.ndarray) -> np.ndarray:
        """
        rgb: (4,6,100,3) in [0,1]
        返回 (4,6,3) 的块统计
        """
        x = rgb
        # 可选每通道像素剪裁（对原始 100 像素层面进行）
        if self.clip_percentile and self.clip_percentile > 0:
            q = float(self.clip_percentile)
            low = np.percentile(x, q, axis=2, keepdims=True)
            high = np.percentile(x, 100 - q, axis=2, keepdims=True)
            x = np.clip(x, low, high)

        if self.rgb_stat == 'trimmed_mean':
            agg = _trimmed_mean(x, axis=2, q=self.trim_q)
        elif self.rgb_stat == 'median':
            agg = _median(x, axis=2)
        elif self.rgb_stat == 'mean':
            agg = x.mean(axis=2)
        else:
            raise ValueError(f"未知 rgb_stat: {self.rgb_stat}")
        return agg  # (4,6,3)

    def __getitem__(self, idx: int):
        rgb_path = os.path.join(self.data_dir, self.rgb_files[idx])
        spectral_path = os.path.join(self.data_dir, self.spectral_files[idx])

        # ---------- 读入 ----------
        rgb = np.load(rgb_path).astype(np.float32) / 255.0  # (4,6,100,3)
        y = np.load(spectral_path).astype(np.float32)       # (76,)

        # 基本形状校验
        if rgb.shape != (4, 6, 100, 3):
            raise ValueError(f"RGB 形状异常，应为 (4,6,100,3)，当前为 {rgb.shape} @ {rgb_path}")
        if y.ndim != 1 or y.shape[0] != 76:
            raise ValueError(f"光谱形状异常，应为 (76,)，当前为 {y.shape} @ {spectral_path}")

        # ---------- 稳健聚合 (4,6,3) ----------
        blocks = self._aggregate_blocks(rgb)  # (4,6,3)

        # ---------- 灰度/白平衡（逐样本） ----------
        if self.gray_balance:
            blocks = _apply_gray_balance(blocks, gray_rc=self.gray_rc, target='avg')

        # ---------- 组装为网络输入 (3,4,6) ----------
        # 注意：通道顺序为 RGB → (C,H,W)
        x_tensor = torch.tensor(blocks, dtype=torch.float32).permute(2, 0, 1).contiguous()
        y_tensor = torch.tensor(y.reshape(-1), dtype=torch.float32)

        # ---------- 可选：光谱归一化 ----------
        if self.normalize_spectra:
            mean = torch.tensor(self.mean, dtype=torch.float32)
            std = torch.tensor(self.std, dtype=torch.float32)
            y_tensor = (y_tensor - mean) / std

        return x_tensor, y_tensor
