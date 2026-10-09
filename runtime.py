"""Shared reproduction runner; reconstructed in 2026, not a historical file."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import model


def load_config(path='configs/current.json'):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build_model(config):
    return getattr(model, config['model_class'])(**config['model_kwargs'])


def prediction(model_instance, x):
    result = model_instance(x)
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


def preprocess(array, config):
    x = np.asarray(array, dtype=np.float32)
    if tuple(x.shape) != tuple(config['feature_shape']):
        raise ValueError(f"Expected {config['feature_shape']}, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError('Feature contains NaN or infinity')
    x = x / config['feature_scale']
    layout = config['layout']
    if layout == 'hwkc':
        x = x.transpose(3, 0, 1, 2)  # C,H,W,K
    elif layout not in ('chw', 'chwk', 'clnk'):
        raise ValueError(f'Unsupported layout: {layout}')
    mode = config['aggregation']
    if mode == 'sequence':
        x = x.transpose(0, 3, 1, 2)  # C,K,H,W
    elif mode in ('mean', 'trimmed_mean'):
        if mode == 'trimmed_mean':
            low = np.percentile(x, 5, axis=-1, keepdims=True)
            high = np.percentile(x, 95, axis=-1, keepdims=True)
            x = np.clip(x, low, high)
        x = x.mean(axis=-1)
    elif mode != 'none':
        raise ValueError(f'Unsupported aggregation: {mode}')
    if config.get('gray_balance'):
        chroma = np.abs(x[0] - x[1]) + np.abs(x[1] - x[2])
        lum = x.mean(axis=0)
        neutral = (lum >= .15) & (lum <= .85)
        score = np.where(neutral, chroma, np.inf) if neutral.any() else chroma
        row, col = np.unravel_index(score.argmin(), score.shape)
        gray = x[:, row, col]
        x = np.clip(x * ((gray.mean() + 1e-6) / (gray + 1e-6))[:, None, None], 0, 1)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


class PairedDataset(Dataset):
    def __init__(self, root, config, feature_prefix='rgb_'):
        self.root = Path(root)
        self.config = config
        self.prefix = feature_prefix
        self.features = sorted(self.root.glob(feature_prefix + '*.npy'))
        if not self.features:
            raise ValueError(f'No {feature_prefix}*.npy in {root}')
        self.ids = [p.stem[len(feature_prefix):] for p in self.features]
        for sample_id in self.ids:
            target = self.root / ('spectral_' + sample_id + '.npy')
            if not target.is_file():
                raise FileNotFoundError(target)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = preprocess(np.load(self.features[index], allow_pickle=False), self.config)
        y = np.load(self.root / ('spectral_' + self.ids[index] + '.npy'), allow_pickle=False)
        if y.shape != (self.config['output_dim'],) or not np.isfinite(y).all():
            raise ValueError(f'Invalid spectrum for {self.ids[index]}: {y.shape}')
        return x, torch.tensor(y, dtype=torch.float32)


def split_check(train, val):
    overlap = set(train.ids) & set(val.ids)
    if overlap:
        raise ValueError(f'Sample IDs occur in both splits: {sorted(overlap)[:5]}')


def loss(pred, target, coef, config, weight=None):
    cfg = config['loss']
    error = (pred - target).square()
    mse = error.mean() if weight is None else (weight * error).mean()
    dc = (pred.mean(1) - target.mean(1)).square().mean()
    tv = (pred[:, 1:] - pred[:, :-1]).square().mean()
    diff = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs().mean()
    reg = pred.new_zeros(()) if coef is None else coef.square().mean()
    return (mse + cfg.get('dc', 0) * dc + cfg.get('tv', 0) * tv
            + cfg.get('coef', 0) * reg + cfg.get('diff', 0) * diff)


def peak_weight(dataset, alpha):
    from scipy.signal import find_peaks
    count = np.zeros(dataset.config['output_dim'])
    for _, y in dataset:
        peaks, _ = find_peaks(y.numpy(), height=.05)
        for p in peaks:
            count[max(0, p - 1):min(len(count), p + 2)] += 1
    if count.max() > 0:
        count /= count.max() + 1e-8
    return torch.tensor(1 + (alpha - 1) * count, dtype=torch.float32)


def metrics(pred, target):
    err = pred - target
    norm_product = pred.norm(dim=1) * target.norm(dim=1)
    valid = norm_product > 1e-8
    cosine = ((pred * target).sum(1) / norm_product.clamp_min(1e-8)).clamp(-1, 1)
    sam = torch.acos(cosine[valid]).mean().item() if valid.any() else None
    sample_mse = err.square().mean(1)
    worst = sample_mse.topk(max(1, int(np.ceil(len(sample_mse) * .1)))).values.mean()
    return {'rmse': err.square().mean().sqrt().item(), 'mae': err.abs().mean().item(),
            'sam_rad': sam, 'sam_valid_samples': int(valid.sum()),
            'worst10_mse': worst.item()}
