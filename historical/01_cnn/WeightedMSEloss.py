import torch
import numpy as np
from torch import nn
from scipy.signal import find_peaks


def build_peak_weight(dataset, output_dim=76, peak_width=1, alpha=3.0):
    count = np.zeros(output_dim)
    for i in range(len(dataset)):
        _, spectrum = dataset[i]
        spectrum = spectrum.numpy()
        peaks, _ = find_peaks(spectrum, height=0.05)
        for p in peaks:
            for offset in range(-peak_width, peak_width + 1):
                idx = p + offset
                if 0 <= idx < output_dim:
                    count[idx] += 1
    weight = np.ones(output_dim)
    if np.max(count) > 0:
        count = count / (np.max(count) + 1e-8)
        weight += (alpha - 1.0) * count
    return torch.tensor(weight, dtype=torch.float32)

class WeightedMSELoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.register_buffer('weight', weight if weight is not None else torch.ones(76))

    def forward(self, input, target):
        loss = self.weight * (input - target) ** 2
        return loss.mean()