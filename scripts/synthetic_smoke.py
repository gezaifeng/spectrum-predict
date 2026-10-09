"""Check execution on synthetic data; this is not a scientific accuracy test."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

repo = Path(__file__).resolve().parents[1]
cfg = json.loads((repo / 'configs/current.json').read_text())
with tempfile.TemporaryDirectory(prefix='spectrum_smoke_') as temp:
    root = Path(temp)
    rng = np.random.default_rng(42)
    for split, count in [('train', 6), ('val', 3)]:
        directory = root / split
        directory.mkdir()
        for i in range(count):
            sample_id = f'{split}_{i}'
            x = rng.uniform(.1, .8, cfg['feature_shape']).astype(np.float32) * cfg['feature_scale']
            y = (.2 + .1 * np.sin(np.linspace(0, 3, cfg['output_dim']))).astype(np.float32)
            np.save(directory / f'rgb_{sample_id}.npy', x)
            np.save(directory / f'spectral_{sample_id}.npy', y)
    subprocess.run([sys.executable, 'train.py', '--train-dir', str(root / 'train'),
                    '--val-dir', str(root / 'val'), '--output', str(root / 'run'),
                    '--epochs', '1', '--batch-size', '3'], cwd=repo, check=True)
    subprocess.run([sys.executable, 'predict.py', '--checkpoint', str(root / 'run/best.pt'),
                    '--input', str(root / 'val/rgb_val_0.npy'), '--output', str(root / 'pred.npy')],
                   cwd=repo, check=True)
    pred = np.load(root / 'pred.npy')
    assert pred.shape == (cfg['output_dim'],) and np.isfinite(pred).all()
print('Synthetic training/checkpoint/prediction checks passed; real accuracy remains unverified.')
