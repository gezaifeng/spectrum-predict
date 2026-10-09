"""Run a recovered model with a consistent train/predict data contract."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from runtime import build_model, load_config, loss, metrics, PairedDataset, peak_weight, prediction, split_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/current.json')
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--val-dir', required=True)
    parser.add_argument('--feature-prefix', default='rgb_')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('epochs >= 1 and batch-size >= 2 are required')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    config = load_config(args.config)
    train = PairedDataset(args.train_dir, config, args.feature_prefix)
    val = PairedDataset(args.val_dir, config, args.feature_prefix)
    split_check(train, val)
    if len(train) < 2:
        raise ValueError('Training needs at least two samples for BatchNorm')
    # Avoid one-sample batches for recovered models with BatchNorm1d.
    batch_size = min(args.batch_size, len(train))
    loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                        drop_last=len(train) % batch_size == 1)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    device = torch.device(args.device)
    net = build_model(config).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    alpha = config['loss'].get('peak_alpha', 1)
    weight = peak_weight(train, alpha).to(device) if alpha != 1 else None
    base_weight = None if weight is None else weight.clone()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    history = []
    for epoch in range(1, args.epochs + 1):
        net.train()
        train_sum, train_n = 0., 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred, coef = prediction(net, x)
            objective = loss(pred, y, coef, config, weight)
            if not torch.isfinite(objective):
                raise ValueError('Training produced a non-finite loss')
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            train_sum += objective.item() * len(x)
            train_n += len(x)
        net.eval()
        preds, targets, val_sum = [], [], 0.
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred, coef = prediction(net, x)
                val_sum += loss(pred, y, coef, config, weight).item() * len(x)
                preds.append(pred)
                targets.append(y)
        pred, target = torch.cat(preds), torch.cat(targets)
        row = {'epoch': epoch, 'train_loss': train_sum / train_n,
               'val_loss': val_sum / len(val), **metrics(pred, target)}
        history.append(row)
        if row['val_loss'] < best:
            best = row['val_loss']
            torch.save({'model': net.state_dict(), 'config': copy.deepcopy(config),
                        'epoch': epoch, 'metrics': row, 'seed': args.seed,
                        'training_feature_prefix': args.feature_prefix}, output / 'best.pt')
        interval = config['loss'].get('dynamic_interval', 0)
        if interval and epoch % interval == 0:
            mae = (pred - target).abs().mean(0)
            weight = base_weight * (mae / mae.mean().clamp_min(1e-8)).clamp_min(.1).sqrt()
            weight = weight / weight.mean()
        print(json.dumps(row, allow_nan=False))
        (output / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False), encoding='utf-8')
    # The checkpoint stores the model and preprocessing config for inference.
    (output / 'run.json').write_text(json.dumps({'config': config, 'arguments': vars(args)}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
