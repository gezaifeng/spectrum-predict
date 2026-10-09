# -*- coding: utf-8 -*-
"""
train_multi_block_ablation.py

用途：
    对 selected_blocks_24/36/48/60/72.npy 进行一键式多组色块数量消融训练。
    每组色块会独立训练一个模型，并在训练结束后统一导出：
      1) 每个模型的 best.pth / training_history.csv / prediction npy
      2) 总体性能表 summary_runs.csv / summary_by_blocks.csv
      3) 逐波长误差表 per_wavelength_metrics_all.csv
      4) 多种项目展示图：指标趋势图、雷达图、逐波长误差图、误差热力图、预测曲线对比图、色块选择掩膜图等。

适配数据：
    rgb_xxxx.npy      -> (3,6,12,9) 或 (10,3,6,12,9)
    spectral_xxxx.npy -> (190,)

使用方法：
    1. 把本脚本放在包含 dataset.py / model.py / SpectralReconstructionLoss.py / WeightedMSEloss.py 的工程目录下。
    2. 修改下方 TRAIN_DIR / VAL_DIR / SAVE_ROOT。
    3. 确认 SELECTED_BLOCK_DIR 里有 selected_blocks_24.npy 等文件。
    4. 运行：python train_multi_block_ablation.py
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import DatasetConfig, SpectralRGBSampleDataset, load_block_indices
from model import SpectralTokenNet
from SpectralReconstructionLoss import SpectralReconstructionLoss, first_diff, second_diff, spectral_angle_mapper

try:
    from WeightedMSEloss import build_peak_weight
except Exception:
    build_peak_weight = None


# =============================================================================
# 1. 路径配置：只需要优先修改这里
# =============================================================================
TRAIN_DIR = r"D:\Desktop\zdataset-3.9\train"
VAL_DIR   = r"D:\Desktop\zdataset-3.9\val"
SAVE_ROOT = r"D:\Desktop\model\color_block_ablation_24_36_48_60_72"

# selected_blocks_*.npy 所在目录；默认使用本脚本同级 selected_blocks 文件夹
SCRIPT_DIR = Path(__file__).resolve().parent
SELECTED_BLOCK_DIR = SCRIPT_DIR / "selected_blocks"

# 需要训练的色块数量。严格嵌套：24 ⊂ 36 ⊂ 48 ⊂ 60 ⊂ 72
BLOCK_SETUPS = [
    (24, SELECTED_BLOCK_DIR / "selected_blocks_24.npy"),
    (36, SELECTED_BLOCK_DIR / "selected_blocks_36.npy"),
    (48, SELECTED_BLOCK_DIR / "selected_blocks_48.npy"),
    (60, SELECTED_BLOCK_DIR / "selected_blocks_60.npy"),
    (72, SELECTED_BLOCK_DIR / "selected_blocks_72.npy"),
]

# 如需每个色块数量重复训练多次，改成例如 [42, 2024, 3407]
# 默认只训练一次，节省时间。
SEEDS = [42]


# =============================================================================
# 2. 数据维度配置
# =============================================================================
GRID_ROWS = 6
GRID_COLS = 12
SAMPLE_POINTS = 9
OUTPUT_DIM = 190
WAVELENGTHS = np.arange(380, 760, 2, dtype=np.float32)  # 190 点：380~758 nm

RGB_PREFIX = "rgb_"
SPECTRAL_PREFIX = "spectral_"
RGB_SCALE = "auto"              # "auto" | "255" | "none"
NORMALIZE_MODE = "scale_only"   # "scale_only" | "channel_zscore" | "global_zscore"
CLIP_VALUE = None
SPLIT_REPEATS = True             # 若文件是 (10,3,6,12,9)，拆成10个训练样本


# =============================================================================
# 3. 训练超参数
# =============================================================================
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 30

# 模型超参数
EMBED_DIM = 96
DEPTH = 3
NUM_HEADS = 4
MLP_RATIO = 4.0
DROPOUT = 0.10
OUT_ACTIVATION = "linear"       # 可对比 "softplus"

# 损失函数超参数
ALPHA_PEAK = 3.0
LAMBDA_DIFF1 = 0.30
LAMBDA_DIFF2 = 0.05
LAMBDA_COS = 0.10
LAMBDA_SAM = 0.10
LAMBDA_TOPK = 0.20
LAMBDA_DC = 0.05
TOPK_RATIO = 0.10


# =============================================================================
# 4. 基础工具函数
# =============================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sample_ids_from_dataset(ds: SpectralRGBSampleDataset) -> List[str]:
    ids = []
    for rgb_path, _sp_path, rep in ds.entries:
        base = Path(rgb_path).stem
        if base.startswith(ds.cfg.rgb_prefix):
            base = base[len(ds.cfg.rgb_prefix):]
        ids.append(f"{base}#rep{rep}" if rep is not None else base)
    return ids


@torch.no_grad()
def predict_loader(model, loader, device, return_attention: bool = False):
    model.eval()
    preds, tgts, attns = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        if return_attention:
            out = model(x, return_dict=True)
            pred = out["spectrum"]
            attn = out.get("block_attention", None)
            if attn is not None:
                attns.append(attn.detach().cpu())
        else:
            pred = model(x)
        preds.append(pred.detach().cpu())
        tgts.append(y.detach().cpu())
    pred_np = torch.cat(preds, dim=0).numpy().astype(np.float32)
    tgt_np = torch.cat(tgts, dim=0).numpy().astype(np.float32)
    attn_np = torch.cat(attns, dim=0).numpy().astype(np.float32) if len(attns) else None
    return pred_np, tgt_np, attn_np


@torch.no_grad()
def evaluate_loss_and_metrics(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    comp_sum: Dict[str, float] = {}
    pred_all, target_all = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        pred = model(x)
        comps = criterion.forward_components(pred, y)
        bs = x.size(0)
        total_loss += comps["total"].item() * bs
        total_n += bs
        for k, v in comps.items():
            comp_sum[k] = comp_sum.get(k, 0.0) + v.item() * bs
        pred_all.append(pred.detach())
        target_all.append(y.detach())

    pred_all = torch.cat(pred_all, dim=0)
    target_all = torch.cat(target_all, dim=0)
    metrics = compute_metrics_torch(pred_all, target_all, worst_ratio=TOPK_RATIO)
    metrics["loss"] = total_loss / max(1, total_n)
    for k, v in comp_sum.items():
        metrics[f"loss_{k}"] = v / max(1, total_n)
    return metrics


@torch.no_grad()
def compute_metrics_torch(pred: torch.Tensor, target: torch.Tensor, worst_ratio: float = 0.1) -> dict:
    err = pred - target
    err2 = err.pow(2)
    abs_err = err.abs()

    mse = err2.mean().item()
    rmse = math.sqrt(max(mse, 0.0))
    mae = abs_err.mean().item()

    diff1_mae = (first_diff(pred) - first_diff(target)).abs().mean().item()
    diff2_mae = (second_diff(pred) - second_diff(target)).abs().mean().item() if pred.shape[1] >= 3 else 0.0
    sam = spectral_angle_mapper(pred, target).mean().item()

    sample_mse = err2.mean(dim=1)
    k = max(1, int(sample_mse.numel() * worst_ratio))
    worst_mse = torch.topk(sample_mse, k=k, largest=True).values.mean().item()

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "sam": sam,
        "diff1_mae": diff1_mae,
        "diff2_mae": diff2_mae,
        "worst10_mse": worst_mse,
    }


def compute_metrics_np(pred: np.ndarray, target: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    err = pred - target
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(max(mse, 0.0)))
    mae = float(np.mean(np.abs(err)))

    # R2：所有样本与所有波长整体计算
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    # Pearson：整体拉平后计算相关系数
    p = pred.reshape(-1)
    t = target.reshape(-1)
    corr = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-12 and np.std(t) > 1e-12 else np.nan

    # SAM：逐样本光谱夹角
    dot = np.sum(pred * target, axis=1)
    pn = np.linalg.norm(pred, axis=1)
    tn = np.linalg.norm(target, axis=1)
    cos = dot / (pn * tn + 1e-12)
    sam = float(np.mean(np.arccos(np.clip(cos, -1.0 + 1e-7, 1.0 - 1e-7))))

    # SID：光谱信息散度，要求非负；这里做平移和归一化，主要用于相对对比
    pp = pred - np.min(pred, axis=1, keepdims=True) + 1e-8
    tt = target - np.min(target, axis=1, keepdims=True) + 1e-8
    pp = pp / (np.sum(pp, axis=1, keepdims=True) + 1e-12)
    tt = tt / (np.sum(tt, axis=1, keepdims=True) + 1e-12)
    sid = float(np.mean(np.sum(pp * np.log((pp + 1e-12) / (tt + 1e-12)), axis=1) +
                        np.sum(tt * np.log((tt + 1e-12) / (pp + 1e-12)), axis=1)))

    d1_mae = float(np.mean(np.abs(np.diff(pred, axis=1) - np.diff(target, axis=1))))
    d2_mae = float(np.mean(np.abs(np.diff(pred, n=2, axis=1) - np.diff(target, n=2, axis=1))))

    data_range = float(np.max(target) - np.min(target))
    if data_range < 1e-12:
        data_range = max(float(np.max(np.abs(target))), 1.0)
    psnr = float(20.0 * np.log10(data_range / (rmse + 1e-12)))

    sample_rmse = np.sqrt(np.mean(err ** 2, axis=1))
    sample_mae = np.mean(np.abs(err), axis=1)
    worst10_rmse = float(np.mean(np.sort(sample_rmse)[-max(1, int(len(sample_rmse) * 0.1)):]))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2),
        "pearson_corr": corr,
        "sam_rad": sam,
        "sid": sid,
        "diff1_mae": d1_mae,
        "diff2_mae": d2_mae,
        "psnr_db": psnr,
        "worst10_rmse": worst10_rmse,
        "sample_rmse_mean": float(np.mean(sample_rmse)),
        "sample_rmse_std": float(np.std(sample_rmse)),
        "sample_mae_mean": float(np.mean(sample_mae)),
        "sample_mae_std": float(np.std(sample_mae)),
    }


def compute_per_wavelength_metrics(pred: np.ndarray, target: np.ndarray, wavelengths: np.ndarray) -> pd.DataFrame:
    err = pred - target
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    bias = np.mean(err, axis=0)
    std_abs = np.std(np.abs(err), axis=0)

    r2_each = []
    for j in range(target.shape[1]):
        y = target[:, j]
        p = pred[:, j]
        ss_res = np.sum((p - y) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_each.append(1.0 - ss_res / (ss_tot + 1e-12))

    return pd.DataFrame({
        "wavelength_nm": wavelengths,
        "mae": mae,
        "rmse": rmse,
        "bias_pred_minus_true": bias,
        "abs_error_std": std_abs,
        "r2": np.asarray(r2_each),
    })


def make_loss_weight(train_ds, output_dim: int, alpha_peak: float):
    if build_peak_weight is None:
        print("[提示] build_peak_weight 不可用，使用全 1 权重。")
        return torch.ones(output_dim, dtype=torch.float32)
    try:
        return build_peak_weight(train_ds, output_dim=output_dim, alpha=alpha_peak).float()
    except Exception as e:
        print(f"[提示] build_peak_weight 失败，使用全 1 权重。原因：{e}")
        return torch.ones(output_dim, dtype=torch.float32)


def plot_single_run_history(history_df: pd.DataFrame, save_dir: Path):
    plt.figure(figsize=(8.5, 5.0))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Composite loss")
    plt.title("Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "training_loss_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9.5, 5.2))
    for col in ["val_mae", "val_rmse", "val_sam", "val_diff1_mae", "val_worst10_mse"]:
        if col in history_df.columns:
            plt.plot(history_df["epoch"], history_df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Validation Metrics During Training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "validation_metrics_curve.png", dpi=220)
    plt.close()


# =============================================================================
# 5. 单组训练
# =============================================================================
def train_one_run(block_count: int, block_indices_path: Path, seed: int, device: torch.device, save_root: Path) -> dict:
    seed_everything(seed)

    run_name = f"blocks_{block_count:02d}_seed_{seed}"
    run_dir = ensure_dir(save_root / run_name)
    pred_dir = ensure_dir(run_dir / "predictions")
    fig_dir = ensure_dir(run_dir / "figures")

    total_blocks = GRID_ROWS * GRID_COLS
    block_indices = load_block_indices(
        block_indices_path=str(block_indices_path),
        total_blocks=total_blocks,
    )
    if len(block_indices) != block_count:
        raise ValueError(f"{block_indices_path} 内索引数量为 {len(block_indices)}，但 block_count={block_count}")

    print("\n" + "=" * 90)
    print(f"[Start] {run_name}")
    print(f"[Blocks] {block_count}/{total_blocks}: {block_indices.tolist()}")
    print("=" * 90)

    ds_cfg = DatasetConfig(
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        sample_points=SAMPLE_POINTS,
        output_dim=OUTPUT_DIM,
        rgb_prefix=RGB_PREFIX,
        spectral_prefix=SPECTRAL_PREFIX,
        rgb_scale=RGB_SCALE,
        normalize_mode=NORMALIZE_MODE,
        clip_value=CLIP_VALUE,
        split_repeats=SPLIT_REPEATS,
    )

    train_ds = SpectralRGBSampleDataset(TRAIN_DIR, cfg=ds_cfg, block_indices=block_indices)
    val_ds = SpectralRGBSampleDataset(VAL_DIR, cfg=ds_cfg, block_indices=block_indices)

    train_ld = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model_cfg = {
        "out_dim": OUTPUT_DIM,
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "sample_points": SAMPLE_POINTS,
        "block_indices": block_indices.tolist(),
        "embed_dim": EMBED_DIM,
        "depth": DEPTH,
        "num_heads": NUM_HEADS,
        "mlp_ratio": MLP_RATIO,
        "dropout": DROPOUT,
        "out_activation": OUT_ACTIVATION,
    }
    model = SpectralTokenNet(**model_cfg).to(device)

    weight = make_loss_weight(train_ds, output_dim=OUTPUT_DIM, alpha_peak=ALPHA_PEAK).to(device)
    criterion = SpectralReconstructionLoss(
        weight=weight,
        lambda_diff1=LAMBDA_DIFF1,
        lambda_diff2=LAMBDA_DIFF2,
        lambda_cos=LAMBDA_COS,
        lambda_sam=LAMBDA_SAM,
        lambda_topk=LAMBDA_TOPK,
        lambda_dc=LAMBDA_DC,
        topk_ratio=TOPK_RATIO,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    run_cfg = {
        "run_name": run_name,
        "block_count": block_count,
        "seed": seed,
        "train_dir": TRAIN_DIR,
        "val_dir": VAL_DIR,
        "selected_block_indices": block_indices.tolist(),
        "wavelengths_nm": WAVELENGTHS.tolist(),
        "dataset_cfg": ds_cfg.__dict__,
        "model_cfg": model_cfg,
        "train_cfg": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "grad_clip_norm": GRAD_CLIP_NORM,
        },
        "loss_cfg": {
            "alpha_peak": ALPHA_PEAK,
            "lambda_diff1": LAMBDA_DIFF1,
            "lambda_diff2": LAMBDA_DIFF2,
            "lambda_cos": LAMBDA_COS,
            "lambda_sam": LAMBDA_SAM,
            "lambda_topk": LAMBDA_TOPK,
            "lambda_dc": LAMBDA_DC,
            "topk_ratio": TOPK_RATIO,
        },
        "train_dataset": train_ds.describe(),
        "val_dataset": val_ds.describe(),
    }
    save_json(run_cfg, run_dir / "run_config.json")

    history = []
    best_val = float("inf")
    best_epoch = -1
    wait = 0
    ckpt_path = run_dir / "best.pth"

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for x, y in train_ld:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()

            pred = model(x)
            loss = criterion(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is NaN/Inf @ {run_name}, epoch={ep}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

        train_loss = train_loss_sum / max(1, train_n)
        val_metrics = evaluate_loss_and_metrics(model, val_ld, criterion, device)
        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        row = {
            "run_name": run_name,
            "block_count": block_count,
            "seed": seed,
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_sam": val_metrics["sam"],
            "val_diff1_mae": val_metrics["diff1_mae"],
            "val_diff2_mae": val_metrics["diff2_mae"],
            "val_worst10_mse": val_metrics["worst10_mse"],
            "lr": lr_now,
            "epoch_time_s": elapsed,
        }
        history.append(row)

        print(
            f"[{run_name}] [{ep:03d}/{EPOCHS}] "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"mae={val_metrics['mae']:.6f}  rmse={val_metrics['rmse']:.6f}  "
            f"sam={val_metrics['sam']:.6f}  diff1={val_metrics['diff1_mae']:.6f}  "
            f"lr={lr_now:.2e}  time={elapsed:.1f}s"
        )

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = ep
            wait = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": ep,
                "best_val_loss": best_val,
                "run_cfg": run_cfg,
                "history": history,
            }, ckpt_path)
            print(f"  -> saved best: {ckpt_path}")
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                print(f"[Early Stop] {run_name}: no improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

        # 每个 epoch 保存一次日志，避免中断后丢失
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(run_dir / "training_history.csv", index=False)
        plot_single_run_history(hist_df, fig_dir)

    # 加载最优模型重新做完整验证集预测
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    pred_np, target_np, attn_np = predict_loader(model, val_ld, device, return_attention=True)

    val_ids = sample_ids_from_dataset(val_ds)
    np.save(pred_dir / "val_predictions.npy", pred_np)
    np.save(pred_dir / "val_targets.npy", target_np)
    if attn_np is not None:
        np.save(pred_dir / "val_block_attention.npy", attn_np)
    with open(pred_dir / "val_sample_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_ids))

    full_metrics = compute_metrics_np(pred_np, target_np)
    full_metrics.update({
        "run_name": run_name,
        "block_count": block_count,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "num_train_items": len(train_ds),
        "num_val_items": len(val_ds),
        "ckpt_path": str(ckpt_path),
    })

    per_wl = compute_per_wavelength_metrics(pred_np, target_np, WAVELENGTHS)
    per_wl.insert(0, "run_name", run_name)
    per_wl.insert(1, "block_count", block_count)
    per_wl.insert(2, "seed", seed)
    per_wl.to_csv(run_dir / "per_wavelength_metrics.csv", index=False)

    # 每个样本误差
    sample_rmse = np.sqrt(np.mean((pred_np - target_np) ** 2, axis=1))
    sample_mae = np.mean(np.abs(pred_np - target_np), axis=1)
    sample_df = pd.DataFrame({
        "sample_id": val_ids,
        "sample_rmse": sample_rmse,
        "sample_mae": sample_mae,
    })
    sample_df.to_csv(run_dir / "sample_error_metrics.csv", index=False)

    save_json(full_metrics, run_dir / "final_metrics.json")
    print(f"[Finished] {run_name} | best_epoch={best_epoch}, RMSE={full_metrics['rmse']:.6f}, MAE={full_metrics['mae']:.6f}")

    return full_metrics


# =============================================================================
# 6. 汇总可视化
# =============================================================================
def normalize_score_for_radar(df: pd.DataFrame, cols_low_better: List[str], cols_high_better: List[str]) -> pd.DataFrame:
    out = pd.DataFrame({"block_count": df["block_count"]})
    for c in cols_low_better:
        v = df[c].astype(float).to_numpy()
        lo, hi = np.nanmin(v), np.nanmax(v)
        if hi - lo < 1e-12:
            score = np.ones_like(v)
        else:
            score = 1.0 - (v - lo) / (hi - lo)
        out[c + "_score"] = score
    for c in cols_high_better:
        v = df[c].astype(float).to_numpy()
        lo, hi = np.nanmin(v), np.nanmax(v)
        if hi - lo < 1e-12:
            score = np.ones_like(v)
        else:
            score = (v - lo) / (hi - lo)
        out[c + "_score"] = score
    return out


def plot_summary_figures(save_root: Path):
    cmp_dir = ensure_dir(save_root / "comparison_figures")
    table_dir = ensure_dir(save_root / "comparison_tables")

    summary_runs = pd.read_csv(table_dir / "summary_runs.csv")
    summary_by = pd.read_csv(table_dir / "summary_by_blocks.csv")
    per_wl_all = pd.read_csv(table_dir / "per_wavelength_metrics_all.csv")
    sample_all = pd.read_csv(table_dir / "sample_error_metrics_all.csv")

    # 为展示默认取每个 block_count 的第一个 seed；若多 seed，summary_by 用均值展示。
    summary_by = summary_by.sort_values("block_count")

    # 1) 核心指标随色块数量变化
    metrics = ["rmse_mean", "mae_mean", "sam_rad_mean", "diff1_mae_mean", "sid_mean"]
    ylabels = ["RMSE", "MAE", "SAM (rad)", "First-diff MAE", "SID"]
    for m, yl in zip(metrics, ylabels):
        if m not in summary_by.columns:
            continue
        plt.figure(figsize=(7.5, 4.8))
        plt.plot(summary_by["block_count"], summary_by[m], marker="o", linewidth=2)
        if m.replace("_mean", "_std") in summary_by.columns:
            std_col = m.replace("_mean", "_std")
            if summary_by[std_col].notna().any() and summary_by[std_col].sum() > 0:
                plt.errorbar(summary_by["block_count"], summary_by[m], yerr=summary_by[std_col], fmt="none", capsize=4)
        plt.xlabel("Number of selected color blocks")
        plt.ylabel(yl)
        plt.title(f"{yl} vs. Number of Color Blocks")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(cmp_dir / f"metric_trend_{m}.png", dpi=260)
        plt.close()

    # 2) 综合指标趋势图：低误差指标 + 高拟合指标分开
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    ax = axes[0]
    for m, name in [("rmse_mean", "RMSE"), ("mae_mean", "MAE"), ("sam_rad_mean", "SAM"), ("diff1_mae_mean", "Diff1 MAE")]:
        if m in summary_by.columns:
            y = summary_by[m].to_numpy(dtype=float)
            y_norm = y / (np.nanmax(y) + 1e-12)
            ax.plot(summary_by["block_count"], y_norm, marker="o", label=name)
    ax.set_xlabel("Number of selected color blocks")
    ax.set_ylabel("Normalized metric (lower is better)")
    ax.set_title("Normalized Error Metrics")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for m, name in [("r2_mean", "R²"), ("pearson_corr_mean", "Pearson"), ("psnr_db_mean", "PSNR")]:
        if m in summary_by.columns:
            y = summary_by[m].to_numpy(dtype=float)
            if m == "psnr_db_mean":
                y_norm = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y) + 1e-12)
            else:
                y_norm = y
            ax.plot(summary_by["block_count"], y_norm, marker="o", label=name)
    ax.set_xlabel("Number of selected color blocks")
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Goodness-of-fit Metrics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(cmp_dir / "summary_metric_trends_combined.png", dpi=260)
    plt.close()

    # 3) 表格图，适合直接放 PPT
    display_cols = ["block_count", "rmse_mean", "mae_mean", "r2_mean", "pearson_corr_mean", "sam_rad_mean", "psnr_db_mean", "best_epoch_mean"]
    display_cols = [c for c in display_cols if c in summary_by.columns]
    table_df = summary_by[display_cols].copy()
    for c in table_df.columns:
        if c != "block_count":
            table_df[c] = table_df[c].map(lambda x: f"{x:.5g}" if pd.notnull(x) else "")
    fig, ax = plt.subplots(figsize=(12.5, 2.2 + 0.35 * len(table_df)))
    ax.axis("off")
    tbl = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.35)
    ax.set_title("Performance Summary of Different Color-block Numbers", pad=14)
    plt.tight_layout()
    plt.savefig(cmp_dir / "summary_metrics_table.png", dpi=260)
    plt.close()

    # 4) 雷达图：将低误差指标转成 score，高拟合指标保留高优
    radar_base = summary_by.rename(columns={
        "rmse_mean": "RMSE",
        "mae_mean": "MAE",
        "sam_rad_mean": "SAM",
        "diff1_mae_mean": "Diff1",
        "r2_mean": "R2",
        "pearson_corr_mean": "Corr",
    })
    low = [c for c in ["RMSE", "MAE", "SAM", "Diff1"] if c in radar_base.columns]
    high = [c for c in ["R2", "Corr"] if c in radar_base.columns]
    radar_score = normalize_score_for_radar(radar_base, low, high)
    labels = [c.replace("_score", "") for c in radar_score.columns if c != "block_count"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7.2, 7.2))
    ax = plt.subplot(111, polar=True)
    for _, row in radar_score.iterrows():
        vals = [row[c + "_score"] for c in labels]
        vals += vals[:1]
        ax.plot(angles, vals, marker="o", linewidth=1.8, label=f"{int(row['block_count'])} blocks")
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Normalized Comprehensive Performance Radar\n(higher is better)", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10))
    plt.tight_layout()
    plt.savefig(cmp_dir / "radar_normalized_performance.png", dpi=260)
    plt.close()

    # 5) 逐波长 MAE/RMSE 对比曲线
    for metric in ["mae", "rmse"]:
        plt.figure(figsize=(10.8, 5.2))
        for bc in sorted(per_wl_all["block_count"].unique()):
            sub = per_wl_all[per_wl_all["block_count"] == bc].copy()
            # 多 seed 时对每个波长取均值
            g = sub.groupby("wavelength_nm", as_index=False)[metric].mean()
            plt.plot(g["wavelength_nm"], g[metric], linewidth=1.8, label=f"{int(bc)} blocks")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel(metric.upper())
        plt.title(f"Per-wavelength {metric.upper()} Comparison")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3)
        plt.tight_layout()
        plt.savefig(cmp_dir / f"per_wavelength_{metric}_comparison.png", dpi=260)
        plt.close()

    # 6) 逐波长 bias 热力图
    pivot = per_wl_all.groupby(["block_count", "wavelength_nm"], as_index=False)["bias_pred_minus_true"].mean()
    mat = pivot.pivot(index="block_count", columns="wavelength_nm", values="bias_pred_minus_true")
    plt.figure(figsize=(12.0, 4.2))
    vmax = np.nanmax(np.abs(mat.values))
    plt.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax,
               extent=[mat.columns.min(), mat.columns.max(), mat.index.max() + 6, mat.index.min() - 6])
    plt.colorbar(label="Bias = Prediction - Ground truth")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Number of selected color blocks")
    plt.title("Per-wavelength Prediction Bias Heatmap")
    plt.tight_layout()
    plt.savefig(cmp_dir / "per_wavelength_bias_heatmap.png", dpi=260)
    plt.close()

    # 7) 样本级 RMSE 箱线图，展示稳定性与长尾误差
    plt.figure(figsize=(8.5, 5.0))
    data = []
    labels_bc = []
    for bc in sorted(sample_all["block_count"].unique()):
        data.append(sample_all.loc[sample_all["block_count"] == bc, "sample_rmse"].values)
        labels_bc.append(str(int(bc)))
    plt.boxplot(data, labels=labels_bc, showmeans=True)
    plt.xlabel("Number of selected color blocks")
    plt.ylabel("Sample-level RMSE")
    plt.title("Distribution of Validation Sample Errors")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(cmp_dir / "sample_rmse_boxplot.png", dpi=260)
    plt.close()

    # 8) 预测均值曲线：看整体谱形是否偏移
    plt.figure(figsize=(11.0, 5.4))
    target_saved = None
    for _, run in summary_runs.sort_values(["block_count", "seed"]).groupby("block_count").head(1).iterrows():
        bc = int(run["block_count"])
        run_name = run["run_name"]
        pred_path = save_root / run_name / "predictions" / "val_predictions.npy"
        target_path = save_root / run_name / "predictions" / "val_targets.npy"
        pred = np.load(pred_path)
        target = np.load(target_path)
        target_saved = target
        plt.plot(WAVELENGTHS, pred.mean(axis=0), linewidth=1.8, label=f"Pred mean · {bc} blocks")
    if target_saved is not None:
        plt.plot(WAVELENGTHS, target_saved.mean(axis=0), color="black", linewidth=2.6, linestyle="--", label="Ground truth mean")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectrum value")
    plt.title("Mean Spectrum Prediction Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(cmp_dir / "mean_spectrum_prediction_comparison.png", dpi=260)
    plt.close()

    # 9) 代表性困难样本曲线：选择平均 RMSE 最大的样本
    # 先按照第一组 run 的 sample_id 对齐；通常各组验证集顺序一致。
    first_run = summary_runs.sort_values(["block_count", "seed"]).iloc[0]
    ids_path = save_root / first_run["run_name"] / "predictions" / "val_sample_ids.txt"
    ids = [line.strip() for line in open(ids_path, "r", encoding="utf-8").read().splitlines()]
    err_stack = []
    pred_by_bc = {}
    target_ref = None
    for _, run in summary_runs.sort_values(["block_count", "seed"]).groupby("block_count").head(1).iterrows():
        bc = int(run["block_count"])
        run_name = run["run_name"]
        pred = np.load(save_root / run_name / "predictions" / "val_predictions.npy")
        target = np.load(save_root / run_name / "predictions" / "val_targets.npy")
        pred_by_bc[bc] = pred
        target_ref = target
        err_stack.append(np.sqrt(np.mean((pred - target) ** 2, axis=1)))
    if target_ref is not None:
        avg_rmse = np.mean(np.stack(err_stack, axis=0), axis=0)
        hard_idx = int(np.argmax(avg_rmse))
        median_idx = int(np.argsort(avg_rmse)[len(avg_rmse) // 2])
        for idx, tag in [(median_idx, "median_error_sample"), (hard_idx, "hardest_sample")]:
            plt.figure(figsize=(11.0, 5.4))
            plt.plot(WAVELENGTHS, target_ref[idx], color="black", linewidth=2.6, linestyle="--", label="Ground truth")
            for bc in sorted(pred_by_bc):
                plt.plot(WAVELENGTHS, pred_by_bc[bc][idx], linewidth=1.7, label=f"{bc} blocks")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Spectrum value")
            sid = ids[idx] if idx < len(ids) else str(idx)
            plt.title(f"Representative Prediction Curves · {tag}\nSample: {sid}")
            plt.grid(True, alpha=0.3)
            plt.legend(ncol=3)
            plt.tight_layout()
            plt.savefig(cmp_dir / f"representative_curves_{tag}.png", dpi=260)
            plt.close()

    # 10) 色块选择掩膜图：直观看到不同 K 保留哪些色块
    fig, axes = plt.subplots(len(BLOCK_SETUPS), 1, figsize=(10.5, 1.9 * len(BLOCK_SETUPS)))
    if len(BLOCK_SETUPS) == 1:
        axes = [axes]
    for ax, (bc, path) in zip(axes, BLOCK_SETUPS):
        idx = np.load(path)
        mask = np.zeros(GRID_ROWS * GRID_COLS, dtype=float)
        mask[idx] = 1.0
        mask = mask.reshape(GRID_ROWS, GRID_COLS)
        ax.imshow(mask, cmap="Greys", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"Selected color blocks: {bc}/72")
        ax.set_xticks(range(GRID_COLS))
        ax.set_yticks(range(GRID_ROWS))
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                v = int(r * GRID_COLS + c)
                ax.text(c, r, str(v), ha="center", va="center", fontsize=7,
                        color="white" if mask[r, c] > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(cmp_dir / "selected_block_masks_nested.png", dpi=260)
    plt.close()

    print(f"[Figures Saved] {cmp_dir}")


def make_summary_tables(save_root: Path, run_metrics: List[dict]):
    table_dir = ensure_dir(save_root / "comparison_tables")
    summary_runs = pd.DataFrame(run_metrics)
    summary_runs = summary_runs.sort_values(["block_count", "seed"]).reset_index(drop=True)
    summary_runs.to_csv(table_dir / "summary_runs.csv", index=False)

    # Excel 可选，方便复制进论文/汇报
    try:
        summary_runs.to_excel(table_dir / "summary_runs.xlsx", index=False)
    except Exception:
        pass

    metric_cols = [
        "mse", "rmse", "mae", "r2", "pearson_corr", "sam_rad", "sid",
        "diff1_mae", "diff2_mae", "psnr_db", "worst10_rmse",
        "sample_rmse_mean", "sample_rmse_std", "sample_mae_mean", "sample_mae_std",
        "best_epoch", "best_val_loss",
    ]
    agg_dict = {}
    for c in metric_cols:
        if c in summary_runs.columns:
            agg_dict[c] = ["mean", "std"]
    summary_by = summary_runs.groupby("block_count", as_index=False).agg(agg_dict)
    # flatten multi-index columns
    summary_by.columns = ["_".join([str(x) for x in col if x]).strip("_") for col in summary_by.columns.values]
    summary_by.to_csv(table_dir / "summary_by_blocks.csv", index=False)
    try:
        summary_by.to_excel(table_dir / "summary_by_blocks.xlsx", index=False)
    except Exception:
        pass

    # 合并逐波长与样本级误差
    per_wl_list = []
    sample_list = []
    for row in run_metrics:
        run_dir = save_root / row["run_name"]
        pw = pd.read_csv(run_dir / "per_wavelength_metrics.csv")
        per_wl_list.append(pw)
        sf = pd.read_csv(run_dir / "sample_error_metrics.csv")
        sf.insert(0, "run_name", row["run_name"])
        sf.insert(1, "block_count", row["block_count"])
        sf.insert(2, "seed", row["seed"])
        sample_list.append(sf)

    per_wl_all = pd.concat(per_wl_list, ignore_index=True)
    per_wl_all.to_csv(table_dir / "per_wavelength_metrics_all.csv", index=False)
    sample_all = pd.concat(sample_list, ignore_index=True)
    sample_all.to_csv(table_dir / "sample_error_metrics_all.csv", index=False)

    try:
        with pd.ExcelWriter(table_dir / "ablation_all_tables.xlsx") as writer:
            summary_runs.to_excel(writer, sheet_name="summary_runs", index=False)
            summary_by.to_excel(writer, sheet_name="summary_by_blocks", index=False)
            per_wl_all.to_excel(writer, sheet_name="per_wavelength", index=False)
            sample_all.to_excel(writer, sheet_name="sample_errors", index=False)
    except Exception:
        pass

    print(f"[Tables Saved] {table_dir}")


# =============================================================================
# 7. 主入口
# =============================================================================
def main():
    save_root = ensure_dir(SAVE_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)
    print("[Save Root]", save_root)

    # 检查 selected_blocks 文件
    for bc, p in BLOCK_SETUPS:
        if not Path(p).exists():
            raise FileNotFoundError(f"missing selected block file: {p}")
        idx = np.load(p)
        if len(idx) != bc:
            raise ValueError(f"{p} should contain {bc} indices, got {len(idx)}")

    global_cfg = {
        "train_dir": TRAIN_DIR,
        "val_dir": VAL_DIR,
        "save_root": str(save_root),
        "block_setups": [(bc, str(p)) for bc, p in BLOCK_SETUPS],
        "seeds": SEEDS,
        "data_shape": {
            "grid_rows": GRID_ROWS,
            "grid_cols": GRID_COLS,
            "sample_points": SAMPLE_POINTS,
            "output_dim": OUTPUT_DIM,
            "wavelengths_nm": WAVELENGTHS.tolist(),
        },
        "train_hyperparameters": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "embed_dim": EMBED_DIM,
            "depth": DEPTH,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT,
        },
    }
    save_json(global_cfg, save_root / "multi_block_ablation_config.json")

    run_metrics = []
    for bc, idx_path in BLOCK_SETUPS:
        for seed in SEEDS:
            m = train_one_run(bc, Path(idx_path), int(seed), device, save_root)
            run_metrics.append(m)
            make_summary_tables(save_root, run_metrics)
            plot_summary_figures(save_root)

    make_summary_tables(save_root, run_metrics)
    plot_summary_figures(save_root)
    print("\n" + "=" * 90)
    print("All block ablation training finished.")
    print("Tables:", save_root / "comparison_tables")
    print("Figures:", save_root / "comparison_figures")
    print("=" * 90)


if __name__ == "__main__":
    main()
