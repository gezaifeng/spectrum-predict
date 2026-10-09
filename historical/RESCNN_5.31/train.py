# -*- coding: utf-8 -*-
"""
train.py
用于新数据格式的训练脚本：
    输入 RGB:      (3,6,12,9)
    输出光谱:      190 点，默认 wavelengths = 380,382,...,758 nm

使用方法：
1. 修改下方路径 TRAIN_DIR / VAL_DIR / SAVE_DIR。
2. 默认使用全部 72 个色块。
3. 做色块数量消融时，修改 KEEP_BLOCKS 或 BLOCK_INDICES_PATH：
      KEEP_BLOCKS = 36 / 48 / 60 / None
   更严谨的方式是使用 BLOCK_INDICES_PATH 指定你筛选出的色块索引。
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import DatasetConfig, SpectralRGBSampleDataset, load_block_indices
from model import SpectralTokenNet
from SpectralReconstructionLoss import SpectralReconstructionLoss, first_diff, spectral_angle_mapper
from WeightedMSEloss import build_peak_weight


# ========================= 路径配置：按需修改 =========================
TRAIN_DIR = r"D:\Desktop\zdataset-3.9\train"
VAL_DIR   = r"D:\Desktop\zdataset-3.9\val"
SAVE_DIR  = r"D:\Desktop\model\RESCNN_5.31"

SAVE_CKPT = os.path.join(SAVE_DIR, "best.pth")
SAVE_LOG  = os.path.join(SAVE_DIR, "training_log.json")
SAVE_LOSS_FIG = os.path.join(SAVE_DIR, "loss_curve.png")
SAVE_METRIC_FIG = os.path.join(SAVE_DIR, "val_metrics.png")

# ========================= 数据维度配置 =========================
GRID_ROWS = 6
GRID_COLS = 12
SAMPLE_POINTS = 9
OUTPUT_DIM = 190
WAVELENGTHS = np.arange(380, 760, 2, dtype=np.float32)  # 190 点：380~758 nm

# 文件名前缀：默认 rgb_xxxx.npy / spectral_xxxx.npy
RGB_PREFIX = "rgb_"
SPECTRAL_PREFIX = "spectral_"

# 数据缩放与归一化
# 新采集 RGB 一般建议：rgb_scale="auto"，normalize_mode="scale_only"
# 如果你后续输入的是 log_ratio/ratio 特征，可对比 normalize_mode="channel_zscore"
RGB_SCALE = "auto"              # "auto" | "255" | "none"
NORMALIZE_MODE = "scale_only"   # "scale_only" | "channel_zscore" | "global_zscore"
CLIP_VALUE = None
SPLIT_REPEATS = True             # 若单个文件是 (10,3,6,12,9)，拆成 10 个训练样本

# ========================= 色块数量消融接口 =========================
# 默认 None = 使用全部 72 个色块。
# 快速消融：KEEP_BLOCKS = 36 / 48 / 60。
# 严谨消融：BLOCK_INDICES_PATH = r"D:\...\selected_36.npy"，其中保存 0~71 的色块索引。
KEEP_BLOCKS: Optional[int] = None
BLOCK_INDICES_PATH: Optional[str] = None

# ========================= 训练超参数 =========================
SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
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
OUT_ACTIVATION = "linear"       # 吸光度若严格非负，可对比 "softplus"

# 损失超参数
ALPHA_PEAK = 3.0
LAMBDA_DIFF1 = 0.30
LAMBDA_DIFF2 = 0.05
LAMBDA_COS = 0.10
LAMBDA_SAM = 0.10
LAMBDA_TOPK = 0.20
LAMBDA_DC = 0.05
TOPK_RATIO = 0.10


# ========================= 工具函数 =========================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def extract_sample_ids(file_list, rgb_prefix="rgb_"):
    ids = []
    pattern = re.compile(rf"{re.escape(rgb_prefix)}(.+)\.npy$")
    for f in file_list:
        m = pattern.match(f)
        if m:
            ids.append(m.group(1))
    return set(ids)


def inspect_train_val_overlap(train_ds, val_ds):
    train_ids = extract_sample_ids(train_ds.rgb_files, train_ds.cfg.rgb_prefix)
    val_ids = extract_sample_ids(val_ds.rgb_files, val_ds.cfg.rgb_prefix)
    overlap = sorted(train_ids & val_ids)
    if overlap:
        print("[警告] train/val 发现同名样本ID重叠，可能存在数据泄漏。")
        print("       overlap sample ids (前20个):", overlap[:20])
    else:
        print("[检查] train/val 未发现同名样本ID重叠。")


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor, worst_ratio: float = 0.1):
    err = pred - target
    err2 = err.pow(2)
    abs_err = err.abs()

    mse = err2.mean().item()
    rmse = math.sqrt(max(mse, 0.0))
    mae = abs_err.mean().item()

    diff1_mae = (first_diff(pred) - first_diff(target)).abs().mean().item()
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
        "worst10_mse": worst_mse,
    }


def plot_history(history: dict, save_loss_fig: str, save_metric_fig: str):
    os.makedirs(os.path.dirname(save_loss_fig), exist_ok=True)

    plt.figure(figsize=(9, 4.8))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Composite Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_loss_fig, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5.2))
    plt.plot(history["epoch"], history["val_mae"], label="val_mae")
    plt.plot(history["epoch"], history["val_rmse"], label="val_rmse")
    plt.plot(history["epoch"], history["val_sam"], label="val_sam")
    plt.plot(history["epoch"], history["val_diff1_mae"], label="val_diff1_mae")
    plt.plot(history["epoch"], history["val_worst10_mse"], label="val_worst10_mse")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_metric_fig, dpi=200)
    plt.close()


# ========================= 主训练 =========================
def train():
    seed_everything(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    total_blocks = GRID_ROWS * GRID_COLS
    block_indices = load_block_indices(
        block_indices=None,
        block_indices_path=BLOCK_INDICES_PATH,
        keep_blocks=KEEP_BLOCKS,
        total_blocks=total_blocks,
    )
    selected_blocks = total_blocks if block_indices is None else len(block_indices)
    print(f"[Blocks] selected {selected_blocks}/{total_blocks}")
    if block_indices is not None:
        print("[Blocks] indices:", block_indices.tolist())

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
    inspect_train_val_overlap(train_ds, val_ds)

    print("[Train Dataset]", json.dumps(train_ds.describe(), ensure_ascii=False, indent=2)[:2000])
    print("[Val Dataset]", json.dumps(val_ds.describe(), ensure_ascii=False, indent=2)[:2000])

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
        "block_indices": None if block_indices is None else block_indices.tolist(),
        "embed_dim": EMBED_DIM,
        "depth": DEPTH,
        "num_heads": NUM_HEADS,
        "mlp_ratio": MLP_RATIO,
        "dropout": DROPOUT,
        "out_activation": OUT_ACTIVATION,
    }
    model = SpectralTokenNet(**model_cfg).to(device)

    weight = build_peak_weight(train_ds, output_dim=OUTPUT_DIM, alpha=ALPHA_PEAK).to(device)
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

    @torch.no_grad()
    def evaluate():
        model.eval()
        total_loss = 0.0
        total_n = 0
        pred_all, target_all = [], []
        comp_sum = {}

        for x, y in val_ld:
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
        metrics = compute_metrics(pred_all, target_all, worst_ratio=TOPK_RATIO)
        metrics["loss"] = total_loss / max(1, total_n)
        for k, v in comp_sum.items():
            metrics[f"loss_{k}"] = v / max(1, total_n)
        return metrics

    best_score = float("inf")
    wait = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_sam": [],
        "val_diff1_mae": [],
        "val_worst10_mse": [],
        "lr": [],
    }

    run_cfg = {
        "wavelengths_nm": WAVELENGTHS.tolist(),
        "dataset_cfg": ds_cfg.__dict__,
        "model_cfg": model_cfg,
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
        "train_cfg": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "seed": SEED,
        },
        "selected_block_indices": train_ds.get_selected_block_indices().tolist(),
    }

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0

        for x, y in train_ld:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()

            if not torch.isfinite(x).all():
                raise RuntimeError(f"x has NaN/Inf, range=({x.min().item():.3g},{x.max().item():.3g})")
            if not torch.isfinite(y).all():
                raise RuntimeError(f"y has NaN/Inf, range=({y.min().item():.3g},{y.max().item():.3g})")

            pred = model(x)
            loss = criterion(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError("loss is NaN/Inf")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            bs = x.size(0)
            loss_sum += loss.item() * bs
            n += bs

        train_loss = loss_sum / max(1, n)
        val_metrics = evaluate()
        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_sam"].append(val_metrics["sam"])
        history["val_diff1_mae"].append(val_metrics["diff1_mae"])
        history["val_worst10_mse"].append(val_metrics["worst10_mse"])
        history["lr"].append(cur_lr)

        print(
            f"[{ep:03d}/{EPOCHS}] "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"mae={val_metrics['mae']:.6f}  rmse={val_metrics['rmse']:.6f}  "
            f"sam={val_metrics['sam']:.6f}  diff1_mae={val_metrics['diff1_mae']:.6f}  "
            f"worst10_mse={val_metrics['worst10_mse']:.6f}  "
            f"lr={cur_lr:.2e}  time={dt:.1f}s"
        )

        if val_loss < best_score - 1e-8:
            best_score = val_loss
            wait = 0
            ckpt = {
                "model": model.state_dict(),
                "epoch": ep,
                "best_val_loss": best_score,
                "run_cfg": run_cfg,
                "history": history,
            }
            torch.save(ckpt, SAVE_CKPT)
            print(f"  -> saved best to {SAVE_CKPT}")
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                print(f"[Early Stop] no improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

        with open(SAVE_LOG, "w", encoding="utf-8") as f:
            json.dump({"run_cfg": run_cfg, "history": history}, f, ensure_ascii=False, indent=2)
        plot_history(history, SAVE_LOSS_FIG, SAVE_METRIC_FIG)

    print("========== Training Finished ==========")
    print("best_val_loss:", best_score)
    print("saved:", SAVE_CKPT)
    print("saved:", SAVE_LOG)
    print("saved:", SAVE_LOSS_FIG)
    print("saved:", SAVE_METRIC_FIG)


if __name__ == "__main__":
    train()
