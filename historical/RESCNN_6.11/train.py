# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import DatasetConfig, SpectralMultiLightDataset
from model import SpectralTokenNet
from SpectralReconstructionLoss import SpectralReconstructionLoss, first_diff, spectral_angle_mapper
from WeightedMSEloss import build_peak_weight


# ========================= 路径配置：按需修改 =========================
TRAIN_DIR = r"D:\Desktop\zdataset-3.9\train"
VAL_DIR   = r"D:\Desktop\zdataset-3.9\val"
SAVE_DIR  = r"D:\Desktop\model\MLIGHT_8x72"

SAVE_CKPT = os.path.join(SAVE_DIR, "best.pth")
SAVE_LOG  = os.path.join(SAVE_DIR, "training_log.json")
SAVE_LOSS_FIG = os.path.join(SAVE_DIR, "loss_curve.png")
SAVE_METRIC_FIG = os.path.join(SAVE_DIR, "val_metrics.png")

# ========================= 数据维度配置（新） =========================
LIGHTS = 8
GRID_ROWS = 6
GRID_COLS = 12
SAMPLE_POINTS = 9
OUTPUT_DIM = 190
WAVELENGTHS = np.arange(380, 760, 2, dtype=np.float32)

FEAT_PREFIX = "group_"      # 你的聚合文件名：group_xxxx_....
FEAT_EXT    = ".npz"        # ".npz" 或 ".npy"
SPECTRAL_PREFIX = "spectral_"

RGB_SCALE = "auto"
NORMALIZE_MODE = "scale_only"
CLIP_VALUE = None
SPLIT_REPEATS = True        # 如果单个文件是 (R,3,8,72,9)

# ========================= 训练超参数 =========================
SEED = 42
BATCH_SIZE = 32            # token 变为 576，建议先用 32 或更小
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 30

# 模型超参数
EMBED_DIM = 96
DEPTH = 4                  # token多一点，depth可略加
NUM_HEADS = 4
MLP_RATIO = 4.0
DROPOUT = 0.10
OUT_ACTIVATION = "linear"

# 损失
ALPHA_PEAK = 3.0
LAMBDA_DIFF1 = 0.30
LAMBDA_DIFF2 = 0.05
LAMBDA_COS = 0.10
LAMBDA_SAM = 0.10
LAMBDA_TOPK = 0.20
LAMBDA_DC = 0.05
TOPK_RATIO = 0.10


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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


def train():
    seed_everything(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ds_cfg = DatasetConfig(
        lights=LIGHTS,
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        sample_points=SAMPLE_POINTS,
        output_dim=OUTPUT_DIM,
        feat_prefix=FEAT_PREFIX,
        spectral_prefix=SPECTRAL_PREFIX,
        feat_ext=FEAT_EXT,
        rgb_scale=RGB_SCALE,
        normalize_mode=NORMALIZE_MODE,
        clip_value=CLIP_VALUE,
        split_repeats=SPLIT_REPEATS,
    )

    train_ds = SpectralMultiLightDataset(TRAIN_DIR, cfg=ds_cfg)
    val_ds = SpectralMultiLightDataset(VAL_DIR, cfg=ds_cfg)

    print("[Train Dataset]", json.dumps(train_ds.describe(), ensure_ascii=False, indent=2)[:2000])
    print("[Val Dataset]", json.dumps(val_ds.describe(), ensure_ascii=False, indent=2)[:2000])

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=False)

    model_cfg = dict(
        out_dim=OUTPUT_DIM,
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        lights=LIGHTS,
        sample_points=SAMPLE_POINTS,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        out_activation=OUT_ACTIVATION,
    )
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    @torch.no_grad()
    def evaluate():
        model.eval()
        total_loss, total_n = 0.0, 0
        pred_all, tgt_all = [], []

        for x, y in val_ld:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            pred = model(x)
            loss = criterion(pred, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            pred_all.append(pred.detach())
            tgt_all.append(y.detach())

        pred_all = torch.cat(pred_all, dim=0)
        tgt_all = torch.cat(tgt_all, dim=0)
        metrics = compute_metrics(pred_all, tgt_all, worst_ratio=TOPK_RATIO)
        metrics["loss"] = total_loss / max(1, total_n)
        return metrics

    best = float("inf")
    wait = 0
    history = {k: [] for k in ["epoch","train_loss","val_loss","val_mae","val_rmse","val_sam","val_diff1_mae","val_worst10_mse","lr"]}

    run_cfg = {
        "wavelengths_nm": WAVELENGTHS.tolist(),
        "dataset_cfg": ds_cfg.__dict__,
        "model_cfg": model_cfg,
        "train_cfg": dict(batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, seed=SEED),
        "selected_token_count": LIGHTS * GRID_ROWS * GRID_COLS,
    }

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0

        for x, y in train_ld:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()

            pred = model(x)
            loss = criterion(pred, y)

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
        lr_now = optimizer.param_groups[0]["lr"]

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_sam"].append(val_metrics["sam"])
        history["val_diff1_mae"].append(val_metrics["diff1_mae"])
        history["val_worst10_mse"].append(val_metrics["worst10_mse"])
        history["lr"].append(lr_now)

        dt = time.time() - t0
        print(f"[{ep:03d}/{EPOCHS}] train={train_loss:.6f} val={val_loss:.6f} mae={val_metrics['mae']:.6f} rmse={val_metrics['rmse']:.6f} sam={val_metrics['sam']:.6f} lr={lr_now:.2e} time={dt:.1f}s")

        if val_loss < best - 1e-8:
            best = val_loss
            wait = 0
            torch.save({"model": model.state_dict(), "epoch": ep, "best_val_loss": best, "run_cfg": run_cfg, "history": history}, SAVE_CKPT)
            print("  -> saved best:", SAVE_CKPT)
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                print("[Early Stop]")
                break

        with open(SAVE_LOG, "w", encoding="utf-8") as f:
            json.dump({"run_cfg": run_cfg, "history": history}, f, ensure_ascii=False, indent=2)
        plot_history(history, SAVE_LOSS_FIG, SAVE_METRIC_FIG)

    print("========== Training Finished ==========")
    print("best_val_loss:", best)
    print("saved:", SAVE_CKPT)


if __name__ == "__main__":
    train()