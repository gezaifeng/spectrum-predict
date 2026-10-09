# -*- coding: utf-8 -*-
"""
TokenSpectralNet 训练脚本
------------------------
特点：
1) 直接适配 rgb_*.npy / spectral_*.npy 数据组织；
2) 兼容动态逐波长重加权；
3) 使用更适合光谱任务的 PhysicsSpectralLoss；
4) 保存 best checkpoint、训练日志、损失曲线、逐波长 MAE 曲线。
"""

from __future__ import annotations

import os
import math
import json
import random
from dataclasses import asdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset_token_spectral import SpectralDataset
from model_token_spectral_net import TokenSpectralNet
from physics_spectral_loss import PhysicsSpectralLoss
from WeightedMSEloss import build_peak_weight


# ========================= 参数配置 =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42

train_dir = r"D:\Desktop\dataset\guasnoise1-1020\train"
val_dir   = r"D:\Desktop\dataset\guasnoise1-1020\val"

save_dir  = r"D:\Desktop\model\token-spectral"
ckpt_path = os.path.join(save_dir, "best_token_spectral.pth")
last_path = os.path.join(save_dir, "last_token_spectral.pth")
log_path  = os.path.join(save_dir, "training_log_token_spectral.csv")
fig_path  = os.path.join(save_dir, "loss_curve_token_spectral.png")
mae_fig   = os.path.join(save_dir, "val_mae_by_wavelength_token_spectral.png")
config_js = os.path.join(save_dir, "config_token_spectral.json")

num_epochs = 50
batch_size = 64
num_workers = 0
learning_rate = 1e-3
weight_decay = 1e-4
early_stop_patience = 25

# 数据设置
normalize_rgb = False
normalize_spectra = False
patch_reduce = "mean"

# 模型设置
output_dim = 76
wavelengths = np.arange(380, 760, 5)
model_cfg = {
    "in_ch": 3,
    "grid_h": 4,
    "grid_w": 6,
    "out_dim": output_dim,
    "d_model": 128,
    "depth": 4,
    "decoder_depth": 2,
    "num_heads": 4,
    "mlp_ratio": 4.0,
    "drop": 0.1,
    "num_basis": 12,
    "out_activation": "linear",
    "nonnegative_coef": False,
    "use_patch_confidence": True,
}

# 损失设置
loss_cfg = {
    "lambda_dc": 0.05,
    "lambda_tv": 0.01,
    "lambda_curv": 0.01,
    "lambda_deriv": 0.05,
    "lambda_sam": 0.02,
    "lambda_nonneg": 0.0,
}

# 动态重加权
use_dynamic_reweight = True
reweight_every_epochs = 15
ema_beta = 0.7
dyn_scale_clamp = (0.5, 2.0)
dyn_mix_with_base = 0.6
alpha_peak = 3.0


# ========================= 工具函数 =========================
def set_seed(seed_value: int = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def sam_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = torch.nn.functional.normalize(pred, p=2, dim=1, eps=1e-8)
    targ_n = torch.nn.functional.normalize(target, p=2, dim=1, eps=1e-8)
    cos_sim = (pred_n * targ_n).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_sim).mean()


@torch.no_grad()
def compute_val_mae_by_wavelength(model: nn.Module, loader: DataLoader, dev: str) -> torch.Tensor:
    model.eval()
    mae_sum = torch.zeros(output_dim, device=dev)
    n = 0
    for images, spectra in loader:
        images = images.to(dev).float()
        spectra = spectra.to(dev).float()
        pred = model(images)
        mae_sum += (pred - spectra).abs().mean(dim=0)
        n += 1
    return (mae_sum / max(n, 1)).detach().cpu()


def make_dynamic_weight(base_w: torch.Tensor,
                        val_mae: torch.Tensor,
                        clamp_range=(0.5, 2.0),
                        mix_with_base=0.6) -> torch.Tensor:
    w_base = base_w.clone().detach().float()
    w_dyn = val_mae.clone().detach().float()
    w_dyn = w_dyn / (w_dyn.mean() + 1e-8)
    low, high = clamp_range
    w_dyn = torch.clamp(w_dyn, low, high)

    if mix_with_base >= 1.0:
        return w_base
    if mix_with_base <= 0.0:
        return w_base * w_dyn

    log_w = mix_with_base * torch.log(w_base + 1e-12) + (1.0 - mix_with_base) * torch.log(w_base * w_dyn + 1e-12)
    return torch.exp(log_w)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, dev: str):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_sam = 0.0
    n = 0
    for images, spectra in loader:
        images = images.to(dev).float()
        spectra = spectra.to(dev).float()
        pred = model(images)
        loss = criterion(pred, spectra)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_mse += torch.mean((pred - spectra) ** 2).item() * bs
        total_mae += torch.mean(torch.abs(pred - spectra)).item() * bs
        total_sam += sam_metric(pred, spectra).item() * bs
        n += bs

    avg_loss = total_loss / max(n, 1)
    avg_mse = total_mse / max(n, 1)
    avg_rmse = math.sqrt(max(avg_mse, 0.0))
    avg_mae = total_mae / max(n, 1)
    avg_sam = total_sam / max(n, 1)
    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "rmse": avg_rmse,
        "mae": avg_mae,
        "sam": avg_sam,
    }


# ========================= 主程序 =========================
def main():
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = SpectralDataset(
        train_dir,
        normalize_rgb=normalize_rgb,
        normalize_spectra=normalize_spectra,
        patch_reduce=patch_reduce,
    )
    val_dataset = SpectralDataset(
        val_dir,
        normalize_rgb=normalize_rgb,
        normalize_spectra=normalize_spectra,
        patch_reduce=patch_reduce,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = TokenSpectralNet(**model_cfg).to(device)

    base_weight = build_peak_weight(train_dataset, output_dim=output_dim, alpha=alpha_peak).to(device)
    criterion = PhysicsSpectralLoss(weight=base_weight, **loss_cfg).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, verbose=True)

    with open(config_js, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "device": device,
            "train_dir": train_dir,
            "val_dir": val_dir,
            "model_cfg": model_cfg,
            "loss_cfg": loss_cfg,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "normalize_rgb": normalize_rgb,
            "normalize_spectra": normalize_spectra,
            "patch_reduce": patch_reduce,
        }, f, ensure_ascii=False, indent=2)

    ema_val_mae = None
    best_val = float("inf")
    wait = 0
    mae_last_curve = None
    logs = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_mse": [], "val_rmse": [], "val_mae": [], "val_sam": [], "lr": []
    }

    for epoch in range(1, num_epochs + 1):
        if use_dynamic_reweight and (epoch == 1 or (epoch - 1) % reweight_every_epochs == 0):
            val_mae_curve = compute_val_mae_by_wavelength(model, val_loader, device)
            mae_last_curve = val_mae_curve.numpy().copy()
            if ema_val_mae is None:
                ema_val_mae = val_mae_curve.clone()
            else:
                ema_val_mae = ema_beta * ema_val_mae + (1.0 - ema_beta) * val_mae_curve
            new_w = make_dynamic_weight(
                base_weight.detach().cpu(),
                ema_val_mae,
                clamp_range=dyn_scale_clamp,
                mix_with_base=dyn_mix_with_base,
            )
            criterion.update_weight(new_w.to(device))
            print(f"[Reweight] epoch={epoch}, based on val MAE(λ)")

        model.train()
        total_train = 0.0
        for images, spectra in train_loader:
            images = images.to(device).float()
            spectra = spectra.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            pred = model(images)
            loss = criterion(pred, spectra)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train += loss.item()

        train_loss = total_train / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        logs["epoch"].append(epoch)
        logs["train_loss"].append(train_loss)
        logs["val_loss"].append(val_metrics["loss"])
        logs["val_mse"].append(val_metrics["mse"])
        logs["val_rmse"].append(val_metrics["rmse"])
        logs["val_mae"].append(val_metrics["mae"])
        logs["val_sam"].append(val_metrics["sam"])
        logs["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:03d}/{num_epochs}] "
            f"Train {train_loss:.6f} | "
            f"Val {val_metrics['loss']:.6f} | "
            f"MAE {val_metrics['mae']:.6f} | "
            f"RMSE {val_metrics['rmse']:.6f} | "
            f"SAM {val_metrics['sam']:.6f} | "
            f"LR {current_lr:.6e}"
        )

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
            "model_cfg": model_cfg,
            "normalize_rgb": normalize_rgb,
            "normalize_spectra": normalize_spectra,
            "patch_reduce": patch_reduce,
            "wavelengths": wavelengths.tolist(),
        }
        torch.save(ckpt, last_path)

        if val_metrics["loss"] < best_val - 1e-8:
            best_val = val_metrics["loss"]
            wait = 0
            torch.save(ckpt, ckpt_path)
            print(f"[Save] best checkpoint -> {ckpt_path}")
        else:
            wait += 1

        scheduler.step(val_metrics["loss"])
        if wait >= early_stop_patience:
            print("[Early Stop] triggered.")
            break

    df = pd.DataFrame(logs)
    df.to_csv(log_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TokenSpectralNet Training Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    if mae_last_curve is not None:
        mae_norm = mae_last_curve / (mae_last_curve.mean() + 1e-8)
        plt.figure(figsize=(10, 4.5))
        plt.plot(wavelengths, mae_norm, marker='o')
        plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Val MAE (normalized)")
        plt.title("Validation MAE by Wavelength")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(mae_fig, dpi=200)
        plt.close()

    print(f"[Done] best={ckpt_path}")
    print(f"[Done] last={last_path}")
    print(f"[Done] log={log_path}")
    print(f"[Done] fig={fig_path}")


if __name__ == "__main__":
    main()
