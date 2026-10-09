# -*- coding: utf-8 -*-
"""
predict.py
加载 train.py 保存的 best.pth，对 rgb_*.npy 进行批量预测与可视化。

输入支持：
  rgb_xxxx.npy      -> (3,6,12,9)
  spectral_xxxx.npy -> (190,) 可选；若存在则计算指标并画真值/预测对比。
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from dataclasses import fields
from typing import Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import DatasetConfig, _canonical_single_rgb, _preprocess_rgb
from model import SpectralTokenNet
from SpectralReconstructionLoss import first_diff, spectral_angle_mapper


# ===================== 路径配置：按需修改 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = r"D:\Desktop\zdataset-3.9\val"
MODEL_PATH = r"D:\Desktop\model\RESCNN_5.31\best.pth"
SAVE_DIR = r"D:\Desktop\model\RESCNN_5.31\predict_vis"
# =============================================================


def _dataset_cfg_from_dict(d: dict) -> DatasetConfig:
    valid = {f.name for f in fields(DatasetConfig)}
    kwargs = {k: v for k, v in d.items() if k in valid}
    return DatasetConfig(**kwargs)


def load_model_and_cfg(model_path: str, device: str = DEVICE):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("This predict.py expects checkpoint saved by the new train.py, containing keys: model/run_cfg")

    run_cfg = ckpt.get("run_cfg", {})
    model_cfg = run_cfg.get("model_cfg", {})
    dataset_cfg = _dataset_cfg_from_dict(run_cfg.get("dataset_cfg", {}))
    wavelengths = np.asarray(run_cfg.get("wavelengths_nm", np.arange(380, 760, 2)), dtype=np.float32)

    model = SpectralTokenNet(**model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, dataset_cfg, wavelengths, run_cfg


def _preprocess_one_rgb_path(rgb_path: str, cfg: DatasetConfig, block_indices: Optional[np.ndarray]) -> torch.Tensor:
    arr = np.load(rgb_path)

    # predict 阶段如果遇到 5D，比如 (10,3,6,12,9)，默认对 10 次预测取平均输入。
    if arr.ndim == 5:
        xs = []
        for i in range(arr.shape[0]):
            x_i = _canonical_single_rgb(arr[i], cfg)
            x_i = _preprocess_rgb(x_i, cfg)
            xs.append(x_i)
        x = np.mean(np.stack(xs, axis=0), axis=0)  # (3,R,C,K)
    else:
        x = _canonical_single_rgb(arr, cfg)
        x = _preprocess_rgb(x, cfg)

    c, r, col, k = x.shape
    x = x.reshape(c, r * col, k)
    if block_indices is not None:
        x = x[:, block_indices, :]
    return torch.from_numpy(x).unsqueeze(0).float()  # (1,3,N,K)


@torch.no_grad()
def predict_one(model, rgb_path: str, dataset_cfg: DatasetConfig, run_cfg: dict, device: str = DEVICE) -> np.ndarray:
    selected = np.asarray(run_cfg.get("selected_block_indices", []), dtype=np.int64)
    total = dataset_cfg.grid_rows * dataset_cfg.grid_cols
    block_indices = None if selected.size == 0 or selected.size == total else selected
    x = _preprocess_one_rgb_path(rgb_path, dataset_cfg, block_indices).to(device)
    pred = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return pred


def _load_optional_gt(rgb_path: str, dataset_cfg: DatasetConfig) -> Optional[np.ndarray]:
    base = os.path.basename(rgb_path)
    if not base.startswith(dataset_cfg.rgb_prefix):
        return None
    sp_name = dataset_cfg.spectral_prefix + base[len(dataset_cfg.rgb_prefix):]
    sp_path = os.path.join(os.path.dirname(rgb_path), sp_name)
    if not os.path.exists(sp_path):
        return None
    y = np.load(sp_path).astype(np.float32).reshape(-1)
    if y.shape[0] != dataset_cfg.output_dim:
        print(f"[Skip GT] {sp_path} dim={y.shape}, expected {dataset_cfg.output_dim}")
        return None
    return y


@torch.no_grad()
def compute_metrics_np(pred: np.ndarray, target: np.ndarray):
    pred_t = torch.from_numpy(pred).float().unsqueeze(0)
    tgt_t = torch.from_numpy(target).float().unsqueeze(0)

    mse = float(np.mean((pred - target) ** 2))
    rmse = math.sqrt(max(mse, 0.0))
    mae = float(np.mean(np.abs(pred - target)))
    sam = float(spectral_angle_mapper(pred_t, tgt_t).mean().item())
    diff1_mae = float((first_diff(pred_t) - first_diff(tgt_t)).abs().mean().item())
    return {"mse": mse, "rmse": rmse, "mae": mae, "sam": sam, "diff1_mae": diff1_mae}


def save_curve_figure(wavelengths, pred, gt, sample_name: str, save_path: str):
    plt.figure(figsize=(10, 5.5))
    if gt is not None:
        plt.plot(wavelengths, gt, label="Ground Truth", linewidth=2)
    plt.plot(wavelengths, pred, label="Prediction", linewidth=2, linestyle="--")

    if gt is not None:
        metrics = compute_metrics_np(pred, gt)
        txt = (
            f"MSE: {metrics['mse']:.6f}\n"
            f"RMSE: {metrics['rmse']:.6f}\n"
            f"MAE: {metrics['mae']:.6f}\n"
            f"SAM: {metrics['sam']:.6f}\n"
            f"diff1_MAE: {metrics['diff1_mae']:.6f}"
        )
        plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment="top", bbox=dict(facecolor="white", alpha=0.75))
    else:
        metrics = {}

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectrum Value")
    plt.title(f"Spectral Prediction - {sample_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return metrics


def predict_and_visualize_all(data_dir: str = DATA_DIR, model_path: str = MODEL_PATH, save_dir: str = SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model, dataset_cfg, wavelengths, run_cfg = load_model_and_cfg(model_path, DEVICE)

    print("[Loaded Config]")
    print(json.dumps(run_cfg, ensure_ascii=False, indent=2)[:4000])

    pattern = os.path.join(data_dir, f"{dataset_cfg.rgb_prefix}*.npy")
    rgb_list = sorted(glob.glob(pattern))
    if len(rgb_list) == 0:
        raise FileNotFoundError(f"No {dataset_cfg.rgb_prefix}*.npy found in {data_dir}")

    metrics_csv = os.path.join(save_dir, "prediction_metrics.csv")
    pred_dir = os.path.join(save_dir, "pred_npy")
    os.makedirs(pred_dir, exist_ok=True)

    rows = []
    ok, bad = 0, 0
    for i, rgb_path in enumerate(rgb_list, 1):
        try:
            base = os.path.basename(rgb_path)
            sample_id = base.replace(dataset_cfg.rgb_prefix, "").replace(".npy", "")
            pred = predict_one(model, rgb_path, dataset_cfg, run_cfg, DEVICE)
            gt = _load_optional_gt(rgb_path, dataset_cfg)

            np.save(os.path.join(pred_dir, f"pred_{sample_id}.npy"), pred.astype(np.float32))
            fig_path = os.path.join(save_dir, f"{sample_id}.png")
            metrics = save_curve_figure(wavelengths, pred, gt, sample_id, fig_path)

            row = {"sample_id": sample_id, "rgb_path": rgb_path, **metrics}
            rows.append(row)
            ok += 1
            if metrics:
                print(
                    f"[{i:04d}/{len(rgb_list)}] {sample_id} | "
                    f"RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f} SAM={metrics['sam']:.6f}"
                )
            else:
                print(f"[{i:04d}/{len(rgb_list)}] {sample_id} | predicted")
        except Exception as e:
            print(f"[Error] {rgb_path} -> {e}")
            bad += 1

    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(metrics_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    print("\n========== 完成 ==========")
    print(f"总样本数: {len(rgb_list)}")
    print(f"成功: {ok}")
    print(f"失败: {bad}")
    print(f"输出目录: {save_dir}")


if __name__ == "__main__":
    predict_and_visualize_all(DATA_DIR, MODEL_PATH, SAVE_DIR)
