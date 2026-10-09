# -*- coding: utf-8 -*-
from __future__ import annotations

import glob
import json
import os
import math
import csv
from dataclasses import fields
from typing import Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import DatasetConfig, _load_feat, _canonical_single_feat, _preprocess_feat
from model import SpectralTokenNet
from SpectralReconstructionLoss import first_diff, spectral_angle_mapper


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = r"D:\Desktop\zdataset-3.9\val"
MODEL_PATH = r"D:\Desktop\model\MLIGHT_8x72\best.pth"
SAVE_DIR = r"D:\Desktop\model\MLIGHT_8x72\predict_vis"


def _dataset_cfg_from_dict(d: dict) -> DatasetConfig:
    valid = {f.name for f in fields(DatasetConfig)}
    kwargs = {k: v for k, v in d.items() if k in valid}
    return DatasetConfig(**kwargs)


def load_model_and_cfg(model_path: str, device: str = DEVICE):
    ckpt = torch.load(model_path, map_location="cpu")
    run_cfg = ckpt.get("run_cfg", {})
    model_cfg = run_cfg.get("model_cfg", {})
    dataset_cfg = _dataset_cfg_from_dict(run_cfg.get("dataset_cfg", {}))
    wavelengths = np.asarray(run_cfg.get("wavelengths_nm", np.arange(380, 760, 2)), dtype=np.float32)

    model = SpectralTokenNet(**model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, dataset_cfg, wavelengths, run_cfg


def _preprocess_one_feat_path(feat_path: str, cfg: DatasetConfig) -> torch.Tensor:
    arr = _load_feat(feat_path)

    # predict 时遇到 (R,3,L,N,K)：默认对 R 平均
    if arr.ndim == 5:
        xs = []
        for i in range(arr.shape[0]):
            x_i = _canonical_single_feat(arr[i], cfg)
            x_i = _preprocess_feat(x_i, cfg)
            xs.append(x_i)
        x = np.mean(np.stack(xs, axis=0), axis=0)  # (3,L,N,K)
    else:
        x = _canonical_single_feat(arr, cfg)
        x = _preprocess_feat(x, cfg)

    c, L, N, K = x.shape
    x = x.reshape(c, L * N, K)
    return torch.from_numpy(x).unsqueeze(0).float()  # (1,3,T,K)


@torch.no_grad()
def predict_one(model, feat_path: str, dataset_cfg: DatasetConfig, device: str = DEVICE) -> np.ndarray:
    x = _preprocess_one_feat_path(feat_path, dataset_cfg).to(device)
    pred = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return pred


def _load_optional_gt(feat_path: str, dataset_cfg: DatasetConfig) -> Optional[np.ndarray]:
    base = os.path.basename(feat_path)
    if not base.startswith(dataset_cfg.feat_prefix):
        return None
    sp_name = dataset_cfg.spectral_prefix + base[len(dataset_cfg.feat_prefix):]
    sp_name = sp_name.replace(dataset_cfg.feat_ext, ".npy")
    sp_path = os.path.join(os.path.dirname(feat_path), sp_name)
    if not os.path.exists(sp_path):
        return None
    y = np.load(sp_path).astype(np.float32).reshape(-1)
    if y.shape[0] != dataset_cfg.output_dim:
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
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def predict_and_visualize_all(data_dir: str = DATA_DIR, model_path: str = MODEL_PATH, save_dir: str = SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model, dataset_cfg, wavelengths, run_cfg = load_model_and_cfg(model_path, DEVICE)

    print("[Loaded Config]")
    print(json.dumps(run_cfg, ensure_ascii=False, indent=2)[:2000])

    pattern = os.path.join(data_dir, f"{dataset_cfg.feat_prefix}*{dataset_cfg.feat_ext}")
    feat_list = sorted(glob.glob(pattern))
    if len(feat_list) == 0:
        raise FileNotFoundError(f"No {pattern} found")

    metrics_csv = os.path.join(save_dir, "prediction_metrics.csv")
    pred_dir = os.path.join(save_dir, "pred_npy")
    os.makedirs(pred_dir, exist_ok=True)

    rows = []
    for i, feat_path in enumerate(feat_list, 1):
        base = os.path.basename(feat_path)
        sample_id = base.replace(dataset_cfg.feat_prefix, "").replace(dataset_cfg.feat_ext, "")
        pred = predict_one(model, feat_path, dataset_cfg, DEVICE)
        gt = _load_optional_gt(feat_path, dataset_cfg)

        np.save(os.path.join(pred_dir, f"pred_{sample_id}.npy"), pred.astype(np.float32))
        fig_path = os.path.join(save_dir, f"{sample_id}.png")
        save_curve_figure(wavelengths, pred, gt, sample_id, fig_path)

        if gt is not None:
            metrics = compute_metrics_np(pred, gt)
            rows.append({"sample_id": sample_id, "feat_path": feat_path, **metrics})

        print(f"[{i:04d}/{len(feat_list)}] {sample_id}")

    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(metrics_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    print("Done. Output:", save_dir)


if __name__ == "__main__":
    predict_and_visualize_all(DATA_DIR, MODEL_PATH, SAVE_DIR)