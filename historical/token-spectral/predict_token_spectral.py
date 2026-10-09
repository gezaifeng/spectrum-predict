# -*- coding: utf-8 -*-
"""
TokenSpectralNet 预测脚本
------------------------
功能：
1) 加载单个 rgb_*.npy 并预测光谱；
2) 若提供真值，可计算 MSE / RMSE / MAE / R² / SAM；
3) 可视化预测曲线；
4) 可将结果导出到 Excel。
"""

from __future__ import annotations

import os
import math
import glob
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataset_token_spectral import SpectralDataset
from model_token_spectral_net import TokenSpectralNet


# ===================== 路径与设备（按需修改） =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
val_dir = r"D:\Desktop\dataset\guasnoise1-1020\val"
model_path = r"D:\Desktop\model\token-spectral\best_token_spectral.pth"
WAVELENGTHS = np.arange(380, 760, 5)
# ===============================================================


def spectral_angle_mapper_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    pred_norm = pred / (np.linalg.norm(pred) + 1e-12)
    target_norm = target / (np.linalg.norm(target) + 1e-12)
    cos_sim = np.clip(np.sum(pred_norm * target_norm), -1.0 + 1e-7, 1.0 - 1e-7)
    return float(np.arccos(cos_sim))



def load_model(model_path: str) -> tuple[torch.nn.Module, dict]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weight not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_cfg" in ckpt and "model" in ckpt:
        model_cfg = ckpt["model_cfg"]
        state_dict = ckpt["model"]
    else:
        # 兼容只保存 state_dict 的情况
        model_cfg = {
            "in_ch": 3,
            "grid_h": 4,
            "grid_w": 6,
            "out_dim": 76,
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
        state_dict = ckpt

    model = TokenSpectralNet(**model_cfg).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}


@torch.no_grad()
def predict(test_rgb_path: str, dataset_for_preprocess: Optional[SpectralDataset] = None) -> np.ndarray:
    model, _ = load_model(model_path)

    if dataset_for_preprocess is None:
        dataset_for_preprocess = SpectralDataset(
            val_dir,
            normalize_rgb=False,
            normalize_spectra=False,
            patch_reduce="mean",
        )

    x = dataset_for_preprocess._load_rgb(test_rgb_path)  # [3,H,W]
    if dataset_for_preprocess.normalize_rgb and dataset_for_preprocess.rgb_mean is not None:
        x = (x - dataset_for_preprocess.rgb_mean) / dataset_for_preprocess.rgb_std

    x = torch.from_numpy(x).unsqueeze(0).float().to(device)
    pred = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)

    if dataset_for_preprocess.normalize_spectra:
        pred = dataset_for_preprocess.inverse_transform_spectrum(pred)

    return pred



def evaluate_prediction(true_spectrum: np.ndarray, predicted_spectrum: np.ndarray) -> dict:
    y_true = true_spectrum.astype(np.float32)
    y_pred = predicted_spectrum.astype(np.float32)

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    sam = spectral_angle_mapper_np(y_pred, y_true)

    max_val = max(float(y_true.max()), float(y_pred.max()), 1e-6)
    psnr = 20.0 * math.log10(max_val / (rmse + 1e-12))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "sam": sam,
        "psnr": psnr,
    }



def visualize_prediction(true_spectrum: Optional[np.ndarray], predicted_spectrum: np.ndarray, title: str = "TokenSpectralNet Prediction"):
    plt.figure(figsize=(10, 5.5))

    if true_spectrum is not None:
        metrics = evaluate_prediction(true_spectrum, predicted_spectrum)
        plt.plot(WAVELENGTHS, true_spectrum, label="Ground Truth", linewidth=2)
        metrics_text = (
            f"MSE: {metrics['mse']:.4f}\n"
            f"RMSE: {metrics['rmse']:.4f}\n"
            f"MAE: {metrics['mae']:.4f}\n"
            f"PSNR: {metrics['psnr']:.2f} dB\n"
            f"R²: {metrics['r2']:.4f}\n"
            f"SAM: {metrics['sam']:.4f}"
        )
        plt.text(
            0.02, 0.98, metrics_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.6)
        )

    plt.plot(WAVELENGTHS, predicted_spectrum, label="Prediction", linewidth=2, linestyle="--")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)



def save_to_excel(predicted: np.ndarray, true: Optional[np.ndarray] = None, save_path: str = "spectrum_prediction_token_spectral.xlsx"):
    df = pd.DataFrame({
        "Wavelength (nm)": WAVELENGTHS,
        "Prediction": predicted.astype(np.float32),
    })
    if true is not None:
        df["Ground Truth"] = true.astype(np.float32)
    df.to_excel(save_path, index=False)
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    rgb_list = sorted(glob.glob(os.path.join(val_dir, "rgb_*.npy")))
    if len(rgb_list) == 0:
        raise FileNotFoundError(f"No rgb_*.npy found in {val_dir}")

    test_rgb_path = rgb_list[0]
    spectral_path = test_rgb_path.replace("rgb_", "spectral_")
    true_spectrum = np.load(spectral_path).astype(np.float32) if os.path.exists(spectral_path) else None

    dataset = SpectralDataset(
        val_dir,
        normalize_rgb=False,
        normalize_spectra=False,
        patch_reduce="mean",
    )
    pred = predict(test_rgb_path, dataset)
    print("Prediction first 10 values:", pred[:10])

    if true_spectrum is not None:
        metrics = evaluate_prediction(true_spectrum, pred)
        print("Metrics:", metrics)

    visualize_prediction(true_spectrum, pred)
    # save_to_excel(pred, true_spectrum, save_path="token_spectral_prediction.xlsx")
