# -*- coding: utf-8 -*-
"""
evaluate_val_spectrum_errors.py

功能：
1. 调用现有 model.py 中的 ResCNN
2. 调用现有 dataset.py 中的预处理函数
3. 对 val/test 文件夹全部样本做预测
4. 输出每个样本的总体指标：
   - MSE
   - RMSE
   - MAE
   - Bias
5. 输出所有样本“逐波段”的误差统计：
   - mean_abs_error_per_wavelength
   - rmse_per_wavelength
   - mean_bias_per_wavelength
   - std_abs_error_per_wavelength
6. 保存图像与 csv/json，方便分析模型在哪些波段更差
"""

import os
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== 直接复用你现有工程里的模块 =====
from model import ResCNN
from dataset import _to_chw, sanitize_feature, per_image_channel_zscore


# ========================= 路径配置 =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = r"D:\Desktop\Picture 3.9 2nm\ratio\val"
MODEL_PATH = r"D:\Desktop\model\RESCNN_4.12\best.pth"
SAVE_DIR = r"D:\Desktop\model\RESCNN_4.12\val_error_analysis"

# 全部样本就设为 None；调试时可以改成 50、100
MAX_SAMPLES = None

# 这里一定和你的真实光谱维度一致
# 如果你现在光谱是 190 维，就改成 np.arange(...) 对应长度，或者直接用 OUT_DIM 指定
OUT_DIM = 190

# 仅用于画图横轴
# 你如果是 190 维且步长为 2nm/或别的，请按真实情况改
WAVELENGTHS = np.arange(380, 380 + OUT_DIM * 2, 2)

# ==========================================================


def load_model_and_cfg(model_path: str, device: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        clip_value = ckpt.get("clip_value", 8.0)
        use_zscore = ckpt.get("use_zscore", True)
        out_activation = ckpt.get("out_activation", "linear")
    else:
        state_dict = ckpt
        clip_value = 8.0
        use_zscore = True
        out_activation = "linear"

    model = ResCNN(
        out_dim=OUT_DIM,
        base_ch=64,
        n_blocks=6,
        mlp_hidden=256,
        dropout=0.10,
        out_activation=out_activation,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    cfg = {
        "clip_value": clip_value,
        "use_zscore": use_zscore,
        "out_activation": out_activation,
    }
    return model, cfg


def preprocess_feature(rgb_npy: np.ndarray, clip_value: float, use_zscore: bool) -> torch.Tensor:
    x = rgb_npy.astype(np.float32)
    x = _to_chw(x)
    x = sanitize_feature(x, clip_value=clip_value)
    if use_zscore:
        x = per_image_channel_zscore(x)
    if not np.isfinite(x).all():
        raise ValueError("non-finite feature after preprocess")
    return torch.from_numpy(x).unsqueeze(0).float()  # (1,3,H,W)


@torch.no_grad()
def predict_one(model, rgb_path: str, clip_value: float, use_zscore: bool) -> np.ndarray:
    rgb = np.load(rgb_path)
    x = preprocess_feature(rgb, clip_value=clip_value, use_zscore=use_zscore).to(DEVICE)
    pred = model(x).squeeze(0).cpu().numpy().astype(np.float32)
    return pred


def compute_sample_metrics(pred: np.ndarray, gt: np.ndarray):
    err = pred - gt
    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(max(mse, 0.0)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
    }


def plot_per_wavelength_curves(wavelengths, mae_curve, rmse_curve, bias_curve, std_abs_curve, save_dir: Path):
    # 图1：逐波段 MAE / RMSE
    plt.figure(figsize=(11, 5.5))
    plt.plot(wavelengths, mae_curve, label="MAE per wavelength")
    plt.plot(wavelengths, rmse_curve, label="RMSE per wavelength")
    plt.xlabel("Wavelength")
    plt.ylabel("Error")
    plt.title("Per-wavelength Error Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "01_per_wavelength_error_curve.png", dpi=220)
    plt.close()

    # 图2：逐波段 Bias
    plt.figure(figsize=(11, 5.5))
    plt.plot(wavelengths, bias_curve, label="Mean Bias per wavelength")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Wavelength")
    plt.ylabel("Bias")
    plt.title("Per-wavelength Mean Bias")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "02_per_wavelength_bias_curve.png", dpi=220)
    plt.close()

    # 图3：逐波段绝对误差波动
    plt.figure(figsize=(11, 5.5))
    plt.plot(wavelengths, std_abs_curve, label="Std of Abs Error per wavelength")
    plt.xlabel("Wavelength")
    plt.ylabel("Std(|error|)")
    plt.title("Per-wavelength Error Stability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "03_per_wavelength_error_std_curve.png", dpi=220)
    plt.close()

    # 图4：逐波段热力柱状
    plt.figure(figsize=(12, 4.8))
    plt.bar(wavelengths, mae_curve, width=(wavelengths[1] - wavelengths[0]) * 0.8 if len(wavelengths) > 1 else 1.0)
    plt.xlabel("Wavelength")
    plt.ylabel("MAE")
    plt.title("Per-wavelength MAE Bar")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "04_per_wavelength_mae_bar.png", dpi=220)
    plt.close()


def plot_sample_metric_histograms(sample_df: pd.DataFrame, save_dir: Path):
    for col in ["mse", "rmse", "mae", "bias"]:
        plt.figure(figsize=(7, 4.5))
        plt.hist(sample_df[col].values, bins=30)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Sample-level {col.upper()} Distribution")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(save_dir / f"hist_{col}.png", dpi=220)
        plt.close()


def main():
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 加载模型...")
    model, cfg = load_model_and_cfg(MODEL_PATH, DEVICE)
    print(json.dumps(cfg, ensure_ascii=False, indent=2))

    print("[2/5] 读取样本...")
    rgb_list = sorted(glob.glob(os.path.join(DATA_DIR, "rgb_*.npy")))
    if len(rgb_list) == 0:
        raise FileNotFoundError(f"No rgb_*.npy found in {DATA_DIR}")

    if MAX_SAMPLES is not None:
        rgb_list = rgb_list[:MAX_SAMPLES]

    print(f"样本数: {len(rgb_list)}")

    print("[3/5] 逐样本预测并统计...")
    sample_rows = []
    pred_all = []
    gt_all = []

    for i, rgb_path in enumerate(rgb_list, start=1):
        sp_path = rgb_path.replace("rgb_", "spectral_")
        if not os.path.exists(sp_path):
            print(f"[Skip] missing gt: {sp_path}")
            continue

        pred = predict_one(
            model,
            rgb_path,
            clip_value=cfg["clip_value"],
            use_zscore=cfg["use_zscore"]
        )
        gt = np.load(sp_path).astype(np.float32).reshape(-1)

        if gt.shape[0] != OUT_DIM:
            print(f"[Skip] bad gt shape: {sp_path}, got {gt.shape}, expect ({OUT_DIM},)")
            continue

        metrics = compute_sample_metrics(pred, gt)
        sample_id = os.path.basename(rgb_path).replace("rgb_", "").replace(".npy", "")

        sample_rows.append({
            "sample_id": sample_id,
            **metrics
        })
        pred_all.append(pred)
        gt_all.append(gt)

        print(
            f"[{i:04d}/{len(rgb_list):04d}] {sample_id} | "
            f"MSE={metrics['mse']:.6f}  RMSE={metrics['rmse']:.6f}  "
            f"MAE={metrics['mae']:.6f}  Bias={metrics['bias']:.6f}"
        )

    if len(pred_all) == 0:
        raise RuntimeError("No valid sample pairs found.")

    pred_all = np.stack(pred_all, axis=0)   # [N, D]
    gt_all = np.stack(gt_all, axis=0)       # [N, D]
    err_all = pred_all - gt_all             # [N, D]
    abs_err_all = np.abs(err_all)

    print("[4/5] 计算逐波段误差统计...")
    mean_abs_error_per_wavelength = abs_err_all.mean(axis=0)
    rmse_per_wavelength = np.sqrt((err_all ** 2).mean(axis=0))
    mean_bias_per_wavelength = err_all.mean(axis=0)
    std_abs_error_per_wavelength = abs_err_all.std(axis=0)

    per_wave_df = pd.DataFrame({
        "wavelength": WAVELENGTHS,
        "mae": mean_abs_error_per_wavelength,
        "rmse": rmse_per_wavelength,
        "bias": mean_bias_per_wavelength,
        "std_abs_error": std_abs_error_per_wavelength,
    })

    sample_df = pd.DataFrame(sample_rows)

    global_summary = {
        "num_samples": int(len(sample_df)),
        "sample_mse_mean": float(sample_df["mse"].mean()),
        "sample_rmse_mean": float(sample_df["rmse"].mean()),
        "sample_mae_mean": float(sample_df["mae"].mean()),
        "sample_bias_mean": float(sample_df["bias"].mean()),
        "worst_sample_mae": float(sample_df["mae"].max()),
        "best_sample_mae": float(sample_df["mae"].min()),
        "worst_wavelength_mae": float(mean_abs_error_per_wavelength.max()),
        "worst_wavelength_rmse": float(rmse_per_wavelength.max()),
        "worst_wavelength_bias_abs": float(np.abs(mean_bias_per_wavelength).max()),
        "worst_wavelength_std_abs": float(std_abs_error_per_wavelength.max()),
        "worst_mae_wavelength": float(WAVELENGTHS[np.argmax(mean_abs_error_per_wavelength)]),
        "worst_rmse_wavelength": float(WAVELENGTHS[np.argmax(rmse_per_wavelength)]),
        "worst_abs_bias_wavelength": float(WAVELENGTHS[np.argmax(np.abs(mean_bias_per_wavelength))]),
        "worst_std_abs_wavelength": float(WAVELENGTHS[np.argmax(std_abs_error_per_wavelength)]),
    }

    print(json.dumps(global_summary, ensure_ascii=False, indent=2))

    print("[5/5] 保存结果与可视化...")
    sample_df.to_csv(save_dir / "sample_level_metrics.csv", index=False, encoding="utf-8-sig")
    per_wave_df.to_csv(save_dir / "per_wavelength_error_metrics.csv", index=False, encoding="utf-8-sig")

    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    plot_per_wavelength_curves(
        WAVELENGTHS,
        mean_abs_error_per_wavelength,
        rmse_per_wavelength,
        mean_bias_per_wavelength,
        std_abs_error_per_wavelength,
        save_dir
    )

    plot_sample_metric_histograms(sample_df, save_dir)

    # 额外保存一个“逐波段误差排序表”
    rank_df = per_wave_df.copy()
    rank_df["abs_bias"] = np.abs(rank_df["bias"].values)
    rank_df = rank_df.sort_values(by="mae", ascending=False)
    rank_df.to_csv(save_dir / "per_wavelength_rank_by_mae.csv", index=False, encoding="utf-8-sig")

    print("结果已保存到：", str(save_dir.resolve()))
    print("重点看：")
    print("  sample_level_metrics.csv")
    print("  per_wavelength_error_metrics.csv")
    print("  per_wavelength_rank_by_mae.csv")
    print("  01_per_wavelength_error_curve.png")
    print("  02_per_wavelength_bias_curve.png")
    print("  03_per_wavelength_error_std_curve.png")
    print("  04_per_wavelength_mae_bar.png")


if __name__ == "__main__":
    main()