# -*- coding: utf-8 -*-
import os
import glob
import math
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # 批量保存图片，不弹窗
import matplotlib.pyplot as plt

from model import ResCNN
from SpectralReconstructionLoss import spectral_angle_mapper, first_diff


# ===================== 路径配置（按需修改） =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 待预测数据目录：里面放 rgb_xxxx.npy / spectral_xxxx.npy
data_dir = r"D:\Desktop\Picture 3.9 2nm\ratio\val"

# 训练得到的最优模型
model_path = r"D:\Desktop\model\RESCNN_4.12\best.pth"

# 输出图片目录
save_dir = r"D:\Desktop\model\RESCNN_4.12\Picture 3.9 2nm\ratio"
os.makedirs(save_dir, exist_ok=True)
# =============================================================


WAVELENGTHS = np.arange(380, 760, 2)   # 76维


def _to_chw(x: np.ndarray) -> np.ndarray:
    """(3,H,W) or (H,W,3) -> (3,H,W)"""
    if x.ndim != 3:
        raise ValueError(f"feature must be 3D, got {x.shape}")
    if x.shape[0] == 3:
        return x
    if x.shape[-1] == 3:
        return np.transpose(x, (2, 0, 1))
    raise ValueError(f"unrecognized feature shape: {x.shape}")


def sanitize_feature(
    x_chw: np.ndarray,
    clip_value: float = 8.0,
    nan_repl: float = 0.0,
    posinf_repl: float = 0.0,
    neginf_repl: float = 0.0
) -> np.ndarray:
    x = np.nan_to_num(x_chw, nan=nan_repl, posinf=posinf_repl, neginf=neginf_repl)
    if clip_value is not None:
        x = np.clip(x, -clip_value, clip_value)
    return x.astype(np.float32)


def per_image_channel_zscore(x_chw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x_chw.astype(np.float32)
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    std_safe = np.where(std < 1e-4, 1.0, std)
    return ((x - mean) / (std_safe + eps)).astype(np.float32)


def preprocess_feature(rgb_npy: np.ndarray, clip_value: float, use_zscore: bool) -> torch.Tensor:
    x = rgb_npy.astype(np.float32)
    x = _to_chw(x)
    x = sanitize_feature(x, clip_value=clip_value)
    if use_zscore:
        x = per_image_channel_zscore(x)
    if not np.isfinite(x).all():
        raise ValueError("non-finite feature after preprocess")
    x = torch.from_numpy(x).unsqueeze(0).float()   # (1,3,H,W)
    return x


def load_model_and_cfg(model_path: str):
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
        out_dim=190,
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


@torch.no_grad()
def predict_one(model, rgb_path: str, clip_value: float, use_zscore: bool) -> np.ndarray:
    rgb = np.load(rgb_path)
    x = preprocess_feature(rgb, clip_value=clip_value, use_zscore=use_zscore).to(device)
    pred = model(x).squeeze(0).cpu().numpy().astype(np.float32)
    return pred


@torch.no_grad()
def compute_metrics_np(pred: np.ndarray, target: np.ndarray):
    pred_t = torch.from_numpy(pred).float().unsqueeze(0)
    tgt_t = torch.from_numpy(target).float().unsqueeze(0)

    mse = float(np.mean((pred - target) ** 2))
    rmse = math.sqrt(max(mse, 0.0))
    mae = float(np.mean(np.abs(pred - target)))

    sam = float(spectral_angle_mapper(pred_t, tgt_t).mean().item())

    d1_pred = first_diff(pred_t)
    d1_tgt = first_diff(tgt_t)
    diff1_mae = float((d1_pred - d1_tgt).abs().mean().item())

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "sam": sam,
        "diff1_mae": diff1_mae,
    }


def save_curve_figure(true_spectrum, pred_spectrum, sample_name, save_path):
    metrics = compute_metrics_np(pred_spectrum, true_spectrum)

    plt.figure(figsize=(10, 5.5))
    plt.plot(WAVELENGTHS, true_spectrum, label="Ground Truth", linewidth=2)
    plt.plot(WAVELENGTHS, pred_spectrum, label="Prediction", linewidth=2, linestyle="--")

    txt = (
        f"MSE: {metrics['mse']:.6f}\n"
        f"RMSE: {metrics['rmse']:.6f}\n"
        f"MAE: {metrics['mae']:.6f}\n"
        f"SAM: {metrics['sam']:.6f}\n"
        f"diff1_MAE: {metrics['diff1_mae']:.6f}"
    )

    plt.text(
        0.02, 0.98, txt,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.75)
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectrum Value")
    plt.title(f"Spectral Prediction - {sample_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return metrics


def predict_and_visualize_all(data_dir: str, model_path: str, save_dir: str):
    model, cfg = load_model_and_cfg(model_path)

    print("[Loaded Model Config]")
    print(json.dumps(cfg, ensure_ascii=False, indent=2))

    rgb_list = sorted(glob.glob(os.path.join(data_dir, "rgb_*.npy")))
    if len(rgb_list) == 0:
        raise FileNotFoundError(f"No rgb_*.npy found in {data_dir}")

    total_count = 0
    ok_count = 0
    bad_count = 0

    for rgb_path in rgb_list:
        total_count += 1
        try:
            base_name = os.path.basename(rgb_path)
            sample_id = base_name.replace("rgb_", "").replace(".npy", "")
            spectral_path = rgb_path.replace("rgb_", "spectral_")

            if not os.path.exists(spectral_path):
                print(f"[Skip] 缺少真值文件: {spectral_path}")
                bad_count += 1
                continue

            pred = predict_one(
                model,
                rgb_path,
                clip_value=cfg["clip_value"],
                use_zscore=cfg["use_zscore"]
            )

            gt = np.load(spectral_path).astype(np.float32).reshape(-1)
            if gt.shape[0] !=190:
                print(f"[Skip] 光谱维度错误: {spectral_path}, shape={gt.shape}")
                bad_count += 1
                continue

            fig_path = os.path.join(save_dir, f"{sample_id}.png")
            metrics = save_curve_figure(gt, pred, sample_id, fig_path)

            print(
                f"[{total_count:04d}] {sample_id} | "
                f"MSE={metrics['mse']:.6f}  "
                f"RMSE={metrics['rmse']:.6f}  "
                f"MAE={metrics['mae']:.6f}  "
                f"SAM={metrics['sam']:.6f}  "
                f"diff1_MAE={metrics['diff1_mae']:.6f}"
            )
            ok_count += 1

        except Exception as e:
            print(f"[Error] {rgb_path} -> {e}")
            bad_count += 1

    print("\n========== 完成 ==========")
    print(f"总样本数: {total_count}")
    print(f"成功保存图像: {ok_count}")
    print(f"失败/跳过: {bad_count}")
    print(f"输出目录: {save_dir}")


if __name__ == "__main__":
    predict_and_visualize_all(data_dir, model_path, save_dir)