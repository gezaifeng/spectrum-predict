# -*- coding: utf-8 -*-
"""
predict_dual_head.py
优化：预测结束立即可视化；关闭窗口后才保存PNG/Excel/summary。
"""

import os, glob, math
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # 交互式后端：先显示，再保存
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model_dual_head import DualHeadSpectralNet  # 确保与训练时的双头模型同路径

# ========================= 在此处配置 =========================
# 1) 模型权重
MODEL_PATH = r"D:\Desktop\model\double-RESCNN\best_dual_head.pth"

# 2) 手动点选的文件对（优先使用）。(rgb_path, spectral_path或None)
PAIRS = [
    # 例：
    # (r"D:\Desktop\dataset\guasnoise1-1020\val\rgb_0001.npy",
    #  r"D:\Desktop\dataset\guasnoise1-1020\val\spectral_0001.npy"),
]

# 3) 若 PAIRS 为空：遍历目录批量；自动匹配 spectral_*.npy（若缺失则仅显示预测）
RGB_DIR = r"D:\Desktop\dataset\guasnoise1-1020\val"

# 4) 输出
OUT_DIR = r"D:\Desktop\model\double-RESCNN\predicted"   # 保存曲线与Excel的目录
SAVE_EXCEL = True
# ===========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTHS = np.arange(380, 760, 5, dtype=np.float32)  # 76 维


# ------------------------ 工具函数 ------------------------
def robust_block_stat(x, axis=2, q=5):
    """截断均值：去除两端各 q% 噪声后求均值。x: (4,6,100,3) -> (4,6,3)"""
    low  = np.percentile(x, q, axis=axis, keepdims=True)
    high = np.percentile(x, 100 - q, axis=axis, keepdims=True)
    return np.clip(x, low, high).mean(axis=axis)

def _prepare_rgb_grid(rgb_npy: np.ndarray) -> torch.Tensor:
    """
    将单样本 rgb.npy 转为网络输入 (1,3,4,6)：
     - (4,6,100,3) -> 归一化到[0,1] + robust_block_stat -> (4,6,3)
     - (4,6,3) -> 若>1则归一化
    """
    arr = rgb_npy.astype(np.float32)
    if arr.ndim == 4 and arr.shape == (4, 6, 100, 3):
        if arr.max() > 1.0: arr = arr / 255.0
        arr = robust_block_stat(arr, axis=2, q=5)
    elif arr.ndim == 3 and arr.shape == (4, 6, 3):
        if arr.max() > 1.0: arr = arr / 255.0
    else:
        raise ValueError(f"Unexpected rgb.npy shape: {arr.shape} (expect (4,6,100,3) or (4,6,3)).")
    x = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1,3,4,6)
    return x

def load_model(model_path: str, n_wave: int = 76, token_grid=(4,6)) -> torch.nn.Module:
    model = DualHeadSpectralNet(
        n_wave=n_wave, token_grid=token_grid,
        base_ch=64, n_blocks=4, k_basis=12, out_activation="softplus"
    ).to(DEVICE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weight not found: {model_path}")
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        model.load_state_dict(state["model"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.eval()
    return model

@torch.no_grad()
def predict_from_rgb_npy(model, rgb_npy_path: str) -> np.ndarray:
    rgb = np.load(rgb_npy_path)
    x = _prepare_rgb_grid(rgb).to(DEVICE)  # (1,3,4,6)
    out = model(x)                         # (spec, coef)
    spec = out[0] if isinstance(out, (tuple, list)) else out
    return spec.squeeze(0).detach().cpu().numpy().astype(np.float32)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse + 1e-12)
    mae  = mean_absolute_error(y_true, y_pred)
    psnr = 10 * math.log10(1 / (rmse ** 2 + 1e-12))  # 以1为满幅值；如需以max(y_true)为峰值可自行改
    r2   = r2_score(y_true, y_pred)
    return dict(mse=mse, rmse=rmse, mae=mae, psnr=psnr, r2=r2)

def plot_interactive(y_pred: np.ndarray, y_true: np.ndarray = None, title="Dual-Head"):
    """先显示交互式图；窗口关闭后返回。"""
    plt.figure(figsize=(9, 5.0))
    if y_true is not None:
        plt.plot(WAVELENGTHS, y_true, 'b-', label='Ground Truth', linewidth=2)
        plt.plot(WAVELENGTHS, y_pred, 'r--', label=f'Prediction ({title})', linewidth=2)
    else:
        plt.plot(WAVELENGTHS, y_pred, 'r-', label=f'Prediction ({title})', linewidth=2)
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Absorbance")
    plt.title(f"Spectral Prediction ({title})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    # 关键：阻塞显示。用户关闭窗口后，函数才返回，随后我们再保存PNG/Excel。
    plt.show(block=True)

def save_png_and_excel(base_name: str, y_pred: np.ndarray, y_true: np.ndarray = None):
    # 保存PNG
    fig_png = os.path.join(OUT_DIR, f"{base_name}_pred.png")
    plt.figure(figsize=(9, 5.0))
    if y_true is not None:
        plt.plot(WAVELENGTHS, y_true, 'b-', label='Ground Truth', linewidth=2)
        plt.plot(WAVELENGTHS, y_pred, 'r--', label='Prediction (Dual-Head)', linewidth=2)
    else:
        plt.plot(WAVELENGTHS, y_pred, 'r-', label='Prediction (Dual-Head)', linewidth=2)
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Absorbance")
    plt.title("Spectral Prediction (Dual-Head)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(fig_png, dpi=150); plt.close()
    print(f"[Saved] figure -> {fig_png}")

    # 保存Excel（可选）
    if SAVE_EXCEL:
        xlsx_path = os.path.join(OUT_DIR, f"{base_name}_pred.xlsx")
        df = pd.DataFrame({"Wavelength (nm)": WAVELENGTHS, "Prediction": y_pred.astype(np.float32)})
        if y_true is not None:
            df["Ground Truth"] = y_true.astype(np.float32)
        df.to_excel(xlsx_path, index=False)
        print(f"[Saved] excel  -> {xlsx_path}")

    return fig_png

# ------------------------ 主流程：先显示，后保存 ------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 加载模型
    model = load_model(MODEL_PATH, n_wave=len(WAVELENGTHS), token_grid=(4,6))
    print(f"Model loaded: {MODEL_PATH}")

    # 2) 组装待预测文件对
    pairs = list(PAIRS)
    if not pairs:
        rgb_list = sorted(glob.glob(os.path.join(RGB_DIR, "rgb_*.npy")))
        for rp in rgb_list:
            sp = rp.replace("rgb_", "spectral_")
            pairs.append((rp, sp if os.path.exists(sp) else None))
        if not pairs:
            raise FileNotFoundError(f"No rgb_*.npy found under {RGB_DIR}")

    # 3) 逐个样本：预测 -> 交互式显示 -> 关闭后保存文件
    rows = []
    for i, (rgb_path, spec_path) in enumerate(pairs, 1):
        if not os.path.isfile(rgb_path):
            print(f"[Skip] rgb file not found: {rgb_path}")
            continue

        base = "D:\Desktop\model\double-RESCNN\predicted " # rgb_xxx
        print(f"[{i:03d}] Predicting: {base}")

        # 预测
        y_pred = predict_from_rgb_npy(model, rgb_path)

        # 真值与指标
        y_true = None
        metrics = {}
        if spec_path and os.path.isfile(spec_path):
            y_true = np.load(spec_path).astype(np.float32)
            metrics = compute_metrics(y_true, y_pred)
            print(f"  MSE={metrics['mse']:.4f}  RMSE={metrics['rmse']:.4f}  "
                  f"MAE={metrics['mae']:.4f}  PSNR={metrics['psnr']:.2f} dB  R2={metrics['r2']:.4f}")
        else:
            print("  (no GT)")

        # —— 关键逻辑：先显示，等你关闭窗口后再保存 —— #
        plot_interactive(y_pred, y_true, title="Dual-Head")

        # 关闭窗口后保存
        save_png_and_excel(base, y_pred, y_true)

        rows.append({"file": os.path.basename(rgb_path),
                     "has_gt": y_true is not None,
                     **({k: float(v) for k, v in metrics.items()} if metrics else {})})

    # 4) 汇总CSV
    summary_csv = os.path.join(OUT_DIR, "summary.csv")
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"[Saved] summary -> {summary_csv}")


if __name__ == "__main__":
    main()
