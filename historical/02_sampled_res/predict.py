# -*- coding: utf-8 -*-
# 极简预测：给样本路径 -> 预测 -> 可视化
import torch
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('TkAgg')  # 需要弹窗显示；若服务器无GUI请改为 'Agg'
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import ResCNN

# ----------------- 配置区：请改成你的真实路径 -----------------
device    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"D:\Desktop\model\Res spectrum predict\best_rescnn.pth"
TEST_RGB   = r"D:\Desktop\dataset\guasnoise1-1020\val\rgb_0081.npy"
TEST_GT    = r"D:\Desktop\dataset\guasnoise1-1020\val\spectral_0081.npy"
# -------------------------------------------------------------

def safe_load_state_dict(model: torch.nn.Module, ckpt_path: str, dev: str):
    """
    安全、兼容地加载权重：
    - 先尝试 weights_only=True（新PyTorch）
    - 兼容常见外层键（state_dict / model / net / weights）
    - strict=False 允许键名不完全一致
    """
    try:
        state = torch.load(ckpt_path, map_location=dev, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=dev)

    if isinstance(state, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

    info = model.load_state_dict(state, strict=False)
    if info is not None:
        missing = getattr(info, "missing_keys", [])
        unexpected = getattr(info, "unexpected_keys", [])
        if missing:
            print(f"⚠️ Missing keys（随机初始化这些参数）: {missing}")
        if unexpected:
            print(f"ℹ️ Unexpected keys（忽略这些多余权重）: {unexpected}")
    print("✅ 权重加载完成。")
    return model

def robust_block_stat(x, axis=2, q=5):
    """
    抗异常像素的截断均值：剔除两端各 q% 再取均值。
    输入: x 形状 (4,6,100,3)
    输出: (4,6,3)
    """
    low  = np.percentile(x, q, axis=axis, keepdims=True)
    high = np.percentile(x, 100 - q, axis=axis, keepdims=True)
    x_clip = np.clip(x, low, high)
    return x_clip.mean(axis=axis)

def predict_from_rgb_npy(model, rgb_npy_path: str):
    """
    输入: (4,6,100,3) 的 RGB npy 文件
    输出: 预测光谱 (76,)
    说明: 为适配最初版 ResCNN，这里在前向前补一个 seq 维 -> (1,3,1,4,6)
    """
    rgb_data = np.load(rgb_npy_path).astype(np.float32)
    if rgb_data.shape != (4,6,100,3):
        raise ValueError(f"RGB shape 错误，期望 (4,6,100,3)，实际 {rgb_data.shape}")

    # 归一化到 [0,1] 再做截断均值
    rgb_data = rgb_data / 255.0
    rgb_stat = robust_block_stat(rgb_data, axis=2, q=5)   # (4,6,3)
    rgb_input = np.transpose(rgb_stat, (2, 0, 1))         # (3,4,6)
    rgb_tensor = torch.from_numpy(rgb_input).unsqueeze(0).to(device)  # (1,3,4,6)

    # ★ 关键：ResCNN 需要 seq 维
    rgb_tensor = rgb_tensor.unsqueeze(2)                  # (1,3,1,4,6)

    model.eval()
    with torch.no_grad():
        out = model(rgb_tensor)                           # 兼容 y 或 (y, base_coef)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = out.float().cpu().numpy().flatten()
    return pred

def visualize_prediction(true_spectrum, predicted_spectrum, wavelengths=None):
    if wavelengths is None:
        wavelengths = np.arange(380, 760, 5)

    y_true = np.asarray(true_spectrum, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(predicted_spectrum, dtype=np.float32).reshape(-1)

    mse  = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse + 1e-12)
    mae  = mean_absolute_error(y_true, y_pred)
    # 与你原脚本一致：以 1 为满幅值计算 PSNR（如需以 max(y) 为满幅值可自行改）
    psnr = 10 * math.log10(1 / (rmse ** 2 + 1e-12))
    r2   = r2_score(y_true, y_pred)

    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, y_true, 'bo-', label='real spectrum')
    plt.plot(wavelengths, y_pred, 'ro-', label='predict spectrum')
    plt.xlabel("wavelength (nm)")
    plt.ylabel("absorbance")
    plt.title("Spectral prediction (ResCNN)")
    metrics_text = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nPSNR: {psnr:.2f} dB\nR²: {r2:.4f}"
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.legend(); plt.grid(True); plt.show(block=True)

if __name__ == "__main__":
    # 1) 构建模型并加载权重
    model = ResCNN().to(device)
    model = safe_load_state_dict(model, MODEL_PATH, device)

    # 2) 预测
    pred = predict_from_rgb_npy(model, TEST_RGB)
    print("预测（前10个点）:", np.round(pred[:10], 4))

    # 3) 有真值就做可视化（没有就跳过）
    try:
        gt = np.load(TEST_GT)
        visualize_prediction(gt, pred)
        # 如需保存Excel，可取消下方注释
        # wl = np.arange(380, 760, 5)
        # pd.DataFrame({"Wavelength (nm)": wl, "Predicted": pred, "True": gt}).to_excel("rescnn_pred.xlsx", index=False)
    except Exception as e:
        print("未找到或无法读取真值，仅打印预测；错误：", e)
