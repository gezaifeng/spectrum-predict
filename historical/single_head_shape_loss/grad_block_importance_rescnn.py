# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 为了避免 PyCharm backend 问题，默认只保存图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import SpectralDatasetLogRatio
from model import ResCNN


# =======================
# 配置区：按你实际路径修改
# =======================
CKPT_PATH = r"D:\Desktop\model\RESCNN_1.23\best.pth"
VAL_DIR   = r"D:\Desktop\dataset-1.23\test_logratio_346\val"
OUT_DIR   = r"D:\Desktop\model\RESCNN_1.23\grad_importance"

BATCH_SIZE = 64
NUM_WORKERS = 0

# 重要性计算方式：
# 1) "grad_x_input"：|grad * x|（推荐，最稳）
# 2) "abs_grad"    ：|grad|
IMPORTANCE_MODE = "grad_x_input"

# 标量目标：
# 1) "mse"：MSE(pred,y)（推荐，与训练一致）
# 2) "l2" ：||pred||^2（无标签也可用）
SCALAR_OBJECTIVE = "mse"

# 可选：只看某个波段输出对输入的敏感性（例如 band_index=10），
# 若为 None，则不启用
BAND_INDEX = None

# =======================


def minmax_norm(x: np.ndarray, eps: float = 1e-12):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + eps)


@torch.no_grad()
def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = ResCNN(out_dim=76, base_ch=64, n_blocks=6, mlp_hidden=256, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    clip_value = ckpt.get("clip_value", 8.0)
    use_zscore = ckpt.get("use_zscore", True)
    return model, clip_value, use_zscore


def compute_scalar(pred, y, criterion):
    """
    返回一个标量，用于 backward。
    """
    if BAND_INDEX is not None:
        # 只关心某个波段输出的敏感性
        # pred[:, BAND_INDEX] 的均值作为标量
        return pred[:, BAND_INDEX].mean()

    if SCALAR_OBJECTIVE == "mse":
        return criterion(pred, y)
    elif SCALAR_OBJECTIVE == "l2":
        return (pred ** 2).mean()
    else:
        raise ValueError(f"Unknown SCALAR_OBJECTIVE={SCALAR_OBJECTIVE}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, clip_value, use_zscore = load_model(device)
    print("Loaded ckpt:", CKPT_PATH)
    print("Preprocess clip_value=", clip_value, "use_zscore=", use_zscore)

    val_ds = SpectralDatasetLogRatio(VAL_DIR, clip_value=clip_value, use_zscore=use_zscore)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    criterion = nn.MSELoss()

    total_map = None
    total_count = 0

    # 为了更稳定：关闭 BN/Dropout 的训练态（eval 已经做了）
    model.eval()

    for step, (x, y) in enumerate(val_ld, start=1):
        x = x.to(device).float()
        y = y.to(device).float()

        # 关键：对输入求梯度
        x.requires_grad_(True)

        pred = model(x)
        scalar = compute_scalar(pred, y, criterion)

        # 清梯度
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()

        # 反传
        scalar.backward()

        grad = x.grad  # (B,3,H,W)
        if grad is None:
            raise RuntimeError("x.grad is None, backward failed.")

        # 计算每个色块的重要性 (B,H,W)
        if IMPORTANCE_MODE == "grad_x_input":
            imp = (grad * x).abs().mean(dim=1)  # 通道均值
        elif IMPORTANCE_MODE == "abs_grad":
            imp = grad.abs().mean(dim=1)
        else:
            raise ValueError(f"Unknown IMPORTANCE_MODE={IMPORTANCE_MODE}")

        imp_sum = imp.detach().sum(dim=0).cpu().numpy()  # (H,W)
        bs = x.size(0)

        if total_map is None:
            total_map = imp_sum
        else:
            total_map += imp_sum
        total_count += bs

        if step % 10 == 0 or step == 1:
            print(f"[{step:03d}] accumulated {total_count} samples")

        # 释放图计算图
        x.requires_grad_(False)

    # 全局平均重要性
    heat = total_map / max(1, total_count)  # (H,W)

    # 归一化便于显示
    heat_norm = minmax_norm(heat)

    # 保存数值
    npy_path = os.path.join(OUT_DIR, "block_importance_heatmap.npy")
    np.save(npy_path, heat)

    # 画热力图
    fig_path = os.path.join(OUT_DIR, "block_importance_heatmap.png")
    plt.figure(figsize=(7, 4))
    im = plt.imshow(heat_norm, aspect="auto")
    plt.title(f"Block Importance Heatmap ({IMPORTANCE_MODE}, objective={SCALAR_OBJECTIVE})")
    plt.xlabel("W (block index)")
    plt.ylabel("H (block index)")
    plt.xticks(range(heat_norm.shape[1]))
    plt.yticks(range(heat_norm.shape[0]))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    # 输出 Top-K 色块（按重要性从高到低）
    flat = heat.reshape(-1)
    k = min(10, flat.size)
    top_idx = np.argsort(-flat)[:k]
    H, W = heat.shape
    print("\nTop blocks:")
    for r in top_idx:
        h = r // W
        w = r % W
        print(f"  (h={h}, w={w})  importance={heat[h,w]:.6g}")

    print("\nSaved:")
    print("  heatmap npy :", npy_path)
    print("  heatmap png :", fig_path)


if __name__ == "__main__":
    main()
