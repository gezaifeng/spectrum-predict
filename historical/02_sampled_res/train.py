# -*- coding: utf-8 -*-
# train_rescnn.py
# 说明：最小化改动，仅适配 ResCNN 的输入维度；所有参数与路径在本文件内部提供。
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ====== 你的工程内组件（保持不变）======
from model import ResCNN                         # 最初版模型
from dataset import SpectralDataset              # 你的数据集
from SpectralReconstructionLoss import SpectralReconstructionLoss
from WeightedMSEloss import build_peak_weight    # 若你有峰权重构造

# ================= 内部配置（按需改这里） =================
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR      = r"D:\Desktop\dataset\guasnoise1-4080\train"
VAL_DIR        = r"D:\Desktop\dataset\guasnoise1-4080\val"
SAVE_PATH      = r"D:\Desktop\model\Res spectrum predict\best_rescnn.pth"

BATCH_SIZE     = 64
EPOCHS         = 100
LR             = 1e-3
WEIGHT_DECAY   = 1e-4

OUTPUT_DIM     = 76
LAMBDA_DC      = 0.10
LAMBDA_TV      = 0.02
LAMBDA_BASE    = 1e-4
ALPHA_PEAK     = 3.0      # 峰权重强度

# ================ 内部工具：仅补齐 ResCNN 所需的 seq 维 ================
def _add_seq_dim(x: torch.Tensor) -> torch.Tensor:
    """(B,3,4,6) -> (B,3,1,4,6)；若已含 seq 维则直接返回。"""
    return x.unsqueeze(2) if x.ndim == 4 else x

# ============================= 训练主流程 =============================
def run_training():
    # 数据
    train_ds = SpectralDataset(TRAIN_DIR, normalize_spectra=False)
    val_ds   = SpectralDataset(VAL_DIR,   normalize_spectra=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # 模型
    model = ResCNN().to(DEVICE)

    # 损失（保持你原来的不变）
    base_weight = build_peak_weight(train_ds, output_dim=OUTPUT_DIM, alpha=ALPHA_PEAK).to(DEVICE)
    criterion   = SpectralReconstructionLoss(
        weight=base_weight, lambda_dc=LAMBDA_DC, lambda_tv=LAMBDA_TV, lambda_base=LAMBDA_BASE
    ).to(DEVICE)

    # 优化与调度
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

    # 训练
    best_val = float("inf")
    if SAVE_PATH:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    for ep in range(1, EPOCHS + 1):
        # train
        model.train()
        tr_sum = 0.0
        for rgb, gt in train_loader:
            rgb = _add_seq_dim(rgb.to(DEVICE).float())  # ★ ResCNN 需要 (B,3,1,4,6)
            gt  = gt.to(DEVICE).float()
            optimizer.zero_grad(set_to_none=True)
            pred = model(rgb)                           # (B,76)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()
            tr_sum += loss.item()
        tr_loss = tr_sum / max(1, len(train_loader))

        # val
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for rgb, gt in val_loader:
                rgb = _add_seq_dim(rgb.to(DEVICE).float())
                gt  = gt.to(DEVICE).float()
                pred = model(rgb)
                val_sum += criterion(pred, gt).item()
        val_loss = val_sum / max(1, len(val_loader))
        scheduler.step(val_loss)

        print(f"[Epoch {ep:03d}] train {tr_loss:.6f} | val {val_loss:.6f} | lr {optimizer.param_groups[0]['lr']:.6g}")

        # save best
        if SAVE_PATH and val_loss < best_val - 1e-9:
            best_val = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ saved best -> {SAVE_PATH} (val={val_loss:.6f})")

    print("训练完成。最优验证损失：", best_val)

if __name__ == "__main__":
    run_training()
