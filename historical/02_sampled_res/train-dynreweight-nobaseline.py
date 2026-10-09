import os
import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import ResCNN
from dataset import SpectralDataset
from WeightedMSEloss import build_peak_weight  # 复用你的峰权重构造

# ========================= 参数配置 =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ⚠️ 路径按需修改（Windows 下建议原始字符串 r"..."）
train_dir = r"D:\Desktop\dataset\guasnoise1-4080\train"
val_dir   = r"D:\Desktop\dataset\guasnoise1-4080\val"
save_path = r"D:\Desktop\model\Res spectrum predict\best_model.pth"
log_path  = r"D:\Desktop\model\Res spectrum predict\training_log.csv"
fig_path  = r"D:\Desktop\model\Res spectrum predict\loss_curve.png"

# —— 动态权重可视化输出 —— #
dyn_weight_csv   = r"D:\Desktop\model\Res spectrum predict\dynamic_weight_final.csv"
dyn_weight_line  = r"D:\Desktop\model\Res spectrum predict\dynamic_weight_final.png"
dyn_weight_heat  = r"D:\Desktop\model\Res spectrum predict\dynamic_weight_over_epochs.png"
mae_line_fig     = r"D:\Desktop\model\Res spectrum predict\per_wavelength_mae.png"

num_epochs    = 200
batch_size    = 64
learning_rate = 1e-3
weight_decay  = 1e-4
output_dim    = 76
wavelengths   = np.arange(380, 760, 5)  # 76 个点

# ===== 动态重加权策略 =====
dyn_reweight           = True     # 开关
reweight_every_epochs  = 5        # 每隔多少个 epoch 更新一次权重
ema_beta               = 0.7      # EMA 平滑系数（越大越平滑）
dyn_scale_clamp        = (0.5, 2.0)  # 动态因子截断范围，防止过度倾斜
dyn_mix_with_base      = 0.5      # 与基础权重的混合比例（0~1）
alpha_peak             = 3.0      # 基础峰权重强度

# ===== 早停 =====
early_stop_patience = 20

# ========================= 自定义综合损失（无基线学习） =========================
class SpectralReconstructionLoss(nn.Module):
    """
    将三项损失合并：
      - WeightedMSE：逐波长加权（峰形更准）
      - DC：基线均值约束（抑制整体漂移）
      - TV：一阶差分平滑（抑制不合理抖动）
    """
    def __init__(self, weight: torch.Tensor, lambda_dc: float = 0.10, lambda_tv: float = 0.02):
        super().__init__()
        assert weight is not None, "weight 不能为空（使用 build_peak_weight 构建）"
        self.register_buffer("weight", weight.float())   # [76]
        self.lambda_dc = float(lambda_dc)
        self.lambda_tv = float(lambda_tv)

    @torch.no_grad()
    def update_weight(self, new_weight: torch.Tensor):
        """训练中动态更新逐波长权重。"""
        assert new_weight.shape[-1] == self.weight.shape[-1]
        self.weight.data.copy_(new_weight.to(self.weight.device).float())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: [B, 76]
        mse = (self.weight * (pred - target) ** 2).mean()
        dc  = ((pred.mean(dim=1) - target.mean(dim=1)) ** 2).mean()
        tv  = ((pred[:, 1:] - pred[:, :-1]) ** 2).mean()
        return mse + self.lambda_dc * dc + self.lambda_tv * tv

# ========================= 工具函数 =========================
@torch.no_grad()
def compute_val_mae_by_wavelength(model: nn.Module, loader: DataLoader, dev: str) -> torch.Tensor:
    """统计验证集逐波长 MAE，返回 [76] 张量（CPU）"""
    model.eval()
    mae_sum = torch.zeros(output_dim, device=dev)
    n = 0
    for images, spectra in loader:
        images  = images.to(dev).float()
        spectra = spectra.to(dev).float()
        out = model(images)
        if isinstance(out, (tuple, list)):  # 兼容可能返回 (y, ...)
            out = out[0]
        mae_sum += (out - spectra).abs().mean(dim=0)
        n += 1
    return (mae_sum / max(n, 1)).detach().cpu()

def make_dynamic_weight(base_w: torch.Tensor,
                        val_mae: torch.Tensor,
                        clamp_range=(0.5, 2.0),
                        mix_with_base=0.5) -> torch.Tensor:
    """
    基于验证误差的动态重加权：
      - 逐波长 MAE 归一化（相对均值）
      - 截断到 [low, high]，防止过强/过弱
      - 与基础权重融合（log 域线性混合更稳定）
    """
    w_base = base_w.clone().detach().float()
    w_dyn  = val_mae.clone().detach().float()
    w_dyn  = w_dyn / (w_dyn.mean() + 1e-8)
    low, high = clamp_range
    w_dyn = torch.clamp(w_dyn, low, high)

    if mix_with_base >= 1.0:
        return w_base
    if mix_with_base <= 0.0:
        return (w_base * w_dyn)

    log_w  = mix_with_base * torch.log(w_base + 1e-12) + (1.0 - mix_with_base) * torch.log(w_base * w_dyn + 1e-12)
    return torch.exp(log_w)

# ========================= 数据加载 =========================
train_dataset = SpectralDataset(train_dir, normalize_spectra=False)
val_dataset   = SpectralDataset(val_dir,   normalize_spectra=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# ========================= 模型/优化/调度/损失 =========================
model = ResCNN().to(device)

base_weight = build_peak_weight(train_dataset, output_dim=output_dim, alpha=alpha_peak).to(device)
criterion   = SpectralReconstructionLoss(weight=base_weight, lambda_dc=0.10, lambda_tv=0.02).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# ========================= 动态重加权的状态（EMA + 历史记录） =========================
ema_val_mae = None           # EMA 平滑后的 MAE(λ)
weight_history = []          # 保存每次更新后的权重（np.array[76]）
weight_epochs  = []          # 记录更新发生的 epoch 编号
mae_last_curve = None        # 记录最后一次（或 EMA）MAE(λ) 用于可视化

# ========================= 训练循环 =========================
log = {"epoch": [], "train_loss": [], "val_loss": [], "learning_rate": []}
best_val = float("inf")
wait = 0

for epoch in range(1, num_epochs + 1):
    # ----- 动态重加权：先基于验证集评估并更新损失权重 -----
    if dyn_reweight and (epoch == 1 or (epoch - 1) % reweight_every_epochs == 0):
        val_mae = compute_val_mae_by_wavelength(model, val_loader, device)  # CPU
        mae_last_curve = val_mae.numpy().copy()
        # EMA 平滑
        if ema_val_mae is None:
            ema_val_mae = val_mae.clone()
        else:
            ema_val_mae = ema_beta * ema_val_mae + (1.0 - ema_beta) * val_mae
        # 生成新权重并更新
        new_w = make_dynamic_weight(base_weight.detach().cpu(), ema_val_mae,
                                    clamp_range=dyn_scale_clamp,
                                    mix_with_base=dyn_mix_with_base)
        criterion.update_weight(new_w.to(device))
        # 记录权重历史（用于画热力图）
        weight_history.append(new_w.cpu().numpy())
        weight_epochs.append(epoch)
        print(f"🔁 动态重加权 @ epoch {epoch}: 依据验证 MAE(λ) 更新逐波长权重 | 混合={dyn_mix_with_base}, 截断={dyn_scale_clamp}")

    # ----- 训练 -----
    model.train()
    total = 0.0
    for images, spectra in train_loader:
        images  = images.to(device).float()
        spectra = spectra.to(device).float()

        optimizer.zero_grad()
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        loss = criterion(out, spectra)
        loss.backward()
        optimizer.step()
        total += loss.item()

    train_loss = total / len(train_loader)

    # ----- 验证 -----
    model.eval()
    val_total = 0.0
    with torch.no_grad():
        for images, spectra in val_loader:
            images  = images.to(device).float()
            spectra = spectra.to(device).float()
            out = model(images)
            if isinstance(out, (tuple, list)):
                out = out[0]
            val_total += criterion(out, spectra).item()

    val_loss = val_total / len(val_loader)
    current_lr = optimizer.param_groups[0]["lr"]

    # ----- 日志 -----
    log["epoch"].append(epoch)
    log["train_loss"].append(train_loss)
    log["val_loss"].append(val_loss)
    log["learning_rate"].append(current_lr)

    print(f"Epoch [{epoch:03d}/{num_epochs}] Train {train_loss:.4f} | Val {val_loss:.4f} | LR {current_lr:.6f}")

    # ----- 保存最优 & 调度 & 早停 -----
    if val_loss < best_val - 1e-6:
        best_val = val_loss
        wait = 0
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"✅ Save model @ epoch {epoch} (val={val_loss:.6f})")
    else:
        wait += 1

    scheduler.step(val_loss)
    if wait >= early_stop_patience:
        print("⏹️ Early stop triggered.")
        break

# ========================= 保存日志和损失曲线 =========================
df = pd.DataFrame(log)
df.to_csv(log_path, index=False)

plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training & Validation Loss")
plt.grid(True); plt.legend()
txt = (f"Epochs: {num_epochs}\nBatch: {batch_size}\nLR: {learning_rate}\nWD: {weight_decay}\n"
       f"DynReweight: {dyn_reweight}, every {reweight_every_epochs} ep\n"
       f"mix={dyn_mix_with_base}, clamp={dyn_scale_clamp}")
plt.gcf().text(0.98, 0.5, txt, fontsize=9, va='center', ha='right', bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout(); plt.savefig(fig_path); plt.close()

print(f"📑 日志: {log_path}")
print(f"📈 曲线: {fig_path}")

# ========================= 可视化：动态权重 & MAE(λ) =========================
# 1) 最终权重（归一化到均值1）折线
if len(weight_history) > 0:
    final_w = weight_history[-1]
    final_w_norm = final_w / (final_w.mean() + 1e-8)
    df_w = pd.DataFrame({"wavelength_nm": wavelengths, "weight": final_w, "weight_norm": final_w_norm})
    df_w.to_csv(dyn_weight_csv, index=False)

    plt.figure(figsize=(10, 4.5))
    plt.plot(wavelengths, final_w_norm, marker='o')
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Dynamic weight (normalized)")
    plt.title("Final per-wavelength dynamic weight (mean-normalized)")
    plt.grid(True); plt.tight_layout(); plt.savefig(dyn_weight_line); plt.close()
    print(f"🖼 逐波长最终权重: {dyn_weight_line}")
    print(f"📄 最终权重CSV: {dyn_weight_csv}")

# 2) 权重演化热力图（y=发生更新的epoch，x=波长）
if len(weight_history) > 1:
    W = np.vstack([w / (w.mean() + 1e-8) for w in weight_history])  # [K,76]
    plt.figure(figsize=(12, max(3, len(weight_history)*0.4)))
    im = plt.imshow(W, aspect='auto', cmap='viridis',
                    extent=[wavelengths.min(), wavelengths.max(), weight_epochs[-1]+1, weight_epochs[0]-1])
    plt.colorbar(im, label="Dynamic weight (normalized)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Epoch (reweight checkpoints)")
    plt.title("Dynamic weight evolution across wavelengths")
    plt.tight_layout(); plt.savefig(dyn_weight_heat); plt.close()
    print(f"🖼 权重热力图: {dyn_weight_heat}")

# 3) 最后一次（或 EMA）逐波长 MAE 折线（归一化）
if mae_last_curve is not None:
    mae_norm = mae_last_curve / (mae_last_curve.mean() + 1e-8)
    plt.figure(figsize=(10, 4.5))
    plt.plot(wavelengths, mae_norm, marker='s')
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Val MAE (normalized)")
    plt.title("Validation per-wavelength MAE (normalized)")
    plt.grid(True); plt.tight_layout(); plt.savefig(mae_line_fig); plt.close()
    print(f"🖼 验证逐波长MAE: {mae_line_fig}")

print("🎯 训练完成 & 动态权重/误差可视化已导出。")
