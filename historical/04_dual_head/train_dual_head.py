# -*- coding: utf-8 -*-
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from MSELossCompat import  MSELossCompat
from dataset import SpectralDataset          # 直接复用你现有的数据读取
from model_dual_head import DualHeadSpectralNet

# ============ 简单损失：MSE + λ_dc·DC + λ_tv·TV + λ_coef·||coef||² ============
class MSE_DC_TV_Loss(nn.Module):
    def __init__(self, lambda_dc=0.0, lambda_tv=1e-3, lambda_coef=1e-4):
        super().__init__()
        self.lambda_dc = float(lambda_dc)
        self.lambda_tv = float(lambda_tv)
        self.lambda_coef = float(lambda_coef)
    def forward(self, pred, target, coef):
        main = torch.mean((pred - target) ** 2)                                  # MSE
        dc   = torch.mean((pred.mean(1, keepdim=True) - target.mean(1, keepdim=True)) ** 2)  # DC
        tv   = torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2)                     # TV-L2
        cl2  = torch.mean(coef ** 2)                                             # coef L2
        return main + self.lambda_dc*dc + self.lambda_tv*tv + self.lambda_coef*cl2

def train_dual_head():
    # ======== 路径（改成你的本地路径） ========
    train_dir = r"D:\Desktop\dataset\guasnoise1-2040\train"
    val_dir   = r"D:\Desktop\dataset\guasnoise1-2040\val"
    save_path = r"D:\Desktop\model\double-RESCNN\best_dual_head.pth"
    fig_path  = r"D:\Desktop\model\double-RESCNN\loss_curve_dual_head.png"

    # ======== 设备 / 超参 ========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size    = 64
    epochs        = 100
    lr            = 1e-3
    weight_decay  = 1e-4

    # ======== 数据（直接用你现有的 SpectralDataset） ========
    # 它会从 data_dir 中读取配对的 rgb_*.npy (4,6,100,3) 和 spectral_*.npy (76,)
    # 并做聚合/灰平衡，返回 x:(3,4,6), y:(76,) —— 与本模型完全匹配。
    train_dataset = SpectralDataset(train_dir, normalize_spectra=False)
    val_dataset   = SpectralDataset(val_dir,   normalize_spectra=False)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # ======== 模型 / 损失 / 优化器 ========
    model = DualHeadSpectralNet(
        n_wave=76, token_grid=(4,6),
        base_ch=64, n_blocks=4, k_basis=12, out_activation="softplus"
    ).to(device)
    criterion = MSE_DC_TV_Loss(lambda_dc=0.0, lambda_tv=1e-3, lambda_coef=1e-4)
    # 原来：criterion = nn.MSELoss()
    "criterion = MSELossCompat()"
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def evaluate():
        model.eval()
        loss_sum, n = 0.0, 0
        for x, y in val_loader:
            x = x.to(device).float()
            y = y.to(device).float()
            pred, coef = model(x)
            loss = criterion(pred, y, coef)
            bs = x.size(0)
            loss_sum += loss.item() * bs
            n += bs
        return loss_sum / max(1, n)

    # ======== 训练主循环（仅记录 train/val 损失） ========
    best = 1e9
    tr_hist, va_hist = [], []
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for ep in range(1, epochs+1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device).float()
            y = y.to(device).float()
            pred, coef = model(x)
            loss = criterion(pred, y, coef)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            loss_sum += loss.item() * bs
            n += bs
        train_loss = loss_sum / max(1, n)
        val_loss   = evaluate()
        tr_hist.append(train_loss); va_hist.append(val_loss)
        dt = time.time() - t0
        print(f"[{ep:03d}/{epochs}] train={train_loss:.6f}  val={val_loss:.6f}  time={dt:.1f}s")

        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": val_loss}, save_path)
            print(f"  -> saved best to {save_path}")

    # ======== 单张损失曲线 ========
    plt.figure(figsize=(8,4))
    plt.plot(tr_hist, label="train")
    plt.plot(va_hist, label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(fig_path); plt.close()
    print(f"Saved loss curve to {fig_path}")

if __name__ == "__main__":
    train_dual_head()
