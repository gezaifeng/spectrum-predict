import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.utils.data import DataLoader
import torch.optim as optim
from model import SpectralCNNTransformer
from dataset import SpectralDataset
from WeightedMSEloss import WeightedMSELoss, build_peak_weight
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========================= 参数配置 =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = "D:\\Desktop\\dataset\\guasnoise1-1020\\train"
val_dir = "D:\\Desktop\\dataset\\guasnoise1-1020\\val"
save_path = "D:\Desktop\model\CNN spectrum predict\\best_model.pth"
log_path = "D:\Desktop\model\CNN spectrum predict\\training_log.csv"
fig_path = "D:\Desktop\model\CNN spectrum predict\\loss_curve.png"

num_epochs = 50
batch_size = 64
learning_rate = 0.001
weight_decay = 1e-4
step_size = 50
gamma = 0.5

# ========================= 数据加载 =========================
train_dataset = SpectralDataset(train_dir, normalize_spectra=False)
val_dataset = SpectralDataset(val_dir,normalize_spectra=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ========================= 模型与优化器 =========================
model = SpectralCNNTransformer().to(device)
criterion = nn.MSELoss()
"""WeightedMSELoss(weight=build_peak_weight(train_dataset, output_dim=76, alpha=3.0).to(device))"""
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# ========================= 指标计算函数 =========================
def compute_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    psnr = 10 * math.log10(1 / (rmse ** 2 + 1e-8))
    sam = np.mean([
        math.acos(np.clip(np.dot(t, p) / (np.linalg.norm(t) * np.linalg.norm(p) + 1e-8), -1, 1))
        for t, p in zip(y_true, y_pred)
    ])
    sam = math.degrees(sam)
    return rmse, mae, r2, psnr, sam

# ========================= 日志记录变量 =========================
log = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "learning_rate": [],
    "rmse": [],
    "mae": [],
    "r2": [],
    "psnr": [],
    "sam": []
}

best_val_loss = float("inf")

# ========================= 训练循环 =========================
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, spectra in train_loader:
        images, spectra = images.to(device).float(), spectra.to(device).float()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, spectra)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # ===== 验证 & 指标评估 =====
    model.eval()
    val_loss = 0.0
    preds, gts = [], []

    with torch.no_grad():
        for images, spectra in val_loader:
            images, spectra = images.to(device).float(), spectra.to(device).float()
            outputs = model(images)
            val_loss += criterion(outputs, spectra).item()
            preds.append(outputs)
            gts.append(spectra)

    val_loss /= len(val_loader)
    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    rmse, mae, r2, psnr, sam = compute_metrics(gts, preds)
    current_lr = optimizer.param_groups[0]["lr"]

    # ===== 日志记录 =====
    log["epoch"].append(epoch + 1)
    log["train_loss"].append(train_loss)
    log["val_loss"].append(val_loss)
    log["learning_rate"].append(current_lr)
    log["rmse"].append(rmse)
    log["mae"].append(mae)
    log["r2"].append(r2)
    log["psnr"].append(psnr)
    log["sam"].append(sam)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, RMSE: {rmse:.4f}, PSNR: {psnr:.2f}, SAM: {sam:.2f}°, LR: {current_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")

    scheduler.step()

# ========================= 保存日志 & 图像 =========================
df = pd.DataFrame(log)
df.to_csv(log_path, index=False)

xls_path = log_path.replace(".csv", ".xlsx")
df.to_excel(xls_path, index=False)
print(f"📑 指标日志已保存到: {xls_path}")

plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.grid(True)
plt.legend()

hyperparams = (
    f"Epochs: {num_epochs}\n"
    f"Batch Size: {batch_size}\n"
    f"Learning Rate: {learning_rate}\n"
    f"Weight Decay: {weight_decay}\n"
    f"LR Step Size: {step_size}\n"
    f"Gamma: {gamma}"
)
plt.gcf().text(0.98, 0.5, hyperparams, fontsize=10, va='center', ha='right',
               bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(fig_path)
plt.close()

print(f"📊 训练日志已保存到: {log_path}")
print(f"📈 Loss 曲线图已保存到: {fig_path}")
print("🎯 训练完成！")
