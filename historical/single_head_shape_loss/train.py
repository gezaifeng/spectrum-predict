# -*- coding: utf-8 -*-
import os
import re
import time
import math
import json
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import SpectralDatasetLogRatio
from model import ResCNN
from SpectralReconstructionLoss import SpectralReconstructionLoss, spectral_angle_mapper, first_diff

try:
    from WeightedMSEloss import build_peak_weight
except Exception:
    build_peak_weight = None


def extract_sample_ids(file_list):
    ids = []
    for f in file_list:
        m = re.match(r"rgb_(.+)\.npy$", f)
        if m:
            ids.append(m.group(1))
    return set(ids)


def inspect_train_val_overlap(train_ds, val_ds):
    train_ids = extract_sample_ids(train_ds.rgb_files)
    val_ids = extract_sample_ids(val_ds.rgb_files)
    overlap = sorted(train_ids & val_ids)
    if overlap:
        print("[警告] train/val 发现同名样本ID重叠，可能存在表面验证集问题。")
        print("        overlap sample ids (前20个):", overlap[:20])
    else:
        print("[检查] train/val 未发现同名样本ID重叠。")


@torch.no_grad()
def compute_metrics(pred, target, worst_ratio=0.1):
    err2 = (pred - target) ** 2
    abs_err = (pred - target).abs()

    mse = err2.mean().item()
    rmse = math.sqrt(max(mse, 0.0))
    mae = abs_err.mean().item()

    diff1_pred = first_diff(pred)
    diff1_tgt = first_diff(target)
    diff1_mae = (diff1_pred - diff1_tgt).abs().mean().item()

    sam = spectral_angle_mapper(pred, target).mean().item()

    sample_mse = err2.mean(dim=1)
    k = max(1, int(sample_mse.numel() * worst_ratio))
    worst_mse = torch.topk(sample_mse, k=k, largest=True).values.mean().item()

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "sam": sam,
        "diff1_mae": diff1_mae,
        "worst10_mse": worst_mse,
    }


def build_loss_weight(train_ds, output_dim=190, alpha_peak=3.0):
    if build_peak_weight is None:
        print("[提示] 未找到 build_peak_weight，回退为全1权重。")
        return torch.ones(output_dim, dtype=torch.float32)
    return build_peak_weight(train_ds, output_dim=output_dim, alpha=alpha_peak).float()


def train():
    # ========= 改成你的路径 =========
    train_dir = r"D:\Desktop\Picture 3.9 2nm\ratio\train"
    val_dir   = r"D:\Desktop\Picture 3.9 2nm\ratio\val"
    save_ckpt = r"D:\Desktop\model\RESCNN_4.12\best.pth"
    save_fig  = r"D:\Desktop\model\RESCNN_4.12\loss_curve.png"
    save_log  = r"D:\Desktop\model\RESCNN_4.12\training_log.json"
    # ===============================

    os.makedirs(os.path.dirname(save_ckpt), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 100
    lr = 1e-3
    weight_decay = 1e-4

    # 特征预处理
    clip_value = 8.0
    use_zscore = True

    # 单头网络，从头训练，不加载任何预训练权重
    out_activation = "linear"   # 可改成 "softplus" 做非负约束对照

    # 组合损失超参数
    alpha_peak = 3.0
    lambda_diff1 = 0.50
    lambda_cos = 0.20
    lambda_sam = 0.20
    lambda_topk = 0.30
    lambda_dc = 0.05
    topk_ratio = 0.10

    train_ds = SpectralDatasetLogRatio(train_dir, clip_value=clip_value, use_zscore=use_zscore)
    val_ds   = SpectralDatasetLogRatio(val_dir,   clip_value=clip_value, use_zscore=use_zscore)
    inspect_train_val_overlap(train_ds, val_ds)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResCNN(
        out_dim=190,
        base_ch=64,
        n_blocks=6,
        mlp_hidden=256,
        dropout=0.10,
        out_activation=out_activation,
    ).to(device)

    weight = build_loss_weight(train_ds, output_dim=190, alpha_peak=alpha_peak).to(device)
    criterion = SpectralReconstructionLoss(
        weight=weight,
        lambda_diff1=lambda_diff1,
        lambda_cos=lambda_cos,
        lambda_sam=lambda_sam,
        lambda_topk=lambda_topk,
        lambda_dc=lambda_dc,
        topk_ratio=topk_ratio,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def evaluate():
        model.eval()
        total_loss = 0.0
        total_n = 0
        pred_all, target_all = [], []

        for x, y in val_ld:
            x = x.to(device).float()
            y = y.to(device).float()
            pred = model(x)
            comps = criterion.forward_components(pred, y)
            bs = x.size(0)
            total_loss += comps["total"].item() * bs
            total_n += bs
            pred_all.append(pred.detach())
            target_all.append(y.detach())

        pred_all = torch.cat(pred_all, dim=0)
        target_all = torch.cat(target_all, dim=0)
        metrics = compute_metrics(pred_all, target_all, worst_ratio=topk_ratio)
        metrics["loss"] = total_loss / max(1, total_n)
        return metrics

    best_score = float("inf")
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_sam": [],
        "val_diff1_mae": [],
        "val_worst10_mse": [],
        "lr": [],
    }

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0

        for x, y in train_ld:
            x = x.to(device).float()
            y = y.to(device).float()

            if not torch.isfinite(x).all():
                raise RuntimeError(f"x has NaN/Inf, range=({x.min().item():.3g},{x.max().item():.3g})")
            if not torch.isfinite(y).all():
                raise RuntimeError(f"y has NaN/Inf, range=({y.min().item():.3g},{y.max().item():.3g})")

            pred = model(x)
            loss = criterion(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError("loss is NaN/Inf")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            loss_sum += loss.item() * bs
            n += bs

        train_loss = loss_sum / max(1, n)
        val_metrics = evaluate()
        val_loss = val_metrics["loss"]
        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_sam"].append(val_metrics["sam"])
        history["val_diff1_mae"].append(val_metrics["diff1_mae"])
        history["val_worst10_mse"].append(val_metrics["worst10_mse"])
        history["lr"].append(cur_lr)

        print(
            f"[{ep:03d}/{epochs}] "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"mae={val_metrics['mae']:.6f}  rmse={val_metrics['rmse']:.6f}  "
            f"sam={val_metrics['sam']:.6f}  diff1_mae={val_metrics['diff1_mae']:.6f}  "
            f"worst10_mse={val_metrics['worst10_mse']:.6f}  lr={cur_lr:.2e}  time={dt:.1f}s"
        )

        # 用组合 val_loss 作为主保存准则，避免纯 MSE 虚低
        if val_loss < best_score:
            best_score = val_loss
            torch.save({
                "model": model.state_dict(),
                "epoch": ep,
                "best_val_loss": val_loss,
                "clip_value": clip_value,
                "use_zscore": use_zscore,
                "out_activation": out_activation,
                "loss_cfg": {
                    "alpha_peak": alpha_peak,
                    "lambda_diff1": lambda_diff1,
                    "lambda_cos": lambda_cos,
                    "lambda_sam": lambda_sam,
                    "lambda_topk": lambda_topk,
                    "lambda_dc": lambda_dc,
                    "topk_ratio": topk_ratio,
                },
            }, save_ckpt)
            print(f"  -> saved best to {save_ckpt}")

    with open(save_log, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(9, 4.5))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Composite Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_fig, dpi=200)
    plt.close()

    extra_fig = save_fig.replace("loss_curve", "val_metrics")
    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["val_mae"], label="val_mae")
    plt.plot(history["epoch"], history["val_rmse"], label="val_rmse")
    plt.plot(history["epoch"], history["val_sam"], label="val_sam")
    plt.plot(history["epoch"], history["val_diff1_mae"], label="val_diff1_mae")
    plt.plot(history["epoch"], history["val_worst10_mse"], label="val_worst10_mse")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(extra_fig, dpi=200)
    plt.close()

    print("saved:", save_ckpt)
    print("saved:", save_fig)
    print("saved:", extra_fig)
    print("saved:", save_log)


if __name__ == "__main__":
    train()
