# -*- coding: utf-8 -*-
"""
同时训练对比三种模型（CNN 残差 / MicroU-Net / 双头残差），并输出统一表征
- 训练策略：MSE 或 MSE+DC+TV（不使用自定义权重）
- 指标：RMSE / MAE / SAM / PSNR / R2（验证集）
- 输出：
  outdir/
    CNN_MSE_DC_TV/
      best.ckpt, train_log.csv, loss_curve.png, val_metrics_per_sample.csv
    UNet_MSE_DC_TV/
      ...
    DualHead_MSE_DC_TV/
      ...
    summary_baseline.csv   # 三模型的验证集均值指标
    model_profile.csv      # 参数量（M）与估算 FLOPs（可选粗略）
"""
"""python .\train_three_models_compare.py `
  --train_root "D:\Desktop\dataset\guasnoise1-1020\train" `
  --val_root   "D:\Desktop\dataset\guasnoise1-1020\val" `
  --outdir     ".\three_model_runs" `
  --epochs     50 `
  --batch_size 128 `
  --lr         1e-3 `
  --strategy   mse `
  --lambda_dc  0.10 `
  --lambda_tv  0.02 `
  --device     cuda"""

import os, argparse, math, json, random
from typing import Dict, Any, List, Tuple

import math   # ← 新增：修复 NameError
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== 你的工程内模块 =====
from dataset import SpectralDataset                    # (B,3,4,6)->(B,76)
from model_cnn import CNNResidualBaseline              # 上条消息已给出
from model_unet import MicroUNetSpectral               # 上条消息已给出
from model_dual_head import DualHeadSpectralNet        # 你工程已有

# ---------- 指标 ----------
def rmse_vec(pred, gt): return torch.sqrt(torch.mean((pred-gt)**2, dim=-1))
def mae_vec(pred, gt):  return torch.mean(torch.abs(pred-gt), dim=-1)
def sam_vec(pred, gt, eps=1e-8):
    num = torch.sum(pred*gt, dim=-1)
    den = torch.linalg.norm(pred, dim=-1) * torch.linalg.norm(gt, dim=-1) + eps
    cos = torch.clamp(num/den, -1.0, 1.0)
    return torch.arccos(cos)  # radians

def psnr_scalar(y_true, y_pred):
    mse = torch.mean((y_true - y_pred)**2).item()
    return 10.0 * math.log10(1.0 / (mse + 1e-12))

def r2_scalar(y_true, y_pred):
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    ss_res = float(np.sum((yt - yp)**2))
    ss_tot = float(np.sum((yt - yt.mean())**2) + 1e-12)
    return 1.0 - ss_res/ss_tot

# ---------- 损失 ----------
def loss_mse(pred, target):
    return F.mse_loss(pred, target)

def loss_dc_mean(pred, target):
    # DC：均值一致性（输出光谱的均值对齐）
    mp = pred.mean(dim=1, keepdim=True)
    mt = target.mean(dim=1, keepdim=True)
    return torch.mean((mp - mt) ** 2)

def loss_tv_l2(pred):
    # TV：光谱维一阶差分 L2 平滑
    diff = pred[:, 1:] - pred[:, :-1]
    return torch.mean(diff ** 2)

# ---------- 工具 ----------
def set_seed(seed: int = 2025):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6  # M params

def forward_pred(model, x, model_tag: str):
    """
    兼容三模型输出：返回 pred 张量 (B,76)
    model_tag in {"CNN","UNet","DualHead"}
    """
    out = model(x)
    if model_tag == "DualHead":
        if isinstance(out, (tuple, list)):
            return out[0]
    return out

def build_model(tag: str, device: torch.device, n_wave: int = 76):
    """
    tag: CNN / UNet / DualHead
    """
    if tag == "CNN":
        m = CNNResidualBaseline(out_dim=n_wave, base_ch=56, n_blocks=4, mlp_hidden=128, out_activation="linear").to(device)
    elif tag == "UNet":
        m = MicroUNetSpectral(out_dim=n_wave, c1=24, c2=48, c3=96, mlp_hidden=128, out_activation="linear").to(device)
    elif tag == "DualHead":
        m = DualHeadSpectralNet(n_wave=n_wave, token_grid=(4,6), base_ch=64, n_blocks=4, k_basis=12, out_activation="softplus").to(device)
    else:
        raise ValueError("Unknown model tag")
    return m

def train_or_eval_epoch(model, loader, device, model_tag: str,
                        optimizer=None, strategy="mse", lambda_dc=0.1, lambda_tv=0.02):
    """
    strategy: "mse" or "mse_dc_tv"
    """
    training = optimizer is not None
    model.train(training)

    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()

        if training: optimizer.zero_grad()
        pred = forward_pred(model, xb, model_tag)

        loss = loss_mse(pred, yb)
        if strategy == "mse_dc_tv":
            loss = loss + lambda_dc * loss_dc_mean(pred, yb) + lambda_tv * loss_tv_l2(pred)

        if training:
            loss.backward()
            optimizer.step()

        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device, model_tag: str) -> Tuple[pd.DataFrame, float]:
    model.eval()
    rows = []
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        pred = forward_pred(model, xb, model_tag)

        rmse = rmse_vec(pred, yb)
        mae  = mae_vec(pred, yb)
        sam  = sam_vec(pred, yb)

        for i in range(xb.size(0)):
            rows.append({
                "rmse": float(rmse[i].item()),
                "mae":  float(mae[i].item()),
                "sam":  float(sam[i].item()),
                "psnr": psnr_scalar(yb[i], pred[i]),
                "r2":   r2_scalar(yb[i], pred[i])
            })
    df = pd.DataFrame(rows)
    return df, float(df["rmse"].mean())

def save_curve(log_df: pd.DataFrame, title: str, out_png: str):
    plt.figure(figsize=(7.0,4.0), dpi=150)
    plt.plot(log_df["epoch"], log_df["train"], label="Train")
    plt.plot(log_df["epoch"], log_df["val"],   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title)
    plt.grid(alpha=0.3); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(out_png); plt.close()

# ---------- 主流程 ----------
def run_one_model(tag: str, args, device) -> Dict[str, Any]:
    """
    训练并评测单个模型，返回路径与均值指标
    """
    os.makedirs(args.outdir, exist_ok=True)
    model = build_model(tag, device)
    run_name = f"{tag}_{'MSE' if args.strategy=='mse' else 'MSE_DC_TV'}"
    run_dir  = os.path.join(args.outdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 数据
    train_ds = SpectralDataset(args.train_root, normalize_spectra=False)
    val_ds   = SpectralDataset(args.val_root,   normalize_spectra=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 优化器 & 调度
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # 训练
    best_val = float("inf"); best_ep=-1; wait=0
    ckpt_path = os.path.join(run_dir, "best.ckpt")
    log = {"epoch":[], "train":[], "val":[], "lr":[]}

    for ep in range(1, args.epochs+1):
        tr = train_or_eval_epoch(model, train_loader, device, tag, optimizer,
                                 strategy=args.strategy, lambda_dc=args.lambda_dc, lambda_tv=args.lambda_tv)
        va = train_or_eval_epoch(model, val_loader, device, tag, optimizer=None,
                                 strategy=args.strategy, lambda_dc=args.lambda_dc, lambda_tv=args.lambda_tv)
        sched.step(va)
        lr = optimizer.param_groups[0]["lr"]

        log["epoch"].append(ep); log["train"].append(tr); log["val"].append(va); log["lr"].append(lr)
        print(f"[{run_name}] Ep {ep:03d}/{args.epochs} | Train {tr:.4f} | Val {va:.4f} | LR {lr:.6f}")

        if va < best_val - 1e-8:
            best_val = va; best_ep = ep; wait = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Save best @ ep {ep} (val={va:.6f})")
        else:
            wait += 1
            if wait >= args.early_patience:
                print("  -> Early stop.")
                break

    # 曲线
    df_log = pd.DataFrame(log)
    df_log.to_csv(os.path.join(run_dir, "train_log.csv"), index=False)
    save_curve(df_log, f"{run_name} — loss", os.path.join(run_dir, "loss_curve.png"))

    # 载入最佳并评测
    model_best = build_model(tag, device)
    model_best.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model_best.to(device)

    df_val, rmse_mean = evaluate(model_best, val_loader, device, tag)
    df_val.to_csv(os.path.join(run_dir, "val_metrics_per_sample.csv"), index=False)

    # 模型规模与（可选）粗 FLOPs（此处只报参数量，FLOPs 视需要再接 ptflops）
    profile = {
        "model": tag,
        "strategy": ("MSE" if args.strategy=="mse" else "MSE_DC_TV"),
        "params_M": round(count_params(model_best), 3)
    }

    with open(os.path.join(run_dir, "cfg.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), **profile}, f, indent=2, ensure_ascii=False)

    return {
        "model": tag,
        "strategy": ("MSE" if args.strategy=="mse" else "MSE_DC_TV"),
        "ckpt": ckpt_path,
        "rmse_mean": float(df_val["rmse"].mean()),
        "mae_mean":  float(df_val["mae"].mean()),
        "sam_mean":  float(df_val["sam"].mean()),
        "psnr_mean": float(df_val["psnr"].mean()),
        "r2_mean":   float(df_val["r2"].mean()),
        "params_M":  profile["params_M"],
        "run_dir":   run_dir
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, required=True)
    ap.add_argument("--val_root",   type=str, required=True)
    ap.add_argument("--outdir",     type=str, default="./three_model_runs")
    ap.add_argument("--epochs",     type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--strategy",   type=str, choices=["mse","mse_dc_tv"], default="mse_dc_tv",
                    help="训练策略：mse 或 mse_dc_tv")
    ap.add_argument("--lambda_dc",  type=float, default=0.10)
    ap.add_argument("--lambda_tv",  type=float, default=0.02)
    ap.add_argument("--early_patience", type=int, default=20)
    ap.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed",       type=int, default=2025)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for tag in ["CNN", "UNet", "DualHead"]:
        print(f"\n========== Train & Eval [{tag}] with [{args.strategy}] ==========")
        rec = run_one_model(tag, args, device)
        results.append(rec)

    # 汇总表征
    df_sum = pd.DataFrame([{
        "exp": f"{r['model']}_{r['strategy']}",
        "rmse_mean": r["rmse_mean"],
        "mae_mean":  r["mae_mean"],
        "sam_mean":  r["sam_mean"],
        "psnr_mean": r["psnr_mean"],
        "r2_mean":   r["r2_mean"]
    } for r in results])
    df_sum.to_csv(os.path.join(args.outdir, "summary_baseline.csv"), index=False, encoding="utf-8")

    # 模型规模
    df_prof = pd.DataFrame([{
        "exp": f"{r['model']}_{r['strategy']}",
        "params_M": r["params_M"],
        "ckpt": r["ckpt"]
    } for r in results])
    df_prof.to_csv(os.path.join(args.outdir, "model_profile.csv"), index=False, encoding="utf-8")

    # 也保存一个 experiments.json（供鲁棒性评测脚本直接使用）
    exp_map = { f"{r['model']}_{r['strategy']}": {
                    "type": ("dual" if r["model"]=="DualHead" else "single" if r["model"] in ["CNN","UNet"] else "single"),
                    "ckpt": r["ckpt"]
               } for r in results }
    with open(os.path.join(args.outdir, "experiments.json"), "w", encoding="utf-8") as f:
        json.dump(exp_map, f, indent=2, ensure_ascii=False)

    print("\n[OK] All done. Outputs under:", args.outdir)
    for r in results:
        print(" ", f"{r['model']}_{r['strategy']}", "->", r["ckpt"])

if __name__ == "__main__":
    main()
