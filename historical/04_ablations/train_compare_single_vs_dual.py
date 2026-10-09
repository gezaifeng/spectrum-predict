# -*- coding: utf-8 -*-
"""
同时训练 & 对比 单头 / 双头 光谱重建网络
- 模型：SingleHeadSpectralNet / DualHeadSpectralNet
- 损失：MSE / MSE+DC / MSE+TV / MSE+DC+TV
- 统一数据、统一超参、统一评测，公平对比
- 输出：
  outdir/
    SingleHead_MSE_DC_TV/
      best.ckpt, train_log.csv, loss_curve.png, val_metrics_per_sample.csv
    DualHead_MSE_DC_TV/
      best.ckpt, train_log.csv, loss_curve.png, val_metrics_per_sample.csv
    summary_baseline.csv  # 跨模型的基线均值指标（RMSE/MAE/SAM）
"""
"""python .\train_compare_single_vs_dual.py `
  --train_root "D:\Desktop\dataset\guasnoise1-1020\train" `
  --val_root   "D:\Desktop\dataset\guasnoise1-1020\val" `
  --outdir     ".\compare_runs" `
  --models     "single,dual" `
  --losses     "mse_dc_tv" `
  --epochs     50 `
  --batch_size 64 `
  --lr         1e-5 `
  --lambda_dc  0.10 `
  --lambda_tv  0.02 `
  --device     "cuda"""

import os, json, math, argparse, random
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==== 你的工程内模块（按你现有文件命名）====
from dataset import SpectralDataset                      # (B,3,4,6)->(B,76)
from model_single_head import SingleHeadSpectralNet
from model_dual_head import DualHeadSpectralNet          # forward -> (pred, attn)

# ------------------ 指标 ------------------
def rmse_vec(pred, gt): return torch.sqrt(torch.mean((pred-gt)**2, dim=-1))
def mae_vec(pred, gt):  return torch.mean(torch.abs(pred-gt), dim=-1)
def sam_vec(pred, gt, eps=1e-8):
    num = torch.sum(pred*gt, dim=-1)
    den = torch.linalg.norm(pred, dim=-1) * torch.linalg.norm(gt, dim=-1) + eps
    cos = torch.clamp(num/den, -1.0, 1.0)
    return torch.arccos(cos)  # [rad]

def psnr_scalar(y_true, y_pred):
    mse = torch.mean((y_true - y_pred)**2).item()
    return 10.0 * math.log10(1.0 / (mse + 1e-12))

def r2_scalar(y_true, y_pred):
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    ss_res = np.sum((yt - yp)**2)
    ss_tot = np.sum((yt - yt.mean())**2) + 1e-12
    return float(1.0 - ss_res/ss_tot)

# ------------------ 损失项 ------------------
def loss_mse(pred, target):
    return F.mse_loss(pred, target)

def loss_dc_mean(pred, target):
    # DC：均值一致性（每个样本在光谱维度的均值对齐）
    mp = pred.mean(dim=1, keepdim=True)
    mt = target.mean(dim=1, keepdim=True)
    return torch.mean((mp - mt) ** 2)

def loss_tv_l2(pred):
    # TV：预测谱在波长维度的一阶差分 L2 平滑
    diff = pred[:, 1:] - pred[:, :-1]
    return torch.mean(diff ** 2)

# ------------------ 训练/验证 ------------------
def set_seed(seed: int = 2025):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def forward_pred(model, x):
    """兼容单头/双头：返回 pred 张量 (B,76)"""
    out = model(x)
    if isinstance(out, tuple) or isinstance(out, list):
        return out[0]
    return out

def run_one_epoch(model, loader, device, optimizer=None,
                  use_dc=False, use_tv=False, lambda_dc=0.1, lambda_tv=0.02):
    is_train = optimizer is not None
    model.train(is_train)

    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()

        if is_train: optimizer.zero_grad()

        pred = forward_pred(model, xb)

        loss = loss_mse(pred, yb)
        if use_dc: loss = loss + lambda_dc * loss_dc_mean(pred, yb)
        if use_tv: loss = loss + lambda_tv * loss_tv_l2(pred)

        if is_train:
            loss.backward()
            optimizer.step()

        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def eval_metrics(model, loader, device) -> Tuple[pd.DataFrame, float]:
    model.eval()
    rows = []
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        pred = forward_pred(model, xb)

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
    return df, df["rmse"].mean()

def build_model(model_type: str, device, n_wave: int = 76):
    if model_type == "single":
        model = SingleHeadSpectralNet(out_dim=n_wave, width=64, depth=4,
                                      drop=0.0, out_activation="linear",
                                      global_pool="avg").to(device)
        name = "SingleHead"
    elif model_type == "dual":
        model = DualHeadSpectralNet(n_wave=n_wave, token_grid=(4,6),
                                    base_ch=64, n_blocks=4, k_basis=12,
                                    out_activation="softplus").to(device)
        name = "DualHead"
    else:
        raise ValueError("model_type 必须是 'single' 或 'dual'")
    return name, model

def loss_tag(loss_cfg: str) -> str:
    mp = {
        "mse": "MSE",
        "mse_dc": "MSE_DC",
        "mse_tv": "MSE_TV",
        "mse_dc_tv": "MSE_DC_TV",
    }
    return mp[loss_cfg]

def parse_loss_flags(loss_cfg: str):
    use_dc = (loss_cfg in ["mse_dc", "mse_dc_tv"])
    use_tv = (loss_cfg in ["mse_tv", "mse_dc_tv"])
    return use_dc, use_tv

def train_one_config(model_type: str, loss_cfg: str, args, device) -> Dict[str, Any]:
    """训练单个配置（模型×损失），返回产物路径与指标"""
    # 输出目录
    model_name, model = build_model(model_type, device)
    loss_name = loss_tag(loss_cfg)
    run_dir = os.path.join(args.outdir, f"{model_name}_{loss_name}")
    os.makedirs(run_dir, exist_ok=True)

    # 记录配置
    with open(os.path.join(run_dir, "cfg.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "model_name":model_name, "loss_name":loss_name}, f, indent=2, ensure_ascii=False)

    # 数据
    train_ds = SpectralDataset(args.train_root, normalize_spectra=False)
    val_ds   = SpectralDataset(args.val_root,   normalize_spectra=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 优化器/调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    use_dc, use_tv = parse_loss_flags(loss_cfg)

    # 训练循环
    log = {"epoch": [], "train": [], "val": [], "lr": []}
    best_val = float("inf"); best_ep = -1; wait = 0
    ckpt_path = os.path.join(run_dir, "best.ckpt")

    for ep in range(1, args.epochs + 1):
        tr = run_one_epoch(model, train_loader, device, optimizer,
                           use_dc=use_dc, use_tv=use_tv,
                           lambda_dc=args.lambda_dc, lambda_tv=args.lambda_tv)
        va = run_one_epoch(model, val_loader, device, optimizer=None,
                           use_dc=use_dc, use_tv=use_tv,
                           lambda_dc=args.lambda_dc, lambda_tv=args.lambda_tv)
        scheduler.step(va)
        lr = optimizer.param_groups[0]["lr"]

        log["epoch"].append(ep); log["train"].append(tr); log["val"].append(va); log["lr"].append(lr)
        print(f"[{model_name}-{loss_name}] Ep {ep:03d}/{args.epochs} | Train {tr:.4f} | Val {va:.4f} | LR {lr:.6f}")

        if va < best_val - 1e-8:
            best_val = va; best_ep = ep; wait = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Save best @ epoch {ep} (val={va:.6f})")
        else:
            wait += 1
            if wait >= args.early_patience:
                print("  -> Early stop.")
                break

    # 导出训练曲线
    df_log = pd.DataFrame(log)
    df_log.to_csv(os.path.join(run_dir, "train_log.csv"), index=False)
    plt.figure(figsize=(7.2,4.2), dpi=150)
    plt.plot(df_log["epoch"], df_log["train"], label="Train")
    plt.plot(df_log["epoch"], df_log["val"],   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{model_name}-{loss_name} — loss")
    plt.grid(alpha=0.3); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png")); plt.close()

    # 用 best.ckpt 评估验证集逐样本指标
    model_name2, model2 = build_model(model_type, device)  # fresh
    model2.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model2.to(device)
    df_val, rmse_mean = eval_metrics(model2, val_loader, device)
    df_val.to_csv(os.path.join(run_dir, "val_metrics_per_sample.csv"), index=False)

    return {
        "model": model_name,
        "loss": loss_name,
        "ckpt": ckpt_path,
        "val_rmse_mean": rmse_mean,
        "run_dir": run_dir
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, required=True)
    ap.add_argument("--val_root",   type=str, required=True)
    ap.add_argument("--outdir",     type=str, default="./compare_runs")
    ap.add_argument("--models",     type=str, default="single,dual",
                    help="逗号分隔：single,dual（支持单独训练：single 或 dual）")
    ap.add_argument("--losses",     type=str, default="mse_dc_tv",
                    help="逗号分隔：mse,mse_dc,mse_tv,mse_dc_tv")
    ap.add_argument("--epochs",     type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lambda_dc",  type=float, default=0.10)
    ap.add_argument("--lambda_tv",  type=float, default=0.02)
    ap.add_argument("--early_patience", type=int, default=20)
    ap.add_argument("--seed",       type=int, default=2025)
    ap.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    losses = [l.strip() for l in args.losses.split(",") if l.strip()]

    results: List[Dict[str, Any]] = []

    for m in models:
        for l in losses:
            print(f"\n========== Train [{m}] with [{l}] ==========")
            rec = train_one_config(m, l, args, device)
            results.append(rec)

    # 汇总 baseline（便于鲁棒性脚本读取）
    rows = []
    for r in results:
        # 从各自 run_dir 读取验证逐样本指标，汇总 mean
        dfv = pd.read_csv(os.path.join(r["run_dir"], "val_metrics_per_sample.csv"))
        rows.append({
            "exp": f"{r['model']}_{r['loss']}",
            "rmse_mean": float(dfv["rmse"].mean()),
            "mae_mean":  float(dfv["mae"].mean()),
            "sam_mean":  float(dfv["sam"].mean())
        })
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "summary_baseline.csv"), index=False, encoding="utf-8")

    # 记录 ckpt 路径清单（方便后续鲁棒性评测 exp_json）
    ckpt_map = { f"{r['model']}_{r['loss']}": r["ckpt"] for r in results }
    with open(os.path.join(args.outdir, "experiments.json"), "w", encoding="utf-8") as f:
        json.dump(ckpt_map, f, indent=2, ensure_ascii=False)

    print("\n[OK] All done. Outputs under:", args.outdir)
    for k,v in ckpt_map.items():
        print(" ", k, "->", v)

if __name__ == "__main__":
    main()
