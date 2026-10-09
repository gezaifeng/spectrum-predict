# -*- coding: utf-8 -*-
"""
ablate_losses_train_eval.py
python ablate_losses_train_eval.py --train_root "D:\Desktop\dataset\guasnoise1-1020\train" --val_root "D:\Desktop\dataset\guasnoise1-1020\val" --epochs 30 --batch_size 128 --lr 1e-3 --outdir ".\ablation_losses_out" --freq_bins 40 --freq_y count --freq_style line

保持原训练/验证/输出结构不变：
- 4组损失配置：L1_MSE / L2_MSE_DC / L3_MSE_TV / L4_MSE_DC_TV
- 每组训练、保存 best.ckptS、输出 loss_curve、metrics_per_sample.csv、summary.json、curve_fixed_*.png
- 统一输出 boxplot 图与 summary_stats_boxplot.csv

改动点：
- 将“频数分布图”改回“ECDF 图”进行表征
- 输出 5 个性能指标 ECDF：RMSE / MAE / SAM / PSNR / R2
- 同时导出 ECDF 绘图原始数据：ecdf_data.xlsx（多sheet） + ecdf_data_long.csv
"""

import os, glob, json, math, random, argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_dual_head import DualHeadSpectralNet
from SpectralReconstructionLoss import SpectralReconstructionLoss


# =========================
# 随机性控制
# =========================
def set_global_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 预处理（对齐 predict.py）
# =========================
def robust_block_stat(x, axis=2, q=5):
    low  = np.percentile(x, q, axis=axis, keepdims=True)
    high = np.percentile(x, 100 - q, axis=axis, keepdims=True)
    return np.clip(x, low, high).mean(axis=axis)

def prepare_rgb_grid(rgb_npy: np.ndarray) -> torch.Tensor:
    arr = rgb_npy.astype(np.float32)
    if arr.ndim == 4 and arr.shape == (4, 6, 100, 3):
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = robust_block_stat(arr, axis=2, q=5)  # -> (4,6,3)
    elif arr.ndim == 3 and arr.shape == (4, 6, 3):
        if arr.max() > 1.0:
            arr = arr / 255.0
    else:
        raise ValueError(f"Unexpected rgb.npy shape: {arr.shape}")

    x = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1)  # (3,4,6)
    return x


# =========================
# 数据集
# =========================
def list_pairs(root_dir: str) -> List[Tuple[str, str]]:
    rgb_list = sorted(glob.glob(os.path.join(root_dir, "rgb_*.npy")))
    pairs = []
    for rp in rgb_list:
        sp = rp.replace("rgb_", "spectral_")
        if os.path.exists(sp):
            pairs.append((rp, sp))
    if not pairs:
        raise FileNotFoundError(f"No paired rgb_/spectral_ under {root_dir}")
    return pairs

class NpySpectralDataset(torch.utils.data.Dataset):
    def __init__(self, root: str):
        self.pairs = list_pairs(root)
        self.WAVELENGTHS = np.arange(380, 760, 5, dtype=np.float32)  # 76维

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rp, sp = self.pairs[idx]
        rgb = np.load(rp)
        spec = np.load(sp).astype(np.float32)  # (76,)
        x = prepare_rgb_grid(rgb)              # (3,4,6)
        y = torch.tensor(spec, dtype=torch.float32)
        return x, y


# =========================
# 指标
# =========================
def rmse_vec(pred, gt): return np.sqrt(np.mean((pred - gt) ** 2, axis=-1))
def mae_vec(pred, gt):  return np.mean(np.abs(pred - gt), axis=-1)

def sam_vec(pred, gt, eps=1e-8):
    num = np.sum(pred * gt, axis=-1)
    den = np.linalg.norm(pred, axis=-1) * np.linalg.norm(gt, axis=-1) + eps
    return np.arccos(np.clip(num / den, -1.0, 1.0))

def psnr_scalar(y_true, y_pred):
    mse = float(np.mean((y_true - y_pred) ** 2))
    return 10.0 * math.log10(1.0 / (mse + 1e-12))

def r2_scalar(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


# =========================
# boxplot（原功能保留）
# =========================
def _format_num(x):
    ax = abs(float(x))
    if ax < 0.01: return f"{x:.3f}"
    elif ax < 0.1: return f"{x:.2f}"
    else: return f"{x:.1f}"

def _collect_stats(values):
    arr = np.asarray(values).astype(float)
    return dict(mean=float(np.mean(arr)), median=float(np.median(arr)), std=float(np.std(arr)),
                q25=float(np.percentile(arr,25)), q75=float(np.percentile(arr,75)),
                min=float(np.min(arr)), max=float(np.max(arr)))

def _annotate_box(ax, stats, xpos, color_mean="#2c3e50", color_med="#c0392b", y_offset=0.03):
    y_mean, y_med = stats["mean"], stats["median"]
    ax.scatter([xpos],[y_mean], marker="^", s=38, color=color_mean, zorder=3)
    ax.scatter([xpos],[y_med],  marker="o", s=28, facecolors="none", edgecolors=color_med, zorder=3)
    ylim = ax.get_ylim()
    span = (ylim[1]-ylim[0]) if ylim[1] > ylim[0] else 1.0
    ax.text(xpos+0.14, y_mean+y_offset*span, f"μ={_format_num(y_mean)}", color=color_mean, fontsize=8)
    ax.text(xpos+0.14, y_med -y_offset*span, f"med={_format_num(y_med)}", color=color_med, fontsize=8)

def box_with_labels(save_path, values_dict, title="Metric", yscale="linear", linthresh=0.05, ylabel=None):
    labels = list(values_dict.keys())
    data   = [np.asarray(values_dict[k]).astype(float) for k in labels]
    plt.figure(figsize=(5.8,3.9), dpi=160)
    ax = plt.gca()
    ax.boxplot(data, labels=labels, showmeans=True,
               meanprops=dict(marker='D', markerfacecolor='gray', markersize=5))
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_title(f"{title} (symlog)")
    else:
        ax.set_title(f"{title} (linear)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.28, which='both', linestyle='--')
    for i, lab in enumerate(labels, start=1):
        _annotate_box(ax, _collect_stats(values_dict[lab]), xpos=i)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================
# ECDF：绘图 + 导出原始数据
# =========================
# =========================
# ECDF（对齐公共 x 后导出）：Prism 友好（one X column + several Y）
# =========================
def _ecdf_on_grid(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """
    返回 ECDF 在给定 x_grid 上的值（严格ECDF阶梯定义）
    y(x) = (#samples <= x) / N
    """
    s = np.sort(np.asarray(samples, dtype=float).ravel())
    n = len(s)
    if n == 0:
        return np.zeros_like(x_grid, dtype=float)
    return np.searchsorted(s, x_grid, side="right") / float(n)

def build_ecdf_aligned_table(values_dict: Dict[str, np.ndarray],
                             max_points: int = None) -> pd.DataFrame:
    """
    生成对齐后的 ECDF 宽表：
      x | exp1 | exp2 | exp3 | exp4
    x 默认取“所有组 x 的并集去重排序”（最严格阶梯）
    若 max_points 不为 None 且点太多，则使用分位点下采样（仍保持单调，但不再是全量并集）
    """
    # 公共 x：并集去重排序
    all_x = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values_dict.values()], axis=0)
    x_common = np.sort(np.unique(all_x))

    # 可选：点数太多时下采样（防止 Excel 太大/Prism 卡）
    if (max_points is not None) and (len(x_common) > max_points):
        qs = np.linspace(0, 1, max_points)
        x_common = np.unique(np.quantile(all_x, qs))
        x_common = np.sort(x_common)

    df = pd.DataFrame({"x": x_common})
    for tag, arr in values_dict.items():
        df[tag] = _ecdf_on_grid(arr, x_common)
    return df

def plot_ecdf_aligned(save_path: str,
                      values_dict: Dict[str, np.ndarray],
                      title: str,
                      max_points: int = None) -> pd.DataFrame:
    """
    画图并返回对齐后的宽表（Prism 直接可用）
    """
    df = build_ecdf_aligned_table(values_dict, max_points=max_points)

    plt.figure(figsize=(6.2, 4.0), dpi=180)
    ax = plt.gca()

    x = df["x"].values
    for tag in values_dict.keys():
        y = df[tag].values
        ax.plot(x, y, lw=2.0, label=tag)

    ax.set_title(f"{title} — ECDF (aligned X)")
    ax.set_xlabel(title)
    ax.set_ylabel("F(x)")
    ax.grid(alpha=0.28, linestyle="--")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return df


# ========= 训练与评估 =========
def train_one(model, loss_obj, train_loader, val_loader, device, epochs, lr, save_path, fig_path):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf"); tr_hist=[]; va_hist=[]
    for ep in range(1, epochs+1):
        model.train(); t_loss=0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            pred, coef = model(x)
            loss = loss_obj(pred, y, base_coef=coef)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item()*x.size(0)
        t_loss /= len(train_loader.dataset); tr_hist.append(t_loss)

        model.eval(); v_loss=0.0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                pred, coef = model(x)
                loss = loss_obj(pred, y, base_coef=coef)
                v_loss += loss.item()*x.size(0)
        v_loss /= len(val_loader.dataset); va_hist.append(v_loss)

        print(f"[Epoch {ep:03d}] train={t_loss:.6f}  val={v_loss:.6f}")
        if v_loss < best:
            best = v_loss
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": v_loss}, save_path)
            print(f"  -> saved best to {save_path}")

    plt.figure(figsize=(8,4), dpi=160)
    plt.plot(tr_hist, label="train"); plt.plot(va_hist, label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(alpha=0.3); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(fig_path); plt.close()

def eval_one(model, loader, device):
    model.eval()
    YT, YP = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            pred, _ = model(x)
            YP.append(pred.cpu().numpy())
            YT.append(y.numpy())
    y_true = np.concatenate(YT, 0)
    y_pred = np.concatenate(YP, 0)

    rmse = rmse_vec(y_pred, y_true)
    mae  = mae_vec(y_pred, y_true)
    sam  = sam_vec(y_pred, y_true)
    psnr = np.array([psnr_scalar(gt, pr) for gt, pr in zip(y_true, y_pred)], dtype=float)
    r2   = np.array([r2_scalar(gt, pr)   for gt, pr in zip(y_true, y_pred)], dtype=float)

    mets = {"RMSE": rmse, "MAE": mae, "SAM": sam, "PSNR": psnr, "R2": r2}
    return y_true, y_pred, mets


def main():
    parser = argparse.ArgumentParser(description="Ablation on Losses: MSE / +DC / +TV / +DC+TV")
    parser.add_argument("--train_root", type=str, required=True, help="训练集目录，含 rgb_*.npy / spectral_*.npy")
    parser.add_argument("--val_root",   type=str, required=True, help="验证集目录，含 rgb_*.npy / spectral_*.npy")
    parser.add_argument("--outdir",     type=str, default="./ablation_losses_out")
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    set_global_seed(2025)

    train_ds = NpySpectralDataset(args.train_root)
    val_ds   = NpySpectralDataset(args.val_root)

    g_train = torch.Generator(); g_train.manual_seed(2025)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g_train)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    def new_model():
        return DualHeadSpectralNet(
            n_wave=76, token_grid=(4,6),
            base_ch=64, n_blocks=4,
            k_basis=12, out_activation="softplus"
        ).to(device)

    exps = [
        ("L1_MSE",        dict(dc=0.00, tv=0.00)),
        ("L2_MSE_DC",     dict(dc=0.10, tv=0.00)),
        ("L3_MSE_TV",     dict(dc=0.00, tv=0.02)),
        ("L4_MSE_DC_TV",  dict(dc=0.10, tv=0.02)),
    ]

    WAVELENGTHS = np.arange(380, 760, 5, dtype=np.float32)

    # 固定一组验证样本索引用于画谱线示例（跨实验一致）
    N_vis = 8 if len(val_ds) >= 8 else len(val_ds)
    rng = np.random.RandomState(2025)
    idxs_fixed = rng.choice(len(val_ds), size=N_vis, replace=False)

    # 收集五个指标（用于 boxplot + ECDF）
    all_mets = {"RMSE": {}, "MAE": {}, "SAM": {}, "PSNR": {}, "R2": {}}

    for tag, cfg in exps:
        print(f"\n========== {tag} ==========")
        out_dir = os.path.join(args.outdir, tag)
        os.makedirs(out_dir, exist_ok=True)

        model = new_model()
        loss_obj = SpectralReconstructionLoss(
            weight=torch.ones(76),
            lambda_dc=cfg["dc"],
            lambda_tv=cfg["tv"],
            lambda_base=1e-4
        ).to(device)

        ckpt_path = os.path.join(out_dir, "best.ckptS")
        fig_loss  = os.path.join(out_dir, "loss_curve.png")
        train_one(model, loss_obj, train_loader, val_loader, device,
                  args.epochs, args.lr, ckpt_path, fig_loss)

        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)

        y_true, y_pred, mets = eval_one(model, val_loader, device)

        # per-sample CSV（补齐 PSNR/R2）
        rows = []
        for i in range(len(y_true)):
            rows.append({
                "index": i,
                "rmse": float(mets["RMSE"][i]),
                "mae":  float(mets["MAE"][i]),
                "sam":  float(mets["SAM"][i]),
                "psnr": float(mets["PSNR"][i]),
                "r2":   float(mets["R2"][i]),
            })
        csv_path = os.path.join(out_dir, "metrics_per_sample.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[OK] {tag}: metrics_per_sample.csv saved -> {csv_path}")

        # summary.json（补齐 PSNR/R2）
        summary = {
            "RMSE": {"mean": float(np.mean(mets["RMSE"])), "std": float(np.std(mets["RMSE"]))},
            "MAE":  {"mean": float(np.mean(mets["MAE"])),  "std": float(np.std(mets["MAE"]))},
            "SAM":  {"mean": float(np.mean(mets["SAM"])),  "std": float(np.std(mets["SAM"]))},
            "PSNR": {"mean": float(np.mean(mets["PSNR"])), "std": float(np.std(mets["PSNR"]))},
            "R2":   {"mean": float(np.mean(mets["R2"])),   "std": float(np.std(mets["R2"]))},
            "best_epoch": int(state.get("epoch", -1)),
            "best_val_loss": float(state.get("val_loss", np.nan)),
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 保存谱线示例（避免叠画）
        for j, i in enumerate(idxs_fixed.tolist()):
            fig, ax = plt.subplots(figsize=(6.2, 3.9), dpi=160)
            ax.plot(WAVELENGTHS, y_true[i], lw=2.2, c="k", label="GT")
            ax.plot(WAVELENGTHS, y_pred[i], lw=1.8, c="#2980b9", label=f"Pred ({tag})")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Intensity")
            ax.grid(alpha=0.3, linestyle="--")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"curve_fixed_{j:02d}.png"))
            plt.close(fig)

        # 收集
        for k in all_mets.keys():
            all_mets[k][tag] = mets[k]

    # boxplot（保留原图，RMSE/MAE/SAM）
    box_with_labels(os.path.join(args.outdir, "box_RMSE_linear.png"), all_mets["RMSE"], title="RMSE", yscale="linear", ylabel="RMSE")
    box_with_labels(os.path.join(args.outdir, "box_RMSE_symlog.png"), all_mets["RMSE"], title="RMSE", yscale="symlog", linthresh=0.05, ylabel="RMSE")

    box_with_labels(os.path.join(args.outdir, "box_MAE_linear.png"),  all_mets["MAE"],  title="MAE",  yscale="linear", ylabel="MAE")
    box_with_labels(os.path.join(args.outdir, "box_MAE_symlog.png"),  all_mets["MAE"],  title="MAE",  yscale="symlog", linthresh=0.05, ylabel="MAE")

    box_with_labels(os.path.join(args.outdir, "box_SAM_linear.png"),  all_mets["SAM"],  title="SAM",  yscale="linear", ylabel="SAM (rad)")
    box_with_labels(os.path.join(args.outdir, "box_SAM_symlog.png"),  all_mets["SAM"],  title="SAM",  yscale="symlog", linthresh=0.01, ylabel="SAM (rad)")

    # ====== ECDF 输出：五个指标（对齐公共 x，直接导出 Prism 友好宽表）======
    ecdf_tables_wide = {}

    # 可选：如果你担心点太多导致 Excel 很大，把这里设成比如 3000 或 5000
    # 不想下采样就保持 None
    ECDF_MAX_POINTS = None

    for met_name in ["RMSE", "MAE", "SAM", "PSNR", "R2"]:
        fig_path = os.path.join(args.outdir, f"ecdf_{met_name}.png")
        df_wide = plot_ecdf_aligned(fig_path, all_mets[met_name], title=met_name, max_points=ECDF_MAX_POINTS)
        ecdf_tables_wide[met_name] = df_wide

    # 导出 Prism 友好格式：每个指标一个 sheet（宽表：x + 多列y）
    excel_path = os.path.join(args.outdir, "ecdf_data.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for met_name, df_wide in ecdf_tables_wide.items():
            # sheet名保持你之前的习惯
            df_wide.to_excel(writer, sheet_name=f"{met_name}_ecdf", index=False)

    # 同时导出一个长表（可选，便于你自己追溯/检查）
    long_rows = []
    for met_name, df_wide in ecdf_tables_wide.items():
        x = df_wide["x"].values
        for tag in df_wide.columns:
            if tag == "x":
                continue
            y = df_wide[tag].values
            for xi, yi in zip(x, y):
                long_rows.append({"metric": met_name, "exp": tag, "x": float(xi), "ecdf": float(yi)})

    long_csv = os.path.join(args.outdir, "ecdf_data_long.csv")
    pd.DataFrame(long_rows).to_csv(long_csv, index=False, encoding="utf-8")

    print(f"[OK] ECDF figures saved: ecdf_RMSE/MAE/SAM/PSNR/R2.png")
    print(f"[OK] Prism-ready ECDF xlsx: {excel_path}")
    print(f"[OK] ECDF long csv: {long_csv}")


if __name__ == "__main__":
    main()
