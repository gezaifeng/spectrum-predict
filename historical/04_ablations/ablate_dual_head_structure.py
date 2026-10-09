# -*- coding: utf-8 -*-
"""
ablate_dual_head_structure_v2.py
与 predict.py 完全对齐的结构消融：Base-only / Residual-only / Dual-Head
- 直接读取 rgb_*.npy / spectral_*.npy（无需自定义 Dataset）
- 复用 robust_block_stat + 归一化 + (4,6,100,3)->(4,6,3) 逻辑
- 模型加载严格 strict=True，兼容 state['model'] / 纯 state_dict
- 指标：RMSE / MAE / SAM(rad) / PSNR(dB) / R2
- 可选关掉逐波长注意力：--no-attn
- 导出：逐样本 CSV、汇总 JSON、曲线图 & 箱线图 PNG
"""
"""python .\ablate_dual_head_structure.py `
  --root  "D:\Desktop\dataset\guasnoise1-1020\val" `
  --ckpt  "D:\Desktop\model\double-RESCNN-Statistical analysis\best_dual_head.pth" `
  --outdir ".\ablation_dualhead_v2_val"""

import os, glob, math, json, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # 非交互式，直接保存图片
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model_dual_head import DualHeadSpectralNet  # 确保路径与训练/预测一致
# ========= 可视化：线性 + symlog（带数值标记） =========
import numpy as np
import matplotlib.pyplot as plt

def _format_num(x):
    """自动选择小数位：小于0.01保留3位，小于0.1保留2位，否则1位。角度/弧度也适用。"""
    ax = abs(float(x))
    if ax < 0.01:
        return f"{x:.3f}"
    elif ax < 0.1:
        return f"{x:.2f}"
    else:
        return f"{x:.1f}"

def _annotate_box(ax, stats, xpos, color_mean="#2c3e50", color_med="#c0392b", y_offset=0.03):
    """
    在箱线图上方/附近标注 mean 和 median 数值。
    stats: dict 包含 'mean'、'median' 值
    xpos: 该箱线的 x 坐标（1-based）
    """
    y_mean = stats["mean"]
    y_median = stats["median"]
    # 画标记
    ax.scatter([xpos], [y_mean], marker="^", s=38, color=color_mean, zorder=3, label=None)
    ax.scatter([xpos], [y_median], marker="o", s=28, facecolors="none", edgecolors=color_med, zorder=3, label=None)
    # 放文本（尽量不与箱体重叠）
    ylim = ax.get_ylim()
    span = (ylim[1] - ylim[0]) if ylim[1] > ylim[0] else 1.0
    ax.text(xpos + 0.14, y_mean + y_offset*span, f"μ={_format_num(y_mean)}", color=color_mean, fontsize=8)
    ax.text(xpos + 0.14, y_median - y_offset*span, f"med={_format_num(y_median)}", color=color_med, fontsize=8)

def _collect_stats(values):
    arr = np.asarray(values).astype(float)
    return {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "q25":    float(np.percentile(arr, 25)),
        "q75":    float(np.percentile(arr, 75)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
    }

def boxplot_with_labels(save_path, values_dict, title="Metrics (linear)", yscale="linear", linthresh=0.05, ylabel=None):
    """
    画箱线图，叠加 mean/median 数值标记；支持 linear / symlog。
      values_dict: {"Base-only":[N], "Residual-only":[N], "Dual-Head":[N]}
      yscale: "linear" 或 "symlog"
      linthresh: symlog 在线性区阈值
    """
    labels = list(values_dict.keys())
    data   = [np.asarray(values_dict[k]).astype(float) for k in labels]

    plt.figure(figsize=(5.6, 3.8), dpi=160)
    ax = plt.gca()
    bp = ax.boxplot(data, labels=labels, showmeans=True, meanprops=dict(marker='D', markerfacecolor='gray', markersize=5))

    # 轴缩放
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_title(f"{title} (symlog)")
    else:
        ax.set_title(title)

    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(alpha=0.28, which='both', linestyle='--')

    # 叠加 mean/median 数值标注
    for i, lab in enumerate(labels, start=1):
        stats = _collect_stats(values_dict[lab])
        _annotate_box(ax, stats, xpos=i)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ===================== 与 predict.py 对齐的工具函数 =====================

def robust_block_stat(x, axis=2, q=5):
    """截断均值：去两端各 q% 噪声后求均值。x:(4,6,100,3)->(4,6,3)"""
    low  = np.percentile(x, q, axis=axis, keepdims=True)
    high = np.percentile(x, 100 - q, axis=axis, keepdims=True)
    return np.clip(x, low, high).mean(axis=axis)

def prepare_rgb_grid(rgb_npy: np.ndarray) -> torch.Tensor:
    """
    把单样本 rgb.npy 转为 (1,3,4,6)：
      - (4,6,100,3) -> [0,1] 归一化 + robust_block_stat -> (4,6,3)
      - (4,6,3)     -> 若>1则除以255
    """
    arr = rgb_npy.astype(np.float32)
    if arr.ndim == 4 and arr.shape == (4, 6, 100, 3):
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = robust_block_stat(arr, axis=2, q=5)
    elif arr.ndim == 3 and arr.shape == (4, 6, 3):
        if arr.max() > 1.0:
            arr = arr / 255.0
    else:
        raise ValueError(f"Unexpected rgb.npy shape: {arr.shape} (expect (4,6,100,3) or (4,6,3)).")
    x = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1,3,4,6)
    return x

def load_model_strict(model_path: str, n_wave: int = 76, token_grid=(4,6), device="cpu") -> torch.nn.Module:
    model = DualHeadSpectralNet(
        n_wave=n_wave, token_grid=token_grid,
        base_ch=64, n_blocks=4, k_basis=12, out_activation="softplus"
    ).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weight not found: {model_path}")
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        model.load_state_dict(state["model"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.eval()
    return model

def list_pairs(root_dir: str):
    """遍历目录，返回 (rgb_path, spectral_path or None) 列表。"""
    rgb_list = sorted(glob.glob(os.path.join(root_dir, "rgb_*.npy")))
    pairs = []
    for rp in rgb_list:
        sp = rp.replace("rgb_", "spectral_")
        pairs.append((rp, sp if os.path.exists(sp) else None))
    if not pairs:
        raise FileNotFoundError(f"No rgb_*.npy under {root_dir}")
    return pairs


# ===================== 指标 =====================

def rmse_vec(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2, axis=-1))

def mae_vec(pred, gt):
    return np.mean(np.abs(pred - gt), axis=-1)

def sam_vec(pred, gt, eps=1e-8):
    # Spectral Angle Mapper, per-sample, rad
    num = np.sum(pred * gt, axis=-1)
    den = np.linalg.norm(pred, axis=-1) * np.linalg.norm(gt, axis=-1) + eps
    cosang = np.clip(num / den, -1.0, 1.0)
    return np.arccos(cosang)

def psnr_scalar(y_true, y_pred):
    # 以 1 为满幅值（与 predict.py 保持一致）
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse + 1e-12)
    return 10 * math.log10(1.0 / (rmse**2 + 1e-12))

def r2_scalar(y_true, y_pred):
    return r2_score(y_true, y_pred)



# ===================== 可选：去注意力 =====================

def replace_attn_with_uniform(model: DualHeadSpectralNet):
    """将逐波长注意力 softmax 权重置为均匀（log 均匀 -> softmax 后均匀）。"""
    with torch.no_grad():
        L, Nt = model.attn_weight.shape
        uni = torch.full_like(model.attn_weight, 1.0 / Nt)
        model.attn_weight.copy_(torch.log(uni))


# ===================== 模型分量前向 =====================

@torch.no_grad()
def forward_components(model: DualHeadSpectralNet, x: torch.Tensor):
    """
    返回：
      spec_base: [B,L]  基线头（内部已对 coef 做 Softplus 非负约束）
      spec_res : [B,L]  残差头（逐波长线性+对角抽取，不再额外激活）
      spec_full: [B,L]  模型最终输出（等价 predict.py 的结果，含 out_act Softplus）
    """
    tokens = model.keep_grid(model.proj(model.blocks(model.stem(x))))
    f_global, f_lambda = model._attn_pool(tokens)  # [B,C], [B,C,L]

    # Base head
    coef = F.softplus(model.head_coef(f_global))            # [B,K]
    spec_base = torch.matmul(coef, model.basis.t())         # [B,L]

    # Residual head
    Bsz, Cc, L = f_lambda.shape
    res_in  = f_lambda.permute(0, 2, 1).contiguous().view(Bsz * L, Cc)  # [B*L,C]
    res_all = model.head_res(res_in).view(Bsz, L, L)                    # [B,L,L]
    idx = torch.arange(L, device=x.device)
    spec_res = res_all[:, idx, idx]                                     # [B,L]

    # Full output（模型自带 out_act）
    spec_full = model.out_act(spec_base + spec_res)
    return spec_base, spec_res, spec_full


# ===================== 绘图 =====================

def plot_sample_curves(save_dir, waves_nm, y_true, y_base, y_res, y_dual, max_plots=8, title_prefix=""):
    os.makedirs(save_dir, exist_ok=True)
    N = y_dual.shape[0]
    idxs = random.sample(range(N), k=min(max_plots, N))
    for i in idxs:
        plt.figure(figsize=(8.5, 4.6), dpi=160)
        if y_true is not None:
            plt.plot(waves_nm, y_true[i], lw=2.2, label="Ground Truth", color="black")
        plt.plot(waves_nm, y_base[i], lw=1.6, label="Base-only", color="#e67e22")
        plt.plot(waves_nm, y_res[i],  lw=1.6, label="Residual-only", color="#8e44ad")
        plt.plot(waves_nm, y_dual[i], lw=2.0, label="Dual-Head", color="#2980b9")
        plt.xlabel("Wavelength (nm)"); plt.ylabel("Intensity / Absorbance")
        plt.title(f"{title_prefix} sample #{i}")
        plt.grid(alpha=0.3); plt.legend(frameon=False, ncol=2); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"curve_sample_{i}.png"))
        plt.close()

def boxplot_metrics(save_dir, metrics_dict, title="Ablation (Dual-Head)"):
    """
    metrics_dict: {
      "RMSE": {"Base-only":[N], "Residual-only":[N], "Dual-Head":[N]},
      "MAE" : {...}, "SAM": {...}
    }
    """
    os.makedirs(save_dir, exist_ok=True)
    for mname, groups in metrics_dict.items():
        plt.figure(figsize=(5.4, 3.6), dpi=160)
        data = [groups["Base-only"], groups["Residual-only"], groups["Dual-Head"]]
        plt.boxplot(data, labels=["Base-only", "Residual-only", "Dual-Head"], showmeans=True)
        plt.title(f"{title} — {mname}")
        plt.ylabel(mname + (" (rad)" if mname == "SAM" else ""))
        plt.grid(alpha=0.25, linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"box_{mname}.png"))
        plt.close()


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(description="Ablation: Base-only vs Residual-only vs Dual-Head (predict-aligned)")
    parser.add_argument("--root",  type=str, required=True, help="数据子目录（例如 D:\\...\\guasnoise1-1020\\val）")
    parser.add_argument("--ckptS",  type=str, required=True, help="双头网络权重 .pth")
    parser.add_argument("--outdir", type=str, default="./ablation_dualhead_v2_out")
    parser.add_argument("--no-attn", action="store_true", help="关闭逐波长注意力（用均匀加权）")
    parser.add_argument("--max_plots", type=int, default=8, help="随机样本曲线的数量")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    # 波段（与 predict.py 一致：380~755, step=5，共76维）
    WAVELENGTHS = np.arange(380, 760, 5, dtype=np.float32)
    L = len(WAVELENGTHS)

    # 模型
    model = load_model_strict(args.ckptS, n_wave=L, token_grid=(4,6), device=device)
    if args.no_attn:
        replace_attn_with_uniform(model)
        print("[INFO] use uniform attention weights (no-attn).")

    # 数据对
    pairs = list_pairs(args.root)

    # 存放预测
    ys_true, ys_base, ys_res, ys_dual = [], [], [], []

    # 前向
    for i, (rgb_path, spec_path) in enumerate(pairs, 1):
        rgb = np.load(rgb_path)
        x = prepare_rgb_grid(rgb).to(device)  # (1,3,4,6)
        with torch.no_grad():
            s_base, s_res, s_dual = forward_components(model, x)
        y_base = s_base.squeeze(0).cpu().numpy().astype(np.float32)  # [L]
        y_res  = s_res.squeeze(0).cpu().numpy().astype(np.float32)   # [L]
        y_dual = s_dual.squeeze(0).cpu().numpy().astype(np.float32)  # [L]

        ys_base.append(y_base)
        ys_res.append(y_res)
        ys_dual.append(y_dual)

        if spec_path and os.path.exists(spec_path):
            ys_true.append(np.load(spec_path).astype(np.float32))
        else:
            ys_true.append(np.full((L,), np.nan, dtype=np.float32))  # 缺GT时占位，后续会过滤

    ys_base = np.stack(ys_base, axis=0)   # [N,L]
    ys_res  = np.stack(ys_res,  axis=0)
    ys_dual = np.stack(ys_dual, axis=0)
    ys_true = np.stack(ys_true, axis=0)

    # 仅对“有GT”的样本计算指标
    has_gt = ~np.isnan(ys_true).any(axis=1)
    y_true = ys_true[has_gt]
    y_base = ys_base[has_gt]
    y_res  = ys_res [has_gt]
    y_dual = ys_dual[has_gt]
    # ===== 新增：导出所有样本的完整光谱曲线数据 =====
    # 结构：每行 = 一个样本在某一波长处的 4 条曲线数值
    # 列: index, wavelength_nm, gt, base, res, dual
    curve_rows = []
    N, L = ys_dual.shape
    for i in range(N):
        gt_i   = ys_true[i]      # 可能含 NaN（无 GT）
        base_i = ys_base[i]
        res_i  = ys_res[i]
        dual_i = ys_dual[i]
        for j in range(L):
            curve_rows.append({
                "index":        int(i),
                "wavelength_nm": float(WAVELENGTHS[j]),
                "gt":           float(gt_i[j]),
                "base":         float(base_i[j]),
                "res":          float(res_i[j]),
                "dual":         float(dual_i[j]),
            })
    curves_csv = os.path.join(args.outdir, "curves_all_samples.csv")
    pd.DataFrame(curve_rows).to_csv(curves_csv, index=False, encoding="utf-8")
    print(f"[OK] curves data -> {curves_csv}")


    # 指标（逐样本向量）
    mets = {
        "RMSE": {
            "Base-only": rmse_vec(y_base, y_true),
            "Residual-only": rmse_vec(y_res,  y_true),
            "Dual-Head": rmse_vec(y_dual, y_true),
        },
        "MAE": {
            "Base-only": mae_vec(y_base, y_true),
            "Residual-only": mae_vec(y_res,  y_true),
            "Dual-Head": mae_vec(y_dual, y_true),
        },
        "SAM": {
            "Base-only": sam_vec(y_base, y_true),
            "Residual-only": sam_vec(y_res,  y_true),
            "Dual-Head": sam_vec(y_dual, y_true),
        },
    }

    # 逐样本 CSV（含 PSNR / R2 便于和 predict 对齐对照）
    rows = []
    idxs = np.where(has_gt)[0].tolist()
    for k, i in enumerate(idxs):
        gt  = y_true[k]
        prB = y_base[k];  prR = y_res[k];  prD = y_dual[k]
        rows.append({
            "index": int(i),
            "rmse_base": float(mets["RMSE"]["Base-only"][k]),
            "rmse_res":  float(mets["RMSE"]["Residual-only"][k]),
            "rmse_dual": float(mets["RMSE"]["Dual-Head"][k]),
            "mae_base":  float(mets["MAE"]["Base-only"][k]),
            "mae_res":   float(mets["MAE"]["Residual-only"][k]),
            "mae_dual":  float(mets["MAE"]["Dual-Head"][k]),
            "sam_base(rad)": float(mets["SAM"]["Base-only"][k]),
            "sam_res(rad)":  float(mets["SAM"]["Residual-only"][k]),
            "sam_dual(rad)": float(mets["SAM"]["Dual-Head"][k]),
            "psnr_base(dB)": float(psnr_scalar(gt, prB)),
            "psnr_res(dB)":  float(psnr_scalar(gt, prR)),
            "psnr_dual(dB)": float(psnr_scalar(gt, prD)),
            "r2_base":  float(r2_scalar(gt, prB)),
            "r2_res":   float(r2_scalar(gt, prR)),
            "r2_dual":  float(r2_scalar(gt, prD)),
        })
    csv_path = os.path.join(args.outdir, f"metrics{'_noattn' if args.no_attn else ''}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] per-sample metrics -> {csv_path}")

    stats_rows = []
    for mname, groups in mets.items():
        for k, arr in groups.items():
            st = _collect_stats(arr)
            st.update({"metric": mname, "method": k})
            stats_rows.append(st)
    pd.DataFrame(stats_rows).to_csv(os.path.join(args.outdir, "summary_stats_boxplot.csv"), index=False,
                                    encoding="utf-8")
    # 汇总 JSON（mean±std）
    def mean_std(a): return float(np.mean(a)), float(np.std(a))
    summary = {}
    for mname, groups in mets.items():
        summary[mname] = {k: {"mean": mean_std(v)[0], "std": mean_std(v)[1]} for k, v in groups.items()}
    json_path = os.path.join(args.outdir, f"summary{'_noattn' if args.no_attn else ''}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[OK] summary:\n", json.dumps(summary, ensure_ascii=False, indent=2))

    # 箱线图
    boxplot_metrics(args.outdir, mets, title=f"Ablation (Aligned){' | no-attn' if args.no_attn else ''}")
    # === 新增：线性 + symlog 两套箱线图，并叠加数值标注 ===
    # RMSE
    boxplot_with_labels(
        save_path=os.path.join(args.outdir, "box_RMSE_linear.png"),
        values_dict=mets["RMSE"],
        title="RMSE (linear y-axis)",
        yscale="linear",
        ylabel="RMSE"
    )
    boxplot_with_labels(
        save_path=os.path.join(args.outdir, "box_RMSE_symlog.png"),
        values_dict=mets["RMSE"],
        title="RMSE",
        yscale="symlog", linthresh=0.05,
        ylabel="RMSE"
    )

    # MAE
    boxplot_with_labels(
        save_path=os.path.join(args.outdir, "box_MAE_linear.png"),
        values_dict=mets["MAE"],
        title="MAE (linear y-axis)",
        yscale="linear",
        ylabel="MAE"
    )
    boxplot_with_labels(
        save_path=os.path.join(args.outdir, "box_MAE_symlog.png"),
        values_dict=mets["MAE"],
        title="MAE",
        yscale="symlog", linthresh=0.05,
        ylabel="MAE"
    )

    # SAM（弧度）
    boxplot_with_labels(
        save_path=os.path.join(args.outdir, "box_SAM_linear.png"),
        values_dict=mets["SAM"],
        title="SAM (linear y-axis)",
        yscale="linear",
        ylabel="SAM (rad)"
    )
    boxplot_with_labels(
        save_path=os.path.join(args.outdir, "box_SAM_symlog.png"),
        values_dict=mets["SAM"],
        title="SAM",
        yscale="symlog", linthresh=0.01,
        ylabel="SAM (rad)"
    )

    # 随机样本曲线（含无 GT 的也一并画对比：只是不显示 GT）
    plot_sample_curves(args.outdir, WAVELENGTHS, y_true if len(y_true)>0 else None,
                       y_base if len(y_true)>0 else ys_base,
                       y_res  if len(y_true)>0 else ys_res,
                       y_dual if len(y_true)>0 else ys_dual,
                       max_plots=min(args.max_plots, (y_dual if len(y_true)>0 else ys_dual).shape[0]),
                       title_prefix=f"Ablation Curves{' | no-attn' if args.no_attn else ''}")

    print(f"[OK] plots saved under: {args.outdir}")

if __name__ == "__main__":
    main()
