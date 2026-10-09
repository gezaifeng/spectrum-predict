# -*- coding: utf-8 -*-
import os
import re
import csv
import shutil
import argparse
import numpy as np

EPS = 1e-12

def luminance_linear(rgb_chw):
    # rgb_chw: (3,H,W) linear domain
    r, g, b = rgb_chw[0], rgb_chw[1], rgb_chw[2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def robust_mad(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + EPS
    # 1.4826 将 MAD 转为接近 std 的尺度
    scale = 1.4826 * mad
    return med, scale

def to_chw(arr):
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={arr.shape}")
    if arr.shape[0] in (3, 15):
        return arr
    if arr.shape[-1] in (3, 15):
        return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unrecognized shape={arr.shape}")

def parse_id(fname):
    # rgb_0004.npy -> 0004
    m = re.match(r"rgb_(\d+)\.npy$", os.path.basename(fname))
    return m.group(1) if m else None

def compute_metrics(x_chw):
    """
    x_chw: (C,H,W) where C in {3,15} (feature npy)
    输出一组适合筛查暗光/异常的指标
    """
    x = x_chw.astype(np.float32)
    C, H, W = x.shape

    finite = np.isfinite(x).all()
    if not finite:
        return {"finite": 0}

    # 通用统计（不依赖于是否 zscore）
    abs_p99 = float(np.percentile(np.abs(x), 99))
    abs_p95 = float(np.percentile(np.abs(x), 95))
    abs_max = float(np.max(np.abs(x)))
    dyn = float(np.percentile(x, 99) - np.percentile(x, 1))

    # outlier 比例：以 robust 方式定义（|x| > p99 其实也可，但这里用固定阈值更直观）
    # 对 log_ratio zscore 后，|x|>6 基本就是很异常的噪声/尖峰
    outlier_frac_6 = float((np.abs(x) > 6.0).mean())
    outlier_frac_10 = float((np.abs(x) > 10.0).mean())

    # 每通道 std（暗光导致的“近常数/被clip”会让 std 很小；噪声爆炸会让 std 很大）
    ch_std = x.reshape(C, -1).std(axis=1)
    ch_std_min = float(ch_std.min())
    ch_std_mean = float(ch_std.mean())

    metrics = {
        "finite": 1,
        "C": C, "H": H, "W": W,
        "abs_p95": abs_p95,
        "abs_p99": abs_p99,
        "abs_max": abs_max,
        "dyn_99_1": dyn,
        "outlier_frac_6": outlier_frac_6,
        "outlier_frac_10": outlier_frac_10,
        "ch_std_min": ch_std_min,
        "ch_std_mean": ch_std_mean,
    }

    # multi 模式：前 3 是 ref_lin，接着 3 是 sam_lin（按你 features.py 的拼接顺序）
    if C == 15:
        ref_lin = x[0:3]
        sam_lin = x[3:6]
        ref_luma = luminance_linear(ref_lin)
        sam_luma = luminance_linear(sam_lin)

        ref_luma_mean = float(ref_luma.mean())
        sam_luma_mean = float(sam_luma.mean())

        # 暗像素比例：linear 域接近 0
        ref_dark_frac = float((ref_luma < 1e-4).mean())
        sam_dark_frac = float((sam_luma < 1e-4).mean())

        metrics.update({
            "ref_luma_mean": ref_luma_mean,
            "sam_luma_mean": sam_luma_mean,
            "ref_dark_frac": ref_dark_frac,
            "sam_dark_frac": sam_dark_frac,
        })

    return metrics

def decide_low_quality(metrics, robust_z, args):
    """
    返回 (is_bad, reasons[list])
    robust_z: dict metric->robust_zscore
    """
    reasons = []
    if metrics.get("finite", 0) == 0:
        return True, ["non_finite"]

    C = metrics["C"]

    # ---- 硬规则：multi 可直接用亮度/暗像素判断 ----
    if C == 15:
        if metrics["ref_luma_mean"] < args.min_ref_luma:
            reasons.append(f"ref_luma_mean<{args.min_ref_luma}")
        if metrics["sam_luma_mean"] < args.min_sam_luma:
            reasons.append(f"sam_luma_mean<{args.min_sam_luma}")
        if metrics["ref_dark_frac"] > args.max_dark_frac:
            reasons.append(f"ref_dark_frac>{args.max_dark_frac}")
        if metrics["sam_dark_frac"] > args.max_dark_frac:
            reasons.append(f"sam_dark_frac>{args.max_dark_frac}")

    # ---- 通用硬规则：极端离群/动态范围异常 ----
    if metrics["outlier_frac_10"] > args.max_outlier_frac_10:
        reasons.append(f"outlier_frac_10>{args.max_outlier_frac_10}")

    # ---- 鲁棒异常：MAD zscore 超阈值 ----
    # 常用：abs_p99、dyn、ch_std_min、outlier_frac_6
    for k in args.robust_keys:
        z = robust_z.get(k, 0.0)
        if z > args.robust_z_th:
            reasons.append(f"robust_outlier:{k}(z={z:.2f})")

    is_bad = len(reasons) > 0
    return is_bad, reasons

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="包含 rgb_*.npy / spectral_*.npy 的目录")
    ap.add_argument("--low_dir", type=str, default=None, help="低质量移动目录（默认 data_dir/low_quality）")
    ap.add_argument("--good_dir", type=str, default=None, help="可选：把好数据移动到该目录（默认不动）")
    ap.add_argument("--move", action="store_true", help="执行移动；不加则只生成报告不移动")

    # multi 亮度硬阈值（linear 域，范围大约 0~1）
    ap.add_argument("--min_ref_luma", type=float, default=0.02)
    ap.add_argument("--min_sam_luma", type=float, default=0.02)
    ap.add_argument("--max_dark_frac", type=float, default=0.35)

    # 通用离群硬阈值
    ap.add_argument("--max_outlier_frac_10", type=float, default=0.02)

    # 鲁棒异常阈值（MAD）
    ap.add_argument("--robust_z_th", type=float, default=4.5)
    ap.add_argument("--robust_keys", type=str, nargs="+",
                    default=["abs_p99", "dyn_99_1", "ch_std_min", "outlier_frac_6"])

    ap.add_argument("--report", type=str, default=None, help="输出csv路径（默认 data_dir/quality_report.csv）")
    args = ap.parse_args()

    data_dir = args.data_dir
    low_dir = args.low_dir or os.path.join(data_dir, "low_quality")
    good_dir = args.good_dir
    report_path = args.report or os.path.join(data_dir, "quality_report.csv")

    rgb_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.startswith("rgb_") and f.endswith(".npy")
    ])
    if not rgb_files:
        raise FileNotFoundError(f"No rgb_*.npy found in {data_dir}")

    # Pass1: 计算每个样本 metrics
    rows = []
    for fp in rgb_files:
        sid = parse_id(fp)
        if sid is None:
            continue
        try:
            arr = np.load(fp)
            x = to_chw(arr)
            metrics = compute_metrics(x)
        except Exception as e:
            metrics = {"finite": 0, "error": str(e), "C": -1, "H": -1, "W": -1}
        metrics["id"] = sid
        metrics["rgb_path"] = fp
        metrics["spectral_path"] = os.path.join(data_dir, f"spectral_{sid}.npy")
        rows.append(metrics)

    # Pass2: 为关键指标计算 robust zscore
    # 只用 finite 的样本估计分布
    finite_rows = [r for r in rows if r.get("finite", 0) == 1]
    robust_stats = {}
    for k in args.robust_keys:
        vals = [float(r.get(k, 0.0)) for r in finite_rows if k in r]
        if len(vals) < 10:
            # 数据太少就跳过鲁棒判定
            robust_stats[k] = (0.0, 1.0)
            continue
        med, scale = robust_mad(vals)
        robust_stats[k] = (med, scale)

    # 判定并可选移动
    os.makedirs(low_dir, exist_ok=True)
    if good_dir is not None:
        os.makedirs(good_dir, exist_ok=True)

    bad_count = 0
    for r in rows:
        robust_z = {}
        for k in args.robust_keys:
            if k in r and r.get("finite", 0) == 1:
                med, scale = robust_stats[k]
                robust_z[k] = abs((float(r[k]) - med) / (scale + EPS))
            else:
                robust_z[k] = 0.0

        is_bad, reasons = decide_low_quality(r, robust_z, args)
        r["is_bad"] = int(is_bad)
        r["reasons"] = ";".join(reasons)

        # 保存 robust z 到报表里
        for k, z in robust_z.items():
            r[f"z_{k}"] = float(z)

        if is_bad:
            bad_count += 1
            if args.move:
                # 移动 rgb
                dst_rgb = os.path.join(low_dir, os.path.basename(r["rgb_path"]))
                if os.path.exists(r["rgb_path"]):
                    shutil.move(r["rgb_path"], dst_rgb)

                # 移动 spectral（若存在）
                sp = r["spectral_path"]
                if os.path.exists(sp):
                    dst_sp = os.path.join(low_dir, os.path.basename(sp))
                    shutil.move(sp, dst_sp)
        else:
            if args.move and good_dir is not None:
                dst_rgb = os.path.join(good_dir, os.path.basename(r["rgb_path"]))
                if os.path.exists(r["rgb_path"]):
                    shutil.move(r["rgb_path"], dst_rgb)
                sp = r["spectral_path"]
                if os.path.exists(sp):
                    dst_sp = os.path.join(good_dir, os.path.basename(sp))
                    shutil.move(sp, dst_sp)

    # 写 csv 报告
    # 统一字段
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    # 常用字段放前面
    front = ["id", "is_bad", "reasons", "C", "H", "W", "rgb_path", "spectral_path",
             "ref_luma_mean", "sam_luma_mean", "ref_dark_frac", "sam_dark_frac",
             "abs_p95", "abs_p99", "abs_max", "dyn_99_1", "outlier_frac_6", "outlier_frac_10",
             "ch_std_min", "ch_std_mean"]
    cols = front + [k for k in sorted(all_keys) if k not in front]

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    print(f"Total: {len(rows)}  Bad: {bad_count}  Good: {len(rows)-bad_count}")
    print("Report:", report_path)
    if args.move:
        print("Moved bad samples to:", low_dir)
        if good_dir is not None:
            print("Moved good samples to:", good_dir)

if __name__ == "__main__":
    main()
