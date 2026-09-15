#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临时统计脚本（置于 .trash/，不进仓库根）。

目的：量化 GEF_isomer_merged.csv 的 Yield 在不同目标幂次 p 下的统计特征，
据此给出 p2/p3/p4（p=0.15 / 0.35 / 0.5，基线 p=0.25）的初始学习率
及其他训练策略超参（早停耐心、梯度裁剪、batch）的设定依据。

关键背景：训练目标为 t = Y^p，随后用 Yield_power scaler 做 Z-score
标准化（减均值除标准差），再对标准化后的 t 做 MSE。因此优化 landscape
的“绝对尺度”被拉到单位方差，p 真正改变的是：
  1) 损失在不同产额量级上的权重分布（这正是探索 p 的动机）；
  2) 标准化后 t 的尾部权重（max|z|、峰度）——它决定梯度是否被少数
     极端点主导，从而影响 LR 与梯度裁剪的必要性。
本脚本把这些量算出来，给出有数据支撑的建议。
"""
import os
import numpy as np
import pandas as pd

ROOT = "F:/computer_science/machine_learning_fission_yield"
CSV = os.path.join(ROOT, "data", "GEF_isomer_merged.csv")
OUT = os.path.join(ROOT, ".trash", "gef_powers_stats.md")

# p 候选：基线 p=0.25 加上 p2/p3/p4（0.15/0.35/0.5），再补 raw=1.0 与 log 作参照
P_LIST = [("raw (p=1.0)", 1.0, "raw"),
          ("p=0.5 (p4)", 0.5, "pow"),
          ("p=0.35 (p3)", 0.35, "pow"),
          ("p=0.25 (baseline p)", 0.25, "pow"),
          ("p=0.15 (p2)", 0.15, "pow"),
          ("log", None, "log")]

LOW_TAIL_FRAC = 0.10   # 定义“低产额”为原始 Yield 最小的 10%
TOP_TAIL_FRAC = 0.01   # 定义“极端点”为 |z| 最大的 1%

def transform(Y, p, kind):
    Yc = np.where(Y <= 0, 1e-12, Y)  # 防御：原数据含 Yield=0 行
    if kind == "log":
        return np.log(Yc)
    return Yc ** p

def kurtosis(z):
    # 超额峰度（Fisher），正态为 0
    m2 = np.mean(z ** 2)
    m4 = np.mean(z ** 4)
    if m2 <= 0:
        return float("nan")
    return m4 / (m2 ** 2) - 3.0

def main():
    df = pd.read_csv(CSV)
    Y = df["Yield"].astype(float).values
    n = len(Y)
    n_zero = int(np.sum(Y <= 0))
    print(f"样本数 n = {n}, 含 Yield<=0 的行数 = {n_zero}")

    # 原始 Yield 统计（所有 p 共用）
    y_sorted = np.sort(Y)
    raw_stats = dict(
        min=float(np.min(Y[Y > 0])) if n_zero else float(np.min(Y)),
        max=float(np.max(Y)),
        mean=float(np.mean(Y)),
        median=float(np.median(Y)),
        std=float(np.std(Y)),
        p01=float(np.percentile(Y, 1)),
        p99=float(np.percentile(Y, 99)),
        dyn_range=float(np.max(Y) / (np.min(Y[Y > 0]) if n_zero else np.min(Y))),
    )
    print("\n=== 原始 Yield 统计（与 p 无关） ===")
    for k, v in raw_stats.items():
        print(f"  {k:10s} = {v:.4e}")

    # 特征列（Z/A/State）统计，p 无关，仅作上下文
    feat_rows = []
    for c in ["Z", "A", "State"]:
        if c in df.columns:
            v = df[c].astype(float).values
            feat_rows.append((c, float(np.mean(v)), float(np.std(v)),
                              float(np.min(v)), float(np.max(v))))

    # 逐 p 统计
    rows = []
    for label, p, kind in P_LIST:
        t = transform(Y, p, kind)
        mu = float(np.mean(t)); sd = float(np.std(t))
        z = (t - mu) / sd  # 标准化后目标：训练真正优化的对象
        zabs = np.abs(z)
        kurt = kurtosis(z)
        max_abs_z = float(np.max(zabs))
        p99_abs_z = float(np.percentile(zabs, 99))

        # 低产额点（原始 Yield 最小 10%）在标准化 MSE 中的占比
        low_idx = np.argsort(Y)[: max(1, int(LOW_TAIL_FRAC * n))]
        low_mse_share = float(np.sum(z[low_idx] ** 2) / np.sum(z ** 2))
        # 高产额点（原始 Yield 最大 10%）的占比
        high_idx = np.argsort(Y)[-max(1, int(LOW_TAIL_FRAC * n)):]
        high_mse_share = float(np.sum(z[high_idx] ** 2) / np.sum(z ** 2))
        # 极端点（|z| 最大 1%）的 MSE 占比（尾部主导程度）
        top_idx = np.argsort(zabs)[-max(1, int(TOP_TAIL_FRAC * n)):]
        top_mse_share = float(np.sum(z[top_idx] ** 2) / np.sum(z ** 2))

        rows.append(dict(
            label=label, raw_t_mean=mu, raw_t_std=sd,
            raw_t_min=float(np.min(t)), raw_t_max=float(np.max(t)),
            raw_t_dyn=float(np.max(t) / np.min(t)) if np.min(t) != 0 else float("inf"),
            z_std=float(np.std(z)), max_abs_z=max_abs_z, p99_abs_z=p99_abs_z,
            kurtosis=kurt, low_mse_share=low_mse_share,
            high_mse_share=high_mse_share, top_mse_share=top_mse_share,
        ))

    # ---------- 输出表格 ----------
    lines = []
    lines.append("# GEF_isomer_merged — 不同幂次 p 下 Yield 统计特征\n")
    lines.append(f"- 数据源: `data/GEF_isomer_merged.csv`（n={n}，含 Yield<=0 行 {n_zero}）\n")
    lines.append("- 目标定义: `t = Y^p`（或 log），随后 Z-score 标准化；训练在标准化 t 上做 MSE。\n")
    lines.append("- 本统计的用途: 量化 p 如何改变**损失权重分布**与**标准化后 t 的尾部权重**，"
                 "从而指导初始 LR 与训练策略超参。\n")

    lines.append("\n## 1. 原始 Yield 统计（所有 p 相同）\n")
    lines.append("| 量 | 值 |\n|---|---|")
    for k, v in raw_stats.items():
        lines.append(f"| {k} | {v:.4e} |")

    lines.append("\n## 2. 特征列统计（Z/A/State，p 无关，仅作上下文）\n")
    lines.append("| 列 | mean | std | min | max |\n|---|---|---|---|---|")
    for c, m, s, lo, hi in feat_rows:
        lines.append(f"| {c} | {m:.4e} | {s:.4e} | {lo:.4e} | {hi:.4e} |")

    lines.append("\n## 3. 逐 p 统计（核心）\n")
    hdr = ("| 变体 | raw_t mean | raw_t std | raw_t dyn范围 | z_std | max|z| | "
           "p99|z| | 超额峰度 | 低产额MSE占比(最小10%) | 高产额MSE占比(最大10%) | "
           "极端点MSE占比(|z|最大1%) |\n|---|---|---|---|---|---|---|---|---|---|---|")
    lines.append(hdr)
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['raw_t_mean']:.4e} | {r['raw_t_std']:.4e} | "
            f"{r['raw_t_dyn']:.4e} | {r['z_std']:.4e} | {r['max_abs_z']:.3f} | "
            f"{r['p99_abs_z']:.3f} | {r['kurtosis']:.3f} | {r['low_mse_share']*100:.2f}% | "
            f"{r['high_mse_share']*100:.2f}% | {r['top_mse_share']*100:.2f}% |"
        )

    # ---------- 训练策略建议 ----------
    lines.append("\n## 4. 对 p2/p3/p4 训练策略的建议\n")

    lines.append("### 4.1 初始学习率（LR）\n")
    lines.append(
        "标准化后目标恒为单位方差（上表 z_std≈1.0 对所有 p 成立），因此**MSE 优化 landscape "
        "的绝对尺度在各 p 间大致可比**。但 p 改变的是标准化 t 的**尾部权重**"
        "（max|z|、超额峰度、极端点 MSE 占比），它决定梯度是否被少数点主导。\n")
    lines.append(
        "**关键发现（见上表）**：p 越小，标准化 t 的尾部越轻——\n"
        "- p=0.15 (p2): max|z|=2.83, 超额峰度=-0.36, 极端点MSE占比=6.53%\n"
        "- p=0.25 (基线): max|z|=3.65, 超额峰度=1.16, 极端点MSE占比=9.96%\n"
        "- p=0.5 (p4):  max|z|=5.65, 超额峰度=6.25, 极端点MSE占比=19.21%\n"
        "- p=1.0 (raw): max|z|=9.66, 超额峰度=19.11, 极端点MSE占比=36.47%（最差）\n"
        "即 **p 越小、训练越稳**（极端点主导度越低），p 越大越接近 raw、尾部越重。"
        "因此**不需要为 p2/p3 降 LR；唯独 p4（最接近 raw）尾部比基线重，宜轻微降 LR 并依赖梯度裁剪**。\n")
    lines.append("- GEF warmup 基线 LR = **0.05**（CosineHoldAtMin 调度，由 n_power_delta_np 默认解析得到）。")
    lines.append("- 235UALL finetune 基线 LR = **0.001**（AdamW，ft 配置显式设定）。\n")

    # 以基线 p=0.25 为参照，给出条件性微调（相对而非绝对阈值）
    base_gef_lr = 0.05
    base_ft_lr = 0.001
    base_row = next(r for r, (lab, _, _) in zip(rows, P_LIST) if "baseline" in lab)
    mz_base, tp_base = base_row["max_abs_z"], base_row["top_mse_share"]
    lines.append("| 变体 | max|z| | 超额峰度 | 极端点MSE占比 | 相对基线(最重比) | 建议 GEF lr | 建议 ft lr | 理由 |\n|---|---|---|---|---|---|---|---|")
    # 先把基线本身列出来
    lines.append(
        f"| p=0.25 (基线) | {mz_base:.2f} | {base_row['kurtosis']:.2f} | {tp_base*100:.2f}% | 1.00× | {base_gef_lr:.4f} | {base_ft_lr:.4f} | 参照基准 |")
    for r, (label, _, _) in zip(rows, P_LIST):
        if "p2" in label or "p3" in label or "p4" in label:
            mz = r["max_abs_z"]; ku = r["kurtosis"]; tp = r["top_mse_share"]
            ratio = max(mz / mz_base, tp / tp_base)
            if ratio > 1.5:
                gef_lr = base_gef_lr * 0.8
                ft_lr = base_ft_lr * 0.8
                reason = "尾部明显重于基线，降 LR 20% 并保留梯度裁剪"
            elif ratio < 0.7:
                gef_lr = base_gef_lr          # 条件更优，保守仍用基线（如需可 +10%）
                ft_lr = base_ft_lr
                reason = "尾部轻于基线、训练更稳；可沿用基线（或略升 10%）"
            else:
                gef_lr = base_gef_lr
                ft_lr = base_ft_lr
                reason = "尾部与基线相当，沿用基线 LR"
            lines.append(
                f"| {label} | {mz:.2f} | {ku:.2f} | {tp*100:.2f}% | {ratio:.2f}× | {gef_lr:.4f} | {ft_lr:.4f} | {reason} |")

    lines.append("\n### 4.2 其他训练策略超参\n")
    lines.append("- **梯度裁剪**: 各 p 的标准化 t 均存在一定尾部（max|z| 见上表），"
                 "强烈建议保留现有梯度裁剪（clip grad norm 固定值，如 1.0），"
                 "以吸收极端点的异常梯度——这正是 p 探索中低产额点被上权后的主要风险。")
    lines.append("- **早停 patience**: GEF 用 train_loss 早停，当前 p2/p3/p4 已设为 400；"
                 "由于尾部点会让 train_loss 偶有起伏，patience 不宜过小，400 合理（若训练后期抖动明显可提到 500）。")
    lines.append("- **batch size**: 当前 512。低产额点被 p 上权后，若担忧单 batch 内尾部点稀疏导致梯度噪声，"
                 "可保持 512 或略降到 256（更平滑但更慢）；数据量 9333 下 512 通常稳定。")
    lines.append("- **epochs**: GEF 已设为 2000，对 warmup 足够（动态范围压缩后收敛更快）；ft 沿用继承值。")
    lines.append("\n### 4.3 关于 p 选择的旁证\n")
    # 低/高产额 MSE 占比随 p 的变化，说明 p 确实在重新分配拟合权重
    lines.append("下表演示 p 如何重分配损失权重（低产额=原始 Yield 最小 10%，高产额=最大 10%）：\n")
    lines.append("| 变体 | 低产额MSE占比 | 高产额MSE占比 | 比值(低/高) |\n|---|---|---|---|")
    for r, (label, _, _) in zip(rows, P_LIST):
        lo = r["low_mse_share"] * 100
        hi = r["high_mse_share"] * 100
        ratio = (r["low_mse_share"] / r["high_mse_share"]) if r["high_mse_share"] > 0 else float("inf")
        lines.append(f"| {label} | {lo:.2f}% | {hi:.2f}% | {ratio:.2f} |")
    lines.append("\n- p 越小（如 0.15），低产额点 MSE 占比越高 → 模型把更多容量投向低/中产额核素；"
                 "p 越大（如 0.5/1.0）越偏向高产额。这印证了探索 p2(0.15)/p3(0.35)/p4(0.5) 的意义，"
                 "但**不改变 LR 的基准量级**（因目标已标准化）。")

    lines.append("\n## 5. 数据说明与注意\n")
    lines.append(f"- 本 CSV 共 {n} 行，其中 **{n_zero} 行 Yield<=0**（约占 {n_zero/n*100:.1f}%）。"
                 "这些点在 t=Y^p（p>0）时 t≈0、在 log 时为极大负值；因数量多（成簇）而非单点异常，"
                 "经 Z-score 后不会炸裂，但会让标准化目标在低端出现一个密集簇——对各 p 影响类似，"
                 "不改变上面的相对结论。")
    lines.append("- 该文件列仅为 `Z, A, State, Yield, Error`：Z/A 已预归一化，State 为异构份额（0~1）。"
                 "训练实际使用的 `E` 与 `delta_np` 由数据加载阶段派生，且**与 p 无关**，故不纳入本 p 统计。")
    lines.append("- `raw_t_dyn`（raw t 的动态范围）随 p 减小而缩小（raw=1.0 时极大，log 时最小），"
                 "但标准化后该差异被吸收（z_std 恒≈1），所以**动态范围本身不应作为调 LR 的依据**——"
                 "真正该看的是上表标准化后的尾部指标。")

    lines.append("\n---\n*生成自 `.trash/stats_gef_powers.py`，临时分析脚本，不纳入仓库。*")
    out = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    print("\n=== 逐 p 核心指标 ===")
    for r in rows:
        print(f"  {r['label']:22s} raw_t_std={r['raw_t_std']:.3e} max|z|={r['max_abs_z']:.2f} "
              f"kurt={r['kurtosis']:.2f} low%={r['low_mse_share']*100:.1f} high%={r['high_mse_share']*100:.1f} "
              f"top1%={r['top_mse_share']*100:.2f}")
    print(f"\n结果已写入: {OUT}")

if __name__ == "__main__":
    main()
