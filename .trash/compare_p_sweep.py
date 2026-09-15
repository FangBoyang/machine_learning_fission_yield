#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临时对比分析脚本（置于 .trash/，不进仓库根）。

对比 p ∈ {0.15(p2), 0.25(基线 p), 0.35(p3), 0.5(p4)} 四个目标幂次变体：
  1) GEF warmup 域内拟合（in-sample R²）—— 来自各 *_gef/results/eval_report_*.json
  2) 235U finetune 域内（held-out val）指标 —— 来自各 *_ft/results/eval_report_*.json
  3) 灾难性遗忘：把微调后模型在 GEF 域 X_train 上重新评估（post-ft GEF R²），
     与 warmup 基线之差 = 遗忘量 Δ。复用 eval_gef_postfinetune.py 的模型加载/反变换逻辑。

运行：从 pipeline/ 目录执行：
  PYTHONPATH=src C:/Users/86138/.conda/envs/fpy_kan/python.exe -u ../.trash/compare_p_sweep.py
结果写入 .trash/compare_p_sweep.md
"""
import os
import sys
import json
import pickle

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)                       # .../pipeline
sys.path.insert(0, os.path.join(PIPE, 'src'))
from kan import KAN
from common import build_kan_from_ckpt, output_path

OUT_MD = os.path.join(HERE, 'compare_p_sweep.md')

# ft 变体 -> gef 变体 映射（与 config 的 reuse_scalers_from/init_from 一致）
PAIRS = [
    ('p2_ft_235UALL_power_delta_np', 'p2_gef_isomer_delta_np', 0.15),
    ('p_ft_235UALL_power_delta_np',  'p_gef_isomer_delta_np',  0.25),
    ('p3_ft_235UALL_power_delta_np', 'p3_gef_isomer_delta_np', 0.35),
    ('p4_ft_235UALL_power_delta_np', 'p4_gef_isomer_delta_np', 0.50),
]


def predict(model, X, device, bs=1024):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], bs):
            out.append(model(Xt[i:i + bs]).detach().cpu().numpy())
    return np.concatenate(out).reshape(-1)


def gef_postft_r2(ftv, gefv, device):
    """微调后模型在 GEF 域 X_train 上的 R²（遗忘后）。"""
    best = output_path(ftv, 'models', f'kan_best_{ftv}.pth')
    ckpt = torch.load(best, map_location=device, weights_only=False)
    model, mc, _ = build_kan_from_ckpt(ckpt, device)

    pkl = output_path(gefv, 'data', f'preprocessed_{gefv}.pkl')
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    X = data['X_train']
    y_true = np.asarray(data['raw_data']['Yield_original'], dtype=np.float32).reshape(-1)

    info = data.get('data_info', {})
    target_power = float(info.get('target_power', 1.0))
    target_space = mc['target']['space']
    clip_min = float(mc['target']['clip_min'])
    scaler_key = info.get('target_key', 'Yield_original')
    scaler = data['scalers'].get(scaler_key) or data['scalers'].get('Yield_original')

    y_pred_norm = predict(model, X, device)
    y = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).reshape(-1)
    if target_space == 'log':
        y = np.exp(y)
    if target_power != 1.0:
        y = np.power(np.clip(y, 0.0, None), 1.0 / target_power)
    y = np.clip(y, clip_min, None)

    return float(r2_score(y_true, y)), y_true, y


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for ftv, gefv, p in PAIRS:
        # --- GEF warmup 基线（in-sample）---
        gr = read_json(output_path(gefv, 'results', f'eval_report_{gefv}.json'))
        gef_base_r2 = gr['metrics']['original_space']['r2']
        gef_base_hr2 = gr['metrics']['high_yield_region']['r2']

        # --- 235U finetune（held-out val）---
        fr = read_json(output_path(ftv, 'results', f'eval_report_{ftv}.json'))
        u_r2 = fr['metrics']['original_space']['r2']
        u_rmse = fr['metrics']['original_space']['rmse']
        u_mae = fr['metrics']['original_space']['mae']
        u_hr2 = fr['metrics']['high_yield_region']['r2']
        ir = fr.get('important_region') or {}
        ir_r2 = ir.get('r2')
        ir_gain = ir.get('finetune_gain_r2')
        zs_r2 = fr.get('zero_shot_pretrained', {}).get('r2')

        # --- GEF 遗忘：微调后模型在 GEF 上的 R² ---
        post_r2, _, _ = gef_postft_r2(ftv, gefv, device)
        forget = gef_base_r2 - post_r2   # >0 表示遗忘

        rows.append(dict(
            p=p, ftv=ftv, gefv=gefv,
            gef_base_r2=gef_base_r2, gef_base_hr2=gef_base_hr2,
            gef_post_r2=post_r2, forget=forget,
            u_r2=u_r2, u_rmse=u_rmse, u_mae=u_mae, u_hr2=u_hr2,
            ir_r2=ir_r2, ir_gain=ir_gain, zs_r2=zs_r2,
        ))
        print(f"p={p:.2f} {ftv:36s} GEF_base={gef_base_r2:.4f} GEF_post={post_r2:.4f} "
              f"forget={forget:+.4f} | 235U R2={u_r2:.4f} RMSE={u_rmse:.4f} MAE={u_mae:.4f} "
              f"hiR2={u_hr2:.4f} impR2={ir_r2 if ir_r2 is None else round(ir_r2,4)} "
              f"gain={ir_gain if ir_gain is None else round(ir_gain,4)} zs={zs_r2 if zs_r2 is None else round(zs_r2,4)}")

    # ---------- Markdown ----------
    L = []
    L.append("# p 幂次扫描对比分析（p ∈ {0.15, 0.25, 0.35, 0.5}）\n")
    L.append("数据源：GEF_isomer_merged（warmup, full_train）+ 235UALL（finetune, held-out val）。\n")
    L.append("架构统一 [16,16]/grid15/k3；GEF epochs=2000/patience=400，ft patience=100；"
             "GEF lr=0.05（p4=0.04），ft lr=0.001（p4=0.0008）。\n")
    L.append("**遗忘 Δ = GEF warmup in-sample R² − 微调后 GEF R²**（越大=忘得越多）。\n")

    L.append("\n## 总表\n")
    L.append("| p | GEF warmup R² | GEF 遗忘Δ | 235U val R² | 235U RMSE | 235U MAE | "
             "235U 高产额R² | 235U 重要区R² | finetune增益 | 零样本(预训练→235U)R² |\n"
             "|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {r['p']:.2f} | {r['gef_base_r2']:.4f} | {r['forget']:+.4f} | {r['u_r2']:.4f} | "
            f"{r['u_rmse']:.4f} | {r['u_mae']:.4f} | {r['u_hr2']:.4f} | "
            f"{(r['ir_r2'] if r['ir_r2'] is not None else float('nan')):.4f} | "
            f"{(r['ir_gain'] if r['ir_gain'] is not None else float('nan')):.4f} | "
            f"{(r['zs_r2'] if r['zs_r2'] is not None else float('nan')):.4f} |")

    L.append("\n## 解读\n")
    # 自动挑“最佳” 235U R² 与最小遗忘
    best_u = max(rows, key=lambda r: r['u_r2'])
    min_forget = min(rows, key=lambda r: r['forget'])
    L.append(f"- **235U 拟合最好**：p={best_u['p']:.2f}（R²={best_u['u_r2']:.4f}, RMSE={best_u['u_rmse']:.4f}）。")
    L.append(f"- **GEF 遗忘最少**：p={min_forget['p']:.2f}（Δ={min_forget['forget']:+.4f}）。")
    L.append(f"- **GEF 遗忘最多**：p={max(rows,key=lambda r:r['forget'])['p']:.2f}"
             f"（Δ={max(rows,key=lambda r:r['forget'])['forget']:+.4f}）。")
    L.append("")
    L.append("### 幂次 p 的影响趋势（结合 .trash/gef_powers_stats.md 的统计）\n")
    L.append("- p 越小（0.15）→ 损失权重越偏向**低/中产额**，235U 低产额拟合更充分，"
             "但 GEF 域内（以中高产额为主）拟合略松。")
    L.append("- p 越大（0.5）→ 越接近 raw，损失被**高产额**主导，235U 高产额 R² 通常更高，"
             "但低产额拟合变差，且标准化后尾部更重（训练更不稳）。")
    L.append("- 灾难性遗忘方面：finetune 主要改写输出/后层，GEF 遗忘量一般随 p 变化不大，"
             "但需看上表 Δ 具体数值判断是否存在某一 p 特别容易忘。")
    L.append("")
    L.append("### 推荐\n")
    L.append("- 若目标是**整体 235U 精度（含低产额）**：优先看 p=0.15/0.25。")
    L.append("- 若目标是**高产额核素精度**：优先看 p=0.35/0.5 的高产额 R²。")
    L.append("- 综合建议：在 235U R² 与 GEF 遗忘 Δ 之间取折中；若某 p 遗忘异常大则弃用。")
    L.append("\n---\n*生成自 `.trash/compare_p_sweep.py`。*")

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    print(f"\n结果已写入: {OUT_MD}")


if __name__ == '__main__':
    main()
