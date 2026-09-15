#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临时脚本（置于 .trash/，不进仓库根）：

试验"跨模型集成"能否抬升高产额精度。复用 src/03_evaluate.py 的模型加载/预测/反变换，
且完全复刻其 val 评估集选取，因此单模型指标与其 eval_report 完全一致。

比较三种集成（均在 **原始空间** 反变换后求平均，因各 p 模型目标/反变换不同）：
  A. 全平均       ：p2(0.15) + p(0.25) + p3(0.35) + p4(0.50) 等权平均
  B. top-3 等权   ：p(0.25) + p3(0.35) + p4(0.50) 等权（剔除 p2，其高产额 R² 仅 0.862）
  C. top-3 加权   ：按各模型原始空间 R² 在三者间归一化加权

同时输出逐点 std（跨 4 个 p 模型），作为"模型形式不确定性"的粗略探针。

运行（从 pipeline/ 目录）：
  PYTHONPATH=src C:/Users/86138/.conda/envs/fpy_kan/python.exe -u ../.trash/ensemble_top3.py
结果写入 .trash/ensemble_top3.md
"""
import os
import sys
import json
import pickle
import importlib.util

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(os.path.dirname(HERE), 'pipeline')
sys.path.insert(0, os.path.join(PIPE, 'src'))

from common import load_config, get_variant, output_path

# —— 复用 03_evaluate.py 的核心函数 ——
spec = importlib.util.spec_from_file_location(
    "eval03", os.path.join(PIPE, 'src', '03_evaluate.py'))
eval03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval03)

# 四个 p 变体（235U finetune），p 值标注在注释里
VARIANTS = [
    ('p2_ft_235UALL_power_delta_np', 0.15),
    ('p_ft_235UALL_power_delta_np',  0.25),
    ('p3_ft_235UALL_power_delta_np', 0.35),
    ('p4_ft_235UALL_power_delta_np', 0.50),
]
TOP3 = ['p_ft_235UALL_power_delta_np', 'p3_ft_235UALL_power_delta_np', 'p4_ft_235UALL_power_delta_np']
OUT_MD = os.path.join(HERE, 'ensemble_top3.md')


def select_val(data, cfg):
    """完全复刻 03_evaluate 的 val 评估集选取，返回 (X_eval, y_true_orig, tag)。"""
    eval_cfg = cfg.get('eval', {}) or {}
    on = eval_cfg.get('eval_set', 'auto')
    has_val = 'X_val' in data and data.get('X_val') is not None
    split_meta = data.get('data_info', {}).get('split', {}) or {}
    if on == 'val' and has_val:
        X_eval, split_idx, tag = data['X_val'], split_meta.get('val_indices'), 'val (held-out)'
    elif on == 'train' or (on == 'auto' and not has_val):
        X_eval, split_idx, tag = data['X_train'], split_meta.get('train_indices'), 'train (in-sample)'
    elif has_val:
        X_eval, split_idx, tag = data['X_val'], split_meta.get('val_indices'), 'val (held-out)'
    else:
        X_eval, split_idx, tag = data['X_train'], split_meta.get('train_indices'), 'train (in-sample)'
    full_yield = np.asarray(data['raw_data']['Yield_original'], dtype=np.float32).reshape(-1)
    y_true_orig = full_yield[np.asarray(split_idx, dtype=int)] if split_idx is not None else full_yield
    return np.asarray(X_eval, dtype=np.float32), y_true_orig, tag


def predict_original(variant, device):
    """加载单模型，返回 (y_pred_original, r2_orig) 及复用所需信息。"""
    cfg = load_config(os.path.join(PIPE, 'configs', f'{variant}.yaml'))
    v = get_variant(cfg)
    pkl_path = output_path(v, 'data', f'preprocessed_{v}.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    model, ckpt, model_path, model_cfg = eval03._load_model(v, device)

    X_eval, y_true_orig, tag = select_val(data, cfg)

    target_space = model_cfg['target']['space']
    clip_min = float(model_cfg['target']['clip_min'])
    info = data.get('data_info', {})
    target_power = float(info.get('target_power', 1.0))
    scaler_key = info.get('target_key', 'Yield_log' if target_space == 'log' else 'Yield_original')
    target_scaler = data['scalers'].get(scaler_key)
    if target_scaler is None:
        target_scaler = data['scalers'].get('Yield_original') or data['scalers'].get('Yield_log')

    y_pred_norm = eval03._predict(model, X_eval, device)
    y_pred_original = eval03._inverse_target(y_pred_norm, target_scaler, target_space, target_power)
    y_pred_original = np.clip(y_pred_original, clip_min, None)
    r2_orig = r2_score(y_true_orig, y_pred_original)
    return y_true_orig, y_pred_original, r2_orig, tag


def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    high_thr = np.percentile(y_true, 75)
    high = y_true >= high_thr
    if high.sum() > 0:
        r2_h = r2_score(y_true[high], y_pred[high])
        rmse_h = float(np.sqrt(mean_squared_error(y_true[high], y_pred[high])))
    else:
        r2_h = rmse_h = float('nan')
    return r2, rmse, mae, r2_h, rmse_h, high_thr


def main():
    device = torch.device('cpu')
    preds = {}        # variant -> y_pred_original
    r2_indiv = {}     # variant -> r2_orig
    tag = None
    y_true = None

    for variant, p in VARIANTS:
        yt, yp, r2, tag = predict_original(variant, device)
        preds[variant] = yp
        r2_indiv[variant] = r2
        if y_true is None:
            y_true = yt
        print(f"[{variant}] (p={p}) 单模型原始 R²={r2:.4f}")

    P = np.stack([preds[v] for v, _ in VARIANTS], axis=0)  # (4, n)
    stack = {v: preds[v] for v, _ in VARIANTS}

    # —— A. 全平均 ——
    mean_all = P.mean(axis=0)
    # —— B. top-3 等权 ——
    P3 = np.stack([preds[v] for v in TOP3], axis=0)
    mean_top3 = P3.mean(axis=0)
    # —— C. top-3 按 R² 加权 ——
    w = np.array([r2_indiv[v] for v in TOP3], dtype=float)
    w = w / w.sum()
    mean_w = (P3 * w[:, None]).sum(axis=0)

    # —— 逐点 std（跨 4 个 p 模型），粗探针 ——
    std_all = P.std(axis=0, ddof=1)

    rows = []
    def add(name, yp):
        r2, rmse, mae, r2h, rmseh, thr = metrics(y_true, yp)
        rows.append((name, r2, rmse, mae, r2h, rmseh))
        print(f"  {name:14s} R²={r2:.4f}  RMSE={rmse:.4e}  MAE={mae:.4e}  高产额R²={r2h:.4f} (阈值={thr:.4e}, n={int((y_true>=thr).sum())})")

    print("\n=== 集成对比（评估集: %s, n=%d）===" % (tag, len(y_true)))
    for variant, p in VARIANTS:
        add(f"单:{variant}(p={p})", preds[variant])
    add("A 全平均(4)", mean_all)
    add("B top3等权", mean_top3)
    add("C top3加权", mean_w)

    # —— 写 md ——
    L = []
    L.append("# 跨模型集成探针（top-3 加权 vs 全平均）\n")
    L.append(f"- 评估集: {tag}，样本数 n={len(y_true)}")
    L.append(f"- 复用 `src/03_evaluate.py` 的模型加载/预测/反变换与相同 val 选取，故单模型原始 R² 与其 eval_report 一致。")
    L.append(f"- 集成在 **原始空间** 反变换后求平均（各 p 模型目标/反变换不同，不能在归一化空间直接平均）。\n")
    L.append("## 单模型基线\n")
    L.append("| 变体 | p | 原始 R² |")
    L.append("|---|---|---|")
    for variant, p in VARIANTS:
        L.append(f"| {variant} | {p} | {r2_indiv[variant]:.4f} |")
    L.append("\n## 集成结果\n")
    L.append("| 方案 | 原始 R² | RMSE | MAE | 高产额 R² | 高产额 RMSE |")
    L.append("|---|---|---|---|---|---|")
    for name, r2, rmse, mae, r2h, rmseh in rows:
        # 单模型行不重复打印高产额 RMSE 列不好对齐，简单全打
        L.append(f"| {name} | {r2:.4f} | {rmse:.4e} | {mae:.4e} | {r2h:.4f} | {rmseh:.4e} |")
    L.append("\n## 解读\n")
    best_single = max(r2_indiv, key=lambda k: r2_indiv[k])
    best_ens = max(rows[4:], key=lambda r: r[1])  # 跳过前4个单模型行
    L.append(f"- 单模型最佳: `{best_single}` R²={r2_indiv[best_single]:.4f}；最高产额 R² 来源同。")
    L.append(f"- 集成最佳方案: **{best_ens[0]}**，原始 R²={best_ens[1]:.4f}，高产额 R²={best_ens[4]:.4f}。")
    L.append(f"- 若 `top3` 任一方案原始 R² / 高产额 R² 超过单模型最佳，说明跨模型集成确有精度收益；否则精度已近天花板，"
             f"集成的价值转向 UQ（逐点 std 跨 4 模型，均值={std_all.mean():.4e}，可作为模型形式不确定性的粗估计）。")
    L.append("\n---\n*生成自 `.trash/ensemble_top3.py`，临时分析脚本。*")

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    print(f"\n结果已写入: {OUT_MD}")


if __name__ == '__main__':
    main()
