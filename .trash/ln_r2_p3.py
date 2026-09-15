#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临时脚本（置于 .trash/，不进仓库根）：

统计 p3_ft（p=0.35）在 **ln(自然对数) 空间** 的 R² / RMSE，以及平均绝对相对误差 MARE，
作为与裂变产额 ML 文献（普遍用 log / 相对误差 而非原始空间 R²）对标用的量。

复用 src/03_evaluate.py 的模型加载/预测/反变换逻辑，且完全复刻其 val 评估集选取路径，
因此得到的 y_true/y_pred 与 03_evaluate 产出的原始空间 R²=0.9784 完全一致（仅后处理指标不同）。

运行（从 pipeline/ 目录）：
  PYTHONPATH=src C:/Users/86138/.conda/envs/fpy_kan/python.exe -u ../.trash/ln_r2_p3.py
结果写入 .trash/ln_r2_p3.md
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

VARIANT = 'p3_ft_235UALL_power_delta_np'
OUT_MD = os.path.join(HERE, 'ln_r2_p3.md')


def main():
    device = torch.device('cpu')
    cfg = load_config(os.path.join(PIPE, 'configs', f'{VARIANT}.yaml'))
    variant = get_variant(cfg)

    pkl_path = output_path(variant, 'data', f'preprocessed_{variant}.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    model, ckpt, model_path, model_cfg = eval03._load_model(variant, device)

    # —— 完全复刻 03_evaluate 的 val 评估集选取 ——
    eval_cfg = cfg.get('eval', {}) or {}
    on = eval_cfg.get('eval_set', 'auto')
    has_val = 'X_val' in data and data.get('X_val') is not None
    split_meta = data.get('data_info', {}).get('split', {}) or {}
    if on == 'val' and has_val:
        X_eval, y_eval_norm, split_idx, tag = data['X_val'], data['y_val'], split_meta.get('val_indices'), 'val (held-out)'
    elif on == 'train' or (on == 'auto' and not has_val):
        X_eval, y_eval_norm, split_idx, tag = data['X_train'], data['y_train'], split_meta.get('train_indices'), 'train (in-sample)'
    elif has_val:
        X_eval, y_eval_norm, split_idx, tag = data['X_val'], data['y_val'], split_meta.get('val_indices'), 'val (held-out)'
    else:
        X_eval, y_eval_norm, split_idx, tag = data['X_train'], data['y_train'], split_meta.get('train_indices'), 'train (in-sample)'

    full_yield = np.asarray(data['raw_data']['Yield_original'], dtype=np.float32).reshape(-1)
    y_true_orig = full_yield[np.asarray(split_idx, dtype=int)] if split_idx is not None else full_yield
    y_eval_norm = np.asarray(y_eval_norm, dtype=np.float32).reshape(-1)

    # —— 反变换到原始空间（与 03_evaluate 完全一致）——
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

    n = len(y_true_orig)
    n_zero = int(np.sum(y_true_orig <= 0))

    # —— 复算已有指标，确认与 03_evaluate 一致 ——
    r2_orig = r2_score(y_true_orig, y_pred_original)
    rmse_orig = float(np.sqrt(mean_squared_error(y_true_orig, y_pred_original)))
    mae_orig = mean_absolute_error(y_true_orig, y_pred_original)
    r2_norm = r2_score(y_eval_norm, y_pred_norm)

    # —— ln / log10 空间指标（排除真值或预测为 0 的点，ln 无定义）——
    m = (y_true_orig > 0) & (y_pred_original > 0)
    n_ln = int(m.sum())
    ln_true = np.log(y_true_orig[m])
    ln_pred = np.log(y_pred_original[m])
    r2_ln = r2_score(ln_true, ln_pred)
    rmse_ln = float(np.sqrt(mean_squared_error(ln_true, ln_pred)))
    r2_log10 = r2_score(np.log10(y_true_orig[m]), np.log10(y_pred_original[m]))
    rmse_log10 = float(np.sqrt(mean_squared_error(np.log10(y_true_orig[m]), np.log10(y_pred_original[m]))))

    # —— 平均绝对相对误差 MARE（文献常用；全量，含 0 用 eps 保护）——
    eps = 1e-12
    rel = np.abs(y_pred_original - y_true_orig) / (np.abs(y_true_orig) + eps)
    mare = float(np.mean(rel))
    mare_pos = float(np.mean(rel[m]))  # 仅在正产额点
    median_rel = float(np.median(rel[m]))

    L = []
    L.append(f"# p3_ft (p=0.35) ln 空间评估\n")
    L.append(f"- 变体: `{VARIANT}`，评估集: {tag}，样本数 n={n}（其中真值 Yield<=0 共 {n_zero} 个，ln 计算已排除）\n")
    L.append(f"- 复用 `src/03_evaluate.py` 的模型加载/预测/反变换与 **完全相同的 val 选取路径**，"
             f"故下列原始空间 R² 应等于 03_evaluate 报告值。\n")
    L.append("\n## 指标对照\n")
    L.append("| 量 | 值 | 说明 |\n|---|---|---|")
    L.append(f"| 原始空间 R² | {r2_orig:.4f} | 与 03_evaluate 报告一致（高产额主导） |")
    L.append(f"| 原始空间 RMSE | {rmse_orig:.4e} | |")
    L.append(f"| 原始空间 MAE | {mae_orig:.4e} | |")
    L.append(f"| 归一化空间 R² | {r2_norm:.4f} | 标准化目标 t 上 |")
    L.append(f"| **ln 空间 R²** | **{r2_ln:.4f}** | 自然对数，排除 {n - n_ln} 个 0 点，n_ln={n_ln} |")
    L.append(f"| ln 空间 RMSE | {rmse_ln:.4f} | 单位：ln(yield) |")
    L.append(f"| log10 空间 R² | {r2_log10:.4f} | 常用对数，便于与文献并排 |")
    L.append(f"| log10 空间 RMSE | {rmse_log10:.4f} | 单位：log10(yield) |")
    L.append(f"| MARE（全量） | {mare*100:.2f}% | 平均绝对相对误差，含 0 点（eps 保护） |")
    L.append(f"| MARE（正产额点） | {mare_pos*100:.2f}% | 仅正产额子集 |")
    L.append(f"| 相对误差中位数（正点） | {median_rel*100:.2f}% | |")
    L.append("\n## 解读\n")
    L.append(f"- **ln 空间 R² = {r2_ln:.4f}**，log10 空间 R² = {r2_log10:.4f}：这是与裂变产额 ML 文献（GPR/BNN/多任务 DNN 多报 log 空间或相对误差）"
             f"可直接对标的数。原始空间 R²=0.978 偏高中产额，ln 空间 R² 才是更公允的‘整体拟合’口径。")
    L.append(f"- ln 空间 RMSE = {rmse_ln:.4f}（即平均约 {np.exp(rmse_ln):.2f}× 的乘性误差幅度），可直接与文献的 log-RMSE 比较。")
    L.append(f"- MARE（正产额点）= {mare_pos*100:.2f}%：文献常报 well-measured 区域 MARE 在几个百分点到十几百分点；"
             f"本结果处于该区间，属‘好模型’水平，但具体高低需对照所用文献的产额覆盖范围。")
    L.append(f"- 注意：{n_zero} 个真值=0 的点无法取 ln，已被排除；若文献也这样处理则可比，否则需确认其零值处理方式。")
    L.append("\n---\n*生成自 `.trash/ln_r2_p3.py`，临时分析脚本。*")

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))

    # 终端也打印
    print(f"[{VARIANT}] n={n} (zeros={n_zero}, ln-valid={n_ln})")
    print(f"  原始空间 R²={r2_orig:.4f} (应≈0.9784) | RMSE={rmse_orig:.4e} | MAE={mae_orig:.4e}")
    print(f"  归一化空间 R²={r2_norm:.4f}")
    print(f"  ln空间   R²={r2_ln:.4f} | RMSE={rmse_ln:.4f}")
    print(f"  log10空间 R²={r2_log10:.4f} | RMSE={rmse_log10:.4f}")
    print(f"  MARE(全量)={mare*100:.2f}% | MARE(正点)={mare_pos*100:.2f}% | 中位相对误差={median_rel*100:.2f}%")
    print(f"\n结果已写入: {OUT_MD}")


if __name__ == '__main__':
    main()
