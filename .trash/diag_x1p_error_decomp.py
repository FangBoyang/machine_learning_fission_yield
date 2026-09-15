# -*- coding: utf-8 -*-
"""w1 / x1 / x1' 三方误差分解（.trash 临时脚本）。

在同一个 val 集（split.seed=42，三者逐点对齐）上预测，按真实产额分位分区，
看各自的误差相对 w1 增量集中在哪里。总指标与 03 报告交叉校验。
"""
import os
import sys
import json
import pickle
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'src'))
from common import load_pretrained_model, output_path  # noqa: E402

OUT = os.path.join(ROOT, 'pipeline', 'output')
VARIANTS = [('w1', 'w1_ft_235UALL_power'),
            ('x1', 'x1_ft_235UALL_power'),
            ("x1'", "x1'_ft_235UALL_power")]


def get_val(variant):
    with open(output_path(variant, 'data', f'preprocessed_{variant}.pkl'), 'rb') as f:
        d = pickle.load(f)
    di = d['data_info']
    idx = np.asarray(di['split']['val_indices'], dtype=int)
    return d['X_val'], d['raw_data']['Yield_original'][idx], d['scalers'][di['target_key']], di


Xv, ytrue, _, di = get_val('w1_ft_235UALL_power')
space, power = di['target_space'], di['target_power']
clip = float(di['config']['target']['clip_min'])

preds = {}
for tag, v in VARIANTS:
    with open(output_path(v, 'data', f'preprocessed_{v}.pkl'), 'rb') as f:
        sc = pickle.load(f)['scalers'][di['target_key']]
    model, cfg, grid, path = load_pretrained_model(v, 'cpu')
    model.eval()
    with torch.no_grad():
        pn = model(torch.tensor(Xv, dtype=torch.float32)).numpy().reshape(-1)
    y = sc.inverse_transform(np.asarray(pn, dtype=np.float32).reshape(-1, 1)).reshape(-1)
    if space == 'log':
        y = np.exp(y)
    if power != 1.0:
        y = np.power(np.clip(y, 0.0, None), 1.0 / power)
    preds[tag] = np.clip(y, clip, None)

print("=" * 80)
print("交叉校验（本脚本复算 vs 03 报告）")
print("=" * 80)
for tag, v in VARIANTS:
    d = json.load(open(os.path.join(OUT, v, 'results', f'eval_report_{v}.json'), encoding='utf-8'))
    o = d['metrics']['original_space']
    thr = np.percentile(ytrue, 75)
    hi = ytrue >= thr
    r2, rmse, mae = (r2_score(ytrue, preds[tag]), np.sqrt(mean_squared_error(ytrue, preds[tag])),
                     mean_absolute_error(ytrue, preds[tag]))
    r2h = r2_score(ytrue[hi], preds[tag][hi])
    ok = (abs(r2 - o['r2']) < 1e-5 and abs(rmse - o['rmse']) < 1e-5
          and abs(mae - o['mae']) < 1e-5 and abs(r2h - d['metrics']['high_yield_region']['r2']) < 1e-5)
    print(f"  {tag:<5} R2={r2:.6f} RMSE={rmse:.6f} MAE={mae:.6f} highR2={r2h:.6f}  {'OK' if ok else 'MISMATCH'}")

qs = [0, 50, 75, 90, 95, 99, 100]
labels = ['0-50%', '50-75%', '75-90%', '90-95%', '95-99%', '99-100%']
masks = []
for j in range(len(qs) - 1):
    lo, hi = np.percentile(ytrue, qs[j]), np.percentile(ytrue, qs[j + 1])
    masks.append((labels[j], (ytrue >= lo) & (ytrue <= hi) if j == len(qs) - 2
                  else (ytrue >= lo) & (ytrue < hi)))

print("\n" + "=" * 80)
print("各区中位相对误差 |log10(pred/true)|（低位产额真值含 0，故只统计真值>0 的点）")
print("=" * 80)
print(f"{'区间':<12}{'n':>5}{'w1':>10}{'x1':>10}{'x1p':>10}{'x1-w1':>10}{'x1p-w1':>10}")
for lab, m in masks:
    v = m & (ytrue > 0)
    if v.sum() < 3:
        continue
    r = {}
    for tag, _ in VARIANTS:
        r[tag] = np.median(np.abs(np.log10(np.clip(preds[tag][v], 1e-30, None) / ytrue[v])))
    rw, r1, rp = r['w1'], r['x1'], r["x1'"]
    print(f"{lab:<12}{v.sum():>5}{rw:>10.4f}{r1:>10.4f}{rp:>10.4f}"
          f"{r1 - rw:>+10.4f}{rp - rw:>+10.4f}")

print("\n" + "=" * 80)
print("相对 w1 的【平方误差增量】分解（原始空间）")
print("=" * 80)
se = {t: (preds[t] - ytrue) ** 2 for t, _ in VARIANTS}
for tag in ('x1', "x1'"):
    tot = se[tag].sum() - se['w1'].sum()
    print(f"\n  {tag} vs w1：总增量 = {tot:+.4e}（w1 总平方误差 = {se['w1'].sum():.4e}）")
    for lab, m in masks:
        c = se[tag][m].sum() - se['w1'][m].sum()
        pct = c / tot * 100 if abs(tot) > 1e-15 else float('nan')
        print(f"    {lab:<10} 增量 {c:+.4e}   占总增量 {pct:+7.1f}%")

print("\n" + "=" * 80)
print("=" * 80)
top = ytrue >= np.percentile(ytrue, 90)
ntop = int(top.sum())
print(f"最高 10% 产额区（n={ntop}）的偏差与离散")
for tag, _ in VARIANTS:
    d = np.log10(np.clip(preds[tag][top], 1e-30, None) / ytrue[top])
    print(f"  {tag:<5} 中位偏差 {np.median(d):+.4f}   中位|偏差| {np.median(np.abs(d)):.4f}   "
          f"RMSE(原始) {np.sqrt(mean_squared_error(ytrue[top], preds[tag][top])):.5f}"
          f"   预测峰值 {preds[tag].max():.4f}（真值峰值 {ytrue.max():.4f}）")
