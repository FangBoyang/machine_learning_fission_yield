# -*- coding: utf-8 -*-
"""x1 vs w1：原始空间逐点误差的分区诊断（.trash 临时脚本）。

用各自 best 模型在【同一个 val 集】上预测，按真实产额分位数分区，
看误差恶化集中在哪里。同时复算 03 的总指标做交叉校验（防止自己算错）。
"""
import os
import sys
import pickle
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pipeline', 'src'))
import torch  # noqa: E402
from common import load_pretrained_model, output_path  # noqa: E402

OUT = os.path.join(ROOT, 'pipeline', 'output')


def get_val(variant):
    with open(output_path(variant, 'data', f'preprocessed_{variant}.pkl'), 'rb') as f:
        d = pickle.load(f)
    di = d['data_info']
    idx = di['split']['val_indices']
    return (d['X_val'], d['raw_data']['Yield_original'][np.asarray(idx, dtype=int)],
            d['scalers'][di['target_key']], di['target_space'], di['target_power'],
            float(di['config']['target']['clip_min']))


def predict(variant, X):
    model, cfg, grid, path = load_pretrained_model(variant, 'cpu')
    model.eval()
    with torch.no_grad():
        pn = model(torch.tensor(X, dtype=torch.float32)).numpy().reshape(-1)
    return pn, path


def inverse(pn, scaler, space, power, clip_min):
    y = scaler.inverse_transform(np.asarray(pn, dtype=np.float32).reshape(-1, 1)).reshape(-1)
    if space == 'log':
        y = np.exp(y)
    if power != 1.0:
        y = np.power(np.clip(y, 0.0, None), 1.0 / power)
    return np.clip(y, clip_min, None)


Xv, ytrue, _, space, power, clip = get_val('w1_ft_235UALL_power')
Xv2, ytrue2, _, _, _, _ = get_val('x1_ft_235UALL_power')
assert np.array_equal(Xv, Xv2) and np.allclose(ytrue, ytrue2), "两套 val 不一致，无法逐点对比"

res = {}
for v in ('w1_ft_235UALL_power', 'x1_ft_235UALL_power'):
    pn, path = predict(v, Xv)
    sc = get_val(v)[2]
    pred = inverse(pn, sc, space, power, clip)
    res[v] = pred
    print(f"{v}: {os.path.basename(path)}")

pw, px = res['w1_ft_235UALL_power'], res['x1_ft_235UALL_power']

print("\n" + "=" * 88)
print("交叉校验：本脚本复算 vs 03_evaluate 报告")
print("=" * 88)
for v, p in (('w1_ft_235UALL_power', pw), ('x1_ft_235UALL_power', px)):
    import json
    d = json.load(open(os.path.join(OUT, v, 'results', f'eval_report_{v}.json'), encoding='utf-8'))
    o = d['metrics']['original_space']
    hi = d['metrics']['high_yield_region']
    thr = np.percentile(ytrue, 75)
    high = ytrue >= thr
    print(f"  {v}")
    print(f"    R2      {r2_score(ytrue, p):.6f}  vs 报告 {o['r2']:.6f}"
          f"   {'OK' if abs(r2_score(ytrue,p)-o['r2'])<1e-5 else 'MISMATCH'}")
    print(f"    RMSE    {np.sqrt(mean_squared_error(ytrue,p)):.6f}  vs 报告 {o['rmse']:.6f}"
          f"   {'OK' if abs(np.sqrt(mean_squared_error(ytrue,p))-o['rmse'])<1e-5 else 'MISMATCH'}")
    print(f"    MAE     {mean_absolute_error(ytrue,p):.6f}  vs 报告 {o['mae']:.6f}"
          f"   {'OK' if abs(mean_absolute_error(ytrue,p)-o['mae'])<1e-5 else 'MISMATCH'}")
    print(f"    highR2  {r2_score(ytrue[high],p[high]):.6f}  vs 报告 {hi['r2']:.6f}"
          f"   {'OK' if abs(r2_score(ytrue[high],p[high])-hi['r2'])<1e-5 else 'MISMATCH'}")

print("\n" + "=" * 88)
print("按真实产额分位分区诊断（log10 相对误差 = log10(pred/true)）")
print("=" * 88)
print(f"{'区间':<22}{'n':>5}{'w1 中位|Δlog|':>16}{'x1 中位|Δlog|':>16}{'Δ(x1-w1)':>13}"
      f"{'w1 偏差':>12}{'x1 偏差':>12}")
print("-" * 88)
qs = [0, 25, 50, 75, 90, 95, 99, 100]
labels = ['0-25%', '25-50%', '50-75%', '75-90%', '90-95%', '95-99%', '99-100%']
rows = []
for j in range(len(qs) - 1):
    lo, hi = np.percentile(ytrue, qs[j]), np.percentile(ytrue, qs[j + 1])
    m = (ytrue >= lo) & (ytrue <= hi) if j == len(qs) - 2 else (ytrue >= lo) & (ytrue < hi)
    if m.sum() == 0:
        continue
    dw = np.log10(np.clip(pw[m], 1e-30, None) / ytrue[m])
    dx = np.log10(np.clip(px[m], 1e-30, None) / ytrue[m])
    rows.append((labels[j], int(m.sum()),
                 np.median(np.abs(dw)), np.median(np.abs(dx)),
                 np.median(dx) - np.median(dw), np.median(dw), np.median(dx)))
    print(f"{labels[j]:<22}{m.sum():>5}{np.median(np.abs(dw)):>16.4f}{np.median(np.abs(dx)):>16.4f}"
          f"{np.median(np.abs(dx))-np.median(np.abs(dw)):>+13.4f}"
          f"{np.median(dw):>+12.4f}{np.median(dx):>+12.4f}")

print("\n" + "=" * 88)
print("对 RMSE 恶化的贡献分解（原始空间，平方误差之和）")
print("=" * 88)
se_w = (pw - ytrue) ** 2
se_x = (px - ytrue) ** 2
tot = se_x.sum() - se_w.sum()
print(f"  总平方误差: w1={se_w.sum():.6e}  x1={se_x.sum():.6e}  增量={tot:.6e}")
for lab, lo, hi in (('0-25%', 0, 25), ('25-50%', 25, 50), ('50-75%', 75 - 25, 75),
                    ('75-90%', 75, 90), ('90-95%', 90, 95), ('95-99%', 95, 99), ('99-100%', 99, 100)):
    pass
qs2 = [0, 25, 50, 75, 90, 95, 99, 100]
for j in range(len(qs2) - 1):
    lo, hi = np.percentile(ytrue, qs2[j]), np.percentile(ytrue, qs2[j + 1])
    m = (ytrue >= lo) & (ytrue <= hi) if j == len(qs2) - 2 else (ytrue >= lo) & (ytrue < hi)
    contrib = (se_x[m].sum() - se_w[m].sum())
    print(f"  {labels[j]:<10} 贡献增量 = {contrib:+.4e}  （占总增量 {contrib/tot*100:+.1f}%）")
