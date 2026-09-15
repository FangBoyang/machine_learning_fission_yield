# -*- coding: utf-8 -*-
"""1) GEF 与 235UALL 在重合 (Z,A,E) 点上差多少；2) R² 被多少个点主导。（.trash 临时脚本）"""
import os
import pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'pipeline', 'output')

print("=" * 88)
print("一、GEF vs 235UALL：重合点上的直接对比")
print("=" * 88)
g = pd.read_csv(os.path.join(ROOT, 'data/GEF_isomer_merged.csv'))
u = pd.read_csv(os.path.join(ROOT, 'data/235UALL.csv'))


def key(df, dec=6):
    z, a, e = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    return (z.round(dec).astype(str) + '|' + a.round(dec).astype(str)
            + '|' + e.round(dec).astype(str))


gk, uk = key(g), key(u)
mg = pd.DataFrame({'k': gk, 'y': g.iloc[:, 3].values})
mu = pd.DataFrame({'k': uk, 'y': u.iloc[:, 3].values})
# 同一 key 可能出现多次（异构态合并等），先求和
mg = mg.groupby('k', as_index=False)['y'].sum()
mu = mu.groupby('k', as_index=False)['y'].sum()
j = mg.merge(mu, on='k', suffixes=('_gef', '_exp'))
print(f"  GEF 唯一点 {len(mg)}，235UALL 唯一点 {len(mu)}，重合点 {len(j)}")
if len(j) > 10:
    a, b = j['y_gef'].values, j['y_exp'].values
    both = (a > 0) & (b > 0)
    print(f"  两边都 >0 的点：{both.sum()}")
    r = np.log10(b[both] / a[both])
    print(f"  log10(实验/GEF) 偏差：中位 {np.median(r):+.4f}（= 实验是 GEF 的 {10**np.median(r):.3f} 倍）")
    print(f"                        均值 {r.mean():+.4f}   "
          f"68% 区间 [{np.percentile(r,16):+.3f}, {np.percentile(r,84):+.3f}]   "
          f"中位|偏差| {np.median(np.abs(r)):.4f}（= 倍数 {10**np.median(np.abs(r)):.2f}x）")
    lg = np.log10(np.clip(a[both], 1e-30, None))
    le = np.log10(np.clip(b[both], 1e-30, None))
    print(f"  log 空间相关性 r = {np.corrcoef(lg, le)[0,1]:.4f}   "
          f"log 空间 R² = {np.corrcoef(lg, le)[0,1]**2:.4f}")
    hi = b[both] >= np.percentile(b[both], 90)
    if hi.sum() > 3:
        rh = np.log10(b[both][hi] / a[both][hi])
        print(f"  最高 10% 产额点（n={hi.sum()}）：log10(实验/GEF) 中位 {np.median(rh):+.4f}"
              f"（{10**np.median(rh):.3f} 倍），中位|偏差| {np.median(np.abs(rh)):.4f}"
              f"（{10**np.median(np.abs(rh)):.2f}x）")

print("\n" + "=" * 88)
print("二、R² / MSE 被多少个点主导（w1_ft, val n=619）")
print("=" * 88)
with open(os.path.join(OUT, 'w1_ft_235UALL_power', 'data',
                       'preprocessed_w1_ft_235UALL_power.pkl'), 'rb') as fh:
    pk = pickle.load(fh)
idx = np.asarray(pk['data_info']['split']['val_indices'], dtype=int)
y = pk['raw_data']['Yield_original'][idx].astype(float)
df = pd.read_csv(os.path.join(OUT, 'w_ens_seed_p3', 'results',
                              'ensemble_perpoint_w_ens_seed_p3.csv'))
mem = [c for c in df.columns if '_ft_' in c]
pred = df[mem].values.mean(axis=1)
assert np.allclose(df['y_true'].values, y)
se = (pred - y) ** 2
order = np.argsort(-se)
tot = se.sum()
print(f"  总平方误差 {tot:.6e}")
for n in (1, 5, 10, 31, 62, 124):
    print(f"    误差最大的 {n:>3} 个点（占 {n/len(y)*100:5.1f}%）占总平方误差的 "
          f"{se[order[:n]].sum()/tot*100:5.1f}%")

print("\n  按真值产额分区：各区对总平方误差的贡献")
qs = [0, 50, 75, 90, 95, 99, 100]
for jj in range(len(qs) - 1):
    lo, hi = np.percentile(y, qs[jj]), np.percentile(y, qs[jj + 1])
    m = (y >= lo) & (y <= hi) if jj == len(qs) - 2 else (y >= lo) & (y < hi)
    print(f"    {qs[jj]:>3}-{qs[jj+1]:>3}%   n={m.sum():>4}   "
          f"贡献 {se[m].sum()/tot*100:5.1f}%   该区 RMSE {np.sqrt(se[m].mean()):.4e}")

print("\n  若只看真值 > 0.01 的点（产额可见的区）：")
m = y > 0.01
print(f"    n={m.sum()}，贡献总平方误差的 {se[m].sum()/tot*100:.1f}%")
print(f"  若只看真值 <= 0.01 的点：n={(~m).sum()}，贡献 {se[~m].sum()/tot*100:.2f}%")
