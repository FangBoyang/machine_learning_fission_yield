# -*- coding: utf-8 -*-
"""
analyze_gef_yield_stats.py — 统计 GEF.csv 产额分布，为损失函数设计提供数据依据。

只做统计，不改动任何训练/评估/数据加载脚本。

GEF.csv 无表头，列顺序（见 00_experimental_data_analysis.py）：
    col0 = Z_norm, col1 = A_norm, col2 = E_norm, col3 = Yield(原始尺度), col4 = Error
"""
import numpy as np
import pandas as pd

CSV = "data/GEF.csv"
df = pd.read_csv(CSV, header=None)
print("=" * 70)
print(f"GEF.csv shape: {df.shape}")
print("各列描述 (col0..col4 = Z_norm,A_norm,E_norm,Yield,Error):")
print(df.describe())

y = df[3].values.astype(float)
e = df[2].values.astype(float)
n_zero = int((y == 0).sum())
n_pos = int((y > 0).sum())
print("\n" + "=" * 70)
print(f"Yield: 零点数={n_zero}, 正点数={n_pos}, 零占比={n_zero/len(y):.3%}")

yp = y[y > 0]  # 只看正产额（0 对损失设计意义有限，但保留计数）

print("\n--- RAW YIELD 分位数 (绝对尺度) ---")
for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  P{p:>3}: {np.percentile(y, p):.6e}")
print(f"  min={y.min():.6e}  max={y.max():.6e}  mean={y.mean():.6e}  median={np.median(y):.6e}  std={y.std():.6e}")
print(f"  偏度 skew = {pd.Series(y).skew():.3f}")

print("\n--- 动态范围 ---")
decades = np.log10(yp.max()) - np.log10(yp.min())
print(f"  非零最大值/最小值 = 10^{decades:.2f} 个数量级")
print(f"  max / median     = {yp.max()/np.median(yp):.1e}x")
print(f"  max / P1         = {yp.max()/np.percentile(yp,1):.1e}x")

print("\n--- LOG10(Yield) 分布 (非零) ---")
ly = np.log10(yp)
print(f"  mean={ly.mean():.3f}  std={ly.std():.3f}  min={ly.min():.3f}  max={ly.max():.3f}")
print(f"  (std 越小说明 log 空间越对称 -> log 变换越自然)")

print("\n--- 低/高产额占比 (按绝对阈值) ---")
for thr in [1e-4, 1e-3, 1e-2, 1e-1]:
    frac = (y < thr).mean()
    print(f"  Yield < {thr:.0e} : 占比 {frac:.3%}")

print("\n--- 幂变换后的“平衡度”对比 (越大越偏向高产额主导) ---")
# 用 变换空间里 高产额组/低产额组 的方差比 衡量“谁主导损失”
def high_low_ratio(p):
    # t = y^p (p=1 raw, 0.5 sqrt, 0 log)
    if p == 0:
        t = np.log10(yp)
    else:
        t = yp ** p
    hi = t[yp >= np.percentile(yp, 75)]   # 高产额 25%
    lo = t[yp <= np.percentile(yp, 25)]   # 低产额 25%
    return hi.var() / lo.var()
for p, name in [(1.0, "raw (p=1)"), (0.75, "p=0.75"), (0.5, "sqrt (p=0.5)"),
                (0.25, "p=0.25"), (0.0, "log (p=0)")]:
    print(f"  {name:14s}: 高产额组方差/低产额组方差 = {high_low_ratio(p):.2e}")

print("\n--- 能量依赖：按 E_norm 十分位看产额分布 ---")
edges = np.quantile(e, np.linspace(0, 1, 11))
print("  E_norm 区间 | 该区 Yield 中位数 | 最大值 | 非零占比")
for i in range(10):
    m = (e >= edges[i]) & (e < edges[i + 1])
    if i == 9:
        m = (e >= edges[i]) & (e <= edges[i + 1])
    yi = y[m]
    if len(yi) == 0:
        continue
    print(f"  [{edges[i]:.3f},{edges[i+1]:.3f}] | med={np.median(yi):.3e} | max={yi.max():.3e} | 非零={ (yi>0).mean():.2%}")
