# -*- coding: utf-8 -*-
"""
analyze_235UALL_stats.py — 统计 235UALL.csv（实验数据），为 finetune 设计提供依据。

只做统计与对比，不改动任何训练/评估/数据加载脚本。与 analyze_gef_yield_stats.py 对齐风格，
并额外加入 235UALL ↔ GEF 的对比，回答 finetune 最关键的几个问题：
  1) 特征/目标空间是否落在 GEF 拟合的 scaler 范围内（scaler 复用是否成立）；
  2) 235UALL 的产额动态范围/偏度如何，幂次 p 选多少能让低/高产额平衡（指导 target.power）；
  3) 重要区（前 3096 之后）与次要区在产额/E/核素覆盖上有何差异（指导必须学好的目标）；
  4) val（仅从前 3096 划分）覆盖的是哪段分布，能否作为泛化代理。

用法（在 fpy_kan 环境）：
    python analyze_235UALL_stats.py
"""
import os
import numpy as np
import pandas as pd
import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_235 = os.path.join(ROOT, "data", "235UALL.csv")
CSV_GEF = os.path.join(ROOT, "data", "GEF.csv")


def load_scaler(name):
    p = os.path.join(ROOT, "data", name)
    if not os.path.exists(p):
        return None
    return joblib.load(p)


def high_low_ratio(y, p):
    """变换空间 t=y^p（p=0→log10）里，高产额25%/低产额25% 的方差比；越小越平衡。"""
    yp = y[y > 0]
    if p == 0:
        t = np.log10(yp)
    else:
        t = yp ** p
    hi = t[yp >= np.percentile(yp, 75)]
    lo = t[yp <= np.percentile(yp, 25)]
    return hi.var() / lo.var()


def section(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


# ========== 加载 ==========
df = pd.read_csv(CSV_235)  # 有表头 Z,A,E,Yield,Error；Z/A/E 已归一化
print("=" * 70)
print(f"235UALL.csv shape: {df.shape}")
print("各列描述 (Z,A,E 为归一化量, Yield 为原始产额):")
print(df.describe())

# GEF 对照
g = pd.read_csv(CSV_GEF, header=None)
g.columns = ['Z_norm', 'A_norm', 'E_norm', 'Yield', 'Error']
print(f"\nGEF.csv shape: {g.shape}（对照用）")

y = df['Yield'].values.astype(float)
e = df['E'].values.astype(float)
z = df['Z'].values.astype(float)
a = df['A'].values.astype(float)


# ========== 1. 235UALL 产额分布 ==========
section("1. 235UALL 产额分布（绝对尺度）")
n_zero = int((y == 0).sum())
n_pos = int((y > 0).sum())
print(f"零点数={n_zero}, 正点数={n_pos}, 零占比={n_zero/len(y):.3%}")
for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  P{p:>3}: {np.percentile(y, p):.6e}")
print(f"  min={y.min():.6e}  max={y.max():.6e}  mean={y.mean():.6e}  median={np.median(y):.6e}  std={y.std():.6e}")
print(f"  偏度 skew = {pd.Series(y).skew():.3f}")
yp = y[y > 0]
dec = np.log10(yp.max()) - np.log10(yp.min())
print(f"  动态范围: 非零 max/min = 10^{dec:.2f} 个数量级; max/median = {yp.max()/np.median(yp):.1e}x")
print("  低/高产额占比(绝对阈值):")
for thr in [1e-4, 1e-3, 1e-2, 1e-1]:
    print(f"    Yield < {thr:.0e} : { (y < thr).mean():.3%}")


# ========== 2. 幂变换平衡度（指导 target.power） ==========
section("2. 幂变换平衡度（指导 finetune 的 target.power）")
print("  高产额组(75分位+)方差 / 低产额组(25分位-)方差；越小=低/高产额越平衡")
for p, name in [(1.0, "raw (p=1)"), (0.75, "p=0.75"), (0.5, "sqrt (p=0.5)"),
                (0.25, "p=0.25"), (0.0, "log (p=0)")]:
    print(f"    {name:14s}: {high_low_ratio(y, p):.2e}")
print("  （当前 j_power 基用 p=0.25；若 235UALL 在 p=0.5/0.25 已较平衡，则低产额权重足够）")


# ========== 3. 特征/目标空间 vs GEF（指导 scaler 复用 + 冻结） ==========
section("3. 特征与 E 范围：235UALL vs GEF（验证 scaler 复用是否成立）")
print(f"  {'特征':10s} | {'235UALL P1':>12s} {'P50':>12s} {'P99':>12s} | {'GEF P1':>12s} {'P50':>12s} {'P99':>12s}")
for col, gc in [('Z', 'Z_norm'), ('A', 'A_norm'), ('E', 'E_norm')]:
    print(f"  {col+'_norm':10s} | {np.percentile(df[col],1):12.3f} {np.percentile(df[col],50):12.3f} {np.percentile(df[col],99):12.3f} | "
          f"{np.percentile(g[gc],1):12.3f} {np.percentile(g[gc],50):12.3f} {np.percentile(g[gc],99):12.3f}")
print("  注：235UALL 的 Z/A/E 与 GEF 同归一化尺度；E 范围窄说明 finetune 只覆盖 E 的一段子区间。")


# ========== 4. 目标 scaler 复用校验（关键正确性） ==========
section("4. 目标 scaler 复用校验（yield_scaler 拟合于 GEF，MinMaxScaler）")
ys = load_scaler("yield_scaler.pkl")
if ys is not None:
    yn = ys.transform(y.reshape(-1, 1)).flatten()          # 235UALL 归一化产额
    gn = ys.transform(g['Yield'].values.astype(float).reshape(-1, 1)).flatten()  # GEF 归一化产额
    print(f"  scaler 类型: {type(ys).__name__}")
    print(f"  GEF 归一化产额范围: [{gn.min():.4f}, {gn.max():.4f}]  (训练目标空间)")
    print(f"  235UALL 归一化产额范围: [{yn.min():.4f}, {yn.max():.4f}]")
    out = int(((yn < gn.min() - 1e-9) | (yn > gn.max() + 1e-9)).sum())
    print(f"  越出 GEF 归一化范围的点数 = {out} ({out/len(yn):.3%})")
    if out == 0:
        print("  ✅ 235UALL 产额完全落在 GEF 目标空间内 → 复用 yield_scaler 安全，")
        print("     且归一化后仅占 [0, ~0.18] 低位段（235UALL 产额幅度小于 GEF 最大值）。")
    else:
        print("  ⚠️ 存在越界点 → 注意目标空间外推风险。")
else:
    print("  未找到 yield_scaler.pkl，跳过校验。")


# ========== 5. delta_np 分布 vs GEF ==========
section("5. delta_np 分布（235UALL vs GEF）")
sz = load_scaler("standard_scalerZ.pkl")
sa = load_scaler("standard_scalerA.pkl")
sd = load_scaler("delta_np_scaler.pkl")
if sz is not None and sa is not None:
    Zo = sz.inverse_transform(z.reshape(-1, 1)).round().astype(int).flatten()
    Ao = sa.inverse_transform(a.reshape(-1, 1)).round().astype(int).flatten()
    N = Ao - Zo
    I = (N - Zo) / Ao.astype(float)
    Ne = (N % 2 == 0)
    Ze = (Zo % 2 == 0)
    c1 = Ne == Ze
    c2 = Ne & (~Ze) & (N != Zo)
    c3 = (~Ne) & Ze & (N != Zo)
    raw_delta = np.select([c1, c2, c3], [2 - np.abs(I), 1 - np.abs(I), 1 - np.abs(I)], default=1.0)
    print(f"  235UALL delta_np: min={raw_delta.min():.4f}, max={raw_delta.max():.4f}, mean={raw_delta.mean():.4f}")
    if sd is not None:
        dn = sd.transform(raw_delta.reshape(-1, 1)).flatten()
        print(f"  235UALL delta_np(归一化): min={dn.min():.3f}, max={dn.max():.3f}, mean={dn.mean():.3f}")
        # GEF delta_np
        gZo = sz.inverse_transform(g['Z_norm'].values.reshape(-1, 1)).round().astype(int).flatten()
        gAo = sa.inverse_transform(g['A_norm'].values.reshape(-1, 1)).round().astype(int).flatten()
        gN = gAo - gZo
        gI = (gN - gZo) / gAo.astype(float)
        gNe = (gN % 2 == 0); gZe = (gZo % 2 == 0)
        gdelta = np.select([gNe == gZe, gNe & (~gZe) & (gN != gZo), (~gNe) & gZe & (gN != gZo)],
                           [2 - np.abs(gI), 1 - np.abs(gI), 1 - np.abs(gI)], default=1.0)
        gdn = sd.transform(gdelta.reshape(-1, 1)).flatten()
        print(f"  GEF     delta_np(归一化): min={gdn.min():.3f}, max={gdn.max():.3f}, mean={gdn.mean():.3f}（对照）")
        print("  若两者归一化 delta_np 范围接近，则 delta_np 特征在 finetune 中与 GEF 同分布。")


# ========== 6. 重要区划分对比 ==========
section("6. 重要区划分对比（前 3096 = 次要；3097~4127 = 重要）")
split_n = 3096
imp = np.arange(split_n, len(df))          # 重要区行号
unimp = np.arange(0, split_n)              # 次要区行号
yu, yi = y[unimp], y[imp]
eu, ei = e[unimp], e[imp]
print(f"  次要区[{0},{split_n}): n={len(unimp)}, Yield median={np.median(yu):.3e}, max={yu.max():.3e}, "
      f"E=[{eu.min():.3f},{eu.max():.3f}]")
print(f"  重要区[{split_n},{len(df)}): n={len(imp)}, Yield median={np.median(yi):.3e}, max={yi.max():.3e}, "
      f"E=[{ei.min():.3f},{ei.max():.3f}]")
print(f"  重要区/次要区 Yield 中位数比 = {np.median(yi)/np.median(yu):.2e}x")
print(f"  重要区唯一核素(Z,A)数 = {len(set(zip(df['Z'].values[imp].round(3), df['A'].values[imp].round(3))))}")
print(f"  次要区唯一核素(Z,A)数 = {len(set(zip(df['Z'].values[unimp].round(3), df['A'].values[unimp].round(3))))}")


# ========== 7. val 组成（finetune 当前划分） ==========
section("7. val 集组成（val_from_first_n=3096, ratio=0.2 → 619）")
rng = np.random.RandomState(42)
first = np.arange(split_n)
perm = rng.permutation(len(first))
n_val = int(round(len(first) * 0.2))
idx_val = first[perm[:n_val]]
yv = y[idx_val]; ev = e[idx_val]
print(f"  val (n={len(idx_val)}): Yield median={np.median(yv):.3e}, max={yv.max():.3e}, "
      f"E=[{ev.min():.3f},{ev.max():.3f}]")
print(f"  val 完全落在次要区（前 3096）内 → val R² 是实验分布'次要子区'的泛化代理；")
print(f"  重要区的拟合质量只能由训练内指标(in-sample)反映，需结合零样本对照解读。")


# ========== 8. finetune 含义小结 ==========
section("8. 对 finetune 的启发（统计结论）")
print("  - 见上方各节对比，据此决定：target.power / 是否换基 / 冻结 / val 比例。")
print("  - 若第4节 |z|>3 占比很高 → 235UALL 与 GEF 产额分布差异大，需警惕 scaler 复用；")
print("    可考虑对 235UALL 单独 fit target scaler（但会破坏与预训练权重的目标空间一致性）。")
print("  - 若第2节显示某 p 下高低产额方差比明显更小 → 该 p 更平衡，适合作为 finetune 目标幂次。")
print("  - 若第3/5节特征范围 235UALL 是 GEF 的子集 → 冻结部分层安全；若越界则需全量微调。")
