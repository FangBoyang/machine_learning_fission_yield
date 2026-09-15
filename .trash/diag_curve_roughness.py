# -*- coding: utf-8 -*-
"""yield_vs_A 曲线在【高产额区】的光滑度定量指标。（.trash 临时脚本）

曲线来源：04_energy_dep 的 yield_sum_by_A_*.csv，即对每个 E，Y(A) = Σ_Z Y(Z,A,E)。
A 为 66..172 步长 1 的均匀网格，故可直接用离散二阶差分。

指标（全部在 log10 空间，因而跨数量级可比的"相对崎岖"）：
  对高产额区内每个有效三点组 (A-1, A, A+1)：
    d2 = L(A-1) - 2·L(A) + L(A+1)
    W  = |d2| / 2        ← 该点偏离其左右邻居连线的距离，单位 dex
  - W_med ：W 的中位数（典型崎岖程度）
  - W_p90 ：W 的 90 分位（最刺眼的那些毛刺）
  - W_rms ：d2 的 RMS
  换算成百分比：10^W - 1

  辅助指标：
  - 多余极值数：高产额连续区间内，局部极值个数减去该区间应有的 1 个峰
                （双峰谷被阈值切断后，每段应当只有 1 个极大、0 个极小）
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'pipeline', 'output')
THR = float(os.environ.get("ROUGH_THR", 0.10))   # 高产额区阈值：Y >= THR * max(Y)
MIN_RUN = 6         # 连续区间至少这么多点才统计

GROUPS = [
    ('w_ens', 'w_ens_seed_p3', 'w_ens_seed_p3'),
    ('v_ens', 'v_ens_seed_p3', 'v_ens_seed_p3'),
] + [(f'w{i}', f'w{i}_ft_235UALL_power', f'w{i}_ft_235UALL_power') for i in range(1, 7)] \
  + [(f'v{i}', f'v{i}_ft_235UALL_power_delta_np', f'v{i}_ft_235UALL_power_delta_np') for i in range(1, 7)]


def curve(variant_dir, variant):
    p = os.path.join(OUT, variant_dir, 'results', f'yield_sum_by_A_{variant}.csv')
    d = pd.read_csv(p)
    d = d.sort_values(['E_physical', 'A_physical'])
    return d['A_physical'].values, d['E_physical'].values, d['Yield_pred'].values.astype(float)


try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

SG_WIN, SG_POLY = 9, 3      # 中波长基准：9 点三次多项式平滑


def roughness(A, Y, thr=THR):
    """对一条 Y(A) 曲线（A 已排序、步长 1）返回
    (W_med, W_p90, W_rms, n_extra_extrema, n_used, S_med, S_rms)。
    前 5 个是单点尺度（二阶差分），后 2 个是中波长尺度（与平滑样条的偏离）。"""
    n = len(A)
    assert np.all(np.diff(A) == 1), 'A 必须步长 1'
    mx = Y.max()
    if mx <= 0:
        return (np.nan,) * 5
    mask = Y >= thr * mx
    # 只保留长度足够的连续区间（阈值会把双峰切成两段，谷被排除）
    runs, cur = [], []
    for i in range(n):
        if mask[i]:
            cur.append(i)
        else:
            if len(cur) >= MIN_RUN:
                runs.append(cur)
            cur = []
    if len(cur) >= MIN_RUN:
        runs.append(cur)

    W, S, WL = [], [], []
    n_extra = 0
    n_used = 0
    for r in runs:
        r = np.asarray(r)
        y = Y[r]
        if not np.all(y > 0):
            continue
        L = np.log10(y)
        n_used += len(r)
        if len(r) >= 3:
            d2 = L[:-2] - 2 * L[1:-1] + L[2:]
            W.extend(np.abs(d2) / 2.0)
        # ---- 线性空间：二阶差分的绝对量级被峰区主导 → 天然是「高产额光滑性」----
        if len(r) >= 3:
            d2l = y[:-2] - 2 * y[1:-1] + y[2:]
            WL.extend(np.abs(d2l) / 2.0)
        # 中波长：与该段自身的平滑版本（Savitzky-Golay）的偏离
        if _HAS_SCIPY and len(r) >= SG_WIN:
            Ls = savgol_filter(L, SG_WIN, SG_POLY)
            res = L - Ls
            # 去掉两端各 SG_WIN//2 个边界点（平滑在端点不可靠）
            res = res[SG_WIN // 2: len(r) - SG_WIN // 2]
            S.extend(np.abs(res))
        # 极值：内部点的局部极大/极小
        if len(r) >= 3:
            loc_max = (L[1:-1] > L[:-2]) & (L[1:-1] > L[2:])
            loc_min = (L[1:-1] < L[:-2]) & (L[1:-1] < L[2:])
            n_extra += int(loc_min.sum()) + max(0, int(loc_max.sum()) - 1)
    if not W:
        return (np.nan,) * 10
    W = np.asarray(W)
    s_med = float(np.median(S)) if S else np.nan
    s_rms = float(np.sqrt(np.mean(np.asarray(S) ** 2))) if S else np.nan
    if WL:
        WL = np.asarray(WL)
        wl_med, wl_p90 = float(np.median(WL)), float(np.percentile(WL, 90))
        wl_rel = wl_med / float(mx)          # 归一化到峰值高度
    else:
        wl_med = wl_p90 = wl_rel = np.nan
    return float(np.median(W)), float(np.percentile(W, 90)), float(np.sqrt(np.mean((2 * W) ** 2))), \
        n_extra, n_used, s_med, s_rms, wl_med, wl_p90, wl_rel


rows = []
for tag, vdir, var in GROUPS:
    try:
        A, E, Y = curve(vdir, var)
    except FileNotFoundError:
        print(f'  [跳过] {tag}: 无 yield_sum_by_A')
        continue
    Es = np.unique(E)
    per_E = []
    extra_tot = 0
    for e in Es:
        m = E == e
        a_e, y_e = A[m], Y[m]
        o = np.argsort(a_e)
        w_med, w_p90, w_rms, ex, nu, s_med, s_rms, wl, wl9, wlr = roughness(a_e[o], y_e[o])
        per_E.append((e, w_med, w_p90, w_rms, ex, nu, s_med, s_rms, wl, wl9, wlr))
        extra_tot += ex
    med = np.nanmedian([r[1] for r in per_E])
    p90 = np.nanmedian([r[2] for r in per_E])
    rms = np.nanmedian([r[3] for r in per_E])
    smed = np.nanmedian([r[6] for r in per_E])
    wlmed = np.nanmedian([r[8] for r in per_E])
    wlrel = np.nanmedian([r[10] for r in per_E])
    rows.append((tag, med, p90, rms, extra_tot, per_E, smed, wlmed, wlrel))

print("=" * 100)
print(f"yield_vs_A 曲线光滑度（高产额区：Y ≥ {THR:.0%} × 峰值；log10 空间；各能量点取中位数）")
print("=" * 100)
print(f"{'变体':<8}{'W_med(log)':>12}{'≈%':>7}{'S_med(log)':>12}{'≈%':>7}"
      f"{'V_med(线性)':>13}{'占峰值%':>9}{'多余极值':>10}")
print("-" * 100)
for tag, med, p90, rms, ex, _, smed, wlmed, wlrel in rows:
    print(f"{tag:<8}{med:>12.5f}{100*(10**med-1):>6.2f}%{smed:>12.5f}{100*(10**smed-1):>6.2f}%"
          f"{wlmed:>13.3e}{100*wlrel:>8.2f}%{ex:>10d}")

print("\n" + "=" * 100)
print("分组对比：单个成员 vs 集成（W_med，越小越光滑）")
print("=" * 100)
d = {t: (m, p, r, e, s, wl) for t, m, p, r, e, _, s, wl, wr_ in rows}
for grp, ens in (('w', 'w_ens'), ('v', 'v_ens')):
    mem = [f'{grp}{i}' for i in range(1, 7) if f'{grp}{i}' in d]
    mv = [d[m][0] for m in mem]
    ev = d[ens][0]
    print(f"\n  {grp} 系列：")
    for m in mem:
        print(f"    {m:<6} W_med={d[m][0]:.5f} ({100*(10**d[m][0]-1):5.2f}%)   "
              f"W_p90={d[m][1]:.5f}   多余极值={d[m][3]}")
    print(f"    {'成员均值':<6} W_med={np.mean(mv):.5f} ({100*(10**np.mean(mv)-1):5.2f}%)")
    print(f"    {ens:<6} W_med={ev:.5f} ({100*(10**ev-1):5.2f}%)   多余极值={d[ens][3]}")
    print(f"    → 集成相对成员均值改善 {100*(1-ev/np.mean(mv)):.1f}%；"
          f"相对最好的成员（{mem[int(np.argmin(mv))]}）改善 {100*(1-ev/np.min(mv)):.1f}%；"
          f"相对最差的成员（{mem[int(np.argmax(mv))]}）改善 {100*(1-ev/np.max(mv)):.1f}%")
    print(f"    → 集成比【所有】成员都光滑? {'是' if ev < min(mv) else '否'}")

print("\n" + "=" * 100)
print("w_ens 逐能量点的 W_med（看是否随能量变化）")
print("=" * 100)
for tag, med, p90, rms, ex, per_E, smed, wlmed, wlrel in rows:
    if tag == 'w_ens':
        for e, wm, wp, wr, x, nu, sm, sr, vl, v9, vr in per_E:
            print(f"    E={e:>5.1f}  W_med(log)={wm:.5f} ({100*(10**wm-1):5.2f}%)  "
                  f"V_med(线性)={vl:.3e}  多余极值={x}")
