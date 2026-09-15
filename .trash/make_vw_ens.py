# -*- coding: utf-8 -*-
"""make_vw_ens.py — 生成 vw_ens（v 的 6 个 + w 的 6 个 = 12 模型平均）曲线 + 图。

与 04_energy_dep.py 完全一致的产物：
  - yield_sum_by_A_vw_ens.csv / yield_sum_by_Z_vw_ens.csv  （按 A/Z 聚合，含 ±1σ）
  - energy_dep_vw_ens.csv                                    （逐核素预测，含 ±1σ）
  - yield_vs_A_vw_ens.png / yield_vs_Z_vw_ens.png            （复刻 04 的画图代码）
    —— 线性 y 轴、英文标注、viridis、±1σ 带，与 w_ens/v_ens 的 04 图同款。

做法：w_ens / v_ens 的 04 阶段 CSV 本身已是各自 6 模型的平均（原始产额空间求和后平均），
因此直接把这两张曲线 CSV 逐点平均 == 对全部 12 个模型平均（求和线性，等价）。
无需加载任何模型、无需重训。
注：04 的 ensemble 模式要求所有成员特征一致，而 w(3维) 与 v(4维) 特征不同无法并入同一次
04 运行；故此处对「已聚合曲线」平均，正是等价且合规的做法。

输入（04 已生成）：
  pipeline/output/w_ens_seed_p3/results/{yield_sum_by_A, yield_sum_by_Z, energy_dep}_w_ens_seed_p3.csv
  pipeline/output/v_ens_seed_p3/results/{yield_sum_by_A, yield_sum_by_Z, energy_dep}_v_ens_seed_p3.csv

输出：pipeline/output/vw_ens/results/*

调度：在 fpy_kan 环境运行  python .trash/make_vw_ens.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根
OUT = os.path.join(ROOT, 'pipeline', 'output')
DST = os.path.join(OUT, 'vw_ens', 'results')
VARIANT = 'vw_ens'
os.makedirs(DST, exist_ok=True)

# 每个 ensemble 由 6 个成员组成（Yield_pred_std 是其 6 个成员的【样本】标准差 ddof=1）
N_PER = 6

SRC = {
    'A': ('w_ens_seed_p3', 'v_ens_seed_p3', ['A_physical', 'E_physical']),
    'Z': ('w_ens_seed_p3', 'v_ens_seed_p3', ['Z_physical', 'E_physical']),
}
ED_KEYS = ['Z_physical', 'A_physical', 'E_physical']


def src_csv(kind, tag):
    if kind == 'energy_dep':
        return os.path.join(OUT, tag, 'results', f'energy_dep_{tag}.csv')
    return os.path.join(OUT, tag, 'results', f'yield_sum_by_{kind}_{tag}.csv')


def combine(kind, dir_w, dir_v, key_cols, out_name):
    """逐点平均 w 与 v 的曲线，并合成 12 模型【样本】标准差（ddof=1 池化）。"""
    w = pd.read_csv(src_csv(kind, dir_w))
    v = pd.read_csv(src_csv(kind, dir_v))

    m = w.merge(v, on=key_cols, suffixes=('_w', '_v'), how='inner')
    if len(m) != len(w) or len(m) != len(v):
        print(f'  [警告] {kind} 网格不完全对齐: w={len(w)} v={len(v)} 交集={len(m)}')
    mw = m['Yield_pred_w'].values.astype(float)
    mv = m['Yield_pred_v'].values.astype(float)
    sw = m['Yield_pred_std_w'].values.astype(float)
    sv = m['Yield_pred_std_v'].values.astype(float)

    ybar = 0.5 * (mw + mv)

    # 12 模型池化【样本】方差（与 04 ensemble 的 .std(ddof=1) 口径一致）：
    # sp² = [(n1-1)s1² + (n2-1)s2² + n1(m1-ȳ)² + n2(m2-ȳ)²] / (n1+n2-1)
    n1 = n2 = N_PER
    total = n1 + n2
    yg = (n1 * mw + n2 * mv) / total
    var = ((n1 - 1) * sw**2 + (n2 - 1) * sv**2
           + n1 * (mw - yg)**2 + n2 * (mv - yg)**2) / (total - 1)
    scomb = np.sqrt(var)

    out = m[key_cols].copy()
    out['Yield_pred'] = ybar
    out['Yield_pred_std'] = scomb
    out = out.sort_values(key_cols).reset_index(drop=True)

    dst = os.path.join(DST, out_name)
    out.to_csv(dst, index=False)
    print(f'  写出 {dst}  ({len(out)} 行)')
    return out


def plot_energy_dep(df_sum_by, kind, key_col):
    """复刻 04_energy_dep.py 步骤 12 的画图（线性 y 轴、英文标注、viridis、±1σ）。"""
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    cmap = plt.cm.viridis

    E_grid = np.sort(df_sum_by['E_physical'].unique())
    n_E = len(E_grid)
    colors = [cmap(i) for i in np.linspace(0, 0.85, n_E)]

    fig, ax = plt.subplots(figsize=(12, 7))
    for idx, E_phy in enumerate(E_grid):
        sub = df_sum_by[df_sum_by['E_physical'] == E_phy]
        ax.plot(sub[key_col], sub['Yield_pred'], color=colors[idx], alpha=0.7,
                linewidth=1.5, label=f'{E_phy:.0f} MeV' if idx % 3 == 0 else None)
        if 'Yield_pred_std' in sub.columns:   # 集成模式：成员间 ±1σ 带
            ax.fill_between(sub[key_col],
                            np.clip(sub['Yield_pred'] - sub['Yield_pred_std'], 0.0, None),
                            sub['Yield_pred'] + sub['Yield_pred_std'],
                            color=colors[idx], alpha=0.15, linewidth=0)
        if E_phy == E_grid[0] or E_phy == E_grid[-1]:
            lab = (f'{E_phy:.0f} MeV (pts)'
                   if (E_phy == E_grid[0] or E_phy == E_grid[-1]) and idx % 3 != 0 else None)
            ax.scatter(sub[key_col], sub['Yield_pred'], color=colors[idx],
                       s=18, alpha=0.8, label=lab)
    ax.set_xlabel('Mass Number (A)' if kind == 'A' else 'Atomic Number (Z)')
    ax.set_ylabel('Fission Yield Sum')
    ax.set_title(f'Fission Yield vs {"Mass Number (A)" if kind == "A" else "Atomic Number (Z)"}'
                 f' — {VARIANT}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_ylim(0, (df_sum_by['Yield_pred'] + df_sum_by.get('Yield_pred_std', 0.0)).max() * 1.1)
    plt.tight_layout()
    p = os.path.join(DST, f'yield_vs_{kind}_{VARIANT}.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  画出 {p}')


if __name__ == '__main__':
    print('=' * 70)
    print(f'生成 {VARIANT} = avg(w_ens 6模型, v_ens 6模型) = 12 模型平均（与 04 同款产物）')
    print('=' * 70)
    dfs = {}
    for kind, (dw, dv, keys) in SRC.items():
        print(f'[{kind}] 聚合曲线')
        dfs[kind] = combine(kind, dw, dv, keys, f'yield_sum_by_{kind}_{VARIANT}.csv')

    # 逐核素预测（energy_dep）也平均，保证与 04 产物一致
    print('[energy_dep] 逐核素预测')
    try:
        combine('energy_dep', 'w_ens_seed_p3', 'v_ens_seed_p3', ED_KEYS,
                f'energy_dep_{VARIANT}.csv')
    except FileNotFoundError as e:
        print(f'  [跳过] 未找到 energy_dep CSV（{e}）；仅输出聚合曲线。')

    print('-' * 70)
    print('画图（复刻 04_energy_dep）')
    plot_energy_dep(dfs['A'], 'A', 'A_physical')
    plot_energy_dep(dfs['Z'], 'Z', 'Z_physical')
    print('完成 →', DST)
