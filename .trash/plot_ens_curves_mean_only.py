# -*- coding: utf-8 -*-
"""
plot_ens_curves_mean_only.py — 只画集成「均值」曲线的能量依赖图（无 σ 阴影）

用途：04_energy_dep.py 在集成模式下会给曲线加 ±1σ 成员间离散带。这里出一份
      「干净版」——同样的曲线，但只有均值，不带阴影，便于直接放进报告/汇报。

数据源：04 已经算好的按 A / 按 Z 聚合 CSV（其中的 Yield_pred 已是 6 成员均值）。
   pipeline/output/v_ens_seed_p3/results/yield_sum_by_A_v_ens_seed_p3.csv
   pipeline/output/v_ens_seed_p3/results/yield_sum_by_Z_v_ens_seed_p3.csv
绘图风格与 04_energy_dep.py 保持一致（viridis 配色、15 条能量曲线、每 3 条标注、
首尾能量加点、dpi=150），唯一区别是不再 fill_between。

产出（.trash/）：
   yield_vs_A_<variant>.png
   yield_vs_Z_<variant>.png

用法：
   python .trash/plot_ens_curves_mean_only.py [--variant w_ens_seed_p3]
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_ROOT, '.trash')
DEFAULT_VARIANT = 'w_ens_seed_p3'


def _plot(df, xcol, xlabel, title, out_path):
    """画「产额-核素」曲线族：每个能量点一条均值曲线，无不确定度带。"""
    E_grid = np.sort(df['E_physical'].unique())
    n_E = len(E_grid)

    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    colors = [plt.cm.viridis(i) for i in np.linspace(0, 0.85, n_E)]

    fig, ax = plt.subplots(figsize=(12, 7))
    for idx, E in enumerate(E_grid):
        sub = df[df['E_physical'] == E].sort_values(xcol)
        ax.plot(sub[xcol], sub['Yield_pred'], color=colors[idx], alpha=0.7, linewidth=1.5,
                label=f'{E:.0f} MeV' if idx % 3 == 0 else None)
        # 首尾能量额外画散点，便于看清采样位置
        if idx == 0 or idx == n_E - 1:
            ax.scatter(sub[xcol], sub['Yield_pred'], color=colors[idx], s=18, alpha=0.8,
                       label=f'{E:.0f} MeV (pts)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Fission Yield Sum')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_ylim(0, df['Yield_pred'].max() * 1.1)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="画集成均值曲线（不带 σ 阴影）")
    ap.add_argument('--variant', type=str, default=DEFAULT_VARIANT,
                    help="集成变体名，如 v_ens_seed_p3 / w_ens_seed_p3")
    args = ap.parse_args()
    variant = args.variant
    res = os.path.join(PROJECT_ROOT, 'pipeline', 'output', variant, 'results')

    dfA = pd.read_csv(os.path.join(res, f'yield_sum_by_A_{variant}.csv'))
    dfZ = pd.read_csv(os.path.join(res, f'yield_sum_by_Z_{variant}.csv'))

    p1 = _plot(dfA, 'A_physical', 'Mass Number (A)',
               f'Fission Yield vs Mass Number (A) — {variant}',
               os.path.join(OUT, f'yield_vs_A_{variant}.png'))
    p2 = _plot(dfZ, 'Z_physical', 'Atomic Number (Z)',
               f'Fission Yield vs Atomic Number (Z) — {variant}',
               os.path.join(OUT, f'yield_vs_Z_{variant}.png'))

    print('已保存（仅均值曲线，无 σ 阴影）:')
    print(' ', p1)
    print(' ', p2)
    print(f'\n数据源: {res}')
    print(f'A 曲线: {len(dfA)} 行, {dfA.E_physical.nunique()} 个能量点, A∈[{dfA.A_physical.min()},{dfA.A_physical.max()}]')
    print(f'Z 曲线: {len(dfZ)} 行, {dfZ.E_physical.nunique()} 个能量点, Z∈[{dfZ.Z_physical.min()},{dfZ.Z_physical.max()}]')
    print(f'产额范围: A图 [{dfA.Yield_pred.min():.3e}, {dfA.Yield_pred.max():.3e}]；'
          f'Z图 [{dfZ.Yield_pred.min():.3e}, {dfZ.Yield_pred.max():.3e}]')


if __name__ == '__main__':
    main()
