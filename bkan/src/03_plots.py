# -*- coding: utf-8 -*-
"""
bkan/src/03_plots.py — ★ MVP：产出 yield_vs_A / yield_vs_Z

**移植自 `pipeline/src/04_energy_dep.py`，只改预测那一行**：
    mu, var = model(Xb)          # 原来: pred = model(Xb)
其余可比性约定**逐条照抄**，以便与 pykan 的图并排对比：
  - 参考核素 = `235UALL.csv` **前 1032 行**（`04:218-220`）
  - Z/A/E 用 `data/standard_scaler*.pkl` 归一化
  - 特征拼接顺序按 config 的 `features`
  - 能量网格 [0,14] 步长 1
  - 按 (A,E) / (Z,E) 求和聚合
  - 画图：每 E 一条线、每 3 条标 legend

升级点：±1σ 带来自**模型自己的 σ**（而非 6-seed 成员间的 std）。

用法：
    python -u bkan/src/03_plots.py --config bkan/configs/gef_log.yaml
"""

import os
import sys
import argparse
import pickle

import numpy as np
import torch
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vendor  # noqa: F401
from svgp_kan import GPKAN                 # noqa: E402
from data import (load_config, get_variant, output_path,
                  compute_delta_np, PROJECT_ROOT)     # noqa: E402


def _inverse_target(y_norm, scaler, target_space):
    """标准化空间 -> 原始产额空间（log 空间需再 exp）。"""
    y = scaler.inverse_transform(np.asarray(y_norm, dtype=np.float32).reshape(-1, 1)).reshape(-1)
    if target_space == 'log':
        y = np.exp(y)
    return np.clip(y, 0.0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    variant = get_variant(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ed = cfg.get('energy_dep', {})

    print("=" * 70)
    print(f"BKAN 出图  variant={variant}")
    print("=" * 70)

    # ---- 1. pkl（特征/scaler/目标空间）----
    pkl = output_path(variant, 'data', f'preprocessed_{variant}.pkl')
    with open(pkl, 'rb') as f:
        d = pickle.load(f)
    info = d['data_info']
    feats = d['feature_names']
    scalers = d['scalers']
    target_space = info['target_space']
    use_delta_np = cfg['data'].get('use_delta_np', False)
    delta_mode = cfg['data'].get('delta_np_mode', 'discrete')
    target_scaler = scalers['Yield_log' if target_space == 'log' else 'Yield_original']

    # ---- 2. 模型 ----
    mp = output_path(variant, 'models', f'bkan_{variant}.pth')
    ckpt = torch.load(mp, map_location=device, weights_only=False)
    arch = ckpt['arch']
    model = GPKAN(layers_hidden=arch['layers_hidden'],
                  num_inducing=arch['num_inducing'],
                  kernel_type=arch['kernel']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"[1] 载入模型: {mp}   arch={arch['layers_hidden']} M={arch['num_inducing']}")

    # ---- 3. 参考核素表（前 1032 行，与 04g/04i 对齐）----
    ref_rel = ed.get('reference_csv', 'data/235UALL.csv')
    ref_path = ref_rel if os.path.isabs(ref_rel) else os.path.join(PROJECT_ROOT, ref_rel)
    nuc = pd.read_csv(ref_path).iloc[:1032].reset_index(drop=True)
    print(f"[2] 参考核素: {ref_path}  前 1032 行 -> {len(nuc)}")

    Z_norm = nuc['Z'].values.astype(np.float32)
    A_norm = nuc['A'].values.astype(np.float32)
    Z_phys = np.round(scalers['standard_Z'].inverse_transform(
        Z_norm.reshape(-1, 1))).astype(int).flatten()
    A_phys = np.round(scalers['standard_A'].inverse_transform(
        A_norm.reshape(-1, 1))).astype(int).flatten()

    # ---- 4. delta_np（与能量无关）----
    if use_delta_np:
        N_phys = A_phys - Z_phys
        df_dn = pd.DataFrame({'Z_original': Z_phys, 'A_original': A_phys,
                              'N': N_phys, 'I': (N_phys - Z_phys) / A_phys.astype(float)})
        delta_norm = scalers['delta_np'].transform(
            compute_delta_np(df_dn, delta_mode).reshape(-1, 1)).astype(np.float32).flatten()
    else:
        delta_norm = None

    # ---- 5. 能量网格 ----
    e_lo, e_hi = ed.get('E_range', [0, 14])
    e_step = float(ed.get('E_step', 1))
    E_grid = np.arange(float(e_lo), float(e_hi) + e_step / 2, e_step, dtype=np.float32)
    e_scaler = scalers.get('standard_E') or joblib.load(
        os.path.join(PROJECT_ROOT, 'data', 'standard_scalerE.pkl'))
    E_norm = e_scaler.transform(E_grid.reshape(-1, 1)).astype(np.float32).flatten()
    n_E = len(E_grid)
    print(f"[3] 能量网格: {e_lo}~{e_hi} MeV, step {e_step}, {n_E} 点")

    # ---- 6. 拼预测输入 ----
    n_nuc = len(nuc)
    ni = np.repeat(np.arange(n_nuc), n_E)
    ei = np.tile(np.arange(n_E), n_nuc)
    X = np.zeros((n_nuc * n_E, len(feats)), dtype=np.float32)
    for c, fn in enumerate(feats):
        if fn == 'Z_norm':
            X[:, c] = Z_norm[ni]
        elif fn == 'A_norm':
            X[:, c] = A_norm[ni]
        elif fn == 'E_norm':
            X[:, c] = E_norm[ei]
        elif fn == 'delta_np':
            X[:, c] = delta_norm[ni]
        else:
            raise ValueError(f"未知特征名: {fn}")
    print(f"[4] 预测输入: {X.shape[0]} 行 ({n_nuc} 核素 × {n_E} 能量)")

    # ---- 7. 预测 -> (mu, var)  ★ 与 04 的唯一差别 ----
    bs = int(ed.get('batch_size', 256))
    Xt = torch.tensor(X, device=device)
    mus, vars_ = [], []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], bs):
            mu, var = model(Xt[i:i + bs])          # 04 这里是 pred = model(...)
            mus.append(mu.cpu().numpy()); vars_.append(var.cpu().numpy())
    mu_n = np.concatenate(mus).reshape(-1)
    var_n = np.concatenate(vars_).reshape(-1)

    # ---- 8. 反变换（σ 用 delta 法：σy = y·σ_z）----
    y_pred = _inverse_target(mu_n, target_scaler, target_space)
    sigma_z = np.sqrt(np.clip(var_n, 0, None))
    if target_space == 'log':
        y_std = y_pred * (sigma_z / target_scaler.scale_[0])   # log 空间 σ 的尺度换算
    else:
        y_std = sigma_z / target_scaler.scale_[0]
    print(f"[5] 预测完成。原始产额范围 [{y_pred.min():.2e}, {y_pred.max():.2e}]，"
          f"σ 中位 {np.median(y_std):.2e}")

    # ---- 9. 聚合（按 A / Z 求和；σ 按方差相加）----
    A_full = A_phys[ni]; Z_full = Z_phys[ni]; E_full = E_grid[ei]
    dfout = pd.DataFrame({'Z': Z_full, 'A': A_full, 'E': E_full,
                          'y': y_pred, 'var': y_std ** 2})

    def agg(by):
        g = dfout.groupby([by, 'E'], as_index=False).agg(y=('y', 'sum'), var=('var', 'sum'))
        g['std'] = np.sqrt(g['var'])
        return g
    sumA, sumZ = agg('A'), agg('Z')

    # ---- 10. 画图（约定照抄 04_energy_dep.py）----
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    colors = [plt.cm.viridis(i) for i in np.linspace(0, 0.85, n_E)]

    def draw(g, xcol, xlabel, title, fname):
        fig, ax = plt.subplots(figsize=(12, 7))
        for idx, Ep in enumerate(E_grid):
            sub = g[g['E'] == Ep]
            ax.plot(sub[xcol], sub['y'], color=colors[idx], alpha=0.7, linewidth=1.5,
                    label=f'{Ep:.0f} MeV' if idx % 3 == 0 else None)
            ax.fill_between(sub[xcol],
                            np.clip(sub['y'] - sub['std'], 0.0, None),
                            sub['y'] + sub['std'],
                            color=colors[idx], alpha=0.15, linewidth=0)
            if Ep in (E_grid[0], E_grid[-1]):
                ax.scatter(sub[xcol], sub['y'], color=colors[idx], s=18, alpha=0.8)
        ax.set_xlabel(xlabel); ax.set_ylabel('Fission Yield Sum')
        ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.set_ylim(0, (g['y'] + g['std']).max() * 1.1)
        plt.tight_layout()
        p = output_path(variant, 'plots', fname)
        fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f"[6] 已保存: {p}")

    draw(sumA, 'A', 'Mass Number (A)',
         f'BKAN: Fission Yield vs Mass Number (A) — {variant}', f'yield_vs_A_{variant}.png')
    draw(sumZ, 'Z', 'Atomic Number (Z)',
         f'BKAN: Fission Yield vs Atomic Number (Z) — {variant}', f'yield_vs_Z_{variant}.png')

    sumA.to_csv(output_path(variant, 'plots', f'yield_sum_by_A_{variant}.csv'), index=False)
    sumZ.to_csv(output_path(variant, 'plots', f'yield_sum_by_Z_{variant}.csv'), index=False)
    print("=" * 70)


if __name__ == '__main__':
    main()
