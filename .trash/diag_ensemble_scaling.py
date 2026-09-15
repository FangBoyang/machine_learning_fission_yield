# -*- coding: utf-8 -*-
"""用现有 6 成员集成估算「成员数 n 增加还能赚多少」。（.trash 临时脚本）

做法：对 k=1..6，穷举所有 C(6,k) 个子集，算子集均值预测的 MSE，取平均。
理论上有  MSE(k) = c + a/k
    c = 共享分量（成员共同的偏差 + 相关误差），k→∞ 时的地板
    a = 可被平均掉的分量
用 k=1 与 k=6 两点定出 a、c，再外推 k=10/20/50。

同时在两个空间各算一遍：
  A) 原始产额空间   —— 决定 R² / RMSE（被峰区主导）
  B) 相对误差空间   —— 决定「曲线看起来光不光滑」（用户观察到的崎岖）
"""
import os
import itertools
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'pipeline', 'output')

VARIANTS = [('w', 'w_ens_seed_p3'), ('v', 'v_ens_seed_p3')]

for tag, ens in VARIANTS:
    df = pd.read_csv(os.path.join(RES, ens, 'results', f'ensemble_perpoint_{ens}.csv'))
    mem_cols = [c for c in df.columns if c.startswith(f'{tag}_ft') or (tag == 'v' and c.startswith('v') and 'ft' in c)]
    mem_cols = [c for c in df.columns if '_ft_' in c]
    y = df['y_true'].values.astype(float)
    P = df[mem_cols].values.astype(float)          # [N, 6]
    n_mem = P.shape[1]
    print("=" * 84)
    print(f"{tag} 集成：成员数 {n_mem}，评估点 {P.shape[0]}")
    print(f"  成员: {mem_cols}")
    # 校验：报告的 y_ens_mean 是否等于成员均值
    chk = np.abs(df['y_ens_mean'].values - P.mean(axis=1)).max()
    print(f"  校验 y_ens_mean == 成员均值?  最大偏差 {chk:.3e}")

    # ---------- A) 原始空间 ----------
    E = P - y[:, None]                             # 逐成员误差 [N,6]
    # ---------- B) 相对误差空间 ----------
    eps = 1e-10
    L = np.log10((np.clip(P, 0, None) + eps) / (y[:, None] + eps))   # [N,6]
    mask = y > 0                                   # 真值为 0 的点相对误差无意义
    Lm = L[mask]

    for spc_name, Err, note in (('原始产额空间（决定 R²/RMSE）', E, 'all'),
                                ('相对误差空间 log10(pred/true)（决定曲线光滑度）', Lm, 'pos')):
        N = Err.shape[0]
        print(f"\n  --- {spc_name}   n_pts={N} ---")
        mse = {}
        for k in range(1, n_mem + 1):
            vals = []
            for comb in itertools.combinations(range(n_mem), k):
                m = Err[:, list(comb)].mean(axis=1)
                vals.append(np.mean(m ** 2))
            mse[k] = float(np.mean(vals))
        for k in sorted(mse):
            print(f"    k={k:<3} MSE={mse[k]:.6e}   RMSE={np.sqrt(mse[k]):.6e}")
        # 拟合 MSE(k) = c + a/k，用 k=1 与 k=kmax 定参
        kmax = n_mem
        a = (mse[1] - mse[kmax]) / (1.0 - 1.0 / kmax)
        c = mse[kmax] - a / kmax
        print(f"    拟合 MSE(k) = {c:.6e} + {a:.6e}/k")
        print(f"      共享地板 c 占单成员 MSE 的 {c/mse[1]*100:.1f}%（这部分再多的成员也平均不掉）")
        # 用其余 k 检验拟合质量
        pred = {k: c + a / k for k in mse}
        rel = max(abs(pred[k] - mse[k]) / mse[k] for k in mse)
        print(f"      对 k=1..{kmax} 的最大相对残差 {rel*100:.2f}%（检验 c+a/k 形式是否成立）")
        print(f"    外推（RMSE）：")
        for k in (1, 6, 10, 20, 50, 10 ** 6):
            v = c + a / k
            lbl = '∞' if k == 10 ** 6 else str(k)
            print(f"      k={lbl:<7} RMSE={np.sqrt(v):.6e}   相对单成员改善 {100*(1-np.sqrt(v)/np.sqrt(mse[1])):.2f}%"
                  f"   相对 k=6 改善 {100*(1-np.sqrt(v)/np.sqrt(mse[6])):.2f}%")
        # σ 的抽样误差
        print(f"    σ 估计本身的抽样误差 std(s)/s ≈ 1/sqrt(2(n-1))：")
        for n in (6, 10, 20, 50):
            print(f"      n={n:<3} {1/np.sqrt(2*(n-1))*100:5.1f}%")

    # 成员两两误差相关
    E = P - y[:, None]
    C = np.corrcoef(E, rowvar=False)
    iu = np.triu_indices(n_mem, k=1)
    print(f"\n  成员误差的【两两相关系数】（原始空间）：均值 {C[iu].mean():.4f}，"
          f"范围 [{C[iu].min():.4f}, {C[iu].max():.4f}]")
    Lm = L[y > 0]
    C2 = np.corrcoef(Lm, rowvar=False)
    print(f"  成员误差的【两两相关系数】（相对误差空间）：均值 {C2[iu].mean():.4f}，"
          f"范围 [{C2[iu].min():.4f}, {C2[iu].max():.4f}]")
    print()
