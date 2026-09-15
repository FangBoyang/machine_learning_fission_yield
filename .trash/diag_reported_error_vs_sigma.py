# -*- coding: utf-8 -*-
"""三个数据源的第 5 列（Error）到底是什么，以及它和成员间 σ、实际误差的对比。
（.trash 临时脚本）
"""
import os
import pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_path = os.path.join(ROOT, 'pipeline', 'src')
OUT = os.path.join(ROOT, 'pipeline', 'output')

print("=" * 92)
print("一、三个 CSV 第 5 列的性质")
print("=" * 92)
for f, hdr in [('data/GEF_isomer_merged.csv', True), ('data/GEF.csv', False), ('data/235UALL.csv', True)]:
    d = pd.read_csv(os.path.join(ROOT, f), header=0 if hdr else None)
    e, y = d.iloc[:, 4].values.astype(float), d.iloc[:, 3].values.astype(float)
    eq = np.mean(np.isclose(e, y, rtol=1e-6, atol=1e-12)) * 100
    print(f"\n{f}  (rows={len(d)})")
    print(f"  Error 绝对值: 中位 {np.median(e):.3e}  均值 {e.mean():.3e}  max {e.max():.3e}")
    print(f"  Error == Yield 的比例: {eq:.1f}%     Error > Yield 的比例: {np.mean(e>y)*100:.1f}%")
    m = y > 0
    if m.sum():
        rel = e[m] / y[m]
        print(f"  Error/Yield 相对: 中位 {np.median(rel):.3e}  均值 {rel.mean():.3e}  max {rel.max():.3e}")
    vc = pd.Series(e).value_counts().head(3)
    print(f"  最常见取值: {[(float(k), int(v)) for k, v in vc.items()]}")

print("\n" + "=" * 92)
print("二、235UALL val 集（n=619）：报告的 Error vs 成员间 σ vs 实际误差")
print("=" * 92)

# val 索引
with open(os.path.join(OUT, 'w1_ft_235UALL_power', 'data',
                       'preprocessed_w1_ft_235UALL_power.pkl'), 'rb') as fh:
    pk = pickle.load(fh)
idx = np.asarray(pk['data_info']['split']['val_indices'], dtype=int)
err_rep = pk['raw_data']['Error'][idx].astype(float)
y_true_full = pk['raw_data']['Yield_original'][idx].astype(float)

rows = []
for tag, ens in [('w', 'w_ens_seed_p3'), ('v', 'v_ens_seed_p3')]:
    df = pd.read_csv(os.path.join(OUT, ens, 'results', f'ensemble_perpoint_{ens}.csv'))
    mem = [c for c in df.columns if '_ft_' in c]
    P = df[mem].values.astype(float)
    y = df['y_true'].values.astype(float)
    assert np.allclose(y, y_true_full), f'{tag}: y_true 与 pkl 不一致'
    mu = P.mean(axis=1)
    sig = P.std(axis=1, ddof=1)
    aerr = np.abs(mu - y)
    rows.append((tag, err_rep, sig, aerr, y, P))

for tag, err_rep, sig, aerr, y, P in rows:
    print(f"\n  --- {tag} 集成（6 成员）---")
    print(f"  {'量':<26}{'中位':>14}{'均值':>14}{'75分位':>14}{'95分位':>14}")
    for lab, v in [('报告的 Error（数据列）', err_rep),
                   ('成员间 σ（认知）', sig),
                   ('实际 |集成均值 − 真值|', aerr)]:
        print(f"  {lab:<26}{np.median(v):>14.3e}{v.mean():>14.3e}"
              f"{np.percentile(v,75):>14.3e}{np.percentile(v,95):>14.3e}")

    m = y > 0
    print(f"\n  相对量（除以真值，只取真值>0 的 {m.sum()} 点）:")
    print(f"    报告 Error/Yield   中位 {np.median(err_rep[m]/y[m]):.3e}")
    print(f"    成员 σ/Yield       中位 {np.median(sig[m]/y[m]):.3e}")
    print(f"    实际误差/Yield     中位 {np.median(aerr[m]/y[m]):.3e}")

    r = sig / np.maximum(err_rep, 1e-30)
    ok = err_rep > 0
    print(f"\n  σ / 报告Error 的比值（只取 Error>0 的 {ok.sum()} 点）:")
    print(f"    中位 {np.median(r[ok]):.3e}   均值 {r[ok].mean():.3e}   "
          f"σ 更大的点占比 {np.mean(sig[ok] > err_rep[ok])*100:.1f}%")

    # 高产额区
    hi = y >= np.percentile(y, 90)
    print(f"\n  最高 10% 产额区（n={hi.sum()}）:")
    print(f"    报告 Error 中位 {np.median(err_rep[hi]):.3e}   成员 σ 中位 {np.median(sig[hi]):.3e}   "
          f"实际误差中位 {np.median(aerr[hi]):.3e}")
    hm = hi & (y > 0)
    print(f"    相对: 报告 {np.median(err_rep[hm]/y[hm]):.3e}  σ {np.median(sig[hm]/y[hm]):.3e}  "
          f"实际 {np.median(aerr[hm]/y[hm]):.3e}")
