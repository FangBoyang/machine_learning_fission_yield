# -*- coding: utf-8 -*-
"""
bkan/src/check_equivalence.py — 数据层等价性校验

目的：把"移植导致静默分歧"变成"开机就报错"。

做法：在**同一份配置、同一份数据**下，分别用
  - 新代码  bkan/src/data.py
  - 旧代码  pipeline/src/common.py（pykan 那条线的真源）
算出 (X, y, 划分)，断言完全一致。

用法：
    python -u bkan/src/check_equivalence.py --config bkan/configs/gef_log.yaml
"""

import os
import sys
import argparse

import numpy as np
import joblib

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from data import (load_config, compute_delta_np, make_features_and_target,
                  apply_split, PROJECT_ROOT)          # noqa: E402

# 旧真源
_OLD_SRC = os.path.join(PROJECT_ROOT, 'pipeline', 'src')
sys.path.insert(0, _OLD_SRC)
import common as pl_common                                # noqa: E402  (需要 kan 已安装)


def _load_scalers(cfg, df):
    """复刻 01_preprocess 的 scaler 加载（非 finetune 路径）。"""
    from sklearn.preprocessing import StandardScaler
    s = {}
    s['standard_Z'] = joblib.load(os.path.join(PROJECT_ROOT, 'data', 'standard_scalerZ.pkl'))
    s['standard_A'] = joblib.load(os.path.join(PROJECT_ROOT, 'data', 'standard_scalerA.pkl'))
    if cfg['data'].get('use_delta_np', False):
        s['delta_np'] = joblib.load(os.path.join(PROJECT_ROOT, 'data', 'delta_np_scaler.pkl'))
    space = cfg['target']['space']
    if space == 'log':
        s['Yield_log'] = joblib.load(os.path.join(PROJECT_ROOT, 'data', 'log_yield_scaler.pkl'))
    elif space == 'raw':
        s['Yield_original'] = joblib.load(os.path.join(PROJECT_ROOT, 'data', 'yield_scaler.pkl'))
    return s


def _read_df(cfg):
    import pandas as pd
    src = cfg['data'].get('source', 'gef')
    path = os.path.join(PROJECT_ROOT, cfg['data']['csv_path'])
    header = 0 if (src != 'gef' and cfg['data'].get('header', False)) else None
    df = pd.read_csv(path, header=header)
    if df.shape[1] >= 5:
        df = df.iloc[:, :5]
    df.columns = ['Z_norm', 'A_norm', 'E_norm', 'Yield', 'Error'][:df.shape[1]]
    return df


def _prep_df(cfg, df, scalers):
    """补出 delta_np 需要的物理列（两边共用同一份 df 的副本）。"""
    df = df.copy()
    if cfg['data'].get('use_delta_np', False):
        Z = scalers['standard_Z'].inverse_transform(
            df['Z_norm'].values.reshape(-1, 1)).round().astype(int).flatten()
        A = scalers['standard_A'].inverse_transform(
            df['A_norm'].values.reshape(-1, 1)).round().astype(int).flatten()
        df['Z_original'], df['A_original'] = Z, A
        df['N'] = A - Z
        df['I'] = (df['N'] - Z) / (A.astype(float) + 1e-12)
        df['delta_np'] = compute_delta_np(df, cfg['data'].get('delta_np_mode', 'discrete'))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("=" * 68)
    print(f"等价性校验  variant={cfg['experiment']['name']}")
    print(f"  target.space = {cfg['target']['space']}   features = {cfg['data']['features']}")
    print("=" * 68)

    scalers = _load_scalers(cfg, None)
    df0 = _read_df(cfg)

    # 两份互相独立的 df 副本，避免新旧代码互相污染（e.g. 一方写了列）
    df_new = _prep_df(cfg, df0, scalers)
    df_old = _prep_df(cfg, df0, scalers)

    Xn, yn, names_n, key_n, space_n = make_features_and_target(df_new, cfg, scalers)
    Xo, yo, names_o, key_o, power_o, space_o = pl_common.make_features_and_target(
        df_old, cfg, scalers)

    err = cfg['data']['features'] and df_new['Error'].values.astype(np.float32).reshape(-1, 1)
    Xtr_n, ytr_n, Xva_n, yva_n, meta_n = apply_split(Xn, yn, cfg, error=err)
    Xtr_o, ytr_o, Xva_o, yva_o, meta_o = pl_common.apply_split(Xo, yo, cfg, error=err)

    checks = [
        ("feature_names", names_n == names_o),
        ("target_key", key_n == key_o),
        ("target_space", space_n == space_o),
        ("X_train 全等", np.allclose(Xtr_n, Xtr_o, rtol=0, atol=0)),
        ("y_train 全等", np.allclose(ytr_n, ytr_o, rtol=0, atol=0)),
        ("split mode 一致", meta_n['mode'] == meta_o['mode']),
        ("train_indices 全等", np.array_equal(meta_n['train_indices'], meta_o['train_indices'])),
    ]

    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= bool(passed)

    if not checks[3][1]:
        d = np.abs(Xtr_n - Xtr_o)
        print(f"         X 最大绝对差 = {d.max():.3e}  (形状 {Xtr_n.shape} vs {Xtr_o.shape})")
    if not checks[4][1]:
        print(f"         y 最大绝对差 = {np.abs(ytr_n - ytr_o).max():.3e}")

    print()
    if ok:
        print("✅ 数据层等价：新代码与旧真源逐位一致。")
    else:
        print("❌ 数据层不等价 —— 移植引入了分歧，先修这个再往下走。")
        sys.exit(1)


if __name__ == '__main__':
    main()
