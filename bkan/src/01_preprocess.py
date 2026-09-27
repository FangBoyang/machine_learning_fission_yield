# -*- coding: utf-8 -*-
"""
bkan/src/01_preprocess.py — BKAN 数据预处理

与 pipeline/src/01_preprocess.py 的关系：
  - 特征/目标/scaler/划分语义**逐字保留**（由 bkan/src/data.py 提供）
  - 输出到 bkan/output/<variant>/data/preprocessed_<variant>.pkl
  - **去掉 power 分支**（目标空间只有 log / raw）
  - **去掉噪声增广**（那条是 pykan 线的路线；BKAN 换用 σ_expt 加权）
  - **新增 σ**：由第 4/5 列算 log 空间 1σ，存进 pkl 供训练用

用法：
    python -u bkan/src/01_preprocess.py --config bkan/configs/<variant>.yaml
"""

import os
import sys
import argparse
import pickle

import numpy as np
import pandas as pd
import joblib
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (load_config, get_variant, output_path, compute_delta_np,
                  make_features_and_target, load_scalers_from_pretrained,
                  apply_split, compute_sigma_ln, PROJECT_ROOT)


def _safe_load_scaler(filename):
    path = os.path.join(PROJECT_ROOT, 'data', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 scaler 文件: {path}")
    return joblib.load(path)


def _read_df(cfg):
    """按 data.source / csv_path / header 读 CSV，统一列名。"""
    source = cfg['data'].get('source', 'gef')
    csv_rel = cfg['data']['csv_path']
    csv_path = csv_rel if os.path.isabs(csv_rel) else os.path.join(PROJECT_ROOT, csv_rel)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")

    header = 0 if (source != 'gef' and cfg['data'].get('header', False)) else None
    df = pd.read_csv(csv_path, header=header)
    if df.shape[1] >= 5:
        df = df.iloc[:, :5]
    df.columns = ['Z_norm', 'A_norm', 'E_norm', 'Yield', 'Error'][:df.shape[1]]
    print(f"[1] 读取 {source} 数据: {csv_path}")
    print(f"    样本数: {df.shape[0]}, 列: {list(df.columns)}")
    return df


def main():
    ap = argparse.ArgumentParser(description="BKAN 数据预处理")
    ap.add_argument('--config', required=True, help="YAML 配置路径")
    args = ap.parse_args()

    cfg = load_config(args.config)
    variant = get_variant(cfg)
    use_delta_np = cfg['data'].get('use_delta_np', False)
    delta_np_mode = cfg['data'].get('delta_np_mode', 'discrete')

    print("=" * 64)
    print(f"BKAN 数据预处理  variant={variant}")
    print("=" * 64)

    df = _read_df(cfg)

    # ---- 2. scalers：finetune 复用预训练，否则从 data/ 加载 ----
    reuse = (cfg.get('finetune') or {}).get('reuse_scalers_from')
    scalers = {}
    if reuse:
        print(f"\n[2] 复用预训练 scalers (reuse_scalers_from={reuse})")
        scalers, _ = load_scalers_from_pretrained(reuse)
        print(f"    已载入: {list(scalers.keys())}")
    else:
        print("\n[2] 从 data/ 加载 scalers")
        standard_Z = _safe_load_scaler('standard_scalerZ.pkl')
        standard_A = _safe_load_scaler('standard_scalerA.pkl')
        scalers['standard_Z'] = standard_Z
        scalers['standard_A'] = standard_A
        if use_delta_np:
            scalers['delta_np'] = _safe_load_scaler('delta_np_scaler.pkl')
        target_space = cfg['target']['space']
        if target_space == 'log':
            scalers['Yield_log'] = _safe_load_scaler('log_yield_scaler.pkl')
        elif target_space == 'raw':
            scalers['Yield_original'] = _safe_load_scaler('yield_scaler.pkl')
        else:
            raise ValueError(f"BKAN 只支持 target.space ∈ {{log, raw}}，实得 {target_space}")
        print(f"    已载入: {list(scalers.keys())}")
    standard_Z = scalers['standard_Z']
    standard_A = scalers['standard_A']

    # ---- 3. delta_np（先反归一化出物理 Z/A/N）----
    if use_delta_np:
        print("\n[3] 计算 delta_np")
        Z_orig = standard_Z.inverse_transform(
            df['Z_norm'].values.reshape(-1, 1)).round().astype(int).flatten()
        A_orig = standard_A.inverse_transform(
            df['A_norm'].values.reshape(-1, 1)).round().astype(int).flatten()
        df['Z_original'] = Z_orig
        df['A_original'] = A_orig
        df['N'] = A_orig - Z_orig
        df['I'] = (df['N'] - Z_orig) / (A_orig.astype(float) + 1e-12)
        raw_delta = compute_delta_np(df, delta_np_mode)
        df['delta_np'] = raw_delta
        print(f"    delta_np 范围: [{raw_delta.min():.6f}, {raw_delta.max():.6f}]")

    # ---- 4. 特征 / 目标 ----
    print("\n[4] 构建特征 X 与目标 y")
    X, y, feature_names, target_key, target_space = make_features_and_target(df, cfg, scalers)
    print(f"    特征: {feature_names}  dim={X.shape[1]}")
    print(f"    目标空间: {target_space}  scaler key: {target_key}")
    print(f"    归一化目标范围: [{y.min():.6f}, {y.max():.6f}]")

    # ---- 5. σ（BKAN 新增）：log 空间 1σ，供加权似然用 ----
    print("\n[5] 计算 σ（由第 4/5 列）")
    sigma_cfg = cfg.get('sigma', {}) or {}
    zero_mode = sigma_cfg.get('zero_yield', 'one')
    scale = float(getattr(scalers[target_key], 'scale_', [1.0])[0])
    if target_space == 'log':
        sigma_ln, mask_ok = compute_sigma_ln(
            df['Yield'].values, df['Error'].values, zero_mode=zero_mode, scale=scale)
        print(f"    σ_ln(标准化) 中位={np.median(sigma_ln[mask_ok]):.4f} "
              f"  90分位={np.percentile(sigma_ln[mask_ok], 90):.4f}")
        print(f"    zero_yield 处理={zero_mode}  可用于训练的行={mask_ok.sum()}/{len(mask_ok)}")
    else:
        # raw 空间：σ 直接用绝对误差，除以 scaler.scale_
        sigma_ln = (np.asarray(df['Error'].values, dtype=float) / scale).astype(np.float32)
        mask_ok = np.ones(len(df), dtype=bool)
        print(f"    (raw 空间) σ 用第 5 列绝对误差 / scale={scale:.4f}")

    # ---- 6. 划分 ----
    print("\n[6] 划分")
    error = df['Error'].values.astype(np.float32).reshape(-1, 1) if 'Error' in df else None
    X_tr, y_tr, X_va, y_va, split_meta = apply_split(X, y, cfg, error=error)
    sig_tr = sigma_ln[split_meta['train_indices']]
    msk_tr = mask_ok[split_meta['train_indices']]
    sig_va = sigma_ln[split_meta['val_indices']] if split_meta['val_indices'] is not None else None
    msk_va = mask_ok[split_meta['val_indices']] if split_meta['val_indices'] is not None else None
    print(f"    mode={split_meta['mode']}  train={X_tr.shape[0]}  "
          f"val={0 if X_va is None else X_va.shape[0]}")

    # ---- 7. E_original ----
    E_original = df['E_norm'].values.astype(np.float32)
    e_path = os.path.join(PROJECT_ROOT, 'data', 'standard_scalerE.pkl')
    if os.path.exists(e_path):
        e_scaler = joblib.load(e_path)
        E_original = e_scaler.inverse_transform(
            df['E_norm'].values.reshape(-1, 1)).flatten().astype(np.float32)
        scalers['standard_E'] = e_scaler

    # ---- 8. 存 pkl ----
    raw_data = {
        'Z_original': df['Z_original'].values.astype(np.float32) if 'Z_original' in df else None,
        'A_original': df['A_original'].values.astype(np.float32) if 'A_original' in df else None,
        'N': df['N'].values.astype(np.float32) if 'N' in df else None,
        'E_original': E_original,
        'I': df['I'].values.astype(np.float32) if 'I' in df else None,
        'delta_np': df['delta_np'].values.astype(np.float32) if 'delta_np' in df else None,
        'Yield_original': df['Yield'].values.astype(np.float32),
        'Error': df['Error'].values.astype(np.float32) if 'Error' in df else None,
    }
    data_dict = {
        'X_train': X_tr, 'y_train': y_tr, 'X_val': X_va, 'y_val': y_va,
        'sigma_train': sig_tr, 'sigma_val': sig_va,          # BKAN: 加权似然用
        'mask_train': msk_tr, 'mask_val': msk_va,            # BKAN: 有效行掩码
        'scalers': scalers,
        'feature_names': feature_names,
        'raw_data': raw_data,
        'data_info': {
            'config': cfg, 'config_yaml': yaml.dump(cfg, allow_unicode=True),
            'variant': variant, 'data_source': cfg['data'].get('source', 'gef'),
            'reuse_scalers_from': reuse,
            'target_space': target_space, 'target_key': target_key,
            'split_mode': split_meta['mode'], 'split': split_meta,
            'n_samples': int(X.shape[0]), 'n_train': int(X_tr.shape[0]),
            'n_val': 0 if X_va is None else int(X_va.shape[0]),
            'n_features': int(X.shape[1]),
            'sigma': {'zero_yield': zero_mode, 'scale': scale,
                      'target_space': target_space},
        },
    }
    out = output_path(variant, 'data', f'preprocessed_{variant}.pkl')
    with open(out, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"\n[7] 已保存: {out}")
    print("=" * 64)


if __name__ == '__main__':
    main()
