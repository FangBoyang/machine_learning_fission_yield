# -*- coding: utf-8 -*-
"""
01_preprocess.py — 数据预处理（配置驱动，通用数据源 + 可选留出划分）

支持两种数据源：
- data.source = 'gef' （默认）：读 data/GEF.csv（无表头，5 列），从 data/ 加载 scalers。
  全部作为训练集，不划分验证/测试集（硬约束）。
- data.source = 'csv' （通用）：读任意 CSV（data.csv_path，可带表头），前 3 列视为
  已归一化的 Z/A/E，第 4/5 列为 Yield/Error。用于 235UALL 等实验数据。

finetune.reuse_scalers_from：指定预训练变体名，直接复用其 preprocessed pkl 内的 scalers
与 target 空间元信息，保证特征/目标空间与预训练逐字节一致（续训成立的命门）。

data.split：
- mode='full_train'（默认）：不划分。
- mode='held_out'     ：留出验证集；val_from_first_n>0 时 val 仅从 [0,val_from_first_n)
  内随机划分（重要区全进 train），满足“val 只来自前 N 行、训练集涵盖后面数据”的约束。

data.augment（可选，默认关闭，关闭时行为与历史变体 v/w 逐字节一致）：
在第 3 步之后、第 5 步划分之前，按原始产额分位数分档复制行并加相对高斯噪声
（噪声加在原始产额空间）。详见 common.augment_yield_noise 的文档字符串。
启用后会多出两列：Yield（副本为带噪值，用于训练目标）与 Yield_original
（恒为源行干净值，用于 raw_data，保证 03 的指标仍是对原始 GEF 数据计算）。
与 split.mode=held_out 冲突（同一原始行的副本会同时进 train/val），会直接报错。

特征/目标构建统一走 common.make_features_and_target；划分走 common.apply_split。
用法：
    python src/01_preprocess.py --config configs/<variant>.yaml
"""

import os
import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import StandardScaler

from common import (load_config, get_variant, output_path,
                    build_features, compute_delta_np, make_features_and_target,
                    load_scalers_from_pretrained, apply_split,
                    augment_yield_noise, PROJECT_ROOT)


def _safe_load_scaler(filename):
    """加载 data/ 下的 scaler pkl（基于项目根目录解析）。"""
    path = os.path.join(PROJECT_ROOT, 'data', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 scaler 文件: {path}")
    return joblib.load(path)


def _read_df(cfg):
    """按 data.source / data.csv_path / data.header 读入 DataFrame，并加规范列名。"""
    source = cfg['data'].get('source', 'gef')
    if source == 'gef':
        csv_rel = cfg['data']['csv_path']
        csv_path = csv_rel if os.path.isabs(csv_rel) else os.path.join(PROJECT_ROOT, csv_rel)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"未找到数据文件: {csv_path}")
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] >= 5:
            df = df.iloc[:, :5]
        df.columns = ['Z_norm', 'A_norm', 'E_norm', 'Yield', 'Error'][:df.shape[1]]
        print(f"[1] 读取 GEF 数据: {csv_path}")
    else:
        csv_rel = cfg['data'].get('csv_path', 'data/235UALL.csv')
        csv_path = csv_rel if os.path.isabs(csv_rel) else os.path.join(PROJECT_ROOT, csv_rel)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"未找到数据文件: {csv_path}")
        header = 0 if cfg['data'].get('header', False) else None
        df = pd.read_csv(csv_path, header=header)
        if df.shape[1] >= 5:
            df = df.iloc[:, :5]
        df.columns = ['Z_norm', 'A_norm', 'E_norm', 'Yield', 'Error'][:df.shape[1]]
        print(f"[1] 读取通用 CSV 数据 (source={source}): {csv_path}")
    print(f"    样本数: {df.shape[0]}, 列名: {list(df.columns)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="数据预处理（配置驱动，通用数据源）")
    parser.add_argument('--config', type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant = get_variant(cfg)
    use_delta_np = cfg['data'].get('use_delta_np', False)
    delta_np_mode = cfg['data'].get('delta_np_mode', 'discrete')

    print("=" * 60)
    print("数据预处理（配置驱动，通用数据源）")
    print("=" * 60)

    # 1. 读取数据
    df = _read_df(cfg)

    # 2. 确定 scalers 来源：finetune 复用预训练，否则从 data/ 加载
    reuse = (cfg.get('finetune') or {}).get('reuse_scalers_from')
    scalers = {}
    standard_Z = standard_A = None
    if reuse:
        print(f"\n[2] 复用预训练 scalers (reuse_scalers_from={reuse}) ...")
        scalers, pre_info = load_scalers_from_pretrained(reuse)
        standard_Z = scalers.get('standard_Z')
        standard_A = scalers.get('standard_A')
        if standard_Z is None or standard_A is None:
            raise RuntimeError("预训练 scalers 缺少 standard_Z/standard_A，无法反归一化求 Z_original/A_original")
        print(f"    已载入预训练 scalers: {list(scalers.keys())}")
    else:
        print("\n[2] 从 data/ 加载 scalers ...")
        standard_Z = _safe_load_scaler('standard_scalerZ.pkl')
        standard_A = _safe_load_scaler('standard_scalerA.pkl')
        scalers['standard_Z'] = standard_Z
        scalers['standard_A'] = standard_A
        if use_delta_np:
            scalers['delta_np'] = _safe_load_scaler('delta_np_scaler.pkl')
        target_space = cfg['target']['space']
        if target_space == 'log':
            scalers['Yield_log'] = _safe_load_scaler('log_yield_scaler.pkl')
        else:
            target_power = float(cfg['target'].get('power', 1.0))
            if target_power == 1.0:
                scalers['Yield_original'] = _safe_load_scaler('yield_scaler.pkl')
            else:
                # p != 1：需对 t=(y+eps)^p 重新拟合 scaler（与 make_features_and_target 一致）
                eps = 1e-12
                raw_y = df['Yield'].values.astype(float)
                target_in = np.power(np.clip(raw_y, 0.0, None) + eps, target_power).reshape(-1, 1)
                scalers['Yield_power'] = StandardScaler().fit(target_in)
        print(f"    已从 data/ 载入 scalers: {list(scalers.keys())}")

    # 3. 计算 delta_np（如启用）：先反归一化得 Z_original/A_original/N/I
    raw_delta = None
    if use_delta_np:
        print("\n[3] 计算 delta_np 特征 ...")
        Z_original = standard_Z.inverse_transform(df['Z_norm'].values.reshape(-1, 1)).round().astype(int).flatten()
        A_original = standard_A.inverse_transform(df['A_norm'].values.reshape(-1, 1)).round().astype(int).flatten()
        N = A_original - Z_original
        I = (N - Z_original) / A_original.astype(float)
        df['Z_original'] = Z_original
        df['A_original'] = A_original
        df['N'] = N
        df['I'] = I
        raw_delta = compute_delta_np(df, delta_np_mode)
        df['delta_np'] = raw_delta  # 供 build_features / make_features_and_target 读取
        print(f"    delta_np 范围: [{raw_delta.min():.6f}, {raw_delta.max():.6f}]")

    # 3.5 产额噪声增广（可选，data.augment.enabled 控制）
    #   位置有硬性要求：
    #   - 必须在第 2 步 scaler 拟合【之后】：保证标准化统计量与未增广时一致，
    #     finetune 复用该 scaler 时空间不变。
    #   - 必须在第 5 步 apply_split【之前】：划分基于增广后的行。
    #   增广后 df 多出两列：Yield（副本为带噪值，供训练目标用）与
    #   Yield_original（恒为源行干净值，供 raw_data / 03 评估用）。
    n_before_aug = int(df.shape[0])
    df, aug_source_index = augment_yield_noise(df, cfg)
    if aug_source_index is not None:
        _a = cfg['data']['augment']
        print(f"\n[3.5] 产额噪声增广: {n_before_aug} -> {df.shape[0]} 行 | "
              f"分档分位 {_a.get('quantile_edges')} | 副本数 {_a.get('n_copies')} | "
              f"σ {_a.get('rel_sigma')} | mode={_a.get('noise_mode', 'multiplicative')}")

    # 4. 构建特征 X 与目标 y（统一走 common，GEF 与 finetune 共用）
    print("\n[4] 构建特征 X 与目标 y ...")
    X, y, feature_names, target_key, target_power, target_space = make_features_and_target(df, cfg, scalers)
    print(f"    特征顺序: {feature_names}, 维度: {X.shape[1]}")
    print(f"    目标空间: {target_space}, 幂次 p: {target_power}, scaler key: {target_key}")
    print(f"    归一化目标范围: [{y.min():.6f}, {y.max():.6f}]")

    # 5. 划分（held_out 或 full_train）
    print("\n[5] 划分训练/验证集 ...")
    error = df['Error'].values.astype(np.float32).reshape(-1, 1) if 'Error' in df else None
    X_train, y_train, X_val, y_val, split_meta = apply_split(X, y, cfg, error=error)
    if split_meta['mode'] == 'held_out':
        print(f"    held_out: train={X_train.shape[0]}, val={X_val.shape[0]} "
              f"(val_from_first_n={split_meta['val_from_first_n']}, ratio={split_meta['val_ratio']}, seed={split_meta['seed']})")
    else:
        print(f"    full_train: 训练集 {X_train.shape[0]}（无验证集）")

    # 6. 物理能量 E_original
    E_original = df['E_norm'].values.astype(np.float32)
    if os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'standard_scalerE.pkl')):
        e_scaler = _safe_load_scaler('standard_scalerE.pkl')
        E_original = e_scaler.inverse_transform(df['E_norm'].values.reshape(-1, 1)).flatten().astype(np.float32)
        scalers['standard_E'] = e_scaler

    # 7. 保存 pkl
    print("\n[6] 保存预处理数据 ...")
    # 注意：增广后行数变为 N_aug，故所有逐行数组都必须从【增广后】的 df 取，
    # 不能沿用第 3 步基于原始 df 算出的 Z_original/A_original/N/I/raw_delta。
    # Yield_original 恒为源行干净产额（副本行也是），因此 03 的 R² 仍是对原始
    # GEF 数据计算，只是高产区按副本数被重复计数（K 加权）。
    raw_data = {
        'Z_original': df['Z_original'].values.astype(np.float32) if 'Z_original' in df else None,
        'A_original': df['A_original'].values.astype(np.float32) if 'A_original' in df else None,
        'N': df['N'].values.astype(np.float32) if 'N' in df else None,
        'E_original': E_original,
        'I': df['I'].values.astype(np.float32) if 'I' in df else None,
        'delta_np': df['delta_np'].values.astype(np.float32) if 'delta_np' in df else None,
        'Yield_original': df['Yield_original'].values.astype(np.float32),
        'Error': df['Error'].values.astype(np.float32) if 'Error' in df else None,
        'aug_source_index': aug_source_index,   # 增广行 -> 原始行映射（未增广时为 None）
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'scalers': scalers,
        'feature_names': feature_names,
        'raw_data': raw_data,
        'data_info': {
            'config': cfg,
            'config_yaml': yaml.dump(cfg, allow_unicode=True),
            'variant': variant,
            'data_source': cfg['data'].get('source', 'gef'),
            'reuse_scalers_from': reuse,
            'target_space': target_space,
            'target_key': target_key,
            'target_power': target_power,
            'split_mode': split_meta['mode'],
            'split': split_meta,
            'n_samples': int(X.shape[0]),
            'n_train': int(X_train.shape[0]),
            'n_val': int(X_val.shape[0]) if X_val is not None else 0,
            'n_features': int(X.shape[1]),
            'device': device,
            # 增广审计（只存标量；逐行映射 aug_source_index 在 raw_data 中）
            'augment': {
                'enabled': aug_source_index is not None,
                'n_before': n_before_aug,
                'n_after': int(df.shape[0]),
            },
        },
        'device': device,
    }
    out_path = output_path(variant, 'data', f'preprocessed_{variant}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"    已保存: {out_path}")

    # 8. 摘要（中文）
    print("\n" + "=" * 60)
    print("预处理完成！摘要信息")
    print("=" * 60)
    print(f"变体名    : {variant}")
    print(f"数据源    : {cfg['data'].get('source', 'gef')}"
          + (f" (复用 {reuse} 的 scalers)" if reuse else " (data/ scalers)"))
    print(f"样本数    : {X.shape[0]} (train={X_train.shape[0]}, val={0 if X_val is None else X_val.shape[0]})")
    if aug_source_index is not None:
        print(f"噪声增广  : {n_before_aug} -> {X.shape[0]} 行 | 副本数 "
              f"{cfg['data']['augment'].get('n_copies')} | σ "
              f"{cfg['data']['augment'].get('rel_sigma')} | "
              f"mode={cfg['data']['augment'].get('noise_mode', 'multiplicative')}")
    print(f"特征维度  : {X.shape[1]} ({feature_names})")
    print(f"目标空间  : {target_space} (p={target_power}, key={target_key})")
    print(f"输出文件  : {out_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
