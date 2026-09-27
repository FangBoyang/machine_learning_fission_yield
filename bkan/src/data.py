# -*- coding: utf-8 -*-
"""
bkan/src/data.py — BKAN pipeline 的数据/目标层

**来源**：移植自 `pipeline/src/common.py`（pykan 那条线）。
移植原则：**只改与 pykan/目标空间相关的部分，数据语义逐字保留**。
`bkan/src/check_equivalence.py` 会断言移植后的 X/y 与旧 pkl 完全一致。

相对 common.py 的改动：
  1. `output_path` 指向 `bkan/output/`（而非 `pipeline/output/`）
  2. `make_features_and_target` **去掉 power 分支**，目标空间只支持 `log` / `raw`
  3. 新增 `compute_sigma_ln`：由第 4/5 列算 log 空间的 1σ（BKAN 用）

保留不动的踩坑结论：
  - `compute_delta_np` 的离散表（2026-08-20 修正过 ee/oo bug，7 处）
  - `apply_split` 的 `val_from_first_n` 语义
  - `load_scalers_from_pretrained`（finetune 复用 scaler 的"命门"）
"""

import os
import pickle
import yaml
import numpy as np
import pandas as pd

# bkan/src/data.py -> bkan/src -> bkan -> <repo root>
BKAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BKAN_ROOT)


# ===================== 1. 配置 =====================
def _deep_merge(base, override):
    """深度合并：子 dict 递归更新，标量直接覆盖。inherit 是控制字段，不并入结果。"""
    result = dict(base)
    for k, v in override.items():
        if k == 'inherit':
            continue
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path):
    """加载 YAML；`inherit:` 相对当前配置文件目录解析，递归深度合并。"""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get('inherit'):
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), cfg['inherit'])
        cfg = _deep_merge(load_config(base_path), cfg)
    return cfg


def get_variant(cfg):
    """变体名（cfg['experiment']['name']）。"""
    return cfg['experiment']['name']


def output_path(variant, subdir, filename):
    """bkan/output/{variant}/{subdir}/{filename}，并确保目录存在。"""
    d = os.path.join(BKAN_ROOT, 'output', variant, subdir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


# ===================== 2. delta_np（逐字移植）=====================
def compute_delta_np(df, mode):
    """
    Möller–Nix 对关联修正 delta_np。df 需含 Z_original / A_original / N / I。

    ⚠ 关键：物理 Z/A/N 必须先取整再做奇偶判别。若直接用反归一化的浮点数，
    偶数 N 写成 52.000006 时 `N%2==0` 会因浮点误差判为 False，
    delta_np 会退化成单一分支、特征被破坏（见 2026-08-20 的 ee/oo bug 修复）。
    """
    Z = np.round(np.asarray(df['Z_original'], dtype=float)).astype(int)
    A = np.round(np.asarray(df['A_original'], dtype=float)).astype(int)
    N = A - Z
    if 'I' in df:
        I = np.asarray(df['I'], dtype=float)
    else:
        I = (N - Z) / (A + 1e-12)

    if mode == 'discrete':
        N_even = (N % 2 == 0)
        Z_even = (Z % 2 == 0)
        ee = N_even & Z_even
        oo = (~N_even) & (~Z_even)
        eo = N_even & (~Z_even)
        oe = (~N_even) & Z_even
        out = np.select(
            [ee,
             oo,
             eo & (N > Z),          # eo, N>Z -> 1
             oe & (N < Z),          # oe, N<Z -> 1
             eo & (N < Z),          # eo, N<Z -> 1-|I|
             oe & (N > Z)],         # oe, N>Z -> 1-|I|
            [2.0 - np.abs(I),
             np.abs(I),
             1.0,
             1.0,
             1.0 - np.abs(I),
             1.0 - np.abs(I)],
            default=1.0
        )
        return out.astype(float)
    elif mode == 'continuous_cos':
        return (np.cos(np.pi * N) * np.cos(np.pi * Z)).astype(float)
    raise ValueError(f"不支持的 delta_np_mode: {mode}（可选 discrete / continuous_cos）")


# ===================== 3. 特征与目标（去掉 power 分支）=====================
def build_features(df, cfg):
    """按 cfg['data']['features'] 拼特征矩阵，返回 (X, names)。"""
    cols, names = [], []
    for feat in cfg['data']['features']:
        if feat == 'delta_np':
            if cfg['data'].get('use_delta_np', False):
                mode = cfg['data'].get('delta_np_mode', 'discrete')
                cols.append(compute_delta_np(df, mode).reshape(-1, 1))
                names.append('delta_np')
            # use_delta_np=false 时跳过该特征
        else:
            cols.append(np.asarray(df[feat], dtype=float).reshape(-1, 1))
            names.append(feat)
    if not cols:
        raise ValueError("未能构建任何输入特征，请检查 cfg['data']['features']")
    return np.concatenate(cols, axis=1), names


def make_features_and_target(df, cfg, scalers):
    """
    构建输入特征 X 与归一化目标 y。返回 (X, y, feature_names, target_key, target_space)。

    与 common.py 的差别：**不支持 target.power**，目标空间只有 `log` / `raw`。
      - space='log' : target_in = ln(Yield + 1e-12)，scaler key = 'Yield_log'
      - space='raw' : target_in = Yield，            scaler key = 'Yield_original'
    """
    use_delta_np = cfg['data'].get('use_delta_np', False)
    X, names = build_features(df, cfg)
    if use_delta_np and 'delta_np' in names:
        idx = names.index('delta_np')
        X[:, idx] = scalers['delta_np'].transform(
            df['delta_np'].values.reshape(-1, 1)).flatten()
    X = np.asarray(X, dtype=np.float32)

    target_space = cfg['target']['space']
    if target_space == 'log':
        target_in = np.log(df['Yield'].values + 1e-12).reshape(-1, 1)
        target_key = 'Yield_log'
    elif target_space == 'raw':
        target_in = df['Yield'].values.reshape(-1, 1)
        target_key = 'Yield_original'
    else:
        raise ValueError(f"BKAN 只支持 target.space ∈ {{log, raw}}，实得 {target_space}")
    y = np.asarray(scalers[target_key].transform(target_in), dtype=np.float32)
    return X, y, names, target_key, target_space


# ===================== 4. noise（BKAN 新增）=====================
def compute_sigma_ln(yield_orig, error_abs, zero_mode='one', scale=1.0):
    """
    由第 4 列(Yield) 与第 5 列(Error) 算 **log 空间的 1σ**。

    两列都是绝对量（同量纲）。log 空间下：
        σ_ln = Δ(ln Y) = ΔY / Y = Error / Yield
    若目标再被标准化（除以 scaler.scale_），则 σ_std = σ_ln / scale。

    ⚠ Yield == 0 的行 σ_ln = 0/0 未定义（GEF 878 行、235UALL 472 行）。
      zero_mode:
        'one'  -> σ_ln = 1.0（=「相对不确定度 100%」，有意义的降权；推荐）
        'drop' -> 返回 NaN，由调用方剔除这些行

    返回 (sigma_ln, mask_ok)：mask_ok 为可用于训练的布尔掩码（zero_mode='one' 时全 True）。
    """
    y = np.asarray(yield_orig, dtype=float).reshape(-1)
    e = np.asarray(error_abs, dtype=float).reshape(-1)
    sigma = np.full_like(y, np.nan)
    ok = y > 0
    sigma[ok] = e[ok] / y[ok]
    # 非零产额但 Error 缺失(0) -> 视为无信息，赋 1.0
    sigma[ok & (sigma <= 0)] = 1.0

    if zero_mode == 'one':
        sigma[~ok] = 1.0
        mask = np.ones_like(y, dtype=bool)
    elif zero_mode == 'drop':
        mask = ok & (sigma > 0)
    else:
        raise ValueError(f"zero_mode 只支持 'one' / 'drop'，实得 {zero_mode}")

    return (sigma / float(scale)).astype(np.float32), mask


# ===================== 5. scaler 复用 / 划分（逐字移植）=====================
def load_scalers_from_pretrained(spec):
    """
    从**预训练变体**的 preprocessed pkl 取 scalers 与目标空间元信息，供 finetune 复用，
    保证特征/目标空间与预训练逐字节一致（续训能成立的命门）。

    spec: 变体名（在 bkan/output 下找）或 pkl 的绝对/相对路径。
    """
    if spec.endswith('.pkl') or os.path.isabs(spec):
        pkl_path = spec
    else:
        pkl_path = output_path(spec, 'data', f'preprocessed_{spec}.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"未找到预训练预处理文件: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    info = d.get('data_info', {})
    return d.get('scalers', {}), {
        'target_key': info.get('target_key'),
        'target_space': info.get('target_space'),
        'feature_names': d.get('feature_names'),
        'delta_np_mode': info.get('delta_np_mode'),
    }


def apply_split(X, y, cfg, error=None):
    """
    确定性数据划分，返回 (X_train, y_train, X_val, y_val, split_meta)。

    cfg['data']['split']:
      'full_train'（默认，GEF 硬约束）: 不划分，val=None
      {mode: held_out, val_from_first_n: N, val_ratio: r, seed: s}
        val_from_first_n>0: val 仅从 [0,N) 内抽（重要区全进 train）
    """
    split_raw = cfg['data'].get('split', {}) or {}
    split = dict(split_raw) if isinstance(split_raw, dict) else {'mode': str(split_raw)}
    mode = split.get('mode', 'full_train')
    N = X.shape[0]

    if mode != 'held_out':
        return X, y, None, None, {
            'mode': 'full_train', 'val_indices': None,
            'train_indices': np.arange(N), 'val_from_first_n': 0,
            'val_ratio': 0.0, 'seed': 0}

    val_from_first_n = int(split.get('val_from_first_n', 0))
    val_ratio = float(split.get('val_ratio', 0.2))
    seed = int(split.get('seed', 42))
    rng = np.random.RandomState(seed)

    if val_from_first_n > 0:
        first = np.arange(min(val_from_first_n, N))
        perm = rng.permutation(len(first))
        n_val = int(round(len(first) * val_ratio))
        idx_val = first[perm[:n_val]]
        rest = first[perm[n_val:]]
        idx_train = (np.concatenate([rest, np.arange(val_from_first_n, N)])
                     if val_from_first_n < N else rest)
    else:
        idx = rng.permutation(N)
        n_val = int(round(N * val_ratio))
        idx_val, idx_train = idx[:n_val], idx[n_val:]

    meta = {'mode': 'held_out', 'val_indices': idx_val, 'train_indices': idx_train,
            'val_from_first_n': val_from_first_n, 'val_ratio': val_ratio, 'seed': seed}
    if error is not None:
        meta['error_train'] = error[idx_train]
        meta['error_val'] = error[idx_val]
    return X[idx_train], y[idx_train], X[idx_val], y[idx_val], meta
