# -*- coding: utf-8 -*-
"""
common.py — KAN 裂变产额配置驱动 pipeline 的基石模块

提供：配置加载与继承合并、变体名解析、输出路径拼接、特征构建、
delta_np 计算、损失函数/优化器/调度器工厂等通用能力。

所有路径基于项目根目录（pipeline/ 的上一级）拼接，兼容 Windows，
与后续 01_preprocess / 02_train / 03_evaluate / 04_energy_dep 共用。
"""

import os
import math
import copy
import pickle
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kan import KAN

# 项目根目录：common.py 位于 <root>/pipeline/src/common.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===================== 1. load_config =====================
def _deep_merge(base, override):
    """深度合并：子 dict 递归更新，标量直接覆盖。"""
    result = dict(base)
    for k, v in override.items():
        if k == 'inherit':
            # inherit 是控制字段，不并入最终配置
            continue
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path):
    """加载 YAML 配置；若含 inherit 字段，先递归加载其指向的父配置再深度合并。"""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    if 'inherit' in cfg and cfg['inherit']:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), cfg['inherit'])
        base = load_config(base_path)  # 递归解析多级继承链
        cfg = _deep_merge(base, cfg)
    return cfg


# ===================== 配置驱动 KAN 重建 =====================
def build_kan_from_ckpt(ckpt, device):
    """从 checkpoint 内嵌 config + state_dict 重建 KAN。

    关键：训练中可能经 grid schedule / refine 改变网格，而内嵌 config 的
    model.grid 仍是初始值（02_train 旧版未同步）。这里从 state_dict 的
    act_fun.0 张量形状反推“真实网格”，避免按旧 grid 重建导致尺寸不匹配。
    返回 (model, cfg, grid)，并就地把 cfg['model']['grid'] 修正为真实值。
    """
    cfg = ckpt['config']
    mcfg = cfg['model']
    features = cfg['data']['features']
    width = [len(features)] + list(mcfg['hidden_layers']) + [1]
    k = int(mcfg['k'])
    seed = int(mcfg['seed'])
    sd = ckpt['model_state']
    tg = tuple(sd['act_fun.0.grid'].shape)
    tc = tuple(sd['act_fun.0.coef'].shape)
    grid = int(mcfg.get('grid', 5))
    # 扫描真实网格：与 checkpoint 张量形状完全匹配的那个
    for G in range(3, 61):
        probe = KAN(width=width, grid=G, k=k, seed=seed, save_act=False, auto_save=False)
        if (tuple(probe.act_fun[0].grid.shape) == tg
                and tuple(probe.act_fun[0].coef.shape) == tc):
            grid = G
            break
    try:
        cfg['model']['grid'] = grid
    except Exception:
        pass
    model = KAN(width=width, grid=grid, k=k, seed=seed, save_act=False, auto_save=False)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, cfg, grid
    return cfg


# ===================== 2. get_variant =====================
def get_variant(cfg):
    """返回当前实验变体名（cfg['experiment']['name']）。"""
    return cfg['experiment']['name']


# ===================== 3. output_path =====================
def output_path(variant, subdir, filename):
    """拼出 pipeline/output/{variant}/{subdir}/{filename}，并确保目录存在。"""
    d = os.path.join(PROJECT_ROOT, 'pipeline', 'output', variant, subdir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


# ===================== 4. compute_delta_np =====================
def compute_delta_np(df, mode):
    """
    计算 Möller-Nix 对关联修正 delta_np。
    df 需包含列：Z_original, A_original, N, I。
    mode='discrete'   : 沿用原始阶跃逻辑（ee/oo/oe/eo 分类）
    mode='continuous_cos' : cos(pi*N) * cos(pi*Z) 连续代理
    """
    # 关键：物理 Z/A/N 必须取整后再做奇偶判别。原始 01i 先做
    # .round().astype(int) 再算 N%2；若直接用反归一化的浮点数，
    # 偶数 N 写成浮点(如 52.000006) 时 `N%2==0` 会因浮点误差判为 False，
    # 导致 delta_np 退化成单一分支，特征被破坏。这里对齐原始逻辑。
    Z = np.round(np.asarray(df['Z_original'], dtype=float)).astype(int)
    A = np.round(np.asarray(df['A_original'], dtype=float)).astype(int)
    N = A - Z  # 整数 N，保证奇偶判别精确
    if 'I' in df:
        I = np.asarray(df['I'], dtype=float)
    else:
        I = (N - Z) / (A + 1e-12)

    if mode == 'discrete':
        N_even = (N % 2 == 0)
        Z_even = (Z % 2 == 0)
        ee = N_even & Z_even                           # even-even
        oo = (~N_even) & (~Z_even)                     # odd-odd
        eo = N_even & (~Z_even)                        # even-N, odd-Z
        oe = (~N_even) & Z_even                        # odd-N, even-Z
        # 优先级：先分 ee/oo，再按 N>Z / N<Z 细分 oe/eo，
        # 与论文 Yuan Su 2022, Commun. Theor. Phys. 74 095301 的 δ_np 表一致。
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

    else:
        raise ValueError(f"不支持的 delta_np_mode: {mode}（可选 discrete / continuous_cos）")


# ===================== 5. build_features =====================
def build_features(df, cfg):
    """
    根据 cfg['data']['features'] 构建输入特征矩阵。
    若 features 含 'delta_np' 且 cfg['data']['use_delta_np'] 为 true，
    则调用 compute_delta_np 计算（df 需含 Z_original/A_original/N/I）。
    返回 (X_array, feature_names_list)。
    """
    features = cfg['data']['features']
    cols = []
    names = []
    for feat in features:
        if feat == 'delta_np':
            if cfg['data'].get('use_delta_np', False):
                mode = cfg['data'].get('delta_np_mode', 'discrete')
                delta = compute_delta_np(df, mode).reshape(-1, 1)
                cols.append(delta)
                names.append('delta_np')
            # use_delta_np 为 false 时跳过该特征
        else:
            arr = np.asarray(df[feat], dtype=float).reshape(-1, 1)
            cols.append(arr)
            names.append(feat)
    if len(cols) == 0:
        raise ValueError("未能构建任何输入特征，请检查 cfg['data']['features']")
    X = np.concatenate(cols, axis=1)
    return X, names


# ===================== 4b. make_features_and_target =====================
def make_features_and_target(df, cfg, scalers):
    """
    通用：用给定 scalers 从 df 构建输入特征 X 与归一化目标 y。

    供 GEF 预处理与 finetune 预处理共用，避免两边各写一份特征/目标逻辑。
    df 需含列：Z_norm, A_norm, E_norm, Yield, (Error)；若 use_delta_np 为 true，
    build_features 内部会调用 compute_delta_np，故 df 还需含 Z_original/A_original/N/I。

    返回 (X, y, feature_names, target_key, target_power, target_space)。
    - X: float32 数组 [N, n_features]，delta_np 列已用 scalers['delta_np'] 归一化。
    - y: float32 数组 [N, 1]，按 target_space/power 变换并用 scalers[target_key] 归一化。
    - target_key: 'Yield_original' / 'Yield_log' / 'Yield_power'（供 03 反变换选 scaler）。
    """
    use_delta_np = cfg['data'].get('use_delta_np', False)
    X, names = build_features(df, cfg)  # delta_np 返回原始值
    if use_delta_np and 'delta_np' in names:
        idx = names.index('delta_np')
        delta_np_norm = scalers['delta_np'].transform(
            df['delta_np'].values.reshape(-1, 1)).flatten()
        X[:, idx] = delta_np_norm
    X = np.asarray(X, dtype=np.float32)

    target_space = cfg['target']['space']
    target_power = float(cfg['target'].get('power', 1.0))
    if target_space == 'log':
        eps = 1e-12
        target_in = np.log(df['Yield'].values + eps).reshape(-1, 1)
        target_key = 'Yield_log'
    else:
        if target_power == 1.0:
            target_in = df['Yield'].values.reshape(-1, 1)
            target_key = 'Yield_original'
        else:
            # p != 1：目标变换为 t=(y+eps)^p，必须用对 t 拟合的 scaler（power 变体已含 'Yield_power'）
            eps = 1e-12
            raw_y = df['Yield'].values.astype(float)
            target_in = np.power(np.clip(raw_y, 0.0, None) + eps, target_power).reshape(-1, 1)
            target_key = 'Yield_power'
    y = np.asarray(scalers[target_key].transform(target_in), dtype=np.float32)
    return X, y, names, target_key, target_power, target_space


# ===================== 4c. load_scalers_from_pretrained =====================
def load_scalers_from_pretrained(spec):
    """
    从预训练变体的 preprocessed pkl 取出 scalers 与目标空间元信息，供 finetune 复用，
    保证特征/目标空间与预训练逐字节一致（续训能成立的命门）。

    spec: 变体名（如 'i_rawY_delta_np_smaller'）或 pkl 绝对/相对路径。
    返回 (scalers_dict, info_dict)；info_dict 含 target_key/target_power/target_space/
          feature_names/delta_np_mode。
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
        'target_power': float(info.get('target_power', 1.0)),
        'target_space': info.get('target_space'),
        'feature_names': d.get('feature_names'),
        'delta_np_mode': info.get('delta_np_mode'),
    }


# ===================== 4d. apply_split =====================
def apply_split(X, y, cfg, error=None):
    """
    通用、确定（带 seed）的数据划分。返回 (X_train, y_train, X_val, y_val, split_meta)。

    cfg['data']['split']:
      mode='full_train'（默认）: 不划分，X_val/y_val=None（GEF 硬约束：100% 训练）。
      mode='held_out'         : 留出验证集。
        val_from_first_n>0    : val 仅从 [0, val_from_first_n) 内随机划分（重要区全进 train）。
        val_from_first_n=0/缺省: 从全量随机划分。
        val_ratio / seed      : 比例与随机种子（可配置）。

    split_meta 记录 mode / val_indices / train_indices / 参数，供评估与审计。
    """
    split_raw = cfg['data'].get('split', {}) or {}
    # 兼容两种写法：基配置用字符串 'full_train'（如 base.yaml），
    # finetune 用字典 {mode: held_out, val_from_first_n: ...}（如 m_ft_*）。
    split = dict(split_raw) if isinstance(split_raw, dict) else {'mode': str(split_raw)}
    mode = split.get('mode', 'full_train')
    N = X.shape[0]
    if mode != 'held_out':
        meta = {'mode': 'full_train', 'val_indices': None,
                'train_indices': np.arange(N), 'val_from_first_n': 0,
                'val_ratio': 0.0, 'seed': 0}
        return X, y, None, None, meta

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
        idx_val = idx[:n_val]
        idx_train = idx[n_val:]

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    meta = {'mode': 'held_out', 'val_indices': idx_val, 'train_indices': idx_train,
            'val_from_first_n': val_from_first_n, 'val_ratio': val_ratio, 'seed': seed}
    if error is not None:
        meta['error_train'] = error[idx_train]
        meta['error_val'] = error[idx_val]
    return X_train, y_train, X_val, y_val, meta


# ===================== 4d. augment_yield_noise =====================
def augment_yield_noise(df, cfg):
    """按原始产额分档复制 + 加相对高斯噪声（学长的「第二条路」）。

    配置块 data.augment：
      enabled          : false 时原样返回（向后兼容，v/w 等既有变体零影响）
      seed             : 噪声随机种子
      quantile_edges   : 分档分位边界，如 [0.5, 0.9] → 低/中/高 三档
      n_copies         : 每档「副本」份数（不含原件），长度须为 len(edges)+1
      rel_sigma        : 每档相对噪声 σ/y，长度同 n_copies；0 表示该档副本为纯复制
      noise_mode       : multiplicative（ỹ = y·(1+σz)，clip 到 ≥0）
                         | lognormal（ỹ = y·exp(-σ²/2 + σz)，严格为正且均值保持）

    噪声始终加在【原始产额空间】。注意目标变换 t = Y^p 会压缩相对扰动：
    原始空间 σ_rel 对应变换后约 p·σ_rel，故低产额点在标准化单位下几乎不受影响，
    这正是「产额大的部分才有效」的由来。

    关键设计：
    - 原件（干净）始终保留在 dataset 中，副本是额外追加的。
    - 新增列 df['Yield_original'] 恒为「源行的干净产额」：副本行也带干净值，
      因此 03 的 R² 仍是在原始 GEF 数据上计算（只是高产区按副本数被重复计数）。
    - df['Yield'] 在副本行被覆盖为带噪值，训练目标由此列生成。

    返回 (df_aug, aug_source_index)；未启用时 aug_source_index 为 None。
    aug_source_index[i] = 增广后第 i 行来自原始 df 的哪一行（用于去重指标/审计）。
    """
    a = (cfg['data'].get('augment') or {})
    df = df.copy()
    df['Yield_original'] = np.asarray(df['Yield'].values, dtype=float)
    if not a.get('enabled', False):
        return df, None

    # 防泄漏：增广在 apply_split 之前执行，若再划分 held_out，同一原始行的副本
    # 会同时落入 train/val（标签不同但输入相同），直接判定为泄漏并拒绝。
    split_raw = cfg['data'].get('split', {}) or {}
    split_mode = split_raw.get('mode') if isinstance(split_raw, dict) else str(split_raw)
    if split_mode == 'held_out':
        raise ValueError("data.augment.enabled=true 与 data.split.mode=held_out 冲突："
                         "同一原始行的副本会同时落入 train 与 val，造成数据泄漏。")

    seed = int(a.get('seed', 42))
    rng = np.random.RandomState(seed)
    y = df['Yield_original'].values.astype(float)
    N = len(y)

    q_edges = [float(q) for q in a.get('quantile_edges', [])]
    n_tiers = len(q_edges) + 1
    copies = [int(c) for c in a.get('n_copies', [])]
    sigmas = [float(s) for s in a.get('rel_sigma', [])]
    if len(copies) != n_tiers or len(sigmas) != n_tiers:
        raise ValueError(
            f"data.augment: n_copies/rel_sigma 长度须为 {n_tiers}"
            f"（= len(quantile_edges)={len(q_edges)} + 1），实得 {len(copies)}/{len(sigmas)}")
    if any(c < 0 for c in copies):
        raise ValueError(f"data.augment: n_copies 不可为负，实得 {copies}")
    if any(s < 0 for s in sigmas):
        raise ValueError(f"data.augment: rel_sigma 不可为负，实得 {sigmas}")
    noise_mode = a.get('noise_mode', 'multiplicative')
    if noise_mode not in ('multiplicative', 'lognormal'):
        raise ValueError(f"data.augment: 不支持的 noise_mode={noise_mode}"
                         f"（可选 multiplicative / lognormal）")

    # 分档：digitize 返回 0..n_tiers-1（y < e0 → 0；e0 ≤ y < e1 → 1；…）
    if q_edges:
        tier = np.digitize(y, np.quantile(y, q_edges))
    else:
        tier = np.zeros(N, dtype=int)

    frames = [df]                      # 原件（干净）始终保留
    src_idx = [np.arange(N)]
    for t in range(n_tiers):
        K, s = copies[t], sigmas[t]
        if K <= 0:
            continue
        idx = np.where(tier == t)[0]
        if len(idx) == 0:
            continue
        for _ in range(K):
            sub = df.iloc[idx].copy()
            if s > 0:
                if noise_mode == 'lognormal':
                    sub['Yield'] = y[idx] * np.exp(-0.5 * s * s + s * rng.randn(len(idx)))
                else:
                    sub['Yield'] = np.clip(y[idx] * (1.0 + s * rng.randn(len(idx))), 0.0, None)
            frames.append(sub)
            src_idx.append(idx)

    out = pd.concat(frames, ignore_index=True)
    aug_source_index = np.concatenate(src_idx).astype(np.int32)
    return out, aug_source_index


# ===================== 4e. load_pretrained_model =====================
def load_pretrained_model(spec, device):
    """
    从预训练变体的 best/final checkpoint 重建 KAN 模型（含正确 grid 与权重）。
    spec: 变体名或 .pth 路径。返回 (model, cfg, grid, path)。
    区别于 02_train 的 resume（同数据中断续训）：此处是不同数据源的微调起点。
    """
    if spec.endswith('.pth'):
        path = spec
    else:
        path = output_path(spec, 'models', f'kan_best_{spec}.pth')
        if not os.path.exists(path):
            path = output_path(spec, 'models', f'kan_final_{spec}.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到预训练模型: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model, cfg, grid = build_kan_from_ckpt(ckpt, device)
    return model, cfg, grid, path


# ===================== 6. get_criterion =====================
class WeightedMSELoss(nn.Module):
    """低产额用相对误差平方，高产额用绝对 MSE（乘 high_yield_boost）。"""

    def __init__(self, low_yield_quantile=0.30, high_yield_boost=1.0):
        super().__init__()
        self.low_yield_quantile = low_yield_quantile
        self.high_yield_boost = high_yield_boost

    def forward(self, pred, target):
        q = torch.quantile(target, self.low_yield_quantile)
        low_mask = (target < q).float()
        rel = (pred - target) / (target.abs() + 1e-12)
        loss_low = (rel ** 2) * low_mask
        loss_high = ((pred - target) ** 2) * (1.0 - low_mask)
        return (loss_low + self.high_yield_boost * loss_high).mean()


class RelativeMSELoss(nn.Module):
    """相对误差平方损失：(pred-target)/(target+1e-12) 的平方。"""

    def forward(self, pred, target):
        rel = (pred - target) / (target + 1e-12)
        return (rel ** 2).mean()


def get_criterion(cfg):
    """根据 cfg['train']['criterion']['type'] 返回损失函数。

    幂次 p 通过目标变换实现（见 01_preprocess：目标变为 t=(y+eps)^p，训练对 t 做
    普通 MSE），不在损失函数内做幂变换，以保证训练/评估反变换一致且梯度有界。
    """
    c = cfg['train']['criterion']
    t = c['type']
    # YAML 可能把 1e-6 等解析为字符串，统一转 float 以防下游报错
    low_yield_quantile = float(c.get('low_yield_quantile', 0.30))
    high_yield_boost = float(c.get('high_yield_boost', 1.0))
    huber_delta = float(c.get('huber_delta', 0.1))
    if t == 'MSE':
        return nn.MSELoss()
    elif t == 'WeightedMSE':
        return WeightedMSELoss(
            low_yield_quantile=low_yield_quantile,
            high_yield_boost=high_yield_boost
        )
    elif t == 'RelativeMSE':
        return RelativeMSELoss()
    elif t == 'Huber':
        return nn.HuberLoss(delta=huber_delta)
    else:
        raise ValueError(f"不支持的 criterion 类型: {t}")


# ===================== CosineHoldAtMin scheduler =====================
class CosineHoldAtMin(torch.optim.lr_scheduler.LRScheduler):
    """前半段余弦退火到 eta_min，到达 T_max 后保持 eta_min 恒定（不再回弹）。

    原生 CosineAnnealingLR 在 epoch 超过 T_max 后，cos 是周期函数会回弹、lr 重新变大。
    这里把参与余弦计算的 epoch 钳制在 T_max：last_epoch >= T_max 时 lr 恒为 eta_min，
    实现「降到谷底就固定」。基于 last_epoch，天然兼容 resume（调度器 state_dict 记录
    last_epoch，load_state_dict 即可恢复 lr 位置）。按 epoch step() 即可。
    """

    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        e = min(self.last_epoch, self.T_max)          # 钳制，避免 cos 回弹
        cos = math.cos(math.pi * e / self.T_max)
        return [self.eta_min + (group['initial_lr'] - self.eta_min) * (1.0 + cos) / 2.0
                for group in self.optimizer.param_groups]


# ===================== 7. get_scheduler =====================
def get_scheduler(optimizer, cfg):
    """根据 cfg['train']['scheduler']['type'] 返回学习率调度器；none 返回 None。"""
    s = cfg['train']['scheduler']
    t = s.get('type', 'none')
    # YAML 可能把 1e-6 等解析为字符串，统一转 float/int 以防下游报错
    if t is None or t == 'none':
        return None
    elif t == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(s.get('T_0', 300)),
            T_mult=int(s.get('T_mult', 1)),
            eta_min=float(s.get('eta_min', 1e-6))
        )
    elif t == 'CosineAnnealingLR':
        # 单段余弦衰减：lr 从初始值单调降到 eta_min，之后（epoch > T_max）保持 eta_min。
        # 不重启，避免把已收敛的解在重启点炸回高 loss（j 在 epoch 400 的坑）。
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(s.get('T_max', 300)),
            eta_min=float(s.get('eta_min', 1e-6))
        )
    elif t == 'CosineHoldAtMin':
        # 前半段余弦降到 eta_min，到达 T_max 后保持 eta_min 恒定（见 CosineHoldAtMin 类）。
        # 与 CosineAnnealingLR 的区别：epoch 超过 T_max 不会回弹，而是固定到底。
        return CosineHoldAtMin(
            optimizer,
            T_max=int(s.get('T_max', 300)),
            eta_min=float(s.get('eta_min', 1e-6))
        )
    elif t == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(s.get('step_size', 500)),
            gamma=float(s.get('gamma', 0.5))
        )
    elif t == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(s.get('gamma', 0.5))
        )
    elif t == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=int(s.get('patience', 10)),
            factor=float(s.get('factor', 0.5))
        )
    else:
        raise ValueError(f"不支持的 scheduler 类型: {t}")


# ===================== 8. get_optimizer =====================
def get_optimizer(model, cfg):
    """根据 cfg['train']['optimizer']['type'] 返回优化器（支持 AdamW/Adam/SGD）。"""
    o = cfg['train']['optimizer']
    t = o['type']
    # YAML 可能把 1e-4 等解析为字符串，统一转 float 以防下游报错
    lr = float(o.get('lr', 0.01))
    wd = float(o.get('weight_decay', 1e-4))
    params = model.parameters()
    if t == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif t == 'Adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif t == 'SGD':
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"不支持的 optimizer 类型: {t}")


# ===================== 自测块 =====================
if __name__ == '__main__':
    configs_dir = os.path.join(PROJECT_ROOT, 'pipeline', 'configs')
    yaml_files = sorted(
        f for f in os.listdir(configs_dir)
        if f.endswith('.yaml') or f.endswith('.yml')
    ) if os.path.isdir(configs_dir) else []

    if yaml_files:
        test_path = os.path.join(configs_dir, yaml_files[0])
        try:
            cfg = load_config(test_path)
            print(f"[测试] load_config 成功加载: {test_path}")
            print(f"[测试] 变体名: {get_variant(cfg)}")
        except Exception as e:
            print(f"[测试] load_config 加载失败: {e}")
    else:
        print("common.py loaded OK")

    # 测试 output_path 能正确拼出路径
    p = output_path('test_variant', 'data', 'preprocessed_test.pkl')
    print(f"[测试] output_path 结果: {p}")
