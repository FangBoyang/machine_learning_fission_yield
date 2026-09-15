# -*- coding: utf-8 -*-
"""
04_energy_dep.py — 裂变产额能量依赖性分析（配置驱动）

流程：加载模型 checkpoint（结构取自内嵌 config）→ 加载 235UALL 基准核素表
      → 反归一化得到物理 Z/A/N/I → 预计算 delta_np（与能量无关）
      → 固定核素、扫描 0~14 MeV 激发能 → 批量预测 → 反归一化 + 负值裁剪
      →（可选）质量守恒后处理（按 A 归一到 2）
      → 按 A / Z 聚合 → 画图 → 保存 CSV + 中文摘要

用法：
    python src/04_energy_dep.py --config configs/<variant>.yaml

集成模式：
- 若配置含 ensemble.members，则逐成员加载各自 ckpt 与各自的目标 scaler/幂次 p，
  各自反变换到原始产额空间后跨成员取平均（与 05_ensemble.py 的
  average_space=original 口径一致），并额外输出成员间标准差 σ。
- 关键：集成变体自身没有 preprocessed_<variant>.pkl，绝不能走「按 variant 找 pkl」
  的老路径——那样 target_power 会静默回退成 1.0（实际 p=0.35），反变换全错。
  故 scaler/幂次一律取自各成员自己的 pkl。
- 不含 ensemble.members 时，行为与改造前完全一致（单模型）。

关键约定（与 03_evaluate 一致）：
- 模型结构从 checkpoint 的 'config' 字段读取，不依赖外部 YAML。
- 输入特征顺序严格使用 checkpoint config 的 data.features（delta_np 变体含 4 维，否则 3 维）。
- 目标反归一化使用 data/ 下对应 scaler（raw→yield_scaler.pkl，log→log_yield_scaler.pkl）。
- 所有路径基于项目根目录（common.PROJECT_ROOT）解析，兼容从任意 cwd 运行。
"""

import os
import argparse
import time
import pickle

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use('Agg')  # 无显示环境也能保存图片
import matplotlib.pyplot as plt

from kan import KAN
from common import (load_config, get_variant, output_path, PROJECT_ROOT,
                    compute_delta_np, build_kan_from_ckpt)


def _load_model(variant, device):
    """优先 kan_best_{variant}.pth，其次 kan_final_{variant}.pth。返回 (model, ckpt)。"""
    best = output_path(variant, 'models', f'kan_best_{variant}.pth')
    final = output_path(variant, 'models', f'kan_final_{variant}.pth')
    path = best if os.path.exists(best) else final
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到模型文件: {best} 或 {final}（请先运行 02_train.py）")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model, cfg, grid = build_kan_from_ckpt(ckpt, device)  # 自动推断 refine 后的真实 grid
    return model, ckpt


def _load_target_scaler(target_space):
    """按目标空间加载原始空间反归一化 scaler。"""
    if target_space == 'log':
        path = os.path.join(PROJECT_ROOT, 'data', 'log_yield_scaler.pkl')
    else:
        path = os.path.join(PROJECT_ROOT, 'data', 'yield_scaler.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到目标 scaler: {path}")
    return joblib.load(path)


def _inverse_target(y_norm, scaler, target_space, clip_min, target_power=1.0):
    """归一化空间 → 原始空间。log 空间需先反变换再 exp；幂次目标需再 ^(1/p)。"""
    y = scaler.inverse_transform(np.asarray(y_norm, dtype=np.float32).reshape(-1, 1)).reshape(-1)
    if target_space == 'log':
        y = np.exp(y)
    if target_power != 1.0:
        y = np.power(np.clip(y, 0.0, None), 1.0 / target_power)
    return np.clip(y, clip_min, None)


def _load_member_bundle(variant, device):
    """加载单个变体的完整「预测单元」：模型 + ckpt 内嵌配置 + 目标 scaler/幂次 p。

    scaler 与幂次 p 一律取自该变体自己的 preprocessed_<variant>.pkl——
    幂次变体（p=0.35）的 scaler 是对 t=y^p 重新拟合的，用 data/yield_scaler.pkl
    反变换会错。集成变体自身没有 pkl，故只能从成员处取，不能从 variant 处取。
    """
    model, ckpt = _load_model(variant, device)
    model_cfg = ckpt['config']
    data_cfg = model_cfg['data']
    target_space = model_cfg['target']['space']
    clip_min = float(model_cfg['target']['clip_min'])

    target_power = 1.0
    target_key = 'Yield_log' if target_space == 'log' else 'Yield_original'
    pkl_scaler = None
    pkl_path = output_path(variant, 'data', f'preprocessed_{variant}.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            pdata = pickle.load(f)
        info = pdata.get('data_info', {}) or {}
        target_power = float(info.get('target_power', 1.0))
        target_key = info.get('target_key', target_key)
        pkl_scaler = (pdata.get('scalers', {}) or {}).get(target_key)
    scaler = pkl_scaler if pkl_scaler is not None else _load_target_scaler(target_space)

    _best = output_path(variant, 'models', f'kan_best_{variant}.pth')
    _final = output_path(variant, 'models', f'kan_final_{variant}.pth')

    return {
        'name': variant,
        'model': model,
        'model_cfg': model_cfg,
        'model_path': _best if os.path.exists(_best) else _final,
        'features': list(data_cfg['features']),
        'use_delta_np': bool(data_cfg.get('use_delta_np', False)),
        'delta_np_mode': data_cfg.get('delta_np_mode', 'discrete'),
        'target_space': target_space,
        'clip_min': clip_min,
        'target_power': target_power,
        'scaler': scaler,
        'n_params': int(sum(p.numel() for p in model.parameters())),
    }


def _check_bundles_consistent(bundles):
    """校验各成员的特征/目标空间/幂次等是否一致（不一致则无法在同一输入网格上集成）。"""
    ref = bundles[0]
    for key in ('features', 'use_delta_np', 'delta_np_mode', 'target_space',
                'clip_min', 'target_power'):
        vals = {b['name']: b[key] for b in bundles}
        uniq = {str(v) for v in vals.values()}
        if len(uniq) > 1:
            raise ValueError(f"集成成员在 `{key}` 上不一致，无法共同预测：{vals}")
    return True


def _combine_member_aggs(preds_orig, Z_physical, A_physical, E_full, nuc_idx_full, keys):
    """集成模式下按 keys 聚合：先逐成员聚合，再跨成员求均值与标准差。

    不能直接对逐点 σ 求和——求和操作会引入成员间的相关性，先各自聚合才是对的。
    """
    frames = []
    for j in range(preds_orig.shape[0]):
        d = pd.DataFrame({
            'Z_physical': Z_physical[nuc_idx_full],
            'A_physical': A_physical[nuc_idx_full],
            'E_physical': E_full,
            'Yield_pred': preds_orig[j],
        })
        frames.append(d.groupby(keys, as_index=False)['Yield_pred'].sum())
    return (pd.concat(frames, axis=0)
            .groupby(keys, as_index=False)
            .agg(Yield_pred=('Yield_pred', 'mean'),
                 Yield_pred_std=('Yield_pred', 'std')))


def main():
    parser = argparse.ArgumentParser(description="裂变产额能量依赖性分析（配置驱动）")
    parser.add_argument('--config', type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant = get_variant(cfg)
    ed_cfg = cfg.get('energy_dep', {})
    pp_cfg = cfg.get('postprocess', {})

    print("=" * 60)
    print("裂变产额能量依赖性分析（配置驱动）")
    print("=" * 60)

    # 2. 加载模型（结构取自 checkpoint 内嵌 config）
    #    集成模式：cfg 含 ensemble.members 时逐成员加载，各自 ckpt + 各自的目标
    #    scaler/幂次 p，预测后在「原始产额空间」平均（与 05_ensemble 一致）。
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ens_cfg = cfg.get('ensemble', {}) or {}
    members = list(ens_cfg.get('members', []) or [])
    is_ensemble = len(members) > 0
    member_names = list(members) if is_ensemble else [variant]
    if is_ensemble:
        drop = set(ens_cfg.get('drop_members', []) or [])
        member_names = [m for m in member_names if m not in drop]
        if not member_names:
            raise ValueError("ensemble.members 为空（或全部被 drop_members 排除）")

    bundles = []
    for mn in member_names:
        b = _load_member_bundle(mn, device)
        bundles.append(b)
        print(f"[1] 加载模型: {os.path.basename(b['model_path'])}")
        print(f"    架构: KAN{list(b['model_cfg']['model']['hidden_layers'])} "
              f"(grid={b['model_cfg']['model']['grid']}, k={b['model_cfg']['model']['k']})")
        print(f"    特征: {b['features']}, 使用delta_np={b['use_delta_np']}, "
              f"目标空间={b['target_space']}, p={b['target_power']}")

    ref = bundles[0]
    model_cfg, mcfg = ref['model_cfg'], ref['model_cfg']['model']
    features = ref['features']
    use_delta_np = ref['use_delta_np']
    delta_np_mode = ref['delta_np_mode']
    target_space = ref['target_space']
    clip_min = ref['clip_min']
    target_power = ref['target_power']
    model_path = ref['model_path']
    n_params = ref['n_params']
    if is_ensemble:
        _check_bundles_consistent(bundles)
        print(f"    ✅ 集成模式：{len(bundles)} 个成员，一致性校验通过，"
              f"原始产额空间平均")

    # 3. 加载 235UALL.csv 基准核素表
    ref_rel = ed_cfg.get('reference_csv', 'data/235UALL.csv')
    ref_path = ref_rel if os.path.isabs(ref_rel) else os.path.join(PROJECT_ROOT, ref_rel)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"未找到基准核素表: {ref_path}")
    df_base = pd.read_csv(ref_path)
    # 对齐外部 04g/04i：取前 1032 行（与 .iloc[:1032] 完全一致）。
    # 注意：235UALL.csv 有 4127 行、1045 个唯一(Z,A)，iloc[:1032] 与去重集合不同，
    # 必须用 iloc[:1032] 才能保证能量依赖曲线与外部脚本逐点一致。
    nuc = df_base.iloc[:1032].reset_index(drop=True)
    print(f"[2] 加载基准核素表: {ref_path}")
    print(f"    核素数: {len(nuc)}（与外部 04g/04i 的 iloc[:1032] 一致）")

    # 4. 反归一化获取物理 Z/A（用 Z/A scaler）
    z_scaler = joblib.load(os.path.join(PROJECT_ROOT, 'data', 'standard_scalerZ.pkl'))
    a_scaler = joblib.load(os.path.join(PROJECT_ROOT, 'data', 'standard_scalerA.pkl'))
    Z_norm = nuc['Z'].values.astype(np.float32)
    A_norm = nuc['A'].values.astype(np.float32)
    Z_physical = z_scaler.inverse_transform(Z_norm.reshape(-1, 1)).round().astype(int).flatten()
    A_physical = a_scaler.inverse_transform(A_norm.reshape(-1, 1)).round().astype(int).flatten()
    N_physical = A_physical - Z_physical
    I_physical = (N_physical - Z_physical) / A_physical.astype(np.float64)
    print(f"[3] 物理 Z 范围: [{Z_physical.min()}, {Z_physical.max()}], "
          f"A 范围: [{A_physical.min()}, {A_physical.max()}]")

    # 5. 预计算 delta_np（与能量无关）
    if use_delta_np:
        delta_scaler = joblib.load(os.path.join(PROJECT_ROOT, 'data', 'delta_np_scaler.pkl'))
        df_dn = pd.DataFrame({
            'Z_original': Z_physical,
            'A_original': A_physical,
            'N': N_physical,
            'I': I_physical,
        })
        delta_np_physical = compute_delta_np(df_dn, delta_np_mode)
        delta_np_norm = delta_scaler.transform(delta_np_physical.reshape(-1, 1)).astype(np.float32).flatten()
        print(f"[4] 预计算 delta_np（mode={delta_np_mode}）: "
              f"范围 [{delta_np_physical.min():.3f}, {delta_np_physical.max():.3f}]")
    else:
        delta_np_norm = None
        print(f"[4] 变体未启用 delta_np，跳过预计算")

    # 6. 构建能量网格并归一化（用 E scaler）
    e_range = ed_cfg.get('E_range', [0, 14])
    e_step = float(ed_cfg.get('E_step', 1))
    E_physical_grid = np.arange(float(e_range[0]), float(e_range[1]) + e_step / 2.0, e_step, dtype=np.float32)
    e_scaler = joblib.load(os.path.join(PROJECT_ROOT, 'data', 'standard_scalerE.pkl'))
    E_norm_grid = e_scaler.transform(E_physical_grid.reshape(-1, 1)).astype(np.float32).flatten()
    n_E = len(E_physical_grid)
    print(f"[5] 激发能网格: {e_range[0]}~{e_range[1]} MeV, 步长 {e_step}, 共 {n_E} 点")

    # 7. 构建预测输入：每个核素 × 每个能量点一行，按 features 顺序拼接
    n_nuc = len(nuc)
    nuc_idx_full = np.repeat(np.arange(n_nuc), n_E)
    e_idx_full = np.tile(np.arange(n_E), n_nuc)
    n_rows = n_nuc * n_E
    X = np.zeros((n_rows, len(features)), dtype=np.float32)
    for col, fname in enumerate(features):
        if fname == 'Z_norm':
            X[:, col] = Z_norm[nuc_idx_full]
        elif fname == 'A_norm':
            X[:, col] = A_norm[nuc_idx_full]
        elif fname == 'E_norm':
            X[:, col] = E_norm_grid[e_idx_full]
        elif fname == 'delta_np':
            X[:, col] = delta_np_norm[nuc_idx_full]
        else:
            raise ValueError(f"未知特征名: {fname}（仅支持 Z_norm/A_norm/E_norm/delta_np）")
    print(f"[6] 预测输入构建完成: {n_rows} 行 ({n_nuc}核素 × {n_E}能量点), {len(features)} 维")

    # 8. 批量预测（batch_size 取 energy_dep.batch_size）
    #    每个成员用自己的目标 scaler/幂次 p 反变换到原始空间后，再跨成员平均
    #    （与 05_ensemble.py 的 average_space=original 口径一致）。
    batch_size = int(ed_cfg.get('batch_size', 256))
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    preds_orig = []
    for b in bundles:
        b['model'].eval()
        pred_norm = []
        with torch.no_grad():
            for i in range(0, n_rows, batch_size):
                pred_norm.append(b['model'](Xt[i:i + batch_size]).detach().cpu().numpy())
        pred_norm = np.concatenate(pred_norm, axis=0).reshape(-1)
        preds_orig.append(_inverse_target(pred_norm, b['scaler'], b['target_space'],
                                          b['clip_min'], b['target_power']))
    preds_orig = np.stack(preds_orig, axis=0)      # [k, n_rows]，k = 成员数
    y_pred = preds_orig.mean(axis=0)
    y_std = preds_orig.std(axis=0, ddof=1) if preds_orig.shape[0] > 1 else None

    # 9. 反归一化到原始空间 + 负值裁剪（含幂次反变换）
    print(f"[7] 预测完成。原始空间产额范围: [{y_pred.min():.2e}, {y_pred.max():.2e}]"
          + (f"；成员间 σ 范围: [{y_std.min():.2e}, {y_std.max():.2e}]"
             if y_std is not None else ""))

    # 10. 可选质量守恒后处理（按 A 分组归一到 2）
    mass_conservation = bool(pp_cfg.get('mass_conservation', False))
    if mass_conservation:
        A_full = A_physical[nuc_idx_full]
        sum_by_A = pd.Series(A_full).map(
            pd.Series(y_pred, index=A_full).groupby(level=0).sum()
        ).values
        y_pred_mc = y_pred / np.where(sum_by_A == 0, np.nan, sum_by_A) * 2.0
        y_pred_mc = np.nan_to_num(y_pred_mc, nan=0.0)
        print(f"[8] 质量守恒后处理开启：按 A 归一到 2（仅影响输出列，不改主聚合）")
    else:
        y_pred_mc = None

    # 组装结果 DataFrame
    E_full = E_physical_grid[e_idx_full]
    df_out = pd.DataFrame({
        'Z_physical': Z_physical[nuc_idx_full],
        'A_physical': A_physical[nuc_idx_full],
        'E_physical': E_full,
        'Yield_pred': y_pred,
    })
    if y_std is not None:
        df_out['Yield_pred_std'] = y_std
    if mass_conservation:
        df_out['Yield_pred_mass_conserved'] = y_pred_mc

    # 11. 按 A / Z 聚合（原始预测，未做质量守恒）
    #     集成模式：先逐成员聚合、再跨成员求均值与标准差（见 _combine_member_aggs）。
    if y_std is None:
        df_sum_by_A = df_out.groupby(['A_physical', 'E_physical'], as_index=False)['Yield_pred'].sum()
        df_sum_by_Z = df_out.groupby(['Z_physical', 'E_physical'], as_index=False)['Yield_pred'].sum()
    else:
        df_sum_by_A = _combine_member_aggs(preds_orig, Z_physical, A_physical, E_full,
                                           nuc_idx_full, ['A_physical', 'E_physical'])
        df_sum_by_Z = _combine_member_aggs(preds_orig, Z_physical, A_physical, E_full,
                                           nuc_idx_full, ['Z_physical', 'E_physical'])

    # 12. 画图（英文标注）
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0, 0.85, n_E)]

    # 图1：yield_vs_A_{variant}.png
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    for idx, E_phy in enumerate(E_physical_grid):
        sub = df_sum_by_A[df_sum_by_A['E_physical'] == E_phy]
        ax1.plot(sub['A_physical'], sub['Yield_pred'], color=colors[idx], alpha=0.7, linewidth=1.5,
                 label=f'{E_phy:.0f} MeV' if idx % 3 == 0 else None)
        if 'Yield_pred_std' in sub.columns:   # 集成模式：成员间 ±1σ 带
            ax1.fill_between(sub['A_physical'],
                             np.clip(sub['Yield_pred'] - sub['Yield_pred_std'], 0.0, None),
                             sub['Yield_pred'] + sub['Yield_pred_std'],
                             color=colors[idx], alpha=0.15, linewidth=0)
        if E_phy == E_physical_grid[0] or E_phy == E_physical_grid[-1]:
            ax1.scatter(sub['A_physical'], sub['Yield_pred'], color=colors[idx], s=18, alpha=0.8,
                        label=f'{E_phy:.0f} MeV (pts)' if (E_phy == E_physical_grid[0] or E_phy == E_physical_grid[-1]) and idx % 3 != 0 else None)
    ax1.set_xlabel('Mass Number (A)')
    ax1.set_ylabel('Fission Yield Sum')
    ax1.set_title(f'Fission Yield vs Mass Number (A) — {variant}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.set_ylim(0, (df_sum_by_A['Yield_pred'] + df_sum_by_A.get('Yield_pred_std', 0.0)).max() * 1.1)
    plt.tight_layout()
    fig1_path = output_path(variant, 'results', f'yield_vs_A_{variant}.png')
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"[9] 已保存图1: {fig1_path}")

    # 图2：yield_vs_Z_{variant}.png
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    for idx, E_phy in enumerate(E_physical_grid):
        sub = df_sum_by_Z[df_sum_by_Z['E_physical'] == E_phy]
        ax2.plot(sub['Z_physical'], sub['Yield_pred'], color=colors[idx], alpha=0.7, linewidth=1.5,
                 label=f'{E_phy:.0f} MeV' if idx % 3 == 0 else None)
        if 'Yield_pred_std' in sub.columns:   # 集成模式：成员间 ±1σ 带
            ax2.fill_between(sub['Z_physical'],
                             np.clip(sub['Yield_pred'] - sub['Yield_pred_std'], 0.0, None),
                             sub['Yield_pred'] + sub['Yield_pred_std'],
                             color=colors[idx], alpha=0.15, linewidth=0)
        if E_phy == E_physical_grid[0] or E_phy == E_physical_grid[-1]:
            ax2.scatter(sub['Z_physical'], sub['Yield_pred'], color=colors[idx], s=18, alpha=0.8,
                        label=f'{E_phy:.0f} MeV (pts)' if (E_phy == E_physical_grid[0] or E_phy == E_physical_grid[-1]) and idx % 3 != 0 else None)
    ax2.set_xlabel('Atomic Number (Z)')
    ax2.set_ylabel('Fission Yield Sum')
    ax2.set_title(f'Fission Yield vs Atomic Number (Z) — {variant}')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.set_ylim(0, (df_sum_by_Z['Yield_pred'] + df_sum_by_Z.get('Yield_pred_std', 0.0)).max() * 1.1)
    plt.tight_layout()
    fig2_path = output_path(variant, 'results', f'yield_vs_Z_{variant}.png')
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"    已保存图2: {fig2_path}")

    # 13. 保存 CSV：energy_dep_{variant}.csv
    ed_csv = output_path(variant, 'results', f'energy_dep_{variant}.csv')
    df_out.to_csv(ed_csv, index=False)
    print(f"[10] 已保存逐点预测 CSV: {ed_csv}")

    # 14. 保存聚合 CSV
    sumA_csv = output_path(variant, 'results', f'yield_sum_by_A_{variant}.csv')
    sumZ_csv = output_path(variant, 'results', f'yield_sum_by_Z_{variant}.csv')
    df_sum_by_A.to_csv(sumA_csv, index=False)
    df_sum_by_Z.to_csv(sumZ_csv, index=False)
    print(f"     已保存聚合 CSV: {sumA_csv}")
    print(f"     已保存聚合 CSV: {sumZ_csv}")

    # 15. 中文摘要
    print("\n" + "=" * 60)
    print("能量依赖性分析完成！摘要信息")
    print("=" * 60)
    print(f"变体名        : {variant}")
    if is_ensemble:
        print(f"集成成员      : {len(bundles)} 个（{', '.join(b['name'] for b in bundles)}）")
        print(f"集成方式      : 各成员各自反变换到原始产额空间后平均（附成员间 ±1σ）")
    print(f"模型架构      : KAN{list(mcfg['hidden_layers'])} (grid={mcfg['grid']}, k={mcfg['k']})")
    print(f"基准核素表    : {ref_path}（{n_nuc} 唯一核素）")
    print(f"能量网格      : {e_range[0]}~{e_range[1]} MeV，步长 {e_step}，{n_E} 点")
    print(f"目标空间      : {target_space}，使用delta_np={use_delta_np}")
    print(f"预测产额范围  : [{y_pred.min():.2e}, {y_pred.max():.2e}]")
    if mass_conservation:
        print(f"质量守恒后处理: 开启（按 A 归一到 2）")
    print(f"逐点预测 CSV  : {ed_csv}")
    print(f"按A聚合 CSV   : {sumA_csv}")
    print(f"按Z聚合 CSV   : {sumZ_csv}")
    print(f"图1 (vs A)    : {fig1_path}")
    print(f"图2 (vs Z)    : {fig2_path}")
    print(f"完成时间      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
