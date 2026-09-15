# -*- coding: utf-8 -*-
"""
05_ensemble.py — 同构多种子 KAN 模型集成（精度对比 + 不确定度校准）

定位：流水线第 05 阶段。与 01–04 不同，本阶段不训练，而是读取若干个
「仅初始化随机种子不同」的已训练 finetune 变体，在同一份 held-out 验证集上
做集成预测，并给出集成的不确定度（成员间离散度 σ）及其校准情况。

设计要点：
1. 成员差异来源：finetune 阶段的 model.seed 会被 init_from 的 warmup 权重覆盖
   （见 02_train.py：`model = pre_model`），故 6 个 v*_ft 的差异全部来自各自
   v*_gef warmup 的 seed（1..6）。这是纯「同模型不同 seed」集成。
2. 评估口径与 03_evaluate 完全一致：复用其 _load_model / _predict /
   _inverse_target / _compute_metrics，高产额区阈值同为真实产额 75 分位。
3. 平均空间：各成员用「自己的」target scaler 反变换到原始产额空间后再平均。
   各成员 reuse_scalers_from 指向各自 warmup，脚本不假设 scaler 一致——
   逐个用自己的，并额外校验、报告各成员是否一致。
4. 不做自动剔除：加载失败的成员记入 missing 并跳过；所有个体指标照常列出，
   是否剔除由人在配置的 ensemble.drop_members 里显式登记。

产出（pipeline/output/<variant>/results/）：
  ensemble_report_<variant>.json   完整报告（逐成员 / 集成 / UQ / 一致性校验）
  ensemble_summary_<variant>.md    人读表格
  ensemble_perpoint_<variant>.csv  逐点：y_true / y_mean / y_std / 各成员预测
  ensemble_sigma_<variant>.png     奇偶图(±1σ) + σ vs 产额散点

用法：
    python src/05_ensemble.py --config configs/v_ens_seed_p3.yaml
"""

import os
import argparse
import json
import time
import pickle
import csv
import importlib.util

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from common import load_config, get_variant, output_path


# 03_evaluate.py 以数字开头，不是合法模块名，按文件路径导入以复用其评估原语
_EVAL03_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '03_evaluate.py')


def _import_eval03():
    """按路径导入 03_evaluate.py，复用其加载/预测/反变换/指标函数，保证口径一致。"""
    spec = importlib.util.spec_from_file_location('eval03', _EVAL03_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===================== 成员加载 =====================
def _load_member(eval03, member, eval_set):
    """加载单个成员：pkl + checkpoint → 评估集上的原始空间预测。

    不保留模型对象（用完即弃，避免同时驻留多个 KAN）。
    """
    pkl_path = output_path(member, 'data', f'preprocessed_{member}.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"未找到预处理文件: {pkl_path}（请先运行 01_preprocess.py）")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    device = data['device']
    model, ckpt, model_path, model_cfg = eval03._load_model(member, device)
    n_params = int(sum(p.numel() for p in model.parameters()))

    # 评估集选择（与 03_evaluate 同口径）
    split_meta = data.get('data_info', {}).get('split', {}) or {}
    has_val = 'X_val' in data and data.get('X_val') is not None
    if eval_set == 'train':
        X_eval, split_idx, tag = data['X_train'], split_meta.get('train_indices'), 'train (in-sample)'
    else:  # 'val' / 'auto'
        if not has_val:
            raise ValueError(f"成员 {member} 的 pkl 中没有 X_val（无 held-out 验证集），"
                             f"请将 ensemble.eval_set 改为 train")
        X_eval, split_idx, tag = data['X_val'], split_meta.get('val_indices'), 'val (held-out)'

    # 真实产额（原始空间）：从全量 Yield_original 按划分索引取
    full_yield = np.asarray(data['raw_data']['Yield_original'], dtype=np.float32).reshape(-1)
    if split_idx is not None:
        y_true = full_yield[np.asarray(split_idx, dtype=int)]
        split_idx = np.asarray(split_idx, dtype=int)
    else:
        y_true = full_yield

    # 目标空间 / 幂次 / scaler：取该成员自身（不假设各成员一致）
    target_space = model_cfg['target']['space']
    clip_min = float(model_cfg['target']['clip_min'])
    info = data.get('data_info', {}) or {}
    target_power = float(info.get('target_power', 1.0))
    scaler_key = info.get('target_key', 'Yield_log' if target_space == 'log' else 'Yield_original')
    target_scaler = (data['scalers'].get(scaler_key)
                     or data['scalers'].get('Yield_original')
                     or data['scalers'].get('Yield_log'))

    y_pred_norm = eval03._predict(model, X_eval, device)
    y_pred = eval03._inverse_target(y_pred_norm, target_scaler, target_space, target_power)
    y_pred = np.clip(y_pred, clip_min, None)  # 产额非负（物理约束）

    del model  # 及时释放
    return {
        'name': member,
        'model_path': model_path,
        'n_params': n_params,
        'X_eval': np.asarray(X_eval, dtype=np.float32),
        'y_true': np.asarray(y_true, dtype=np.float64),
        'y_pred': np.asarray(y_pred, dtype=np.float64),
        'split_idx': split_idx,
        'tag': tag,
        'target_space': target_space,
        'target_power': target_power,
        'clip_min': clip_min,
        'scaler': target_scaler,
    }


# ===================== 一致性校验 =====================
def _scaler_vec(scaler):
    """抽取 scaler 的 mean_/scale_ 拼成向量，用于比较各成员 scaler 是否一致。"""
    try:
        mean = np.asarray(scaler.mean_, dtype=float).ravel()
        scale = np.asarray(scaler.scale_, dtype=float).ravel()
        return np.concatenate([mean, scale])
    except Exception:
        return None


def _check_consistency(members):
    """校验各成员的 val 索引 / X_eval / scaler / 目标空间是否一致。

    val 索引不一致 → 直接报错（无法逐点集成）；其余仅记录报告。
    """
    ref = members[0]
    ref_idx = ref['split_idx']

    idx_identical = all(
        m['split_idx'] is not None and ref_idx is not None
        and m['split_idx'].shape == ref_idx.shape
        and np.array_equal(m['split_idx'], ref_idx)
        for m in members
    )
    if not idx_identical:
        raise ValueError(
            "各成员的验证集索引不一致，无法逐点集成。请确认所有成员配置的 "
            "data.split（mode/val_from_first_n/val_ratio/seed）完全相同。"
        )

    x_identical = all(
        m['X_eval'].shape == ref['X_eval'].shape
        and np.allclose(m['X_eval'], ref['X_eval'], rtol=1e-5, atol=1e-7)
        for m in members
    )

    ref_vec = _scaler_vec(ref['scaler'])
    scaler_identical = True
    if ref_vec is None:
        scaler_identical = None  # 无法比较（非 StandardScaler）
    else:
        for m in members:
            v = _scaler_vec(m['scaler'])
            if v is None or v.shape != ref_vec.shape or not np.allclose(v, ref_vec, rtol=1e-5, atol=1e-8):
                scaler_identical = False
                break

    space_set = sorted({m['target_space'] for m in members})
    power_set = sorted({round(m['target_power'], 10) for m in members})
    return {
        'val_indices_identical': bool(idx_identical),
        'x_eval_identical': bool(x_identical),
        'scalers_identical': scaler_identical,
        'target_space': space_set,
        'target_power': power_set,
    }


# ===================== 集成 =====================
def _build_ensemble(P, r2_indiv, methods):
    """P: [k, n] 原始空间预测矩阵。返回 {method: y_ens}。"""
    k = P.shape[0]
    out = {}
    if 'mean_equal' in methods:
        out['mean_equal'] = P.mean(axis=0)
    if 'median' in methods:
        out['median'] = np.median(P, axis=0)
    if 'r2_weighted' in methods:
        r2 = np.asarray(r2_indiv, dtype=float)
        if np.all(r2 > 0):
            w = r2 / r2.sum()
        else:
            print("    ⚠️ 存在 R²<=0 的成员，r2_weighted 退化为等权平均")
            w = np.ones(k) / k
        out['r2_weighted'] = (P * w[:, None]).sum(axis=0)
    return out


# ===================== 不确定度（UQ） =====================
def _safe_z(err, sigma):
    """z = |err| / σ，σ=0 处置 nan 后剔除。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(sigma > 0, err / np.where(sigma > 0, sigma, 1.0), np.nan)
    return z[np.isfinite(z)]


def _uq_block(y_true, y_mean, sigma, label):
    """覆盖率与校准系数。σ 标定良好时 cov1≈68%、cov2≈95%、z_rms≈1。"""
    err = np.abs(y_true - y_mean)
    z = _safe_z(err, sigma)
    if z.size == 0:
        return {'label': label, 'n': int(y_true.size), 'note': 'σ 全为 0，无法校准'}
    return {
        'label': label,
        'n': int(y_true.size),
        'coverage_1sigma': float(np.mean(z <= 1.0)),
        'coverage_2sigma': float(np.mean(z <= 2.0)),
        'target_1sigma': 0.6827,
        'target_2sigma': 0.9545,
        'k_68': float(np.percentile(z, 68)),   # 使覆盖率达 68% 所需的 σ 缩放系数
        'k_95': float(np.percentile(z, 95)),   # 使覆盖率达 95% 所需的 σ 缩放系数
        'z_rms': float(np.sqrt(np.mean(z ** 2))),
        'sigma_mean': float(np.mean(sigma)),
        'abs_err_mean': float(np.mean(err)),
    }


def _sigma_bins(y_true, y_mean, sigma, n_bins):
    """按真实产额等频分箱，观察 σ 是否随产额量级增大（异方差）。"""
    n = y_true.size
    order = np.argsort(y_true)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    err = np.abs(y_true - y_mean)
    rows = []
    for b in range(n_bins):
        idx = order[edges[b]:edges[b + 1]]
        if idx.size == 0:
            continue
        s = sigma[idx]
        z = _safe_z(err[idx], s)
        e_mean = float(np.mean(err[idx]))
        rows.append({
            'bin': b + 1,
            'n': int(idx.size),
            'y_min': float(y_true[idx].min()),
            'y_max': float(y_true[idx].max()),
            'y_mean': float(np.mean(y_true[idx])),
            'sigma_mean': float(np.mean(s)),
            'abs_err_mean': e_mean,
            'coverage_1sigma': float(np.mean(z <= 1.0)) if z.size else None,
            'sigma_over_abs_err': float(np.mean(s) / e_mean) if e_mean > 0 else None,
        })
    return rows


# ===================== 作图 =====================
def _make_figure(y_true, y_mean, sigma, high_thr, img_path, variant):
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Seed Ensemble: Accuracy & Uncertainty ({variant})', fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.scatter(y_true, y_mean, alpha=0.5, s=12, c='steelblue', edgecolors='none', label='ensemble mean')
    lo = max(y_true.min(), y_mean.min(), 1e-15)
    hi = max(y_true.max(), y_mean.max())
    ax1.plot([lo, hi], [lo, hi], 'r--', alpha=0.7, label='Ideal Line')
    lower = np.minimum(sigma, np.maximum(y_mean - 1e-15, 0.0))  # 保证误差棒下界非负（log 轴）
    ax1.errorbar(y_true, y_mean, yerr=[lower, sigma], fmt='none', ecolor='gray',
                 alpha=0.30, elinewidth=0.8, label='±1σ (member spread)')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('True Yield (Original Space)')
    ax1.set_ylabel('Ensemble Mean Yield (Original Space)')
    ax1.set_title('Parity with ±1σ Member Spread')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(y_true, sigma, alpha=0.5, s=12, c='darkorange', edgecolors='none')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.axvline(high_thr, color='r', linestyle='--', alpha=0.7,
                label=f'high-yield thr={high_thr:.2e}')
    ax2.set_xlabel('True Yield (Original Space)')
    ax2.set_ylabel('σ (std across members, Original Space)')
    ax2.set_title('Uncertainty σ vs Yield Magnitude')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return img_path


# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser(description="同构多种子 KAN 模型集成（精度对比 + 不确定度校准）")
    parser.add_argument('--config', type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant = get_variant(cfg)
    ens_cfg = cfg.get('ensemble', {}) or {}

    members_cfg = list(ens_cfg.get('members', []) or [])
    drop = set(ens_cfg.get('drop_members', []) or [])
    eval_set = str(ens_cfg.get('eval_set', 'val'))
    methods = list(ens_cfg.get('methods', ['mean_equal']) or ['mean_equal'])
    sigma_ddof = int(ens_cfg.get('sigma_ddof', 1))
    sigma_bins = int(ens_cfg.get('sigma_bins', 5))
    make_figure = bool(ens_cfg.get('make_figure', True))

    targets = [m for m in members_cfg if m not in drop]

    print("=" * 60)
    print("KAN 多种子集成（配置驱动）")
    print("=" * 60)
    print(f"[1] 集成变体: {variant}")
    print(f"    成员 {len(targets)} 个（配置列出 {len(members_cfg)}，手动排除 {len(members_cfg) - len(targets)}）")
    print(f"    评估集: {eval_set}；平均空间: {ens_cfg.get('average_space', 'original')}")
    print(f"    方法: {methods}")

    eval03 = _import_eval03()

    # ---- [2] 逐个加载成员 ----
    members, missing = [], []
    for i, m in enumerate(targets, 1):
        print(f"\n[2.{i}] 加载成员 {m} ...")
        try:
            info = _load_member(eval03, m, eval_set)
            members.append(info)
            print(f"      ✓ 模型: {os.path.basename(info['model_path'])}；"
                  f"评估集 {info['tag']}，样本 {info['y_true'].size}")
        except Exception as e:
            print(f"      ✗ 加载失败，已跳过：{e}")
            missing.append({'name': m, 'error': str(e)})

    if len(members) < 2:
        raise RuntimeError(f"可用成员仅 {len(members)} 个（<2），无法做集成与不确定度估计。"
                           f"失败详情: {missing}")

    # ---- [3] 一致性校验 ----
    consistency = _check_consistency(members)
    print(f"\n[3] 一致性校验: val索引一致={consistency['val_indices_identical']}, "
          f"X_eval一致={consistency['x_eval_identical']}, "
          f"scaler一致={consistency['scalers_identical']}, "
          f"目标空间={consistency['target_space']}, p={consistency['target_power']}")

    y_true = members[0]['y_true']
    n = y_true.size
    P = np.stack([m['y_pred'] for m in members], axis=0)  # [k, n]
    k = P.shape[0]

    # ---- [4] 逐成员指标 ----
    print(f"\n[4] 逐成员指标（原始空间，高产额区=真实产额 75 分位）")
    per_member, r2_indiv = [], []
    for m in members:
        r2, rmse, mae, thr, r2_high, rmse_high = eval03._compute_metrics(m['y_true'], m['y_pred'])
        r2_indiv.append(r2)
        rec = {
            'name': m['name'],
            'model_path': m['model_path'],
            'n_params': m['n_params'],
            'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae),
            'high_yield_r2': float(r2_high), 'high_yield_rmse': float(rmse_high),
        }
        per_member.append(rec)
        print(f"    {m['name']:<34s} R²={r2:.4f}  高产额R²={r2_high:.4f}  RMSE={rmse:.3e}")

    # ---- [5] 集成预测与指标 ----
    ens = _build_ensemble(P, r2_indiv, methods)
    if not ens:
        raise ValueError(f"没有生成任何集成方法，请检查 ensemble.methods（当前={methods}），"
                         f"可选：mean_equal / median / r2_weighted")
    ens_metrics = {}
    print(f"\n[5] 集成指标（{k} 个成员，原始空间平均）")
    for name, y_ens in ens.items():
        r2, rmse, mae, high_thr, r2_high, rmse_high = eval03._compute_metrics(y_true, y_ens)
        ens_metrics[name] = {
            'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae),
            'high_yield_r2': float(r2_high), 'high_yield_rmse': float(rmse_high),
            'high_yield_threshold': float(high_thr),
        }
        print(f"    {name:<14s} R²={r2:.4f}  高产额R²={r2_high:.4f}  RMSE={rmse:.3e}")

    best_single = max(per_member, key=lambda d: d['r2'])
    canonical = ens['mean_equal'] if 'mean_equal' in ens else next(iter(ens.values()))
    canon_name = 'mean_equal' if 'mean_equal' in ens else next(iter(ens.keys()))
    gain = ens_metrics[canon_name]['r2'] - best_single['r2']
    gain_high = ens_metrics[canon_name]['high_yield_r2'] - best_single['high_yield_r2']
    print(f"\n    最优单成员: {best_single['name']} (R²={best_single['r2']:.4f})")
    print(f"    集成({canon_name}) 增益: R² {gain:+.4f}，高产额R² {gain_high:+.4f}")

    # ---- [6] 不确定度与校准 ----
    high_thr = ens_metrics[canon_name]['high_yield_threshold']  # 与 03 一致：真实产额 75 分位
    sigma = P.std(axis=0, ddof=sigma_ddof)
    high_mask = y_true >= high_thr
    uq = {
        'canonical_method': canon_name,
        'sigma_ddof': sigma_ddof,
        'overall': _uq_block(y_true, canonical, sigma, 'all'),
        'high_yield': _uq_block(y_true[high_mask], canonical[high_mask], sigma[high_mask], 'high_yield'),
        'bins': _sigma_bins(y_true, canonical, sigma, sigma_bins),
        'corr_sigma_yield': float(np.corrcoef(sigma, y_true)[0, 1]) if sigma.std() > 0 else None,
        'note': 'σ = 成员间标准差（认知不确定度代理）；覆盖良好时 cov1≈68%、cov2≈95%、z_rms≈1。'
                'k_68/k_95 = 使覆盖率达到 68%/95% 所需的 σ 缩放系数。',
    }
    print(f"\n[6] 不确定度校准（σ = {k} 成员间标准差, ddof={sigma_ddof}）")
    o = uq['overall']
    print(f"    全部   : 1σ覆盖={o['coverage_1sigma']:.4f} (目标0.6827), "
          f"2σ覆盖={o['coverage_2sigma']:.4f} (目标0.9545), z_rms={o['z_rms']:.4f}")
    print(f"              校准系数 k68={o['k_68']:.4f}, k95={o['k_95']:.4f}")
    h = uq['high_yield']
    if 'coverage_1sigma' in h:
        print(f"    高产额区: 1σ覆盖={h['coverage_1sigma']:.4f}, 2σ覆盖={h['coverage_2sigma']:.4f}, "
              f"z_rms={h['z_rms']:.4f}")
    if uq['corr_sigma_yield'] is not None:
        print(f"    corr(σ, 真实产额) = {uq['corr_sigma_yield']:.4f}")

    # ---- [7] 产出 ----
    # 7a 逐点 CSV
    csv_path = output_path(variant, 'results', f'ensemble_perpoint_{variant}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['y_true', 'y_ens_mean', 'y_ens_sigma'] + [m['name'] for m in members])
        for i in range(n):
            w.writerow([f'{y_true[i]:.10g}', f'{canonical[i]:.10g}', f'{sigma[i]:.10g}']
                       + [f'{P[j, i]:.10g}' for j in range(k)])

    # 7b 图
    img_path = None
    if make_figure:
        img_path = output_path(variant, 'results', f'ensemble_sigma_{variant}.png')
        _make_figure(y_true, canonical, sigma, high_thr, img_path, variant)

    # 7c JSON 报告
    report = {
        'ensemble': {
            'name': variant,
            'n_members': k,
            'members': [m['name'] for m in members],
            'missing_members': missing,
            'dropped_members': sorted(drop),
            'eval_set': members[0]['tag'],
            'n_samples': int(n),
            'average_space': ens_cfg.get('average_space', 'original'),
            'methods': methods,
            'canonical_method': canon_name,
        },
        'per_member': per_member,
        'ensemble_metrics': ens_metrics,
        'best_single': best_single,
        'gain_vs_best_single': {'r2': float(gain), 'high_yield_r2': float(gain_high)},
        'consistency': consistency,
        'uncertainty': uq,
        'outputs': {
            'perpoint_csv': csv_path,
            'figure': img_path,
            'summary_md': output_path(variant, 'results', f'ensemble_summary_{variant}.md'),
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    report_path = output_path(variant, 'results', f'ensemble_report_{variant}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 7d Markdown 摘要
    md_path = report['outputs']['summary_md']
    _write_md(md_path, variant, report, ens_metrics, per_member, uq, canon_name, best_single)

    # ---- [8] 摘要 ----
    print(f"\n[7] 已保存:")
    print(f"    报告  : {report_path}")
    print(f"    摘要  : {md_path}")
    print(f"    逐点  : {csv_path}")
    if img_path:
        print(f"    图    : {img_path}")

    print("\n" + "=" * 60)
    print("集成完成！摘要")
    print("=" * 60)
    print(f"集成变体      : {variant}（{k} 成员）")
    print(f"评估集        : {members[0]['tag']}，样本 {n}")
    print(f"最优单成员 R² : {best_single['r2']:.4f} ({best_single['name']})")
    print(f"集成 R²       : {ens_metrics[canon_name]['r2']:.4f} ({canon_name})，增益 {gain:+.4f}")
    print(f"集成高产额 R² : {ens_metrics[canon_name]['high_yield_r2']:.4f}，增益 {gain_high:+.4f}")
    print(f"1σ 覆盖率     : {o['coverage_1sigma']:.4f}（目标 0.6827，标定系数 k68={o['k_68']:.3f}）")
    print("=" * 60)


def _write_md(md_path, variant, report, ens_metrics, per_member, uq, canon_name, best_single):
    """写出人读的 Markdown 摘要表格。"""
    e = report['ensemble']
    lines = []
    lines.append(f"# 集成摘要 — `{variant}`\n")
    lines.append(f"- 成员数：**{e['n_members']}**（{', '.join(e['members'])}）")
    lines.append(f"- 评估集：{e['eval_set']}，样本 {e['n_samples']}")
    lines.append(f"- 平均空间：{e['average_space']}（各成员各自反变换后平均）")
    lines.append(f"- 生成时间：{report['timestamp']}\n")

    if report['ensemble']['missing_members']:
        lines.append("## ⚠️ 未纳入的成员\n")
        lines.append("| 成员 | 失败原因 |")
        lines.append("|---|---|")
        for m in report['ensemble']['missing_members']:
            lines.append(f"| `{m['name']}` | {m['error']} |")
        lines.append("")

    lines.append("## 1. 逐成员指标（原始空间）\n")
    lines.append("| 成员 | R² | 高产额区 R² | RMSE | MAE |")
    lines.append("|---|---:|---:|---:|---:|")
    for d in per_member:
        lines.append(f"| `{d['name']}` | {d['r2']:.4f} | {d['high_yield_r2']:.4f} | "
                     f"{d['rmse']:.3e} | {d['mae']:.3e} |")
    r2s = [d['r2'] for d in per_member]
    lines.append(f"\n成员 R² 极差 = {max(r2s) - min(r2s):.4f}（min={min(r2s):.4f}, max={max(r2s):.4f}）\n")

    lines.append("## 2. 集成 vs 单成员\n")
    lines.append("| 方案 | R² | 高产额区 R² | RMSE | MAE |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, d in ens_metrics.items():
        mark = " 【主方法/UQ基准】" if name == canon_name else ""
        lines.append(f"| {name}{mark} | {d['r2']:.4f} | {d['high_yield_r2']:.4f} | "
                     f"{d['rmse']:.3e} | {d['mae']:.3e} |")
    lines.append(f"| 最优单成员 `{best_single['name']}` | {best_single['r2']:.4f} | "
                 f"{best_single['high_yield_r2']:.4f} | {best_single['rmse']:.3e} | "
                 f"{best_single['mae']:.3e} |")
    g = report['gain_vs_best_single']
    lines.append(f"\n集成({canon_name}) 相对最优单成员：R² **{g['r2']:+.4f}**，"
                 f"高产额区 R² **{g['high_yield_r2']:+.4f}**\n")

    lines.append("## 3. 不确定度校准\n")
    lines.append("σ = 成员间标准差（认知不确定度代理）。标定良好时 1σ 覆盖≈68%、2σ 覆盖≈95%、z_rms≈1。\n")
    lines.append(r"| 范围 | 样本 | 1σ 覆盖 | 2σ 覆盖 | z_rms | k68 | k95 | 平均 σ | 平均 \|误差\| |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for blk_key in ('overall', 'high_yield'):
        b = uq[blk_key]
        if 'coverage_1sigma' not in b:
            continue
        lines.append(f"| {b['label']} | {b['n']} | {b['coverage_1sigma']:.4f} | "
                     f"{b['coverage_2sigma']:.4f} | {b['z_rms']:.4f} | {b['k_68']:.4f} | "
                     f"{b['k_95']:.4f} | {b['sigma_mean']:.3e} | {b['abs_err_mean']:.3e} |")
    lines.append("")
    if uq['corr_sigma_yield'] is not None:
        lines.append(f"- corr(σ, 真实产额) = {uq['corr_sigma_yield']:.4f}\n")

    lines.append("## 4. σ 随产额量级的分箱（按真实产额等频分箱）\n")
    lines.append(r"| 箱 | 样本 | 产额范围 | 平均产额 | 平均 σ | 平均\|误差\| | 1σ 覆盖 | σ/\|误差\| |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in uq['bins']:
        cov = f"{r['coverage_1sigma']:.4f}" if r['coverage_1sigma'] is not None else "—"
        ratio = f"{r['sigma_over_abs_err']:.4f}" if r['sigma_over_abs_err'] is not None else "—"
        lines.append(f"| {r['bin']} | {r['n']} | [{r['y_min']:.3e}, {r['y_max']:.3e}] | "
                     f"{r['y_mean']:.3e} | {r['sigma_mean']:.3e} | {r['abs_err_mean']:.3e} | "
                     f"{cov} | {ratio} |")

    lines.append("")
    lines.append("## 5. 一致性校验\n")
    c = report['consistency']
    lines.append(f"- 各成员验证集索引一致：**{c['val_indices_identical']}**（不一致则无法逐点集成）")
    lines.append(f"- 各成员 X_eval 一致：{c['x_eval_identical']}")
    lines.append(f"- 各成员 target scaler 一致：{c['scalers_identical']}")
    lines.append(f"- 目标空间：{c['target_space']}，幂次 p：{c['target_power']}")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
