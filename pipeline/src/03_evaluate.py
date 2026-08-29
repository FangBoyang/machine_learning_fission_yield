# -*- coding: utf-8 -*-
"""
03_evaluate.py — KAN 模型评估（配置驱动，支持 held-out 验证集）

流程：加载模型 checkpoint + 预处理 pkl → 预测 → 反归一化 → 裁剪负值 →
      计算指标（归一化空间 / 原始空间 / 高产额区）→ 画图 → 保存报告

评估集选择（cfg.eval.on，默认 auto）：
- 'val'  ：在 pkl 的 X_val（held-out 验证集）上评估 —— finetune 场景的泛化代理。
- 'train'：在 X_train 上评估（向后兼容 GEF 全量训练，等同 in-sample）。
- 'auto' ：有 X_val 用 val，否则用 train。

可选增强：
- eval.important_range=[start, end]：对全量数据中的指定行区间（重要区）单独出 R²，
  该区在 finetune 中属于训练集，明确标注 in-sample。
- finetune.init_from 存在时：额外评估“零样本预训练基”在同评估集上的表现作为对照。

用法：
    python src/03_evaluate.py --config configs/<variant>.yaml
"""

import os
import argparse
import json
import time
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from kan import KAN
from common import (load_config, get_variant, output_path, PROJECT_ROOT,
                    build_kan_from_ckpt, load_pretrained_model)


def _load_model(variant, device):
    """优先 kan_best_{variant}.pth，其次 kan_final_{variant}.pth。返回 (model, ckpt, path, cfg)。"""
    best = output_path(variant, 'models', f'kan_best_{variant}.pth')
    final = output_path(variant, 'models', f'kan_final_{variant}.pth')
    path = best if os.path.exists(best) else final
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到模型文件: {best} 或 {final}（请先运行 02_train.py）")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model, cfg, grid = build_kan_from_ckpt(ckpt, device)
    return model, ckpt, path, cfg


def _predict(model, X, device, batch_size=512):
    """分批前向，返回一维 numpy 预测（归一化空间）。"""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    out = []
    n = Xt.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = Xt[i:i + batch_size]
            out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out, axis=0).reshape(-1)


def _inverse_target(y_norm, scaler, target_space, target_power=1.0):
    """归一化空间 → 原始空间。log 空间需先反变换再 exp；幂次目标需再 ^(1/p)。"""
    y = scaler.inverse_transform(np.asarray(y_norm, dtype=np.float32).reshape(-1, 1)).reshape(-1)
    if target_space == 'log':
        y = np.exp(y)
    if target_power != 1.0:
        y = np.power(np.clip(y, 0.0, None), 1.0 / target_power)
    return y


def _compute_metrics(y_true_orig, y_pred_orig):
    """原始空间指标：R² / RMSE / MAE / 高产额区(75分位) R²+RMSE。"""
    r2_orig = r2_score(y_true_orig, y_pred_orig)
    rmse = float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)))
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    high_thr = np.percentile(y_true_orig, 75)
    high = y_true_orig >= high_thr
    if high.sum() > 0:
        r2_high = r2_score(y_true_orig[high], y_pred_orig[high])
        rmse_high = float(np.sqrt(mean_squared_error(y_true_orig[high], y_pred_orig[high])))
    else:
        r2_high = rmse_high = 0.0
    return r2_orig, rmse, mae, high_thr, r2_high, rmse_high


def _make_figure(variant, target_space, X_eval, feats, y_true_orig, y_pred_orig,
                 r2_orig, r2_high, high_thr, tag):
    """画 2x2 评估图，返回保存路径。"""
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'KAN Evaluation ({variant}, space={target_space}, {tag})', fontsize=14, fontweight='bold')

    ax1 = axes[0, 0]
    ax1.scatter(y_true_orig, y_pred_orig, alpha=0.5, s=12, c='steelblue', edgecolors='none')
    lo = max(y_true_orig.min(), y_pred_orig.min(), 1e-15)
    hi = max(y_true_orig.max(), y_pred_orig.max())
    ax1.plot([lo, hi], [lo, hi], 'r--', alpha=0.7, label='Ideal Line')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('True Yield (Original Space)'); ax1.set_ylabel('Predicted Yield (Original Space)')
    ax1.set_title('Predicted vs True Yield (Log Scale)'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'Total R²={r2_orig:.4f}\nHigh-yield R²={r2_high:.4f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = axes[0, 1]
    resid = y_pred_orig - y_true_orig
    ax2.scatter(y_pred_orig, resid, alpha=0.5, s=12, c='seagreen', edgecolors='none')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax2.set_xscale('log')
    ax2.set_xlabel('Predicted Yield (Original Space)'); ax2.set_ylabel('Residual (Pred - True)')
    ax2.set_title('Residual Plot (Original Space)'); ax2.grid(True, alpha=0.3)
    ax2.text(0.05, 0.95, f'Mean Resid={resid.mean():.2e}\nStd Resid={resid.std():.2e}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax3 = axes[1, 0]
    rel_err = np.abs(resid) / (np.abs(y_true_orig) + 1e-12)
    rel_err_clip = np.clip(rel_err, 0, 10)
    ax3.hist(rel_err_clip, bins=50, alpha=0.8, color='purple', edgecolor='black')
    ax3.axvline(np.median(rel_err), color='r', linestyle='--', label=f'Median={np.median(rel_err):.2f}')
    ax3.axvline(np.percentile(rel_err, 90), color='orange', linestyle='--', label=f'P90={np.percentile(rel_err,90):.2f}')
    ax3.set_xlabel('Relative Error |Pred-True|/|True|'); ax3.set_ylabel('Frequency')
    ax3.set_title('Relative Error Distribution'); ax3.legend(); ax3.set_xlim(-0.5, 10.5); ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    X = np.asarray(X_eval, dtype=np.float32)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(feats), 1)))
    for i, (fname, color) in enumerate(zip(feats, colors)):
        fv = X[:, i]
        edges = np.percentile(fv, np.linspace(0, 100, 15))
        centers = (edges[:-1] + edges[1:]) / 2
        maes = []
        for j in range(len(edges) - 1):
            m = (fv >= edges[j]) & (fv < edges[j + 1])
            maes.append(mean_absolute_error(y_true_orig[m], y_pred_orig[m]) if m.sum() >= 5 else np.nan)
        valid = ~np.isnan(maes)
        if valid.any():
            ax4.plot(centers[valid], np.array(maes)[valid], 'o-', color=color, label=fname, alpha=0.7, markersize=4)
    ax4.set_xlabel('Feature Value (Normalized Space)'); ax4.set_ylabel('MAE (Original Space)')
    ax4.set_title('MAE across Feature Dimensions'); ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_tag = tag.replace(' ', '_').replace('(', '').replace(')', '')
    img_path = output_path(variant, 'results', f'eval_{variant}_{safe_tag}.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return img_path


def main():
    parser = argparse.ArgumentParser(description="KAN 模型评估（配置驱动，支持 val 评估）")
    parser.add_argument('--config', type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant = get_variant(cfg)

    print("=" * 60)
    print("KAN 模型评估（配置驱动）")
    print("=" * 60)

    # 2. 加载预处理 pkl
    pkl_path = output_path(variant, 'data', f'preprocessed_{variant}.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"未找到预处理文件: {pkl_path}（请先运行 01_preprocess.py）")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"[1] 加载预处理数据: {pkl_path}")

    # 3. 加载模型（结构取自 checkpoint 内嵌 config）
    device = data['device']
    model, ckpt, model_path, model_cfg = _load_model(variant, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] 加载模型: {model_path}")
    print(f"    架构: KAN{list(model_cfg['model']['hidden_layers'])} "
          f"(grid={model_cfg['model']['grid']}, k={model_cfg['model']['k']})")
    print(f"    参数量: {n_params:,}, 设备: {device}")

    # 4. 选择评估集
    eval_cfg = cfg.get('eval', {}) or {}
    on = eval_cfg.get('eval_set', 'auto')
    has_val = 'X_val' in data and data.get('X_val') is not None
    split_meta = data.get('data_info', {}).get('split', {}) or {}
    if on == 'val' and has_val:
        X_eval, y_eval_norm, split_idx, tag = data['X_val'], data['y_val'], split_meta.get('val_indices'), 'val (held-out)'
    elif on == 'train' or (on == 'auto' and not has_val):
        X_eval, y_eval_norm, split_idx, tag = data['X_train'], data['y_train'], split_meta.get('train_indices'), 'train (in-sample)'
    elif has_val:
        X_eval, y_eval_norm, split_idx, tag = data['X_val'], data['y_val'], split_meta.get('val_indices'), 'val (held-out)'
    else:
        X_eval, y_eval_norm, split_idx, tag = data['X_train'], data['y_train'], split_meta.get('train_indices'), 'train (in-sample)'

    full_yield = np.asarray(data['raw_data']['Yield_original'], dtype=np.float32).reshape(-1)
    if split_idx is not None:
        y_true_orig = full_yield[np.asarray(split_idx, dtype=int)]
    else:
        y_true_orig = full_yield
    y_eval_norm = np.asarray(y_eval_norm, dtype=np.float32).reshape(-1)
    print(f"[3] 评估集: {tag}, 样本数 {X_eval.shape[0]}")

    # 5. 反归一化 + 负值裁剪
    target_space = model_cfg['target']['space']
    clip_min = float(model_cfg['target']['clip_min'])
    info = data.get('data_info', {})
    target_power = float(info.get('target_power', 1.0))
    scaler_key = info.get('target_key', 'Yield_log' if target_space == 'log' else 'Yield_original')
    target_scaler = data['scalers'].get(scaler_key)
    if target_scaler is None:
        target_scaler = data['scalers'].get('Yield_original') or data['scalers'].get('Yield_log')
    y_pred_norm = _predict(model, X_eval, device)
    y_pred_original = _inverse_target(y_pred_norm, target_scaler, target_space, target_power)
    y_pred_original = np.clip(y_pred_original, clip_min, None)
    print(f"    目标空间: {target_space}，负值裁剪阈值: {clip_min}")

    # 6. 指标 + 图
    r2_orig, rmse_orig, mae_orig, high_thr, r2_high, rmse_high = _compute_metrics(y_true_orig, y_pred_original)
    r2_norm = r2_score(y_eval_norm, y_pred_norm)
    mse_norm = mean_squared_error(y_eval_norm, y_pred_norm)
    img_path = _make_figure(variant, target_space, X_eval, data['feature_names'],
                            y_true_orig, y_pred_original, r2_orig, r2_high, high_thr, tag)
    print(f"[4] 指标 -> 归一化空间 R²={r2_norm:.4f}, MSE={mse_norm:.3e}")
    print(f"    原始空间 R²={r2_orig:.4f}, RMSE={rmse_orig:.3e}, MAE={mae_orig:.3e}")
    print(f"    高产额区(阈值>{high_thr:.2e}) R²={r2_high:.4f}, RMSE={rmse_high:.3e}")
    print(f"[5] 已保存评估图: {img_path}")

    # 7. 可选：重要区单独评估（in-sample，训练集内）+ 零样本对照
    important_report = None
    imp_range = eval_cfg.get('important_range')
    # 预加载零样本基（供 val 与重要区两处对照复用）
    zmodel = None
    init_from = (cfg.get('finetune') or {}).get('init_from')
    if init_from:
        try:
            zmodel, _, _, zpath = load_pretrained_model(init_from, device)
        except Exception as e:
            print(f"    ⚠️ 零样本基加载失败: {e}")
    if imp_range is not None:
        start = int(imp_range[0]); end = int(imp_range[1]) if len(imp_range) > 1 and imp_range[1] is not None else full_yield.shape[0]
        train_idx = np.asarray(split_meta.get('train_indices'), dtype=int) if split_meta.get('train_indices') is not None else np.arange(full_yield.shape[0])
        imp_full = np.arange(start, end)
        pos = np.where(np.isin(train_idx, imp_full))[0]
        if pos.size > 0:
            X_imp = data['X_train'][pos]
            y_imp_true = full_yield[imp_full]
            y_imp_pred = np.clip(_inverse_target(_predict(model, X_imp, device), target_scaler, target_space, target_power), clip_min, None)
            ir2, irmse, imae, ithr, ir2h, irmseh = _compute_metrics(y_imp_true, y_imp_pred)
            important_report = {
                'range': [start, end], 'sample_count': int(pos.size),
                'r2': float(ir2), 'rmse': float(irmse), 'mae': float(imae),
                'high_yield_r2': float(ir2h), 'high_yield_rmse': float(irmseh),
                'note': 'in-sample (该区在 finetune 中属于训练集)',
            }
            # 重要区零样本对照：同一批重要行上，预训练基未微调的预测
            if zmodel is not None:
                z_imp_pred = np.clip(_inverse_target(_predict(zmodel, X_imp, device), target_scaler, target_space, target_power), clip_min, None)
                zir2, zirmse, zimae, _, zir2h, zirmseh = _compute_metrics(y_imp_true, z_imp_pred)
                important_report['zero_shot_r2'] = float(zir2)
                important_report['zero_shot_rmse'] = float(zirmse)
                important_report['zero_shot_high_yield_r2'] = float(zir2h)
                important_report['finetune_gain_r2'] = float(ir2 - zir2)
            print(f"[6] 重要区[{start},{end}) R²={ir2:.4f}, RMSE={irmse:.3e}, MAE={imae:.3e} (in-sample)"
                  + (f"；零样本R²={zir2:.4f}，微调增益={ir2 - zir2:+.4f}" if zmodel is not None else ""))

    # 8. 可选：零样本预训练基对照（评估集）
    zero_shot_report = None
    if init_from and zmodel is not None:
        zpred_norm = _predict(zmodel, X_eval, device)
        zpred_orig = np.clip(_inverse_target(zpred_norm, target_scaler, target_space, target_power), clip_min, None)
        zr2, zrmse, zmae, zthr, zr2h, zrmseh = _compute_metrics(y_true_orig, zpred_orig)
        zero_shot_report = {
            'model': init_from, 'path': zpath,
            'r2': float(zr2), 'rmse': float(zrmse), 'mae': float(zmae),
            'high_yield_r2': float(zr2h), 'high_yield_rmse': float(zrmseh),
            'note': '零样本：预训练基未微调直接预测（对照微调增益）',
        }
        print(f"[7] 零样本预训练基({init_from}) 评估集R²={zr2:.4f}, RMSE={zrmse:.3e}")

    # 9. 报告
    report = {
        'model_info': {
            'name': variant, 'model_path': model_path,
            'architecture': model_cfg['model']['hidden_layers'], 'parameters': n_params,
            'features': data['feature_names'], 'target_space': target_space,
        },
        'eval_set': tag,
        'metrics': {
            'normalized_space': {'r2': float(r2_norm), 'mse': float(mse_norm)},
            'original_space': {'r2': float(r2_orig), 'rmse': rmse_orig, 'mae': float(mae_orig)},
            'high_yield_region': {
                'threshold': float(high_thr), 'sample_count': int((y_true_orig >= high_thr).sum()),
                'r2': float(r2_high), 'rmse': float(rmse_high),
            },
        },
        'important_region': important_report,
        'zero_shot_pretrained': zero_shot_report,
        'data_stats': {
            'eval_samples': int(X_eval.shape[0]),
            'true_yield_range': [float(y_true_orig.min()), float(y_true_orig.max())],
            'pred_yield_range': [float(y_pred_original.min()), float(y_pred_original.max())],
        },
        'visualization_path': img_path,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    report_path = output_path(variant, 'results', f'eval_report_{variant}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[8] 已保存评估报告: {report_path}")

    print("\n" + "=" * 60)
    print("评估完成！摘要信息")
    print("=" * 60)
    print(f"变体名        : {variant}")
    print(f"评估集        : {tag}")
    print(f"归一化空间 R²  : {r2_norm:.4f} (MSE={mse_norm:.3e})")
    print(f"原始空间 R²    : {r2_orig:.4f} (RMSE={rmse_orig:.3e}, MAE={mae_orig:.3e})")
    print(f"高产额区 R²    : {r2_high:.4f} (阈值>{high_thr:.2e})")
    if important_report:
        zs = important_report.get('zero_shot_r2')
        extra = f"；零样本R²={zs:.4f}, 微调增益={important_report.get('finetune_gain_r2',0):+.4f}" if zs is not None else ""
        print(f"重要区 R²      : {important_report['r2']:.4f} (in-sample){extra}")
    if zero_shot_report:
        print(f"零样本预训练 R²: {zero_shot_report['r2']:.4f} (对照)")
    print(f"评估图        : {img_path}")
    print(f"报告          : {report_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
