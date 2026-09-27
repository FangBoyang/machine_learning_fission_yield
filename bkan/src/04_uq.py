# -*- coding: utf-8 -*-
"""
bkan/src/04_uq.py — UQ 标定指标

指标配方照抄官方 `examples/UQ_Examples/study_a_revised_final.py`，
**同时就是裴组论文用的 coverage**，所以可直接与他们的结果对照：

  1. Pearson ρ(σ_pred, |error|)   —— 不确定性与真实误差是否相关
  2. 覆盖率 ±1σ / ±2σ / ±3σ        —— 目标 68% / 95% / 99.7%
  3. 标定误差 |coverage(2σ) − 95|
  4. RMSE
  5. 标准化残差 (y−μ)/σ ~ N(0,1)   —— mean≈0, std≈1

⚠ 在**标准化目标空间**里算（模型输出的 σ 只在这个空间有定义）。
⚠ GEF 变体是 full_train，因此这些指标是 **in-sample**，不是泛化指标。

对照基线：现有 6-seed 集成在高产区 1σ 覆盖率 0.35（目标 0.68）。

用法：
    python -u bkan/src/04_uq.py --config bkan/configs/gef_log.yaml
"""

import os
import sys
import json
import argparse
import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vendor  # noqa: F401
from svgp_kan import GPKAN                        # noqa: E402
from data import load_config, get_variant, output_path   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    variant = get_variant(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pkl = output_path(variant, 'data', f'preprocessed_{variant}.pkl')
    with open(pkl, 'rb') as f:
        d = pickle.load(f)
    mp = output_path(variant, 'models', f'bkan_{variant}.pth')
    ckpt = torch.load(mp, map_location=device, weights_only=False)
    model = GPKAN(layers_hidden=ckpt['arch']['layers_hidden'],
                  num_inducing=ckpt['arch']['num_inducing'],
                  kernel_type=ckpt['arch']['kernel']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # 有 val 用 val，否则用 train（GEF full_train 即 in-sample）
    use_val = d.get('X_val') is not None
    X = np.asarray(d['X_val'] if use_val else d['X_train'], dtype=np.float32)
    y = np.asarray(d['y_val'] if use_val else d['y_train'], dtype=np.float32).reshape(-1)
    sig = np.asarray(d['sigma_val'] if use_val else d['sigma_train'], dtype=np.float32)
    tag = 'val(held-out)' if use_val else 'train(in-sample)'

    print("=" * 70)
    print(f"BKAN UQ 标定  variant={variant}  set={tag}  n={len(y)}")
    print("=" * 70)

    Xt = torch.tensor(X, device=device)
    mu_l, var_l = [], []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], 4096):
            m, v = model(Xt[i:i + 4096])
            mu_l.append(m.cpu().numpy()); var_l.append(v.cpu().numpy())
    mu = np.concatenate(mu_l).reshape(-1)
    fvar = np.concatenate(var_l).reshape(-1)

    # 总预测方差 = 认知(GP) + 已知观测噪声
    tot = np.clip(fvar + sig ** 2, 1e-12, None)
    std = np.sqrt(tot)
    err = np.abs(y - mu)
    z = (y - mu) / std

    rho = float(np.corrcoef(std, err)[0, 1])
    cov = {k: float((err <= k * std).mean() * 100) for k in (1, 2, 3)}
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))

    res = {
        'variant': variant, 'eval_set': tag, 'n': int(len(y)),
        'note': '所有指标在标准化目标空间；GEF 变体为 in-sample',
        'rho_sigma_err': rho,
        'coverage_1sigma_pct': cov[1],
        'coverage_2sigma_pct': cov[2],
        'coverage_3sigma_pct': cov[3],
        'calibration_error_2sigma_pp': abs(cov[2] - 95.0),
        'rmse_target_space': rmse,
        'z_mean': float(z.mean()), 'z_std': float(z.std()),
        'sigma_median': float(np.median(std)),
    }

    print(f"  1. ρ(σ, |err|)          = {rho:+.4f}")
    print(f"  2. 覆盖率  ±1σ {cov[1]:6.2f}% (68)   ±2σ {cov[2]:6.2f}% (95)   ±3σ {cov[3]:6.2f}% (99.7)")
    print(f"  3. 标定误差 |cov2σ−95|  = {res['calibration_error_2sigma_pp']:.2f} pp")
    print(f"  4. RMSE                 = {rmse:.4f}")
    print(f"  5. 标准化残差  mean={z.mean():+.4f} (期望0)  std={z.std():.4f} (期望1)")

    out = output_path(variant, 'uq', f'uq_report_{variant}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {out}")
    print("对照：现有 6-seed 集成在高产区 1σ 覆盖率 0.35（目标 0.68）")


if __name__ == '__main__':
    main()
