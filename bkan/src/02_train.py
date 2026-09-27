# -*- coding: utf-8 -*-
"""
bkan/src/02_train.py — BKAN 训练骨架（底层 GPKAN + 手写循环）

为什么不用 `GPKANRegressor.fit()`：
  - 两阶段 warm-up→finetune 需要权重续接（`state_dict()` 存取）
  - 损失要按"逐点 σ"改（库的 `obs_noise` 是标量）

损失（= 论文 arXiv:2607.04148 的 Eq.(2)，含 log 项）：
    total_var = f_var + σ_i²
    loss = gaussian_nll_loss(mu, total_var, y) + kl_weight·KL + sparsity·L1
其中 σ_i 是**逐点已知**的绝对误差（此处固定在 log 标准化空间，不参与学习）。

训练配方沿用官方 examples 的两段式：
    第 1 段 warm-up : sparsity_weight = 0
    第 2 段 pruning : sparsity_weight = cfg.train.sparsity_weight

用法：
    python -u bkan/src/02_train.py --config bkan/configs/gef_log.yaml
"""

import os
import sys
import time
import argparse
import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vendor  # noqa: F401  （把 bkan/vendor 加入 sys.path）
from svgp_kan import GPKAN, gaussian_nll_loss     # noqa: E402
from data import load_config, get_variant, output_path   # noqa: E402


def _l1_sparsity(model):
    """对第一层的核方差做 L1（库的 ARD 稀疏项），与 GPKANRegressor 一致。"""
    return torch.exp(model.layers[0].log_variance).sum()


def _train_phase(model, X, y, sigma2, mask, cfg, epochs, sparsity_weight,
                 phase_name, device, verbose=True):
    tr = cfg['train']
    lr = float(tr.get('lr', 0.02))
    kl_weight = float(tr.get('kl_weight', 0.01))
    batch_size = int(tr.get('batch_size', 256))
    grad_clip = float(tr.get('grad_clip', 1.0))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    hist = []
    t0 = time.time()
    model.train()

    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = ep_nll = ep_kl = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bx, by = X[idx], y[idx]
            bvar = sigma2[idx].reshape(-1, 1).expand_as(by)   # 逐点已知 σ²
            if mask is not None:
                m = mask[idx].reshape(-1, 1).float()
                keep = m.squeeze(-1) > 0
                if keep.sum() == 0:
                    continue
                bx, by, bvar = bx[keep], by[keep], bvar[keep]

            mu, f_var = model(bx)
            total_var = f_var + bvar
            nll = gaussian_nll_loss(mu, total_var, by)
            kl = model.compute_total_kl()
            loss = nll + kl_weight * kl
            if sparsity_weight > 0:
                loss = loss + sparsity_weight * _l1_sparsity(model)

            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            ep_loss += loss.item(); ep_nll += nll.item(); ep_kl += kl.item(); nb += 1

        rec = {'epoch': ep, 'loss': ep_loss / max(nb, 1),
               'nll': ep_nll / max(nb, 1), 'kl': ep_kl / max(nb, 1)}
        hist.append(rec)
        if verbose and ((ep + 1) % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  [{phase_name}] ep {ep+1:5d}/{epochs} | "
                  f"nll {rec['nll']:+.4f} | kl {rec['kl']:8.2f} | loss {rec['loss']:+.4f}")

    print(f"  [{phase_name}] 完成，{time.time()-t0:.1f}s")
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    variant = get_variant(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(int(cfg['model'].get('seed', 42)))

    print("=" * 70)
    print(f"BKAN 训练  variant={variant}  device={device}")
    print("=" * 70)

    pkl = output_path(variant, 'data', f'preprocessed_{variant}.pkl')
    with open(pkl, 'rb') as f:
        d = pickle.load(f)
    X = torch.tensor(np.asarray(d['X_train'], dtype=np.float32), device=device)
    y = torch.tensor(np.asarray(d['y_train'], dtype=np.float32), device=device)
    sig = np.asarray(d['sigma_train'], dtype=np.float32)
    msk = np.asarray(d['mask_train']) if d.get('mask_train') is not None else None
    sigma2 = torch.tensor(sig ** 2, device=device)
    mask_t = torch.tensor(msk, device=device) if msk is not None else None

    n_feat = X.shape[1]
    dims = list(cfg['model']['hidden_layers'])
    dims = [n_feat] + dims          # 首维由数据决定
    print(f"样本 {X.shape[0]}  特征 {n_feat}  网络 {dims}  "
          f"M={cfg['model']['num_inducing']}  kernel={cfg['model']['kernel']}")
    print(f"σ（固定）：中位 {np.median(sig):.4f}  90分位 {np.percentile(sig,90):.4f}")

    model = GPKAN(layers_hidden=dims,
                  num_inducing=int(cfg['model']['num_inducing']),
                  kernel_type=cfg['model']['kernel']).to(device)

    tr = cfg['train']
    warm = int(tr.get('warmup_epochs', 0))
    total = int(tr.get('epochs', 2000))
    ckpt_every = int(tr.get('checkpoint_every', 100))

    hist = []
    if warm > 0:
        hist += _train_phase(model, X, y, sigma2, mask_t, cfg, warm,
                             sparsity_weight=0.0, phase_name='warm-up', device=device)
    prune_epochs = max(total - warm, 1)
    hist += _train_phase(model, X, y, sigma2, mask_t, cfg, prune_epochs,
                         sparsity_weight=float(tr.get('sparsity_weight', 0.05)),
                         phase_name='pruning', device=device)

    mp = output_path(variant, 'models', f'bkan_{variant}.pth')
    torch.save({'model_state': model.state_dict(), 'config': cfg, 'variant': variant,
                'history': hist, 'arch': {'layers_hidden': dims,
                                          'num_inducing': int(cfg['model']['num_inducing']),
                                          'kernel': cfg['model']['kernel']}}, mp)
    print(f"\n已保存: {mp}")


if __name__ == '__main__':
    main()
