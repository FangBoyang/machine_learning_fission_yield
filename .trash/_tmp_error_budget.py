# -*- coding: utf-8 -*-
"""临时脚本：误差预算分析 A —— 模型误差 vs 实验测量误差(Error 列)。

回答：模型在 235UALL 上的预测误差是否已达到实验测量误差水平？
  - 若 model_MAE <= exp_Error，说明 0.970 R² 就是该数据集天花板，再调参无意义。
  - 对比 o6 的 val（held-out 泛化）与 train（in-sample）。
输出全部打印到 stdout。
"""
import os, pickle, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline', 'src'))
from common import load_config, output_path, build_kan_from_ckpt  # noqa: E402

VARIANT = 'o6_ft_235UALL_lr1e3_hold50_wd5e4_pat50'
CONFIG = f'pipeline/configs/{VARIANT}.yaml'

cfg = load_config(CONFIG)
variant = cfg['experiment']['name']

# ---- 1. 加载数据 ----
pkl_path = output_path(variant, 'data', f'preprocessed_{variant}.pkl')
data = pickle.load(open(pkl_path, 'rb'))
device = data['device']
rd = data['raw_data']
sp = data['data_info']['split']
val_idx = sp['val_indices']
train_idx = sp['train_indices']
info = data['data_info']
target_power = float(info.get('target_power', 1.0))
target_space = info.get('target_space', 'raw')
target_key = info.get('target_key', 'Yield')
scalers = data['scalers']
target_scaler = scalers[target_key]
clip_min = float(cfg['target'].get('clip_min', 0.0))

# 原始数据（按原始行序）
Y_true = rd['Yield_original'].astype(np.float64)
Err    = rd['Error'].astype(np.float64)
A      = rd['A_original'].astype(np.float64)
E      = rd['E_original'].astype(np.float64)
Z      = rd['Z_original'].astype(np.float64)

# ---- 2. 加载模型并预测 ----
best = output_path(variant, 'models', f'kan_best_{variant}.pth')
ckpt = torch.load(best, map_location=device, weights_only=False)
model, _, grid = build_kan_from_ckpt(ckpt, device)
model.eval()

def predict(X):
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], 512):
            out.append(model(Xt[i:i+512]).detach().cpu().numpy())
    return np.concatenate(out, axis=0).reshape(-1)

def inverse(y_norm):
    y = target_scaler.inverse_transform(np.asarray(y_norm, dtype=np.float32).reshape(-1, 1)).reshape(-1)
    if target_space == 'log':
        y = np.exp(y)
    if target_power != 1.0:
        y = np.power(np.clip(y, 0.0, None), 1.0 / target_power)
    return np.clip(y, clip_min, None)

y_pred_val = inverse(predict(data['X_val']))
y_pred_tr  = inverse(predict(data['X_train']))

# ---- 3. 汇总指标 ----
print('=' * 70)
print(f'误差预算分析 A  | 变体 {variant}')
print(f'val = held-out {len(val_idx)} 样本 | train = {len(train_idx)} 样本')
print('=' * 70)

def summarize(tag, idx, y_pred):
    yt = Y_true[idx]
    ye = Err[idx]
    yp = y_pred
    abs_err = np.abs(yt - yp)
    rel_err = abs_err / np.maximum(yt, 1e-300)
    print(f'\n[{tag}] 模型预测误差 vs 实验测量误差（{len(idx)} 样本）')
    print(f'  {'指标':<22}{'模型误差':>14}{'实验Error':>14}{'比值(模/实)':>12}')
    print(f'  {'MAE':<22}{np.mean(abs_err):>14.3e}{np.mean(ye):>14.3e}{np.mean(abs_err)/np.mean(ye):>12.2f}')
    print(f'  {'RMSE':<22}{np.sqrt(np.mean(abs_err**2)):>14.3e}{np.sqrt(np.mean(ye**2)):>14.3e}{np.sqrt(np.mean(abs_err**2))/np.sqrt(np.mean(ye**2)):>12.2f}')
    print(f'  {'中位 abs_err':<22}{np.median(abs_err):>14.3e}')
    print(f'  {'中位 rel_err %':<22}{np.median(rel_err)*100:>14.2f}')
    print(f'  {'实验Error 中位':<22}{np.median(ye):>14.3e}')
    # 模型误差超过实验误差的比例
    exceed = np.mean(abs_err > ye) * 100
    print(f'  模型abs_err > 实验Error 的样本占比: {exceed:.1f}%')
    return yt, ye, abs_err, rel_err

yt_v, ye_v, ae_v, re_v = summarize('VAL', val_idx, y_pred_val)
yt_t, ye_t, ae_t, re_t = summarize('TRAIN', train_idx, y_pred_tr)

# ---- 4. 按分区（val 泛化视角）----
print('\n' + '=' * 70)
print('分区误差预算（VAL，按 A 轻重 / 产额高低 / 能量）')
print('=' * 70)

def partition(tag, mask, yt, ye, ae):
    if mask.sum() == 0:
        return
    m = ae[mask].mean()
    e = ye[mask].mean()
    r = m / e if e > 0 else float('nan')
    print(f'  {tag:<28}n={mask.sum():<6}模型MAE={m:>10.3e}  实验Err={e:>10.3e}  比值={r:>6.2f}')

Am = np.median(A[val_idx])
Ym = np.percentile(yt_v, 75)
for tag, m in [
    (f'轻核 A<{Am:.0f}', A[val_idx] < Am),
    (f'重核 A>={Am:.0f}', A[val_idx] >= Am),
    (f'低产额 Y<75th({Ym:.2e})', yt_v < Ym),
    (f'高产额 Y>={Ym:.2e}', yt_v >= Ym),
    ('E<0.25MeV', E[val_idx] < 0.25),
    ('E>=0.25MeV', E[val_idx] >= 0.25),
]:
    partition(tag, m, yt_v, ye_v, ae_v)

print('\n说明：比值(模/实) < 1 表示模型误差低于实验测量误差（已达数据噪声极限）；')
print('      > 1 表示模型误差仍高于实验误差（还有优化空间）。')
