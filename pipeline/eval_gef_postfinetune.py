# -*- coding: utf-8 -*-
"""
eval_gef_postfinetune.py — 评估每个 *微调后* 模型在 GEF 域上的表现（灾难性遗忘度量）。

做法：加载微调后的 kan_best 检查点（结构/缩放器内嵌），在 GEF 预处理 pkl 的
X_train（9333 个 GEF 样本，使用被微调复用的同一套 GEF 缩放器）上预测，
反变换到原始产额空间并裁剪负值，计算 GEF 域 R²/RMSE/MAE/高产额区 R²。

对照：
- 本脚本输出 = 微调“之后”的 GEF 能力（遗忘后）。
- 各 *_gef_isomer_delta_np/results/eval_report_*.json（train in-sample）= 微调“之前”的 GEF 能力（预训练基线）。
- 两者之差 = 微调造成的 GEF 遗忘量。

微调模型 → 所用 GEF pkl 的映射（o 复用 n 的 warmup；t2 复用 t 的 warmup）：
"""
import os
import sys
import json
import pickle

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from kan import KAN
from common import build_kan_from_ckpt, output_path

PAIRS = [
    ('o_ft_235UALL_power_delta_np', 'n_power_delta_np'),
    ('p_ft_235UALL_power_delta_np', 'p_gef_isomer_delta_np'),
    ('q_ft_235UALL_power_delta_np', 'q_gef_isomer_delta_np'),
    ('r_ft_235UALL_power_delta_np', 'r_gef_isomer_delta_np'),
    ('s_ft_235UALL_power_delta_np', 's_gef_isomer_delta_np'),
    ('t_ft_235UALL_power_delta_np', 't_gef_isomer_delta_np'),
    ('t2_freeze_first_lowlr_ft_235UALL_power_delta_np', 't_gef_isomer_delta_np'),
    # u 系列 finetune（均基于 u_gef_isomer_delta_np warmup）
    ('u_ft_235UALL_power_delta_np', 'u_gef_isomer_delta_np'),          # freeze0 (= u)
    ('u_ft_freeze1_235UALL_power_delta_np', 'u_gef_isomer_delta_np'),
    ('u_ft_freeze2_235UALL_power_delta_np', 'u_gef_isomer_delta_np'),
    ('u_ft_freeze3_235UALL_power_delta_np', 'u_gef_isomer_delta_np'),
    ('u_ft_freeze4_235UALL_power_delta_np', 'u_gef_isomer_delta_np'),
]


def predict(model, X, device, bs=1024):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], bs):
            out.append(model(Xt[i:i + bs]).detach().cpu().numpy())
    return np.concatenate(out).reshape(-1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    for ftv, gefv in PAIRS:
        best = output_path(ftv, 'models', f'kan_best_{ftv}.pth')
        ckpt = torch.load(best, map_location=device, weights_only=False)
        model, mc, _ = build_kan_from_ckpt(ckpt, device)

        pkl = output_path(gefv, 'data', f'preprocessed_{gefv}.pkl')
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        X = data['X_train']
        y_true = np.asarray(data['raw_data']['Yield_original'], dtype=np.float32).reshape(-1)

        info = data.get('data_info', {})
        target_power = float(info.get('target_power', 1.0))
        target_space = mc['target']['space']
        clip_min = float(mc['target']['clip_min'])
        scaler_key = info.get('target_key', 'Yield_original')
        scaler = data['scalers'].get(scaler_key) or data['scalers'].get('Yield_original')

        y_pred_norm = predict(model, X, device)
        y = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).reshape(-1)
        if target_space == 'log':
            y = np.exp(y)
        if target_power != 1.0:
            y = np.power(np.clip(y, 0.0, None), 1.0 / target_power)
        y = np.clip(y, clip_min, None)

        r2 = r2_score(y_true, y)
        rmse = float(np.sqrt(mean_squared_error(y_true, y)))
        mae = mean_absolute_error(y_true, y)
        high = np.percentile(y_true, 75)
        m = y_true >= high
        r2h = r2_score(y_true[m], y[m])
        rmseh = float(np.sqrt(mean_squared_error(y_true[m], y[m])))

        results[ftv] = {
            'gef_base': gefv,
            'n_params': int(sum(p.numel() for p in model.parameters())),
            'architecture': mc['model']['hidden_layers'],
            'r2': float(r2), 'rmse': rmse, 'mae': float(mae),
            'high_yield_r2': float(r2h), 'high_yield_rmse': rmseh,
        }
        print(f"{ftv:52s} | GEF post-ft R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f} highR2={r2h:.4f}")

    out_path = output_path('.', '', 'gef_postfinetune_eval.json').replace('\\', '/')
    # 直接写到 pipeline/output 根便于汇总
    save = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'gef_postfinetune_eval.json')
    os.makedirs(os.path.dirname(save), exist_ok=True)
    with open(save, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {save}")


if __name__ == '__main__':
    main()
