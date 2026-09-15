# -*- coding: utf-8 -*-
"""
compare_u_ft_freeze.py — 对比 u finetune 系列在【不同冻结层数】下的表现。
变体（freeze=N 表示冻结前 N 个 KANLayer，仅余后层可训练）：
    freeze0 = u_ft_235UALL_power_delta_np            (全解冻, = u)
    freeze1/2/3/4 = u_ft_freeze{1,2,3,4}_235UALL_power_delta_np
两轴：
  A) 235UALL held-out val 指标（来自各 eval_report_<v>.json）
  B) GEF 灾难性遗忘（来自 gef_postfinetune_eval.json 的 GEF post-ft R²，
     对照 u_gef warmup 基线，Δ = 基线 - post-ft）
"""
import os
import json
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

VARIANTS = [
    ('freeze0', 'u_ft_235UALL_power_delta_np', 0),
    ('freeze1', 'u_ft_freeze1_235UALL_power_delta_np', 1),
    ('freeze2', 'u_ft_freeze2_235UALL_power_delta_np', 2),
    ('freeze3', 'u_ft_freeze3_235UALL_power_delta_np', 3),
    ('freeze4', 'u_ft_freeze4_235UALL_power_delta_np', 4),
]

GEF_BASE = 'u_gef_isomer_delta_np'   # u 系列统一 warmup 基线


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def gef_baseline_r2():
    p = os.path.join(ROOT, 'output', GEF_BASE, 'results',
                     f'eval_report_{GEF_BASE}.json')
    if os.path.exists(p):
        d = load_json(p)
        return float(d['metrics']['original_space']['r2'])
    return None


def main():
    gef_post = {}
    gp = os.path.join(ROOT, 'output', 'gef_postfinetune_eval.json')
    if os.path.exists(gp):
        gef_post = load_json(gp)

    base_r2 = gef_baseline_r2()
    rows = []
    missing = []
    for label, v, nfreeze in VARIANTS:
        rp = os.path.join(ROOT, 'output', v, 'results', f'eval_report_{v}.json')
        if not os.path.exists(rp):
            missing.append(label)
            rows.append({'label': label, 'nfreeze': nfreeze, 'v': v})
            continue
        d = load_json(rp)
        m = d['metrics']
        ir = d.get('important_region', {})
        rows.append({
            'label': label, 'nfreeze': nfreeze, 'v': v,
            'val_r2': m['original_space']['r2'],
            'val_rmse': m['original_space']['rmse'],
            'val_mae': m['original_space']['mae'],
            'val_high_r2': m['high_yield_region']['r2'],
            'ir_r2': ir.get('r2'),
            'ir_zero_shot_r2': ir.get('zero_shot_r2'),
            'finetune_gain': ir.get('finetune_gain_r2'),
            'gef_post_r2': gef_post.get(v, {}).get('r2'),
        })

    # 文本表格
    hdr = (f"{'variant':10s} {'frz':>3s} {'235U val R2':>11s} {'RMSE':>9s} {'MAE':>9s} "
           f"{'hiY R2':>8s} {'impR2':>8s} {'gain':>7s} {'GEFpostR2':>10s} {'forgetΔ':>8s}")
    lines = [hdr, '-' * len(hdr)]
    for r in rows:
        if 'val_r2' not in r:
            lines.append(f"{r['label']:10s} {r['nfreeze']:>3d}   (missing eval_report)")
            continue
        gp_r2 = r.get('gef_post_r2')
        forget = (f"{base_r2 - gp_r2:+.4f}" if (gp_r2 is not None and base_r2 is not None) else '  -  ')
        lines.append(
            f"{r['label']:10s} {r['nfreeze']:>3d} {r['val_r2']:>11.4f} {r['val_rmse']:>9.4f} "
            f"{r['val_mae']:>9.5f} {r['val_high_r2']:>8.4f} "
            f"{(r['ir_r2'] or 0):>8.4f} {(r['finetune_gain'] or 0):>7.4f} "
            f"{(gp_r2 if gp_r2 is not None else 0):>10.4f} {forget:>8s}")
    if base_r2 is not None:
        lines.append(f"\nGEF warmup 基线 (u_gef) R2 = {base_r2:.4f}；forgetΔ = 基线 - GEF post-ft R2")
    if missing:
        lines.append(f"\n缺失 eval_report 的变体: {missing}")

    out_md = '\n'.join(lines)
    save = os.path.join(ROOT, 'output', 'compare_u_ft_freeze.md')
    with open(save, 'w', encoding='utf-8') as f:
        f.write(out_md + '\n')
    print(out_md)
    print(f"\n已保存: {save}")


if __name__ == '__main__':
    main()
