# -*- coding: utf-8 -*-
"""x1 vs w1 的 03 评估报告逐项对比（.trash 临时脚本）。"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'pipeline', 'output')


def load(v):
    p = os.path.join(RES, v, 'results', f'eval_report_{v}.json')
    return json.load(open(p, encoding='utf-8'))


def flat(d, pre=''):
    out = {}
    for k, v in (d or {}).items():
        key = f"{pre}{k}"
        if isinstance(v, dict) and all(isinstance(x, (int, float, type(None))) and not isinstance(x, bool)
                                       for x in v.values()):
            out.update(flat(v, key + '.'))
        elif isinstance(v, (dict,)):
            out.update(flat(v, key + '.'))
        else:
            out[key] = v
    return out


a, b = flat(load('x1_ft_235UALL_power')), flat(load('w1_ft_235UALL_power'))

SKIP = ('timestamp', 'visualization_path', 'model_path', 'figure', 'png')
rows = []
for k in sorted(set(a) | set(b)):
    if any(s in k for s in SKIP):
        continue
    va, vb = a.get(k), b.get(k)
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
        rows.append((k, vb, va, va - vb, (va - vb) / abs(vb) * 100 if vb else float('nan')))

print("=" * 92)
print("x1_ft (GEF 噪声增广 warmup)  vs  w1_ft (无增广 warmup)   —— val held-out, n=619, seed 相同")
print("=" * 92)
print(f"{'指标':<52}{'w1':>13}{'x1':>13}{'Δ':>13}{'Δ%':>9}")
print("-" * 92)
for k, vb, va, d, pct in rows:
    better = ''
    # R² 越大越好；RMSE/MAE/MSE 越小越好
    if 'r2' in k.lower():
        better = '  <<< 改善' if d > 1e-6 else ('  恶化' if d < -1e-6 else '')
    elif any(s in k.lower() for s in ('rmse', 'mae', 'mse')):
        better = '  <<< 改善' if d < -1e-9 else ('  恶化' if d > 1e-9 else '')
    pstr = f"{pct:+.2f}%" if pct == pct else "—"
    print(f"{k:<52}{vb:>13.6g}{va:>13.6g}{d:>+13.3g}{pstr:>9}{better}")

print("\n" + "=" * 92)
print("非数值/结构字段是否一致")
print("=" * 92)
for k in sorted(set(a) | set(b)):
    if any(s in k for s in SKIP):
        continue
    va, vb = a.get(k), b.get(k)
    if not (isinstance(va, (int, float)) and isinstance(vb, (int, float))):
        flag = '一致' if va == vb else f'不一致: w1={vb!r} | x1={va!r}'
        print(f"  {k:<52}{flag}")
