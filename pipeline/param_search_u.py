# -*- coding: utf-8 -*-
"""搜索 4 隐藏层 KAN 在 G=15,k=3 下接近 GEF 参数平衡(≈9333) 的宽度配置。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from kan import KAN

def count(width, grid=15, k=3):
    m = KAN(width=[4] + width + [1], grid=grid, k=k, save_act=False)
    return sum(p.numel() for p in m.parameters())

TARGET = 9333
print(f"目标参数≈{TARGET} (GEF 样本数)\n")

print("=== 等宽 [4,w,w,w,w,1] (w=8..11) ===")
for w in range(8, 12):
    c = count([w, w, w, w])
    print(f"w={w:2d} -> {c:6d}  (Δ={c-TARGET:+d})")

print("\n=== 4 层非等宽候选 ===")
cands = [
    [9, 9, 9, 9],
    [9, 9, 9, 8],
    [10, 9, 9, 8],
    [10, 9, 8, 8],
    [10, 9, 8, 7],
    [10, 10, 9, 8],
    [9, 9, 8, 8],
    [9, 8, 8, 8],
    [10, 10, 8, 8],
    [10, 8, 8, 8],
    [11, 9, 8, 7],
    [9, 9, 9, 7],
    [10, 9, 9, 7],
]
best = None
for c in cands:
    n = count(c)
    d = abs(n - TARGET)
    if best is None or d < best[0]:
        best = (d, c, n)
    print(f"{str(c):22s} -> {n:6d}  (Δ={n-TARGET:+d})")
print(f"\n最接近: {best[1]} -> {best[2]} (Δ={best[2]-TARGET:+d})")
