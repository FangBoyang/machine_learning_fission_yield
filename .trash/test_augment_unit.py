# -*- coding: utf-8 -*-
"""augment_yield_noise 单元测试（.trash 内，临时脚本）。

用法：
    C:/Users/86138/.conda/envs/fpy_kan/python.exe -u .trash/test_augment_unit.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'pipeline', 'src'))
from common import augment_yield_noise  # noqa: E402

N = 20
rng = np.random.RandomState(0)
Y = np.logspace(-8, -0.3, N)          # 跨 8 个数量级，模拟产额分布


def make_df():
    return pd.DataFrame({
        'Z_norm': rng.rand(N),
        'A_norm': rng.rand(N),
        'E_norm': rng.rand(N),
        'Yield': Y.copy(),
        'Error': np.full(N, 0.01),
        'Z_original': np.arange(N),
    })


def cfg_aug(**over):
    c = {'data': {
        'split': 'full_train',
        'augment': {
            'enabled': True,
            'seed': 42,
            'quantile_edges': [0.5, 0.9],
            'n_copies': [0, 1, 2],
            'rel_sigma': [0.0, 0.05, 0.10],
            'noise_mode': 'multiplicative',
        }}}
    c['data']['augment'].update(over)
    return c


ok = fail = 0


def check(name, cond, extra=''):
    global ok, fail
    if cond:
        ok += 1
        print(f"  [PASS] {name}")
    else:
        fail += 1
        print(f"  [FAIL] {name} {extra}")


print("=" * 60)
print("T1 启用：行数 / 分档 / 副本语义")
print("=" * 60)
df0 = make_df()
out, src = augment_yield_noise(df0, cfg_aug())
edges = np.quantile(Y, [0.5, 0.9])
tier = np.digitize(Y, edges)
copies = np.array([0, 1, 2])
expect = N + int(copies[tier].sum())
check("总行数 = N + Σ副本数", len(out) == expect, f"got {len(out)} expect {expect}")
check("aug_source_index 长度对齐", len(src) == len(out))
check("原件在前 N 行且顺序不变", np.array_equal(src[:N], np.arange(N)))
check("原件 Yield 未被改动", np.allclose(out['Yield'].values[:N], Y))
cnt = np.bincount(src, minlength=N)
check("每个源行出现 1+副本数 次", np.array_equal(cnt, 1 + copies[tier]),
      f"got {cnt[:6]}...")

print("\n" + "=" * 60)
print("T2 Yield_original 恒为源行干净值")
print("=" * 60)
check("Yield_original 按映射等于原始产额",
      np.allclose(out['Yield_original'].values, Y[src]))
check("列 'Yield_original' 存在", 'Yield_original' in out.columns)

print("\n" + "=" * 60)
print("T3 噪声行为")
print("=" * 60)
is_copy = np.zeros(len(out), dtype=bool)
is_copy[N:] = True
tier_of_row = tier[src]
sig_of_row = np.array([0.0, 0.05, 0.10])[tier_of_row]
# σ=0 的档：副本应为纯复制（用专门配置，让该档 n_copies>0）
out_p, src_p = augment_yield_noise(
    make_df(), cfg_aug(n_copies=[0, 2, 2], rel_sigma=[0.0, 0.0, 0.10]))
tier_p = tier[src_p]
m0 = np.zeros(len(out_p), dtype=bool); m0[N:] = True
m0 &= (tier_p == 1)                       # 该档 σ=0
check("σ=0 档副本为纯复制（Yield == 干净值）",
      m0.sum() > 0 and np.allclose(out_p['Yield'].values[m0],
                                   out_p['Yield_original'].values[m0]),
      f"m0.sum()={m0.sum()}")
# σ>0 档（tier 2）的副本：确实被扰动
m1 = is_copy & (sig_of_row > 0)
check("σ>0 档副本被扰动", m1.sum() > 0 and
      not np.allclose(out['Yield'].values[m1], out['Yield_original'].values[m1]))
rel = (out['Yield'].values[m1] / out['Yield_original'].values[m1]) - 1.0
check("相对扰动量级 ≈ σ=0.10", abs(np.abs(rel).mean()) < 0.15 and np.abs(rel).max() < 0.45,
      f"mean|rel|={np.abs(rel).mean():.4f} max={np.abs(rel).max():.4f}")
check("无负产额", (out['Yield'].values >= 0).all())

print("\n" + "=" * 60)
print("T4 副本携带其余列（Z_original 等）正确复制")
print("=" * 60)
check("Z_original 按映射正确", np.array_equal(out['Z_original'].values, np.arange(N)[src]))
check("Error 沿用源行", np.allclose(out['Error'].values, 0.01))

print("\n" + "=" * 60)
print("T5 关闭时向后兼容")
print("=" * 60)
df1 = make_df()
out2, src2 = augment_yield_noise(df1, {'data': {'split': 'full_train', 'augment': {'enabled': False}}})
check("返回 None 映射", src2 is None)
check("行数不变", len(out2) == N)
check("Yield 不变", np.allclose(out2['Yield'].values, Y))
check("新增 Yield_original 且等于 Yield", np.allclose(out2['Yield_original'].values, Y))
check("其余列不变", np.allclose(out2['Z_norm'].values, df1['Z_norm'].values))
check("未污染原始 df（无 Yield_original 列）", 'Yield_original' not in df1.columns)

print("\n" + "=" * 60)
print("T6 参数校验 / 防泄漏")
print("=" * 60)
try:
    augment_yield_noise(make_df(), {'data': {'split': {'mode': 'held_out'},
                                             'augment': {'enabled': True}}})
    check("held_out + 增广 → 报错", False)
except ValueError as e:
    check("held_out + 增广 → 报错", '泄漏' in str(e), str(e)[:80])

for bad in [{'n_copies': [1, 2]}, {'rel_sigma': [0.1]},
            {'n_copies': [-1, 0, 0]}, {'noise_mode': 'nope'}]:
    try:
        augment_yield_noise(make_df(), cfg_aug(**bad))
        check(f"非法参数 {bad} → 报错", False)
    except ValueError:
        check(f"非法参数 {bad} → 报错", True)

print("\n" + "=" * 60)
print("T7 lognormal 模式")
print("=" * 60)
out3, src3 = augment_yield_noise(make_df(), cfg_aug(noise_mode='lognormal'))
check("行数一致", len(out3) == expect)
check("严格为正", (out3['Yield'].values > 0).all())
m = np.zeros(len(out3), dtype=bool); m[N:] = True
sig3 = np.array([0.0, 0.05, 0.10])[tier[src3]]
sel = m & (sig3 > 0)
lr = np.log(out3['Yield'].values[sel] / out3['Yield_original'].values[sel])
check("对数扰动均值≈0（均值保持）", abs(lr.mean()) < 0.15, f"mean={lr.mean():.4f}")

print("\n" + "=" * 60)
print(f"结果: {ok} passed, {fail} failed")
print("=" * 60)
sys.exit(1 if fail else 0)
