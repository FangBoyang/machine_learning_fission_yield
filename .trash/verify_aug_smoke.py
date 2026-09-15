# -*- coding: utf-8 -*-
"""验证 01 增广接线 + 关闭时向后兼容（.trash 临时脚本）。"""
import os
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'pipeline', 'output')


def load(v):
    with open(os.path.join(OUT, v, 'data', f'preprocessed_{v}.pkl'), 'rb') as f:
        return pickle.load(f)


ok = fail = 0


def check(name, cond, extra=''):
    global ok, fail
    print(('  [PASS] ' if cond else '  [FAIL] ') + name + ('' if cond else f'  {extra}'))
    if cond:
        ok += 1
    else:
        fail += 1


print("=" * 60)
print("A. 向后兼容：_smoke_noaug vs v1_gef_isomer_delta_np（增广关闭）")
print("=" * 60)
a, b = load('_smoke_noaug'), load('v1_gef_isomer_delta_np')
check("X_train 完全一致", np.array_equal(a['X_train'], b['X_train']))
check("y_train 完全一致", np.array_equal(a['y_train'], b['y_train']))
check("feature_names 一致", a['feature_names'] == b['feature_names'])
for k in b['raw_data']:
    va, vb = a['raw_data'].get(k), b['raw_data'].get(k)
    if vb is None:
        check(f"raw_data['{k}'] 同为 None", va is None)
    else:
        check(f"raw_data['{k}'] 一致", np.array_equal(va, vb), f"{type(va)} vs {type(vb)}")
check("scalers 键一致", set(a['scalers']) == set(b['scalers']))
for k in b['scalers']:
    sa, sb = a['scalers'][k], b['scalers'][k]
    same = all(np.allclose(getattr(sa, at), getattr(sb, at))
               for at in ('mean_', 'scale_', 'var_') if hasattr(sb, at))
    check(f"scalers['{k}'] 参数一致", same)
di_a, di_b = a['data_info'], b['data_info']
for k in ('target_space', 'target_key', 'target_power', 'split_mode',
          'n_samples', 'n_train', 'n_val', 'n_features', 'data_source'):
    check(f"data_info['{k}'] 一致", di_a.get(k) == di_b.get(k), f"{di_a.get(k)} vs {di_b.get(k)}")
check("split 划分一致", np.array_equal(di_a['split']['train_indices'], di_b['split']['train_indices']))
check("关闭时 augment.enabled=False", di_a['augment']['enabled'] is False)
check("关闭时 aug_source_index 为 None", a['raw_data']['aug_source_index'] is None)

print("\n" + "=" * 60)
print("B. 增广接线：_smoke_aug")
print("=" * 60)
c = load('_smoke_aug')
n = c['X_train'].shape[0]
print(f"  行数: {n}（原始 9333）")
check("data_info.augment.enabled=True", c['data_info']['augment']['enabled'] is True)
check("n_before=9333", c['data_info']['augment']['n_before'] == 9333)
check("n_after=X_train 行数", c['data_info']['augment']['n_after'] == n)

for k, v in c['raw_data'].items():
    if v is None:
        print(f"  raw_data['{k}'] = None")
    else:
        check(f"raw_data['{k}'] 长度 == {n}", len(v) == n, f"got {len(v)}")

src = c['raw_data']['aug_source_index']
check("aug_source_index 存在且长度对齐", src is not None and len(src) == n)
check("aug_source_index 取值在 [0, 9333)", src.min() >= 0 and src.max() < 9333)
cnt = np.bincount(src, minlength=9333)
check("每个原始行至少出现 1 次", cnt.min() >= 1, f"min={cnt.min()}")
check("副本数只取 {0,1,2}", set(np.unique(cnt - 1)).issubset({0, 1, 2}), f"{np.unique(cnt)}")

y_clean = c['raw_data']['Yield_original']
uniq_orig = np.unique(load('_smoke_noaug')['raw_data']['Yield_original'])
check("Yield_original 全部来自原始 GEF 产额（干净）",
      np.all(np.isin(y_clean, uniq_orig)))
check("原件区（前 9333 行）Yield_original 与原始顺序一致",
      np.allclose(y_clean[:9333], load('_smoke_noaug')['raw_data']['Yield_original']))

# 副本行的预测输入与原件相同（X 复制正确）
check("副本行 X 与其源行 X 相同",
      np.allclose(c['X_train'][9333:], c['X_train'][src[9333:]]))
# 训练目标 y 在副本行应当被扰动（y 与原件不同）—— 抽样检查高档副本
y = c['y_train']
diff_mask = ~np.isclose(y[9333:], y[src[9333:]])
check("存在被扰动的副本目标（噪声确实进入了 y_train）", diff_mask.sum() > 0,
      f"diff={diff_mask.sum()}")

print("\n" + "=" * 60)
print(f"结果: {ok} passed, {fail} failed")
print("=" * 60)
raise SystemExit(1 if fail else 0)
