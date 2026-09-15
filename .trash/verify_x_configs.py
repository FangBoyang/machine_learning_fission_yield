# -*- coding: utf-8 -*-
"""验证 x 系列 12 个配置：与对应 w 配置的差异是否恰好符合预期。

用法：
    cd pipeline && PYTHONPATH=src C:/Users/86138/.conda/envs/fpy_kan/python.exe -u ../.trash/verify_x_configs.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'pipeline', 'src'))
from common import load_config  # noqa: E402

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'pipeline', 'configs')


def flat(d, pre=''):
    out = {}
    for k, v in (d or {}).items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + '.'))
        else:
            out[key] = v
    return out


ok = fail = 0


def check(name, cond, extra=''):
    global ok, fail
    print(('  [PASS] ' if cond else '  [FAIL] ') + name + ('' if cond else f'  {extra}'))
    if cond:
        ok += 1
    else:
        fail += 1


IGNORE = {'experiment.name', 'experiment.description'}

print("=" * 66)
print("A. x{i}_gef vs w{i}_gef：差异应仅为 data.augment.*（+ 名称/描述）")
print("=" * 66)
for i in range(1, 7):
    a = flat(load_config(os.path.join(CFG, f'x{i}_gef_isomer.yaml')))
    b = flat(load_config(os.path.join(CFG, f'w{i}_gef_isomer.yaml')))
    keys = set(a) | set(b)
    diffs = {k for k in keys if k not in IGNORE and a.get(k) != b.get(k)}
    added = {k for k in diffs if k not in b}
    changed = {k for k in diffs if k in b}
    check(f"x{i}_gef 新增项恰为 data.augment 的 6 个键",
          added == {'data.augment.enabled', 'data.augment.seed',
                    'data.augment.quantile_edges', 'data.augment.n_copies',
                    'data.augment.rel_sigma', 'data.augment.noise_mode'},
          f"got {sorted(added)}")
    check(f"x{i}_gef 无改动项", not changed, f"got {sorted(changed)}")
    check(f"x{i}_gef model.seed == {i}", a['model.seed'] == i, f"got {a['model.seed']}")
    check(f"x{i}_gef split 仍为 full_train", a['data.split'] == 'full_train', f"got {a['data.split']}")
    check(f"x{i}_gef experiment.name 正确", a['experiment.name'] == f'x{i}_gef_isomer')
    check(f"x{i}_gef 训练超参与 w 相同",
          a['train.epochs'] == b['train.epochs']
          and a['train.early_stopping.patience'] == b['train.early_stopping.patience']
          and a['train.early_stopping.min_delta'] == b['train.early_stopping.min_delta']
          and a['train.optimizer.lr'] == b['train.optimizer.lr'])

print("\n" + "=" * 66)
print("B. x{i}_ft vs w{i}_ft：差异应仅为 finetune.init_from / reuse_scalers_from")
print("=" * 66)
for i in range(1, 7):
    a = flat(load_config(os.path.join(CFG, f'x{i}_ft_235UALL_power.yaml')))
    b = flat(load_config(os.path.join(CFG, f'w{i}_ft_235UALL_power.yaml')))
    diffs = {k for k in (set(a) | set(b)) if k not in IGNORE and a.get(k) != b.get(k)}
    check(f"x{i}_ft 差异恰为 init_from + reuse_scalers_from",
          diffs == {'finetune.init_from', 'finetune.reuse_scalers_from'},
          f"got {sorted(diffs)}")
    check(f"x{i}_ft init_from = x{i}_gef_isomer", a['finetune.init_from'] == f'x{i}_gef_isomer')
    check(f"x{i}_ft reuse = x{i}_gef_isomer", a['finetune.reuse_scalers_from'] == f'x{i}_gef_isomer')
    check(f"x{i}_ft split.seed == 42（与 w/v 对齐）", a['data.split.seed'] == 42)
    check(f"x{i}_ft val_from_first_n == 3096", a['data.split.val_from_first_n'] == 3096)
    check(f"x{i}_ft 无 data.augment", 'data.augment.enabled' not in a,
          f"got {a.get('data.augment.enabled')}")
    check(f"x{i}_ft freeze 仍为空", a['finetune.freeze'] == [] or a['finetune.freeze'] == [None]
          or not a['finetune.freeze'], f"got {a.get('finetune.freeze')}")

print("\n" + "=" * 66)
print("C. 增广参数取值（6 个 seed 必须完全一致 -> 共用同一份增广数据）")
print("=" * 66)
ref = None
for i in range(1, 7):
    a = load_config(os.path.join(CFG, f'x{i}_gef_isomer.yaml'))['data']['augment']
    if ref is None:
        ref = a
    check(f"x{i}_gef augment 与 x1 完全一致", a == ref, f"{a}")
check("enabled=True", ref['enabled'] is True)
check("seed=42", ref['seed'] == 42)
check("quantile_edges=[0.5,0.9]", ref['quantile_edges'] == [0.5, 0.9])
check("n_copies=[0,1,2]", ref['n_copies'] == [0, 1, 2])
check("rel_sigma=[0,0.05,0.10]", ref['rel_sigma'] == [0.0, 0.05, 0.10])
check("noise_mode=multiplicative", ref['noise_mode'] == 'multiplicative')

print("\n" + "=" * 66)
print(f"结果: {ok} passed, {fail} failed")
print("=" * 66)
raise SystemExit(1 if fail else 0)
