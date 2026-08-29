# -*- coding: utf-8 -*-
"""
dump_resolved.py — 配置继承“真源 + 全展开镜像”生成器（方案 B）

设计目标：
- 源 config（configs/*.yaml）继续用 `inherit` 作唯一真源（DRY，机器可校验，不漂）。
- 本脚本把每个变体经 common.load_config 展开后的【全部生效参数】写进
  configs/resolved/<variant>.yaml（自包含、可直接喂回 load_config）。
- 同时生成 configs/CONFIG_TREE.md，列出每个变体的继承链（root→leaf）与父子树，
  使“继承关系”与“全展开参数”都一目了然。

resolved/ 下文件是派生物：请勿手改，改完源 config 后重跑本脚本即可。

用法（必须在 fpy_kan 环境，因为 common 顶层 import torch/kan）：
  cd pipeline
  C:/Users/86138/.conda/envs/fpy_kan/python.exe dump_resolved.py
或者：
  PYTHONPATH=src C:/Users/86138/.conda/envs/fpy_kan/python.exe dump_resolved.py
"""
import os
import sys
import glob
import yaml

# 让脚本能 import common
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from common import load_config  # noqa: E402

CONFIGS_DIR = os.path.join(HERE, 'configs')
RESOLVED_DIR = os.path.join(CONFIGS_DIR, 'resolved')
TREE_MD = os.path.join(CONFIGS_DIR, 'CONFIG_TREE.md')


def read_inherit_chain(path, _seen=None):
    """返回从 root 到本文件的继承链（不含 .yaml 扩展名的 basename 列表）。

    例：['base', 'g_rawY_delta_np', 'j_power_delta_np', 'o_ft_...']
    """
    if _seen is None:
        _seen = set()
    raw = yaml.safe_load(open(path, 'r', encoding='utf-8')) or {}
    name = os.path.splitext(os.path.basename(path))[0]
    chain = [name]
    parent = raw.get('inherit')
    if parent:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), parent)
        if os.path.exists(base_path) and parent not in _seen:
            _seen.add(parent)
            chain = read_inherit_chain(base_path, _seen) + chain
    return chain


def dump_one(path):
    name = os.path.splitext(os.path.basename(path))[0]
    chain = read_inherit_chain(path)
    resolved = load_config(path)  # 已合并、已剔除 inherit

    os.makedirs(RESOLVED_DIR, exist_ok=True)
    out_path = os.path.join(RESOLVED_DIR, f'{name}.yaml')

    header = (
        "# ============================================================================\n"
        "# RESOLVED CONFIG — 自动生成，请勿手改\n"
        "# 改源 config 后请重跑 pipeline/dump_resolved.py 重新生成本文件。\n"
        f"# 源文件 : configs/{name}.yaml\n"
        f"# 继承链 : {' -> '.join(chain)}\n"
        "# 说明   : 以下为 common.load_config 展开后的全部生效参数\n"
        "#          （inherit 已合并、已移除；可直接作为独立配置喂回 load_config）\n"
        "# ============================================================================\n"
    )
    body = yaml.safe_dump(
        resolved,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(body)
    return name, chain, out_path


def render_tree(children):
    lines = []

    def walk(node, prefix):
        kids = sorted(children.get(node, []))
        for i, kid in enumerate(kids):
            last = (i == len(kids) - 1)
            lines.append(f'{prefix}{"└── " if last else "├── "}{kid}')
            walk(kid, prefix + ('    ' if last else '│   '))

    roots = sorted(set(children.keys()) - set().union(*[set(v) for v in children.values()]) if children else [])
    for r in roots:
        lines.append(r)
        walk(r, '')
    return '\n'.join(lines)


def main():
    yaml_files = sorted(
        f for f in glob.glob(os.path.join(CONFIGS_DIR, '*.yaml'))
        if os.path.isfile(f)
    )
    if not yaml_files:
        print(f'未在 {CONFIGS_DIR} 找到任何 .yaml')
        return

    chains = {}
    ok, fail = 0, 0
    for p in yaml_files:
        try:
            name, chain, _ = dump_one(p)
            chains[name] = chain
            ok += 1
        except Exception as e:  # 单个失败不影响其他
            print(f'[WARN] 展开失败: {p} -> {e}')
            fail += 1

    # 父子关系（每条边只记一次，避免多变体共享前缀导致子树指数膨胀）
    children = {}
    for chain in chains.values():
        for parent, child in zip(chain, chain[1:]):
            children.setdefault(parent, [])
            if child not in children[parent]:
                children[parent].append(child)

    # 写 CONFIG_TREE.md
    lines = [
        '# 配置继承树 (CONFIG_TREE)',
        '',
        '> 自动生成（pipeline/dump_resolved.py）。每个变体的**全展开参数**见',
        '> `configs/resolved/<variant>.yaml`；本文件只描述**继承关系**。',
        '> 源 config 仍用 `inherit` 作唯一真源，请勿手改 resolved/ 下文件。',
        '',
        '## 继承树（root → leaf）',
        '',
        '```',
        render_tree(children),
        '```',
        '',
        '## 变体 → 继承链 → 全展开文件',
        '',
        '| 变体 | 继承链 (root→leaf) | 全展开文件 |',
        '|------|--------------------|------------|',
    ]
    for name in sorted(chains):
        chain = chains[name]
        resolved_rel = f'resolved/{name}.yaml'
        lines.append(f'| {name} | {" → ".join(chain)} | {resolved_rel} |')
    lines.append('')

    with open(TREE_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'[OK] 展开 {ok} 个配置{f"；失败 {fail} 个" if fail else ""}')
    print(f'[OK] 全展开写入: {RESOLVED_DIR}/')
    print(f'[OK] 继承树写入: {TREE_MD}')


if __name__ == '__main__':
    main()
