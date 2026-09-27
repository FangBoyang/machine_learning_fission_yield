# -*- coding: utf-8 -*-
"""
bkan/src/_vendor.py — 让 `import svgp_kan` 指向仓库内的 vendored 源码。

各阶段脚本在本文件之后 `import` svgp_kan 即可，无需 `pip install`，
也不污染 conda 环境。

用法（在每个 stage 脚本顶部）：
    import _vendor          # noqa: F401  （把 bkan/vendor 加入 sys.path）
    from svgp_kan import GPKAN, gaussian_nll_loss
"""

import os
import sys

_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vendor')
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

# 记录 vendored 版本，便于复现（源码取自 GitHub 镜像，非 PyPI）
SVGP_KAN_SOURCE = 'github.com/sungjuGit/svgp-kan @ main (vendored, zip snapshot 2026-04-20)'
