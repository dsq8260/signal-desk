"""pytest 引导：把仓库根目录与 src/ 加入 sys.path。

这样 tests 里既能 `import ganzhi`（src/ganzhi 包），
也能 `import data_loader / feature_cycle / coverage`（仓库根目录模块）。
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 归档目录里存着旧测试的副本，不能参与收集（同名模块会互相覆盖）
collect_ignore_glob = ["_archive_v1/*", "_archive_v1/**/*"]
