"""gold 包：MT5 现货黄金的 EA 策略与回测。"""

from __future__ import annotations

import pathlib
import sys

# 让 import gold 在任何入口下都能找到 src/validation
_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gold.backtest import GoldBacktest, compare_all, run  # noqa: E402
from gold.data import Bars, load_gold_bars  # noqa: E402
from gold.strategies import STRATEGIES, apply_strategy  # noqa: E402

__all__ = [
    "Bars", "load_gold_bars", "STRATEGIES", "apply_strategy",
    "GoldBacktest", "run", "compare_all",
]
