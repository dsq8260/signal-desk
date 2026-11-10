"""均线偏离回归（第 9 种进场）的回归测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.combo import ENTRIES, ma_dev_reversion_signal
from gold.forge import Genome, build_position


def _bars(n=600, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.004, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close,
                         "tick_volume": 100.0, "spread": 16.0}, index=idx)


@pytest.fixture(scope="module")
def df():
    return _bars()


def test_registered_in_entries():
    assert "均线偏离回归" in ENTRIES
    fn, defaults = ENTRIES["均线偏离回归"]
    assert fn is ma_dev_reversion_signal
    assert {"ma_n", "entry_thr", "exit_thr"} <= set(defaults)


def test_signal_is_binary_and_indexed(df):
    s = ma_dev_reversion_signal(df, ma_n=10, entry_thr=100.0, exit_thr=100.0)
    assert len(s) == len(df)
    assert set(np.unique(s.to_numpy())) <= {0.0, 1.0}


def test_signal_actually_trades(df):
    """曾经因为阈值单位搞错（百分比 vs 标准差）导致 0 笔交易。"""
    s = ma_dev_reversion_signal(df, ma_n=10, entry_thr=100.0, exit_thr=100.0)
    assert s.sum() > 0, "必须真的产生持仓，否则阈值口径又错了"


def test_threshold_monotonic(df):
    """阈值放宽（更难触发）→ 持仓时间不应增加。"""
    tight = ma_dev_reversion_signal(df, ma_n=10, entry_thr=200.0).sum()
    loose = ma_dev_reversion_signal(df, ma_n=10, entry_thr=50.0).sum()
    assert tight <= loose


def test_build_position_integration(df):
    g = Genome("均线偏离回归", {"ma_n": 10, "entry_thr": 100.0,
                              "exit_thr": 100.0}, "无过滤", False, 0.02, 0.05)
    pos = build_position(df, None, g)
    assert len(pos) == len(df)
    assert pos.abs().sum() > 0
