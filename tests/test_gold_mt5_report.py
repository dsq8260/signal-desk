"""MT5 风格报告测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.mt5_report import (
    _annual_factor,
    _max_dd,
    mt5_metrics,
    period_breakdown,
    trades_from_curve,
)


def _curve(n=40, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-01", periods=n)
    eq = 10000 * np.cumprod(1 + rng.normal(0.002, 0.01, n))
    pos = pd.Series(0.0, index=idx)
    pos.iloc[5:15] = 1.0
    pos.iloc[20:30] = 1.0
    return pd.Series(eq, index=idx), pos


def test_annual_factor_by_tf():
    assert _annual_factor("D1") == 252
    assert _annual_factor("H1") == 252 * 24
    assert _annual_factor("未知") == 252


def test_trades_split_by_holding_segments():
    eq, pos = _curve()
    t = trades_from_curve(eq, pos, 10000)
    assert len(t) == 2
    assert set(t["side"]) == {1}
    assert (t["close_time"] > t["open_time"]).all()


def test_trades_empty_when_flat():
    idx = pd.bdate_range("2026-01-01", periods=10)
    eq = pd.Series(10000.0, index=idx)
    pos = pd.Series(0.0, index=idx)
    assert len(trades_from_curve(eq, pos)) == 0


def test_max_dd_positive_values():
    c = pd.Series([100, 120, 90, 110])
    abs_dd, pct = _max_dd(c)
    assert abs_dd == pytest.approx(30.0)
    assert pct == pytest.approx(0.25)


def test_metrics_core_fields():
    eq, pos = _curve()
    m = mt5_metrics(eq, pos, tf="D1", initial=10000)
    assert m["初始入金"] == 10000
    assert m["交易总计"] == 2
    assert m["总净盈利"] == pytest.approx(float(eq.iloc[-1]) - 10000)
    assert m["盈利交易"] + m["亏损交易"] == 2
    assert m["最大结余亏损"] <= 0
    assert m["最大结余亏损%"] <= 0


def test_profit_factor_and_expectancy():
    idx = pd.bdate_range("2026-01-01", periods=6)
    eq = pd.Series([10000, 10100, 10100, 10050, 10050, 10050], index=idx)
    pos = pd.Series([0, 1, 1, 0, 0, 0], index=idx, dtype=float)
    # 模拟一段：pos=1 时净值从 10050 起……
    m = mt5_metrics(eq, pos, tf="D1", initial=10000)
    assert m["交易总计"] >= 1
    assert m["预期收益"] == pytest.approx(m["总净盈利"] / m["交易总计"])


def test_sharpe_finite_with_variation():
    eq, pos = _curve(60)
    m = mt5_metrics(eq, pos, tf="D1")
    assert m["夏普比率"] == m["夏普比率"]


def test_lr_correlation_range():
    eq, pos = _curve(80, seed=7)
    m = mt5_metrics(eq, pos, tf="D1")
    if m["LR 相关性"] == m["LR 相关性"]:
        assert -1.0 <= m["LR 相关性"] <= 1.0


def test_win_streak_counts():
    idx = pd.bdate_range("2026-01-01", periods=8)
    eq = pd.Series([10000, 10010, 10010, 10020, 10020, 10010, 10010, 10000], index=idx)
    pos = pd.Series([0, 1, 1, 0, 1, 1, 0, 0], index=idx, dtype=float)
    m = mt5_metrics(eq, pos, tf="D1", initial=10000)
    assert m["最大连胜(笔)"] + m["最大连败(笔)"] <= m["交易总计"]


def test_metrics_short_series_safe():
    idx = pd.bdate_range("2026-01-01", periods=2)
    m = mt5_metrics(pd.Series([10000, 10001], index=idx),
                    pd.Series([0, 0], index=idx), tf="D1")
    assert m["交易总计"] == 0
    assert m["总净盈利"] != m["总净盈利"]   # NaN


def test_breakdown_keys():
    eq, pos = _curve(80)
    b = period_breakdown(eq, pos, 10000)
    assert set(b) == {"每小时", "每周", "每月"}


def test_breakdown_empty():
    b = period_breakdown(pd.Series(dtype=float), pd.Series(dtype=float))
    assert all(len(v) == 0 for v in b.values())
