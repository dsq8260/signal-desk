"""backtrader 交叉验证的测试（离线，合成数据）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.bt_verify import HAS_BT, bt_engine, compare, our_engine


def _bars(n=1200, seed=5, drift=0.0004):
    rng = np.random.default_rng(seed)
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.004, n))
    idx = pd.bdate_range("2021-01-04", periods=n, freq="h")
    return pd.DataFrame({"open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close,
                         "volume": 0.0}, index=idx)


def test_has_backtrader():
    assert HAS_BT is True, "backtrader 未安装"


def test_our_engine_signals():
    df = _bars()
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[100:300] = 1.0
    r = our_engine(df, sig, 0.33)
    assert r["成交变动"] == 2          # 一次进、一次出
    assert r["引擎"] == "本系统"


def test_bt_engine_matches_position_count():
    df = _bars()
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[100:300] = 1.0
    sig.iloc[500:700] = 1.0
    a = our_engine(df, sig, 0.33)
    b = bt_engine(df, sig, 0.33)
    assert b["成交变动"] == a["成交变动"], (a, b)


def test_bt_engine_flat_signal_no_trades():
    df = _bars()
    sig = pd.Series(0.0, index=df.index)
    b = bt_engine(df, sig, 0.33)
    assert b["成交变动"] == 0
    assert b["期末净值"] == pytest.approx(10000.0)


def test_two_engines_agree_on_return():
    """同一仓位序列下，两个引擎的收益应当接近（差 < 5 个百分点）。"""
    df = _bars(1500)
    sig = pd.Series(0.0, index=df.index)
    for a, b in ((100, 400), (700, 1000), (1100, 1400)):
        sig.iloc[a:b] = 1.0
    x = our_engine(df, sig, 0.33)
    y = bt_engine(df, sig, 0.33)
    assert abs(x["累计收益"] - y["累计收益"]) < 0.05


def test_compare_returns_structure():
    df = _bars(800)
    r = compare(df, "双均线", {"fast": 10, "slow": 40}, 0.33)
    assert set(r) >= {"策略", "本系统", "backtrader", "笔数差", "收益差"}
    assert isinstance(r["笔数差"], int)
