"""新增 4 个策略的回归测试（离线合成数据）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.strategies import (
    STRATEGIES,
    apply_strategy,
    atr_trailing_signal,
    bollinger_reversion_signal,
    macd_momentum_signal,
    rsi_reversion_signal,
)


def _bars(n: int = 400, seed: int = 11, drift: float = 0.0008) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.01, size=n))
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({
        "open": close,
        "high": close * (1 + rng.uniform(0, 0.005, n)),
        "low": close * (1 - rng.uniform(0, 0.005, n)),
        "close": close, "volume": 0.0}, index=idx)


def _sawtooth(n: int = 300) -> pd.DataFrame:
    """锯齿：先涨后跌循环，用来触发均值回归类信号。"""
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = 2000 + 100 * np.sin(np.arange(n) / 12.0)
    return pd.DataFrame({"open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close,
                         "volume": 0.0}, index=idx)


def test_registry_has_eight_strategies():
    assert len(STRATEGIES) == 8
    for name, spec in STRATEGIES.items():
        assert callable(spec["fn"]), name
        assert isinstance(spec["params"], dict), name
        assert spec["desc"], name


@pytest.mark.parametrize("name", [
    "ATR跟踪止损", "布林带回归", "MACD动量", "RSI反转"])
def test_new_strategies_are_offline_safe(name):
    """新策略必须能算且只输出 {0,1}。"""
    df = _bars()
    pos = apply_strategy(df, name)
    assert len(pos) == len(df)
    assert set(np.unique(pos)).issubset({0.0, 1.0})


@pytest.mark.parametrize("fn", [
    atr_trailing_signal, bollinger_reversion_signal,
    macd_momentum_signal, rsi_reversion_signal])
def test_no_lookahead_on_truncation(fn):
    """截断尾部不改变前面的信号（无未来函数）。"""
    df = _bars(500)
    full = pd.Series(fn(df), index=df.index)
    part = pd.Series(fn(df.iloc[:350]), index=df.iloc[:350].index)
    assert np.allclose(full.iloc[:350].to_numpy(), part.to_numpy())


def test_atr_trailing_exits_after_big_drop():
    """ATR 跟踪：大涨后急跌应触发平仓。"""
    up = _bars(120, drift=0.004, seed=5)
    crash_idx = pd.bdate_range(up.index[-1] + pd.Timedelta(days=1), periods=40)
    base = float(up["close"].iloc[-1])
    crash_close = np.linspace(base, base * 0.55, 40)
    crash = pd.DataFrame({"open": crash_close, "high": crash_close,
                          "low": crash_close, "close": crash_close,
                          "volume": 0.0}, index=crash_idx)
    pos = atr_trailing_signal(pd.concat([up, crash]), n=14, mult=2.0, entry_ma=10)
    assert pos.iloc[-1] == 0.0


def test_bollinger_buys_below_lower_band():
    df = _sawtooth(300)
    pos = bollinger_reversion_signal(df, n=20, k=2.0)
    assert pos.sum() > 0            # 至少进过一次
    assert pos.iloc[-1] in (0.0, 1.0)


def test_macd_long_in_persistent_uptrend():
    n = 300
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = 2000 + np.arange(n) * 2.0
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": 0.0}, index=idx)
    assert macd_momentum_signal(df).iloc[-1] == 1.0


def test_rsi_trades_in_oscillating_market():
    df = _sawtooth(400)
    pos = rsi_reversion_signal(df, n=14, buy=30, sell=55)
    assert pos.sum() > 0


def test_rsi_stays_flat_in_strong_uptrend():
    """单边上涨 RSI 长期高位，不应在低位买入。"""
    n = 300
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = 2000 + np.arange(n) * 3.0
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": 0.0}, index=idx)
    assert rsi_reversion_signal(df).sum() == 0
