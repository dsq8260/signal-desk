"""扩展进场 + 止损止盈的测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.entries_ext import (
    EXT_ENTRIES,
    alligator,
    apply_stops,
    time_varying_sharpe,
    tsmom,
)
from gold.forge import Genome, build_position, genome_space
from gold.paper import signal_series


def _bars(n=400, seed=11, drift=0.0006):
    rng = np.random.default_rng(seed)
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.012, n))
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.008,
                         "low": close * 0.992, "close": close,
                         "volume": 0.0}, index=idx)


# ---------------- 新信号 ----------------
def test_tsmom_outputs_binary_and_respects_lookback():
    df = _bars(400)
    pos = tsmom(df, n=60)
    assert set(np.unique(pos)).issubset({0.0, 1.0})
    assert pos.iloc[:60].sum() == 0        # 前 n 根没有信号


def test_tsmom_threshold_increases_selectivity():
    df = _bars(500)
    loose = tsmom(df, n=60, thr=0.0)
    strict = tsmom(df, n=60, thr=0.10)
    assert strict.sum() <= loose.sum()


def test_time_varying_sharpe_binary():
    df = _bars(500)
    pos = time_varying_sharpe(df, n=60)
    assert set(np.unique(pos)).issubset({0.0, 1.0})
    assert len(pos) == len(df)


def test_alligator_binary_and_trendlike():
    df = _bars(600)
    pos = alligator(df)
    assert set(np.unique(pos)).issubset({0.0, 1.0})
    assert 0.0 <= float(pos.mean()) <= 1.0


def test_ext_entries_registered():
    assert set(EXT_ENTRIES) == {"时序动量TSMOM", "时变夏普", "鳄鱼线"}
    for name, (fn, defaults, grid, minlen) in EXT_ENTRIES.items():
        assert callable(fn) and defaults and grid and minlen > 0


# ---------------- 止损止盈 ----------------
def test_apply_stops_noop_when_disabled():
    df = _bars(200)
    pos = pd.Series(1.0, index=df.index)
    out = apply_stops(df, pos, None, None)
    assert out.equals(pos)


def test_apply_stops_cuts_position_on_drop():
    """构造一路下跌：开仓后应被止损打掉，持仓占比显著下降。"""
    n = 120
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = np.linspace(100, 60, n)          # 单调下跌 40%
    df = pd.DataFrame({"open": close, "high": close * 1.001,
                       "low": close * 0.999, "close": close,
                       "volume": 0.0}, index=idx)
    pos = pd.Series(1.0, index=idx)
    out = apply_stops(df, pos, sl=0.02, tp=None)
    assert float(out.mean()) < float(pos.mean())


def test_apply_stops_takeprofit_caps_upside():
    n = 120
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = np.linspace(100, 200, n)         # 单调上涨
    df = pd.DataFrame({"open": close, "high": close * 1.001,
                       "low": close * 0.999, "close": close,
                       "volume": 0.0}, index=idx)
    pos = pd.Series(1.0, index=idx)
    out = apply_stops(df, pos, sl=None, tp=0.03)
    assert float(out.mean()) < float(pos.mean())


def test_apply_stops_never_uses_future_bars():
    """只应减少持仓，不应产生原本没有的持仓。"""
    df = _bars(300)
    base = pd.Series(0.0, index=df.index)
    base.iloc[50:80] = 1.0
    out = apply_stops(df, base, sl=0.01, tp=0.02)
    assert float(out.sum()) <= float(base.sum()) + 1e-9


# ---------------- 基因空间 ----------------
def test_genome_space_expanded():
    old = genome_space(with_rules=False)
    new = genome_space(with_rules=True)
    assert len(new) == len(old) * 4
    assert any(g.entry == "时序动量TSMOM" for g in new)
    assert any(g.sl is not None for g in new)


def test_genome_label_contains_risk():
    g = Genome("时序动量TSMOM", {"n": 60}, "无过滤", False, 0.02, 0.05)
    lab = g.label()
    assert "SL2%" in lab and "TP5%" in lab


def test_build_position_with_new_entry_and_stops():
    from gold.factors import build_features

    df = _bars(400)
    feat = build_features(df)
    g = Genome("时序动量TSMOM", {"n": 60}, "无过滤", False, 0.02, None)
    pos = build_position(df, feat, g)
    plain = build_position(df, feat,
                           Genome("时序动量TSMOM", {"n": 60}, "无过滤"))
    assert float(pos.mean()) <= float(plain.mean()) + 1e-9


# ---------------- 前向引擎支持 ----------------
def test_paper_signal_series_applies_stops():
    df = _bars(300)
    base = signal_series(df, "唐奇安突破", {"entry": 10, "exit_": 5})
    capped = signal_series(df, "唐奇安突破",
                           {"entry": 10, "exit_": 5, "_sl": 0.01})
    assert float(capped.mean()) <= float(base.mean()) + 1e-9
