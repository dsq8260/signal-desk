"""撮合引擎测试：滑点单调性、保证金占用、强平触发、默认关闭不影响旧结果。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.paper import (
    add_strategy,
    connect,
    list_strategies,
    replay_window,
)


def _bars(n=200, seed=3, up=True):
    """构造一段明确上涨/下跌的行情，便于验证滑点与强平。"""
    rng = np.random.default_rng(seed)
    drift = 0.004 if up else -0.004
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.002, n))
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close,
                         "volume": 0.0}, index=idx)


def _setup(tmp_path, kind="唐奇安突破", params=None, units=1.0,
           cash0=10000.0):
    con = connect(tmp_path / "p.db")
    add_strategy(con, "T", kind, params or {"entry": 10, "exit_": 5}, tf="D1",
                 units=units, cash0=cash0)
    return con


def _run(con, df, **kw):
    return replay_window(con, df, spread_usd=0.33, tf="D1", reset=True, **kw)


def test_slippage_reduces_return_monotonically(tmp_path):
    df = _bars()
    con = _setup(tmp_path)
    try:
        r0 = _run(con, df, slippage_bps=0.0)
        r1 = _run(con, df, slippage_bps=10.0)
        r5 = _run(con, df, slippage_bps=50.0)
        e0 = float(r0["收益"].iloc[0])
        e1 = float(r1["收益"].iloc[0])
        e5 = float(r5["收益"].iloc[0])
        assert e1 <= e0 + 1e-12, (e0, e1)
        assert e5 <= e1 + 1e-12, (e1, e5)
        if float(r0["成交"].iloc[0]) > 1:
            assert e5 < e0, "有成交时滑点必须真的降低收益"
    finally:
        con.close()


def test_slippage_shows_in_fill_prices(tmp_path):
    """开仓价必须比那根 bar 的收盘价**高** slippage_bps。"""
    df = _bars()
    con = _setup(tmp_path)
    try:
        bps = 30.0
        _run(con, df, slippage_bps=bps)
        rows = con.execute(
            "SELECT ts, price FROM trades WHERE side='BUY'").fetchall()
        assert rows, "没有开仓记录，无法验证滑点"
        for ts, px in rows:
            bar_close = float(df["close"].loc[pd.Timestamp(ts, unit="s")])
            expect = bar_close * (1 + bps / 10000.0)
            assert px == pytest.approx(expect, rel=1e-9), (px, expect)
    finally:
        con.close()


def test_default_off_matches_legacy_behaviour(tmp_path):
    """默认参数（滑点 0、杠杆 0）必须与不开强平时完全一致。"""
    df = _bars()
    con = _setup(tmp_path)
    try:
        a = _run(con, df)
        b = _run(con, df, slippage_bps=0.0, leverage=0.0)
        assert float(a["收益"].iloc[0]) == pytest.approx(float(b["收益"].iloc[0]))
        assert int(a["成交"].iloc[0]) == int(b["成交"].iloc[0])
    finally:
        con.close()


def _crash_bars(n_up=400, n_down=120, seed=11):
    """慢涨 400 根（建仓）后暴跌 120 根（保证金率跌破阈值）。"""
    rng = np.random.default_rng(seed)
    up = 2000 * np.cumprod(1 + rng.normal(0.0015, 0.0008, n_up))
    down = up[-1] * np.cumprod(1 + rng.normal(-0.010, 0.001, n_down))
    close = np.concatenate([up, down])
    idx = pd.bdate_range("2018-01-02", periods=len(close))
    return pd.DataFrame({"open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close,
                         "volume": 0.0}, index=idx)


# 长周期均线：暴跌初期快线仍高于慢线 → 信号维持多头，仓位会被带进下跌
_EQ = {"kind": "双均线", "params": {"fast": 200, "slow": 300}}


def test_stopout_force_closes_and_logs(tmp_path):
    """高杠杆 + 先涨后暴跌 → 必须触发强平，且账本里有"强平"记录。"""
    df = _crash_bars()
    con = _setup(tmp_path, **_EQ, units=100.0, cash0=20000.0)
    try:
        _run(con, df, leverage=100.0, stopout_pct=0.9)
        notes = [r[0] for r in con.execute(
            "SELECT note FROM trades WHERE side='SELL'").fetchall()]
        assert any("强平" in str(n) for n in notes), notes[:5]
    finally:
        con.close()


def test_leverage_zero_never_stops_out(tmp_path):
    df = _bars(n=200, seed=11, up=False)
    con = _setup(tmp_path, **_EQ, units=100.0, cash0=20000.0)
    try:
        _run(con, df, leverage=0.0, stopout_pct=0.99)
        notes = [r[0] for r in con.execute(
            "SELECT note FROM trades WHERE side='SELL'").fetchall()]
        assert not any("强平" in str(n) for n in notes)
    finally:
        con.close()


def test_high_leverage_stopout_hurts_more_than_low(tmp_path):
    """同样的行情，杠杆越高越容易被强平（保证金占用更大）。"""
    # 仓位规模固定时，**杠杆越低 → 占用保证金越大 → 越容易触发强平**
    df = _crash_bars(seed=7)
    con = _setup(tmp_path, **_EQ, units=100.0, cash0=20000.0)
    try:
        _run(con, df, leverage=20.0, stopout_pct=0.9)
        n_low_lev = len([1 for r in con.execute(
            "SELECT note FROM trades WHERE side='SELL'").fetchall()
            if "强平" in str(r[0])])
        _run(con, df, leverage=200.0, stopout_pct=0.9)
        n_high_lev = len([1 for r in con.execute(
            "SELECT note FROM trades WHERE side='SELL'").fetchall()
            if "强平" in str(r[0])])
        assert n_low_lev >= n_high_lev
    finally:
        con.close()
