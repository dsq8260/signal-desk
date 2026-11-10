"""模拟盘历史时段回放的测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.symbols import spread_for  # noqa: E402
from gold.paper import (
    add_strategy,
    connect,
    list_strategies,
    load_bars,
    replay_window,
    reset_strategy,
    signal_series,
    step,
    upsert_bars,
)


def _bars(n: int = 400, drift: float = 1.5, start=1000.0, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start + np.arange(n) * drift + rng.normal(0, 0.5, n)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 0.0}, index=idx)


@pytest.fixture()
def con(tmp_path):
    c = connect(tmp_path / "paper.db")
    yield c
    c.close()


def test_signal_series_length_and_values(con):
    df = _bars()
    s = signal_series(df, "双均线", {"fast": 5, "slow": 20})
    assert len(s) == len(df)
    assert set(np.unique(s)).issubset({0.0, 1.0})


def test_signal_series_unknown_kind_is_zero():
    df = _bars(50)
    s = signal_series(df, "不存在的策略", {})
    assert (s == 0).all()


def test_reset_strategy_clears_ledger(con):
    sid = add_strategy(con, "A", "双均线", {"fast": 5, "slow": 20})
    upsert_bars(con, _bars())
    replay_window(con, load_bars(con), tf="D1")
    assert len(con.execute("SELECT 1 FROM snapshots").fetchall()) > 0
    reset_strategy(con, sid)
    assert len(con.execute("SELECT 1 FROM snapshots").fetchall()) == 0
    assert len(con.execute("SELECT 1 FROM trades").fetchall()) == 0
    assert len(list_strategies(con)) == 1        # 登记信息保留


def test_replay_writes_snapshots_within_window(con):
    add_strategy(con, "SMA", "双均线", {"fast": 5, "slow": 20})
    df = _bars(300)
    upsert_bars(con, df)
    start = df.index[100].date()
    end = df.index[200].date()
    out = replay_window(con, load_bars(con), start=start, end=end, tf="D1")
    assert len(out) == 1
    assert out.iloc[0]["bar数"] == 101
    rows = con.execute("SELECT ts FROM snapshots ORDER BY ts").fetchall()
    lo = int(pd.Timestamp(start).timestamp())
    hi = int(pd.Timestamp(end).timestamp()) + 86400
    assert all(lo <= r[0] <= hi for r in rows), "回放写出了窗口之外的数据"


def test_replay_respects_start_date(con):
    add_strategy(con, "SMA", "双均线", {"fast": 5, "slow": 20})
    df = _bars(200)
    upsert_bars(con, df)
    start = df.index[150].date()
    replay_window(con, load_bars(con), start=start, tf="D1")
    first_ts = con.execute("SELECT MIN(ts) FROM snapshots").fetchone()[0]
    assert first_ts >= int(pd.Timestamp(start).timestamp())


def test_replay_too_short_returns_empty(con):
    add_strategy(con, "SMA", "双均线")
    df = _bars(100)
    upsert_bars(con, df)
    out = replay_window(con, load_bars(con), start=df.index[98].date(), tf="D1")
    assert out.empty


def test_replay_no_lookahead_uses_previous_signal(con):
    """第 t 根的执行必须基于 t-1 的信号：整段收益应等于 shift(1) 口径。"""
    add_strategy(con, "SMA", "双均线", {"fast": 5, "slow": 20})
    df = _bars(300, drift=2.0)
    upsert_bars(con, df)
    out = replay_window(con, load_bars(con), tf="D1")
    # 手算：pos = signal.shift(1)，买入持有到末日
    sig = signal_series(df, "双均线", {"fast": 5, "slow": 20})
    pos = sig.shift(1).fillna(0.0)
    fee = spread_for("GOLD")     # 与回放同一口径（2026-11-01 起优先用实测点差）
    cash, held = 10000.0, 0.0
    for ts, price in df["close"].items():
        price = float(price)
        if pos.loc[ts] > 0.5 and held == 0:
            cash -= 1.0 * price + fee
            held = 1.0
        elif pos.loc[ts] <= 0.5 and held > 0:
            cash += held * price - fee
            held = 0.0
    expect = (cash + held * float(df["close"].iloc[-1])) / 10000.0 - 1.0
    assert out.iloc[0]["收益"] == pytest.approx(expect, abs=1e-9)


def test_replay_multiple_strategies_independent(con):
    add_strategy(con, "A", "双均线", {"fast": 5, "slow": 20})
    add_strategy(con, "B", "唐奇安通道突破", {"entry": 20, "exit_": 10})
    upsert_bars(con, _bars(300))
    out = replay_window(con, load_bars(con), tf="D1")
    assert len(out) == 2
    assert len(con.execute("SELECT DISTINCT strategy_id FROM snapshots").fetchall()) == 2


def test_replay_without_reset_accumulates(con):
    add_strategy(con, "SMA", "双均线", {"fast": 5, "slow": 20})
    upsert_bars(con, _bars(200))
    replay_window(con, load_bars(con), tf="D1")
    n1 = len(con.execute("SELECT 1 FROM snapshots").fetchall())
    replay_window(con, load_bars(con), tf="D1", reset=False)
    n2 = len(con.execute("SELECT 1 FROM snapshots").fetchall())
    assert n2 == n1          # 同一批时间戳，REPLACE 不会变多


def test_replay_then_step_continues(con):
    """回放结束后再推进一天，应接在回放末尾之后。"""
    add_strategy(con, "SMA", "双均线", {"fast": 5, "slow": 20})
    df = _bars(250)
    upsert_bars(con, df)
    replay_window(con, load_bars(con), end=df.index[200].date(), tf="D1")
    last = con.execute("SELECT MAX(ts) FROM snapshots").fetchone()[0]
    step(con, tf="D1")
    new_last = con.execute("SELECT MAX(ts) FROM snapshots").fetchone()[0]
    assert new_last >= last


def test_signal_series_drops_legacy_zhi_param():
    """干支已移除：旧账本里残留的 `_zhi` 必须被忽略，不能悄悄改变信号。"""
    df = _bars(600)
    base = signal_series(df, "唐奇安突破", {"entry": 10, "exit_": 5})
    legacy = signal_series(df, "唐奇安突破",
                          {"entry": 10, "exit_": 5, "_zhi": 11})
    assert float(base.mean()) == float(legacy.mean())


def test_signal_series_applies_regime_filter():
    """_filter 元参数也要生效。"""
    df = _bars(600)
    base = signal_series(df, "唐奇安突破", {"entry": 20, "exit_": 10})
    filt = signal_series(df, "唐奇安突破",
                         {"entry": 20, "exit_": 10, "_filter": "低波动"})
    assert float(filt.mean()) <= float(base.mean()) + 1e-12


def test_signal_series_ignores_unknown_meta():
    """未知元参数不应让策略崩掉。"""
    df = _bars(300)
    pos = signal_series(df, "唐奇安突破",
                        {"entry": 10, "exit_": 5, "_unknown": 1})
    assert len(pos) == len(df)
