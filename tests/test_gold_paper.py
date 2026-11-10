"""多策略模拟盘引擎测试（离线，不碰网络/终端）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.paper import (
    INIT_CASH,
    add_strategy,
    connect,
    dashboard,
    list_strategies,
    load_bars,
    remove_strategy,
    seed_defaults,
    step,
    upsert_bars,
)


def _df(n: int = 300, slope: float = 1.0, start=1000.0) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = start + np.arange(n) * slope
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 0.0}, index=idx)


def _down(n: int = 300, start=2000.0) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = start - np.arange(n) * 1.0
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 0.0}, index=idx)


@pytest.fixture()
def con(tmp_path):
    c = connect(tmp_path / "paper.db")
    yield c
    c.close()


# ---------------------------------------------------------------- 库结构
def test_connect_creates_tables(con):
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"bars", "strategies", "trades", "snapshots"} <= names


def test_add_and_list_strategy(con):
    sid = add_strategy(con, "测试A", "双均线", {"fast": 5, "slow": 10})
    rows = list_strategies(con)
    assert len(rows) == 1
    assert int(rows.iloc[0]["id"]) == sid
    assert rows.iloc[0]["name"] == "测试A"
    assert rows.iloc[0]["units"] == pytest.approx(1.0)


def test_remove_strategy_cascades(con):
    sid = add_strategy(con, "A", "双均线")
    upsert_bars(con, _df(60))
    step(con, tf="D1")
    assert len(con.execute("SELECT 1 FROM snapshots").fetchall()) == 1
    remove_strategy(con, sid)
    assert len(list_strategies(con)) == 0
    assert len(con.execute("SELECT 1 FROM snapshots").fetchall()) == 0


def test_seed_defaults_is_idempotent(con):
    n1 = seed_defaults(con)
    n2 = seed_defaults(con)
    assert n1 == 3 and n2 == 0
    assert len(list_strategies(con)) == 3


# ---------------------------------------------------------------- 行情存取
def test_upsert_bars_idempotent(con):
    df = _df(50)
    a = upsert_bars(con, df, "D1")
    b = upsert_bars(con, df, "D1")
    assert a == b == 50


def test_load_bars_sorted_datetime_index(con):
    upsert_bars(con, _df(30).iloc[::-1], "D1")     # 故意倒序写入
    out = load_bars(con, "D1")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing
    assert list(out.columns) == ["open", "high", "low", "close"]


# ---------------------------------------------------------------- 推进逻辑
def test_step_buys_in_uptrend(con):
    add_strategy(con, "SMA", "双均线", {"fast": 5, "slow": 20})
    upsert_bars(con, _df(200, slope=2.0))
    res = step(con, spread_usd=0.33)
    assert len(res) == 1
    trades = con.execute("SELECT side FROM trades").fetchall()
    assert [t[0] for t in trades] == ["BUY"]
    snap = con.execute("SELECT position,units,equity FROM snapshots").fetchone()
    assert snap[0] == 1 and snap[1] == pytest.approx(1.0)
    assert snap[2] < INIT_CASH                  # 买入付了钱 + 点差


def test_step_is_idempotent_on_same_bar(con):
    add_strategy(con, "SMA", "双均线", {"fast": 5, "slow": 20})
    upsert_bars(con, _df(200, slope=2.0))
    step(con)
    step(con)                                   # 同一根 bar 再推进
    assert len(con.execute("SELECT 1 FROM trades").fetchall()) == 1


def test_step_sells_when_trend_breaks(con):
    add_strategy(con, "SMA", "双均线", {"fast": 5, "slow": 20})
    up = _df(150, slope=2.0, start=1000)
    upsert_bars(con, up)
    step(con)
    assert con.execute("SELECT position FROM snapshots").fetchone()[0] == 1

    # 追加一段下跌
    idx = pd.bdate_range(up.index[-1] + pd.Timedelta(days=1), periods=120)
    close = up["close"].iloc[-1] - np.arange(120) * 3.0
    down = pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 0.0}, index=idx)
    upsert_bars(con, down)
    step(con)
    sides = [r[0] for r in con.execute("SELECT side FROM trades ORDER BY id")]
    assert sides == ["BUY", "SELL"]
    assert con.execute("SELECT position FROM snapshots ORDER BY ts DESC").fetchone()[0] == 0


def test_step_no_trade_in_downtrend(con):
    add_strategy(con, "SMA", "双均线", {"fast": 5, "slow": 20})
    upsert_bars(con, _down(200))
    step(con)
    assert len(con.execute("SELECT 1 FROM trades").fetchall()) == 0
    assert con.execute("SELECT position FROM snapshots").fetchone()[0] == 0


def test_multiple_strategies_independent(con):
    add_strategy(con, "快", "双均线", {"fast": 3, "slow": 8})
    add_strategy(con, "唐", "唐奇安通道突破", {"entry": 20, "exit_": 10})
    upsert_bars(con, _df(250, slope=2.0))
    res = step(con)
    assert len(res) == 2
    assert len(con.execute("SELECT DISTINCT strategy_id FROM snapshots").fetchall()) == 2


def test_step_needs_min_bars(con):
    add_strategy(con, "A", "双均线")
    upsert_bars(con, _df(2))
    assert step(con) == []


# ---------------------------------------------------------------- 看板
def test_dashboard_columns_and_values(con):
    add_strategy(con, "SMA", "双均线", {"fast": 5, "slow": 20})
    upsert_bars(con, _df(200, slope=2.0))
    step(con)
    d = dashboard(con)
    assert len(d) == 1
    for col in ("策略", "状态", "持仓", "成交", "净值", "当日盈亏",
                "累计收益", "最大回撤", "最近更新"):
        assert col in d.columns
    assert d.iloc[0]["状态"] == "持仓中"
    assert d.iloc[0]["成交"] == 1


def test_dashboard_waiting_state(con):
    add_strategy(con, "A", "双均线")
    d = dashboard(con)
    assert d.iloc[0]["状态"] == "等待第一根 bar"
    assert d.iloc[0]["净值"] == pytest.approx(INIT_CASH)


def test_dashboard_drawdown_negative_or_zero(con):
    add_strategy(con, "SMA", "双均线", {"fast": 5, "slow": 20})
    upsert_bars(con, _df(200, slope=2.0))
    step(con)
    assert dashboard(con).iloc[0]["最大回撤"] <= 1e-12
