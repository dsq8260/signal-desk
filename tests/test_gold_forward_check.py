"""前向验证（C）测试：回测 vs 前向对照。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.forward_check import compare
from gold.paper import add_strategy, connect, load_bars, upsert_bars


def _bars(n=400, drift=1.2, seed=6):
    rng = np.random.default_rng(seed)
    close = 1000 + np.arange(n) * drift + rng.normal(0, 1.0, n)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 0.0}, index=idx)


def _fill_snapshots(con, sid: int, df: pd.DataFrame, sl=0.01) -> None:
    """造一段人工快照（模拟系统逐日推进的结果）。"""
    for ts, row in df.iterrows():
        con.execute(
            "INSERT OR REPLACE INTO snapshots(strategy_id,ts,position,units,cash,equity,price)"
            " VALUES(?,?,?,?,?,?,?)",
            (sid, int(ts.timestamp()), 0, 0.0, 10000.0,
             10000.0 * (1 + sl) ** (df.index.get_loc(ts)), float(row["close"])))
    con.commit()


@pytest.fixture()
def con(tmp_path):
    c = connect(tmp_path / "p.db")
    yield c
    c.close()


def test_compare_no_strategies(con):
    out = compare(con, _bars())
    assert out.table.empty
    assert "还没有登记" in out.note


def test_compare_needs_enough_snapshots(con):
    sid = add_strategy(con, "S", "双均线", {"fast": 5, "slow": 20})
    df = _bars()
    upsert_bars(con, df)
    for ts in df.index[:3]:
        con.execute(
            "INSERT OR REPLACE INTO snapshots(strategy_id,ts,position,units,cash,equity,price)"
            " VALUES(?,?,?,?,?,?,?)",
            (sid, int(ts.timestamp()), 0, 0.0, 1e4, 1e4, 1000.0))
    con.commit()
    out = compare(con, load_bars(con))
    assert out.table.empty
    assert "至少 5 个快照" in out.note


def test_compare_produces_rows(con):
    sid = add_strategy(con, "SMA", "双均线", {"fast": 5, "slow": 20})
    df = _bars()
    upsert_bars(con, df)
    _fill_snapshots(con, sid, df)
    out = compare(con, load_bars(con))
    assert len(out.table) >= 1
    cols = set(out.table.columns)
    assert {"回测夏普", "前向夏普", "衰减率", "回测收益", "前向收益"} <= cols


def test_compare_decay_ratio_math(con):
    sid = add_strategy(con, "SMA", "双均线", {"fast": 5, "slow": 20})
    df = _bars()
    upsert_bars(con, df)
    _fill_snapshots(con, sid, df)
    out = compare(con, load_bars(con))
    row = out.table.iloc[0]
    bt, fw = float(row["回测夏普"]), float(row["前向夏普"])
    if bt == bt and abs(bt) > 1e-9:
        assert float(row["衰减率"]) == pytest.approx(fw / bt, rel=1e-6)


def test_compare_sorted_by_forward_sharpe(con):
    s1 = add_strategy(con, "A", "双均线", {"fast": 5, "slow": 20})
    s2 = add_strategy(con, "B", "双均线", {"fast": 10, "slow": 40})
    df = _bars()
    upsert_bars(con, df)
    _fill_snapshots(con, s1, df, sl=0.02)
    _fill_snapshots(con, s2, df, sl=-0.01)
    out = compare(con, load_bars(con))
    v = out.table["前向夏普"].to_numpy()
    assert np.all(np.diff(v) <= 1e-9)


def test_compare_ignores_flat_forward(con):
    """快照全平（净值不变）时夏普为 NaN，但不能崩。"""
    sid = add_strategy(con, "FLAT", "双均线", {"fast": 5, "slow": 20})
    df = _bars()
    upsert_bars(con, df)
    for ts in df.index[:10]:
        con.execute(
            "INSERT OR REPLACE INTO snapshots(strategy_id,ts,position,units,cash,equity,price)"
            " VALUES(?,?,?,?,?,?,?)",
            (sid, int(ts.timestamp()), 0, 0.0, 1e4, 1e4, 1000.0))
    con.commit()
    out = compare(con, load_bars(con))
    assert len(out.table) == 1
    assert out.table.iloc[0]["前向收益"] == pytest.approx(0.0)
