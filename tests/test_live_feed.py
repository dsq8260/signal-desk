"""实盘监测模块测试：解析成绩单、算概况、偏差比对。"""

from __future__ import annotations

import pandas as pd
import pytest

from gold import live_feed as L

CSV = """time,equity,balance,position,lots,price
2026-10-01 10:00,10000,10000,0,0,0
2026-10-02 10:00,10050,10000,1,0.01,4000
2026-10-03 10:00,9980,10000,1,0.01,4005
2026-10-04 10:00,10120,10000,0,0,0
"""


def test_parse_basic():
    df = L.parse_equity_csv(CSV)
    assert len(df) == 4
    assert list(df.columns) == L.COLS
    assert pd.api.types.is_datetime64_any_dtype(df["time"])


def test_parse_empty_and_broken():
    assert len(L.parse_equity_csv("")) == 0
    assert len(L.parse_equity_csv("   \n")) == 0
    assert len(L.parse_equity_csv("no,header,here\n1,2,3")) == 0


def test_stats_basic():
    st = L.live_stats(L.parse_equity_csv(CSV))
    assert st.bars == 4
    assert st.equity_end == 10120
    assert st.days == pytest.approx(3.0, abs=0.1)
    assert st.in_position_pct == pytest.approx(0.5)
    assert st.trades_hint == 1
    assert st.max_dd < 0


def test_stats_empty_warns():
    st = L.live_stats(pd.DataFrame())
    assert st.bars == 0 and st.warnings


def test_deviation_rows():
    st = L.live_stats(L.parse_equity_csv(CSV))
    rows = L.deviation({"回撤": -0.02, "天数": 100, "回合": 50}, st)
    names = [r["指标"] for r in rows]
    assert "最大回撤" in names and "样本天数" in names
    assert any("样本" in r["结论"] for r in rows)


def test_deviation_turn_mismatch_flagged():
    st = L.live_stats(L.parse_equity_csv(CSV))
    rows = L.deviation({"回合": 100}, st)
    assert rows[0]["结论"] == "成交率异常"


def test_read_equity_csv_never_raises():
    class Bad:
        def call(self, *a, **k):
            raise RuntimeError("boom")

    assert L.read_equity_csv(Bad(), "x") == ""
