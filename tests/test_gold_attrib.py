"""按 magic 归因的测试（离线伪造 mt5_deals.csv）。"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from gold.attrib import MIN_DEALS, magic_names, magic_stats, save_magic_names

DEAL = ("time,ticket,symbol,type,entry,volume,price,profit,swap,"
        "commission,magic,comment")


def _deals(d: pathlib.Path, rows: list[tuple]) -> None:
    lines = [DEAL] + [",".join(str(x) for x in r) for r in rows]
    (d / "mt5_deals.csv").write_text("\n".join(lines) + "\n", encoding="ascii")


def _row(t: str, ticket: int, profit: float, magic: int,
         swap: float = 0.0, comm: float = -0.5) -> tuple:
    return (t, ticket, "GOLD#", 1, 0, 0.08, 4000.0, profit, swap, comm, magic, "")


def test_empty_returns_columns(tmp_path):
    df = magic_stats(tmp_path)
    assert len(df) == 0
    assert "magic" in df.columns and "综合分" in df.columns


def test_net_includes_swap_and_commission(tmp_path):
    _deals(tmp_path, [
        _row("2026.10.27 10:00", 1, 100.0, 111, swap=-2.0, comm=-1.0),
        _row("2026.10.27 11:00", 2, 50.0, 111, swap=-1.0, comm=-1.0),
    ])
    g = magic_stats(tmp_path)
    # 100-2-1 + 50-1-1 = 145
    assert g.iloc[0]["净收益"] == pytest.approx(145.0)


def test_win_rate_and_profit_factor(tmp_path):
    _deals(tmp_path, [
        _row("2026.10.27 10:00", 1, 100.0, 111, swap=0, comm=0),
        _row("2026.10.27 11:00", 2, -50.0, 111, swap=0, comm=0),
        _row("2026.10.27 12:00", 3, 30.0, 111, swap=0, comm=0),
        _row("2026.10.27 13:00", 4, -20.0, 111, swap=0, comm=0),
    ])
    g = magic_stats(tmp_path)
    row = g.iloc[0]
    assert row["胜率"] == pytest.approx(0.5)
    assert row["盈亏比"] == pytest.approx(130.0 / 70.0)


def test_groups_by_magic_and_sorts(tmp_path):
    _deals(tmp_path, [
        _row("2026.10.27 10:00", 1, 100.0, 111),
        _row("2026.10.27 11:00", 2, -80.0, 222),
    ])
    g = magic_stats(tmp_path)
    assert len(g) == 2
    assert int(g.iloc[0]["magic"]) == 111
    assert g.iloc[0]["净收益"] > g.iloc[1]["净收益"]


def test_drawdown_from_curve(tmp_path):
    _deals(tmp_path, [
        _row("2026.10.27 10:00", 1, 100.0, 111, swap=0, comm=0),
        _row("2026.10.27 11:00", 2, -200.0, 111, swap=0, comm=0),
        _row("2026.10.27 12:00", 3, 50.0, 111, swap=0, comm=0),
    ])
    row = magic_stats(tmp_path).iloc[0]
    assert row["最大回撤"] == pytest.approx(-200.0)


def test_rating_needs_min_deals(tmp_path):
    _deals(tmp_path, [_row("2026.10.27 10:00", 1, 100.0, 111)])
    assert magic_stats(tmp_path).iloc[0]["评级"] == "样本不足"


def test_rating_assigned_with_enough_deals(tmp_path):
    rows = [_row(f"2026.10.2{7 + i // 5} 1{i % 5}:00", i, 20.0, 111,
                 swap=0, comm=0) for i in range(MIN_DEALS + 3)]
    _deals(tmp_path, rows)
    row = magic_stats(tmp_path).iloc[0]
    assert row["评级"] in {"A", "B", "C", "D"}


def test_losing_streak(tmp_path):
    _deals(tmp_path, [
        _row("2026.10.27 10:00", 1, -10.0, 111, swap=0, comm=0),
        _row("2026.10.27 11:00", 2, -10.0, 111, swap=0, comm=0),
        _row("2026.10.27 12:00", 3, -10.0, 111, swap=0, comm=0),
        _row("2026.10.27 13:00", 4, 50.0, 111, swap=0, comm=0),
    ])
    assert magic_stats(tmp_path).iloc[0]["连亏"] == 3


def test_magic_names_roundtrip(tmp_path):
    p = tmp_path / "names.json"
    save_magic_names({20261001: "双均线 H1"}, p)
    got = magic_names(p)
    assert got[20261001] == "双均线 H1"


def test_magic_names_missing(tmp_path):
    assert magic_names(tmp_path / "nope.json") == {}
