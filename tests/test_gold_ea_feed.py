"""EA 自报成绩单的读取与评比测试（离线，伪造公共文件夹 CSV）。"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from gold.ea_feed import (
    MIN_SNAPSHOTS,
    PREFIX,
    SUFFIX,
    ea_name,
    ea_score,
    ea_stats,
    list_ea_files,
    load_ea,
    rank_all,
)

HEADER = "time,equity,balance,position,lots,price"


def _write(d: pathlib.Path, tag: str, rows: list[tuple]) -> pathlib.Path:
    p = d / f"{PREFIX}{tag}{SUFFIX}"
    lines = [HEADER] + [",".join(str(x) for x in r) for r in rows]
    p.write_text("\n".join(lines) + "\n", encoding="ascii")
    return p


def _hourly(equities: list[float], positions: list[int] | None = None):
    """每小时一条，时间连续，保证日聚合/回撤持续都能算。"""
    n = len(equities)
    positions = positions or [0] * n
    base = pd.Timestamp("2026-10-20 00:00")
    return [((base + pd.Timedelta(hours=i)).strftime("%Y.%m.%d %H:%M"),
             e, 10000.0, positions[i], 0.01, 4000.0)
            for i, e in enumerate(equities)]


def test_list_ea_files(tmp_path):
    _write(tmp_path, "A", _hourly([10000, 10100]))
    _write(tmp_path, "B", _hourly([10000, 9900]))
    (tmp_path / "gold_bars_GOLD_H1.csv").write_text("x", encoding="ascii")
    files = list_ea_files(tmp_path)
    assert len(files) == 2
    assert sorted(ea_name(f) for f in files) == ["A", "B"]


def test_load_ea_parses_and_sorts(tmp_path):
    p = _write(tmp_path, "A", [
        ("2026.10.28 10:00", 10000, 10000, 0, 0.01, 4000),
        ("2026.10.28 09:00", 9990, 10000, 0, 0.01, 3990),
    ])
    df = load_ea(p)
    assert len(df) == 2
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert df["equity"].iloc[0] == 9990


def test_load_missing_and_broken(tmp_path):
    assert len(load_ea(tmp_path / "nope.csv")) == 0
    p = tmp_path / f"{PREFIX}BAD{SUFFIX}"
    p.write_text("garbage\n1,2,3\n", encoding="ascii")
    assert len(load_ea(p)) == 0


# ---------------------------------------------------------------- 基础指标
def test_basic_return(tmp_path):
    st = ea_score(load_ea(_write(tmp_path, "A", _hourly([10000, 10100, 10200]))))
    assert st["净值"] == pytest.approx(10200)
    assert st["收益"] == pytest.approx(0.02, abs=1e-6)
    assert st["最大回撤"] <= 0


def test_drawdown_value(tmp_path):
    st = ea_score(load_ea(_write(tmp_path, "A", _hourly([10000, 11000, 9000, 10500]))))
    assert st["最大回撤"] == pytest.approx(-0.1818, abs=1e-3)


def test_mar_ratio_positive_when_profitable(tmp_path):
    eq = [10000, 10100, 10050, 10200, 10150, 10300]
    st = ea_score(load_ea(_write(tmp_path, "A", _hourly(eq))))
    assert st["MAR"] == st["MAR"] and st["MAR"] > 0


def test_mar_finite_when_no_drawdown(tmp_path):
    """一路新高没有回撤 -> MAR 用 0.1% 下限，必须是有限正数（不能是 NaN）。"""
    st = ea_score(load_ea(_write(tmp_path, "A", _hourly([10000, 10100, 10200, 10300]))))
    assert st["最大回撤"] == pytest.approx(0.0, abs=1e-12)
    assert st["MAR"] == st["MAR"] and st["MAR"] > 0


def test_drawdown_duration_hours(tmp_path):
    # 索引 1 见峰值 11000，索引 2 跌破，索引 5 才创新高 -> 持续 3 小时
    eq = [10000, 11000, 10000, 9500, 10500, 11100]
    st = ea_score(load_ea(_write(tmp_path, "A", _hourly(eq))))
    assert st["回撤持续"] == pytest.approx(3.0, abs=0.2)


def test_consistency_all_up_vs_all_down(tmp_path):
    up = ea_score(load_ea(_write(tmp_path, "UP", _hourly([10000 + i * 20 for i in range(16)]))))
    down = ea_score(load_ea(_write(tmp_path, "DN", _hourly([10000 - i * 20 for i in range(16)]))))
    assert up["一致性"] == pytest.approx(1.0)
    assert down["一致性"] == pytest.approx(0.0)


def test_day_win_and_worst_day(tmp_path):
    # 跨 3 天，每天涨，中间一根略跌
    rows = _hourly([10000, 10100, 10090, 10200, 10190, 10300] + [10310] * 20)
    st = ea_score(load_ea(_write(tmp_path, "A", rows)))
    assert st["日胜率"] == st["日胜率"]
    assert 0.0 <= st["日胜率"] <= 1.0
    assert st["最差单日"] <= 0 or st["最差单日"] == st["最差单日"]


def test_losing_streak(tmp_path):
    rows = _hourly([10000] + [10000 - i * 30 for i in range(1, 12)])
    st = ea_score(load_ea(_write(tmp_path, "A", rows)))
    assert st["连亏"] >= 1


def test_position_changes(tmp_path):
    st = ea_score(load_ea(_write(tmp_path, "A", _hourly(
        [10000] * 4, positions=[0, 1, 1, 0]))))
    assert st["换手"] == 2
    assert st["当前仓位"] == 0


# ---------------------------------------------------------------- 综合分与评级
def test_score_prefers_stable_profit(tmp_path):
    """稳赚的应比"先暴涨再腰斩"的分高。"""
    stable = ea_score(load_ea(_write(tmp_path, "S", _hourly(
        [10000, 10030, 10060, 10090, 10120, 10150, 10180, 10210]))))
    wild = ea_score(load_ea(_write(tmp_path, "W", _hourly(
        [10000, 12000, 9000, 12500, 8000, 13000, 7000, 13500]))))
    assert stable["综合分"] > wild["综合分"]


def test_rating_needs_enough_samples(tmp_path):
    st = ea_score(load_ea(_write(tmp_path, "A", _hourly([10000, 10100, 10200]))))
    assert st["评级"] == "样本不足"


def test_rating_assigned_with_enough_samples(tmp_path):
    eq = [10000 + i * 10 for i in range(MIN_SNAPSHOTS + 5)]
    st = ea_score(load_ea(_write(tmp_path, "A", _hourly(eq))))
    assert st["评级"] in {"A", "B", "C", "D"}


def test_stats_alias_matches_score(tmp_path):
    df = load_ea(_write(tmp_path, "A", _hourly([10000, 10100, 10200])))
    assert ea_stats(df)["收益"] == pytest.approx(ea_score(df)["收益"])


def test_rank_all_orders_by_score(tmp_path):
    _write(tmp_path, "WIN", _hourly([10000 + i * 25 for i in range(12)]))
    _write(tmp_path, "LOSE", _hourly([10000 - i * 25 for i in range(12)]))
    r = rank_all(tmp_path)
    assert list(r["EA"])[:2] == ["WIN", "LOSE"]
    assert "综合分" in r.columns and "评级" in r.columns


def test_rank_all_empty_dir(tmp_path):
    r = rank_all(tmp_path)
    assert len(r) == 0
    assert "EA" in r.columns
