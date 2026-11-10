"""数据桥测试（离线，用临时目录伪造 MT5 公共文件夹里的 CSV）。"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from gold.bridge import (
    TFS,
    available,
    bars,
    bars_any,
    latest_tick,
    status,
)

HEADER = "time,open,high,low,close,tick_volume"


def _mk_bars(d: pathlib.Path, tf: str, n: int = 5, start: int = 1_700_000_000) -> None:
    lines = [HEADER]
    for i in range(n):
        t = start + i * 60
        px = 4000.0 + i
        lines.append(f"{t},{px},{px+1},{px-1},{px+0.5},{100+i}")
    (d / f"gold_bars_{tf}.csv").write_text("\n".join(lines) + "\n", encoding="ascii")


def _mk_ticks(d: pathlib.Path, rows: list[str]) -> None:
    (d / "gold_tick.csv").write_text("\n".join(rows) + "\n", encoding="ascii")


# ---------------------------------------------------------------- 目录
def test_common_dir_points_to_metaquotes(tmp_path):
    from gold.bridge import common_dir

    p = common_dir()
    assert "MetaQuotes" in str(p)
    assert p.name == "Files"


def test_available_false_on_empty_dir(tmp_path):
    assert available(tmp_path) is False


def test_available_true_with_bars(tmp_path):
    _mk_bars(tmp_path, "H1")
    assert available(tmp_path) is True


def test_available_true_with_tick(tmp_path):
    _mk_ticks(tmp_path, ["1700000000000,4000.10,4000.40"])
    assert available(tmp_path) is True


# ---------------------------------------------------------------- tick
def test_latest_tick_parses_last_line(tmp_path):
    _mk_ticks(tmp_path, [
        "1700000000000,4000.10,4000.40",
        "1700000001000,4001.20,4001.50",
    ])
    t = latest_tick(tmp_path)
    assert t is not None
    assert t["bid"] == pytest.approx(4001.20)
    assert t["ask"] == pytest.approx(4001.50)
    assert t["time_msc"] == 1700000001000


def test_latest_tick_skips_broken_lines(tmp_path):
    _mk_ticks(tmp_path, [
        "1700000000000,4000.10,4000.40",
        "garbage line",
        "1700000002000,bad,data",
    ])
    t = latest_tick(tmp_path)
    assert t is not None and t["bid"] == pytest.approx(4000.10)


def test_latest_tick_none_when_missing(tmp_path):
    assert latest_tick(tmp_path) is None


def test_latest_tick_handles_big_file(tmp_path):
    rows = [f"{1700000000000 + i*1000},{4000 + i*0.01:.2f},{4000.3 + i*0.01:.2f}"
            for i in range(2000)]
    _mk_ticks(tmp_path, rows)
    t = latest_tick(tmp_path)
    assert t is not None
    assert t["bid"] == pytest.approx(4000 + 1999 * 0.01, abs=1e-6)


# ---------------------------------------------------------------- bars
def test_bars_parses_and_indexes_by_time(tmp_path):
    _mk_bars(tmp_path, "H1", n=5)
    df = bars("H1", tmp_path)
    assert len(df) == 5
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[0] == pytest.approx(4000.5)


def test_bars_missing_file_returns_empty(tmp_path):
    df = bars("H1", tmp_path)
    assert len(df) == 0
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_bars_deduplicates_timestamps(tmp_path):
    lines = [HEADER,
             "1700000000,4000,4001,3999,4000.5,10",
             "1700000000,4000,4001,3999,4099.9,11"]      # 同一时间戳，后者应生效
    (tmp_path / "gold_bars_H1.csv").write_text("\n".join(lines) + "\n", encoding="ascii")
    df = bars("H1", tmp_path)
    assert len(df) == 1
    assert df["close"].iloc[0] == pytest.approx(4099.9)


def test_bars_missing_columns_returns_empty(tmp_path):
    (tmp_path / "gold_bars_H1.csv").write_text("time,foo\n1700000000,1\n",
                                               encoding="ascii")
    assert len(bars("H1", tmp_path)) == 0


def test_bars_max_bars_truncates(tmp_path):
    _mk_bars(tmp_path, "M5", n=50)
    assert len(bars("M5", tmp_path, max_bars=10)) == 10


def test_bars_any_unknown_tf_empty(tmp_path):
    _mk_bars(tmp_path, "H1")
    assert len(bars_any("X9", tmp_path)) == 0
    assert len(bars_any("H1", tmp_path)) == 5


# ---------------------------------------------------------------- 状态
def test_status_reports_files(tmp_path):
    _mk_bars(tmp_path, "D1")
    _mk_ticks(tmp_path, ["1700000000000,4000.10,4000.40"])
    s = status(tmp_path)
    assert s["dir_exists"] is True
    assert s["tick_file"] is True
    assert s["bars"]["D1"] is True
    assert s["bars"]["M1"] is False
    assert s["any"] is True
    assert s["tick"]["bid"] == pytest.approx(4000.10)


def test_status_all_tfs_listed(tmp_path):
    s = status(tmp_path)
    assert set(s["bars"]) == set(TFS)
    assert len(TFS) == 8


def test_bars_merges_paged_files(tmp_path):
    """_p1 分页文件必须被合并进来（桥分页导出 H1 更早历史用）。"""
    import pandas as pd

    from gold.bridge import bars

    def _csv(path, times):
        rows = ["time,open,high,low,close,tick_volume"]
        for t in times:
            rows.append(f"{t},2000,2001,1999,2000,10")
        path.write_text("\n".join(rows) + "\n", encoding="ascii")

    base = [1_700_000_000 + i * 3600 for i in range(3)]
    older = [t - 3 * 3600 for t in base]
    _csv(tmp_path / "gold_bars_GOLD_H1.csv", base)
    _csv(tmp_path / "gold_bars_GOLD_H1_p1.csv", older)

    df = bars("H1", d=tmp_path, sym="GOLD")
    assert len(df) == 6
    assert df.index.is_monotonic_increasing
    assert len(df.index.unique()) == 6
