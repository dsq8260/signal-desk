"""累计设计榜测试。"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from gold.leaderboard import (
    cumulative,
    display_label,
    load_all,
    register_top,
)
from gold.paper import connect, list_strategies


def _run(d: pathlib.Path, ts: str, tf: str, n: int, rows: list[dict]) -> None:
    payload = {"ts": ts, "meta": {"mode": "sample", "tf": tf, "n": n},
               "note": "test", "n_pass": len(rows), "rows": rows,
               "survivors": rows}
    (d / f"run_{ts}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _row(ea: str, sharpe: float, ret: float = 0.1, dd: float = -0.02,
         trades: int = 50, entry: str = "SMA金叉", pf: float | None = None,
         **genome) -> dict:
    g = {"entry": entry, "params": {"fast": 10, "slow": 60},
         "filter": "无过滤", "allow_short": False,
         "sl": None, "tp": None}
    g.update(genome)
    return {"EA": ea, "进场": entry, "过滤": g["filter"],
            "方向": "只多", "OOS收益": ret, "OOS夏普": sharpe,
            "盈利因子": pf,
            "OOS回撤": dd, "OOS方向": 0.55, "笔数": trades, "_genome": g}


def test_load_all_empty(tmp_path):
    assert len(load_all(tmp_path)) == 0


def test_cumulative_counts_repeats(tmp_path):
    """同一 EA 跑过两次：主指标取**最近一次运行**的完整成绩，跨运行只给中位/最差。"""
    _run(tmp_path, "20261001_120000", "H1", 100,
         [_row("A", 0.5), _row("B", 0.3)])
    _run(tmp_path, "20261002_120000", "H1", 100,
         [_row("A", 0.6), _row("C", 0.4)])
    board = cumulative(tmp_path)
    a = board[board["EA"] == "A"].iloc[0]
    assert a["出现次数"] == 2
    assert a["运行次数"] == 2
    assert a["最新夏普"] == pytest.approx(0.6)     # 最近一次运行（20261002）
    assert a["夏普中位"] == pytest.approx(0.55)
    assert a["夏普最差"] == pytest.approx(0.5)


def test_board_never_stitches_metrics_across_runs(tmp_path):
    """禁止跨运行拼接：夏普与盈利因子必须来自**同一次运行**。

    构造：运行1 夏普高、PF 低；运行2 夏普低、PF 高。
    旧实现会拼成（夏普 0.9, PF 1.9）；新实现必须给出（最新夏普 0.5, 最新PF 1.9），
    即这两个数字都属于运行2。
    """
    _run(tmp_path, "20261001_120000", "H1", 10,
         [_row("拼接口径", 0.9, pf=1.2)])
    _run(tmp_path, "20261002_120000", "H1", 10,
         [_row("拼接口径", 0.5, pf=1.9)])
    row = cumulative(tmp_path).iloc[0]
    assert row["最新夏普"] == pytest.approx(0.5)
    assert row["最新盈利因子"] == pytest.approx(1.9)
    # 中位数是允许的跨运行统计
    assert row["夏普中位"] == pytest.approx(0.7)
    # 但不存在"最优夏普"这种拼接列
    assert "最优夏普" not in row.index


def test_cumulative_sorted_by_sharpe_then_appearances(tmp_path):
    _run(tmp_path, "20261001_120000", "H1", 10,
         [_row("A", 0.9), _row("B", 0.2)])
    _run(tmp_path, "20261002_120000", "H1", 10,
         [_row("B", 0.2)])
    board = cumulative(tmp_path)
    assert list(board["EA"])[0] == "A"
    b = board[board["EA"] == "B"].iloc[0]
    assert b["出现次数"] == 2


def test_cumulative_separates_timeframes(tmp_path):
    _run(tmp_path, "20261001_120000", "H1", 10, [_row("A", 0.5)])
    _run(tmp_path, "20261002_120000", "M30", 10, [_row("A", 0.4)])
    board = cumulative(tmp_path)
    assert len(board) == 2
    assert set(board["周期"]) == {"H1", "M30"}


def test_cumulative_min_runs_filter(tmp_path):
    _run(tmp_path, "20261001_120000", "H1", 10,
         [_row("A", 0.5), _row("B", 0.4)])
    _run(tmp_path, "20261002_120000", "H1", 10, [_row("A", 0.6)])
    board = cumulative(tmp_path, min_runs=2)
    assert list(board["EA"]) == ["A"]


def test_cumulative_empty(tmp_path):
    board = cumulative(tmp_path)
    assert len(board) == 0
    assert "出现次数" in board.columns


def test_register_top_writes_meta_params(tmp_path):
    _run(tmp_path, "20261001_120000", "H1", 10,
         [_row("A", 0.9, zhi=11, sl=0.02, filter="低波动")])
    board = cumulative(tmp_path)
    con = connect(tmp_path / "p.db")
    try:
        n = register_top(con, board, top=1)
        assert n == 1
        rows = list_strategies(con)
        assert len(rows) == 1
        params = json.loads(rows.iloc[0]["params"])
        assert "_zhi" not in params, "干支已移除，登记时不该再写 _zhi"
        assert params["_sl"] == pytest.approx(0.02)
        assert params["_filter"] == "低波动"
        assert rows.iloc[0]["name"].startswith("[累计]")
    finally:
        con.close()


def test_register_top_skips_duplicates(tmp_path):
    _run(tmp_path, "20261001_120000", "H1", 10, [_row("A", 0.9)])
    board = cumulative(tmp_path)
    con = connect(tmp_path / "p.db")
    try:
        assert register_top(con, board, top=1) == 1
        assert register_top(con, board, top=1) == 0   # 第二次不重复登记
    finally:
        con.close()


# ---------------------------------------------------------------- 旧口径标记
def test_display_label_strips_zhi_segment():
    assert display_label("SMA金叉[fast=10]/无过滤/亥/只多") == \
        "SMA金叉[fast=10]/无过滤/只多"
    # 不用干支的旧标签里是占位符「—」，同样要去掉
    assert display_label("SMA金叉[fast=10]/低波动/—/双向/SL2%") == \
        "SMA金叉[fast=10]/低波动/双向/SL2%"
    # 新标签原样返回
    assert display_label("SMA金叉[fast=10]/低波动/只多") == \
        "SMA金叉[fast=10]/低波动/只多"


def test_cumulative_marks_legacy_zhi_records(tmp_path):
    _run(tmp_path, "20261001_120000", "H1", 10,
         [_row("带干支", 0.9, zhi=11), _row("新口径", 0.8)])
    board = cumulative(tmp_path)
    old = board[board["EA"] == "带干支"].iloc[0]
    new = board[board["EA"] == "新口径"].iloc[0]
    assert bool(old["含干支"]) is True
    assert bool(new["含干支"]) is False
    # 展示标签去掉干支段，但去重键 EA 保持原样
    assert "/亥/" not in old["EA显示"]
    assert old["EA"] == "带干支"


def test_cumulative_without_genome_falls_back_to_label(tmp_path):
    """老记录可能没有 _genome，仅凭标签里的地支也要能判成旧口径。"""
    row = _row("SMA金叉[fast=10]/无过滤/亥/只多", 0.9)
    row.pop("_genome")
    _run(tmp_path, "20261001_120000", "H1", 10, [row])
    board = cumulative(tmp_path)
    assert bool(board.iloc[0]["含干支"]) is True
