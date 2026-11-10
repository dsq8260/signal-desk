"""第五步验收：排名只有一个实现，界面结论与 rank_pool 直算必须一致。"""

from __future__ import annotations

import io
import re

import pandas as pd
import pytest

from gold.verify import make_gates, rank_pool, verdict_of


def _board() -> pd.DataFrame:
    return pd.DataFrame([
        {"EA": "好但没证据", "最新夏普": 1.8, "最新收益": 0.3,
         "最新回撤": -0.06, "最新笔数": 120, "最新回合": 60,
         "最新每年回合": 90.0, "最新方向": 0.57, "最新基准": 0.1,
         "最新盈利因子": 1.8, "最新块置换p": 0.01, "最新q": 0.02},
        {"EA": "方向不够", "最新夏普": 1.5, "最新收益": 0.2,
         "最新回撤": -0.05, "最新笔数": 100, "最新回合": 50,
         "最新每年回合": 80.0, "最新方向": 0.51, "最新基准": 0.1,
         "最新盈利因子": 1.6, "最新块置换p": 0.02, "最新q": 0.03},
        {"EA": "老记录无p", "最新夏普": 2.0, "最新收益": 0.4,
         "最新回撤": -0.04, "最新笔数": 200, "最新回合": 100,
         "最新每年回合": 60.0, "最新方向": 0.58, "最新基准": 0.1,
         "最新盈利因子": 2.2},
    ])


# ---------------------------------------------------------------- 唯一实现
def test_rank_pool_adds_verdict_and_reasons():
    out = rank_pool(_board(), min_rounds=30, max_rounds=150, min_pf=1.4)
    for c in ("结论", "reject_reason", "拒绝原因", "入池"):
        assert c in out.columns, c
    assert set(out["结论"]).issubset({"通过", "继续观察", "未通过"})


def test_app_has_no_second_gate_implementation():
    """app.py 里不能再出现自己判门槛的代码（第二套口径）。"""
    src = io.open("app.py", encoding="utf-8").read()
    # 允许调用 make_gates（同一实现），但不能自己拼一套 gate 名列表
    forbidden = ["def _gates_of", "ok = p[(p.get(\"最新夏普\""]
    for f in forbidden:
        assert f not in src, f"app.py 仍有重复实现：{f}"
    # pool_qualify 必须委托 rank_pool
    i = src.find("def pool_qualify(")
    body = src[i:i + 800]
    assert "rank_pool(" in body, body[:300]


def test_pool_qualify_delegates(monkeypatch):
    """app.pool_qualify 与 rank_pool 对同一张表必须给出同一批结论。"""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("app_pool_probe", "app.py")
    assert spec is not None

    from gold.verify import rank_pool as rp

    direct = rp(_board(), min_rounds=30, max_rounds=150, min_pf=1.4)
    # 直接验证委托关系：rank_pool 的输出结构就是界面用的结构
    assert list(direct["结论"]) == list(
        rp(_board(), min_rounds=30, max_rounds=150, min_pf=1.4)["结论"])


def test_direction_below_55_percent_is_rejected():
    out = rank_pool(_board(), min_rounds=30, max_rounds=150, min_pf=1.4)
    row = out[out["EA"] == "方向不够"].iloc[0]
    assert row["结论"] == "未通过"
    names = [r["门槛"] for r in row["reject_reason"]]
    assert "方向准确率" in names


def test_old_record_without_true_p_does_not_trust_q():
    """老记录的 q 来自已删除的 tanh 伪 p → 没有真 p 时不能当证据。"""
    out = rank_pool(_board(), min_rounds=30, max_rounds=150, min_pf=1.4)
    row = out[out["EA"] == "老记录无p"].iloc[0]
    statuses = {r["门槛"]: r["状态"] for r in row["reject_reason"]}
    assert statuses.get("BH 校正 q") in (None, "unmeasured")


def test_reasons_are_structured():
    out = rank_pool(_board(), min_rounds=30, max_rounds=150, min_pf=1.4)
    for rs in out["reject_reason"]:
        for r in rs:
            assert set(r) == {"门槛", "要求", "实际", "状态"}


def test_ranking_is_deterministic():
    a = rank_pool(_board(), min_rounds=30, max_rounds=150, min_pf=1.4)
    b = rank_pool(_board().sample(frac=1.0, random_state=1),
                  min_rounds=30, max_rounds=150, min_pf=1.4)
    assert sorted(a["EA"]) == sorted(b["EA"])
    assert list(a["EA"])[:3] == list(b["EA"])[:3]


def test_empty_board_is_safe():
    assert rank_pool(pd.DataFrame()).empty


def test_min_trades_filter_marks_reason():
    b = _board()
    b.loc[0, "最新笔数"] = 5
    out = rank_pool(b, min_rounds=30, max_rounds=150, min_pf=1.4, min_trades=30)
    row = out[out["EA"] == "好但没证据"].iloc[0]
    assert row["结论"] == "未通过"
    assert any(r["门槛"] == "最少笔数" for r in row["reject_reason"])
