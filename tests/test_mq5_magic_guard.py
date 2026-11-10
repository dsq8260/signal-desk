"""回归测试：EA 必须只动自己 magic 的持仓（否则多 EA 同账户会互相平仓→持仓闪烁）。"""

from __future__ import annotations

import pathlib

import pytest

from gold.mq5 import generate


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    r = generate("双均线", {"fast": 20, "slow": 60},
                 out_dir=tmp_path_factory.mktemp("mq5magic"),
                 name="GoldEA_MagicGuard")
    assert r.ok, r.error
    return pathlib.Path(r.mq5_path).read_text(encoding="utf-8",
                                              errors="replace")


def test_has_position_filters_by_magic(src):
    i = src.find("bool HasPosition(int &dir)")
    body = src[i:src.find("void CloseAll", i)]
    assert "POSITION_MAGIC" in body, "持仓判定必须按 magic 过滤"
    assert "PositionsTotal()" in body, "必须遍历持仓，而不是 PositionSelect(_Symbol)"
    assert "if(PositionSelect(_Symbol))" not in body


def test_close_all_only_closes_own_positions(src):
    i = src.find("void CloseAll()")
    body = src[i:src.find("double PctToPrice", i)]
    assert "POSITION_MAGIC" in body, "平仓必须按 magic 过滤"
    assert "PositionClose(ticket)" in body, "必须按 ticket 平，不能 PositionClose(_Symbol)"
    assert "trade.PositionClose(_Symbol)" not in body


def test_no_bare_position_select_anywhere(src):
    """模板里任何按品种取持仓都必须带 magic 过滤，否则会误伤别人的仓。"""
    assert "PositionSelect(_Symbol)" not in src


def test_report_state_uses_own_position_only(src):
    i = src.find("void ReportState()")
    body = src[i:src.find("int OnInit", i)]
    assert "POSITION_MAGIC" in body, "成绩单也只能统计自己的持仓"


def test_dry_run_still_guards_close(src):
    i = src.find("void CloseAll()")
    body = src[i:src.find("double PctToPrice", i)]
    assert "InpDryRun" in body and "return;" in body
