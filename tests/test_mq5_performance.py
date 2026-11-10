"""性能回归测试：EA 必须是 bar 级而不是 tick 级。

背景（2026-11-01 用户实测）：生成的 EA 在 MT5 测试器里"一根根慢慢走"。
根因三条，全部在模板里：
1. `VolFilterOK()` 每个 tick 都 CopyBuffer(250)+循环 250；
2. `OnTick()` 每个 tick 都重算指标；
3. `ReportState()` 每次 FileOpen/FileClose。
"""

from __future__ import annotations

import pathlib

import pytest

from gold.mq5 import generate


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    r = generate("双均线", {"fast": 20, "slow": 60},
                 out_dir=tmp_path_factory.mktemp("mq5"),
                 name="GoldEA_PerfTest")
    assert r.ok, r.error
    return pathlib.Path(r.mq5_path).read_text(encoding="utf-8",
                                              errors="replace")


def test_on_tick_has_new_bar_gate(src):
    i = src.find("void OnTick()")
    body = src[i:i + 400]
    assert "if(bt == lastBar) return;" in body, "OnTick 缺少新 bar 门"
    assert "static datetime lastBar" in body


def test_vol_filter_is_cached_per_bar(src):
    i = src.find("bool VolFilterOK()")
    body = src[i:i + 500]
    assert "static datetime cachedBar" in body, "VolFilterOK 没有 per-bar 缓存"
    assert "VolFilterCompute" in src, "应把重计算拆到 VolFilterCompute"


def test_vol_filter_compute_kept_intact(src):
    """拆分不能把里面的分位算法弄丢。"""
    i = src.find("bool VolFilterCompute()")
    body = src[i:i + 600]
    assert "iATR" in body and "CopyBuffer" in body
    assert "InpVolLookback" in body


def test_report_state_is_per_bar_and_keeps_handle(src):
    i = src.find("void ReportState()")
    body = src[i:i + 700]
    assert "if(bt == lastBar) return;" in body, "ReportState 缺 per-bar 门"
    assert "static int h = INVALID_HANDLE" in body, "句柄应常驻，不要每次 open/close"


def test_no_legacy_zhi_filter(src):
    """干支过滤早已删除，模板里不能复活。"""
    assert "ZhiFilterOK" not in src


def test_report_does_not_run_per_tick(src):
    """OnTick 里 ReportState 只能出现在新 bar 门之后。"""
    i = src.find("void OnTick()")
    tick = src[i:i + 900]
    gate = tick.find("if(bt == lastBar) return;")
    rep = tick.find("ReportState()")
    assert gate > 0 and rep > gate, "ReportState 在 bar 门之前被调用"
