"""EA 实盘保护参数的测试：点差上限 / 风险上限 / 日内停机 / 同 bar 一次。"""

from __future__ import annotations

import pathlib

import pytest

from gold.mq5 import generate


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    r = generate("唐奇安通道突破", {"entry": 20, "exit_": 10},
                 out_dir=tmp_path_factory.mktemp("guards"), name="GoldEA_G")
    assert r.ok, r.error
    return pathlib.Path(r.mq5_path).read_text(encoding="utf-8", errors="replace")


def test_inputs_exist_with_defaults(src):
    assert "input int    InpMaxSpreadPoints   = 40;" in src
    assert "input double InpMaxRiskPct        = 2.0;" in src
    assert "input double InpMaxDailyLossPct   = 5.0;" in src
    assert "input bool   InpOnePerBar         = true;" in src


def test_spread_guard_uses_symbol_spread(src):
    i = src.find("bool SpreadOK()")
    body = src[i:i + 320]
    assert "SYMBOL_SPREAD" in body
    assert "InpMaxSpreadPoints" in body


def test_risk_guard_computes_pct_and_rejects(src):
    i = src.find("bool RiskOK(")
    body = src[i:i + 900]
    assert "ACCOUNT_EQUITY" in body and "InpMaxRiskPct" in body
    assert "拒单" in body, "超限必须打印拒单原因"


def test_daily_loss_guard_stops_trading(src):
    i = src.find("bool DailyLossOK()")
    body = src[i:i + 1000]
    assert "InpMaxDailyLossPct" in body
    assert "今日停止开仓" in body


def test_open_trade_checks_all_guards(src):
    i = src.find("void OpenTrade(int dir)")
    body = src[i:i + 1400]
    for k in ("InpOnePerBar", "SpreadOK()", "DailyLossOK()", "RiskOK("):
        assert k in body, f"OpenTrade 缺少 {k} 检查"
    # 拒单必须发生在下单之前
    assert body.find("RiskOK(") < body.find("trade.Buy(")


def test_guards_can_be_disabled(src):
    """0 = 关闭。否则用户想关掉点差限制也关不了。"""
    assert src.count("<= 0") >= 3
