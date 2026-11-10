"""账户自动识别 + 一键方案测试（XM 四种账户 + 美分 + 股票）。"""

from __future__ import annotations

import pytest

from gold.account_detect import auto_plan, detect_account, ea_defaults


def test_detect_standard_gold():
    a = detect_account(contract_size=100, min_lot=0.01, spread_points=30)
    assert a.account_class == "standard" and a.cent is False
    assert a.confidence == "high"


def test_detect_ultra_low_standard_by_tight_spread():
    a = detect_account(100, 0.01, spread_points=16)
    assert a.account_class == "ultra_standard"


def test_detect_micro_gold():
    a = detect_account(contract_size=1, min_lot=0.1, spread_points=30)
    assert a.account_class == "micro"
    assert a.contract_size == 1 and a.min_lot == 0.1


def test_detect_ultra_micro():
    a = detect_account(1, 0.1, spread_points=16)
    assert a.account_class == "ultra_micro"


def test_detect_cent_by_currency():
    a = detect_account(100, 0.01, 30, currency="USC")
    assert a.account_class == "cent" and a.cent is True


def test_detect_shares():
    a = detect_account(contract_size=1, min_lot=1.0, spread_points=None)
    assert a.account_class in ("shares", "micro")


def test_unknown_contract_falls_back_to_standard():
    a = detect_account(None, None, None)
    assert a.account_class == "standard" and a.confidence == "low"


# ---------------------------------------------------------------- 一键方案
def test_plan_micro_feasible_for_500():
    p = auto_plan(500, account="micro")
    assert p["可行"] is True
    assert p["推荐品种"] in ("GOLD", "EURUSD")
    assert "可" in p["建议"]


def test_plan_standard_infeasible_for_500():
    p = auto_plan(500, account="standard")
    assert p["可行"] is False
    assert "资金至少" in p["建议"] or "Micro" in p["建议"]


def test_plan_prefers_lower_risk_symbol():
    p = auto_plan(500, account="micro")
    # 外汇 Micro 的风险比黄金更低 → 应推荐 EURUSD
    assert p["推荐品种"] == "EURUSD"


def test_ea_defaults_are_bounded():
    d = ea_defaults(auto_plan(500, account="micro"), 500)
    assert d["InpLotMode"] == "LOT_PERCENT"
    assert 0.2 <= d["InpRiskPct"] <= 2.0
    assert d["InpMaxRiskPct"] == 2.0
    assert d["InpMaxSpreadPoints"] == 40
    assert d["InpMaxDailyLossPct"] == 5.0
    assert d["InpOnePerBar"] is True
    assert d["InpMaxLots"] >= 0.01


def test_plan_detail_has_both_symbols():
    p = auto_plan(1000, account="micro")
    syms = {x["品种"] for x in p["明细"]}
    assert syms == {"GOLD", "EURUSD"}
