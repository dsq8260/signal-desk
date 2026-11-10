"""账户类型的测试：Micro 对黄金**不降低**最小风险，对小资金外汇才有效。"""

from __future__ import annotations

import pytest

from gold.accounts import (ACCOUNT_TYPES, compare_table, min_risk, min_units,
                           units_per_lot)


def test_standard_and_micro_have_different_lot_units():
    assert units_per_lot("GOLD", "standard") == pytest.approx(100)
    assert units_per_lot("GOLD", "micro") == pytest.approx(1)


def test_micro_cuts_gold_min_risk_by_ten():
    """标准 0.01 手 = 1.0 盎司；Micro 0.1 手 = 0.1 盎司 → 小 10 倍。"""
    assert min_units("GOLD", "standard") == pytest.approx(1.0)
    assert min_units("GOLD", "micro") == pytest.approx(0.1)
    assert min_units("GOLD", "micro") == pytest.approx(
        min_units("GOLD", "standard") / 10)


def test_micro_helps_a_lot_for_fx():
    """外汇标准最小 1000 单位，Micro 最小只有 100 单位。"""
    assert min_units("EURUSD", "standard") == pytest.approx(1000)
    assert min_units("EURUSD", "micro") == pytest.approx(100)


def test_gold_min_risk_on_500_account():
    r = min_risk("GOLD", 4000, 0.02, 500, "standard")
    assert r.risk_usd == pytest.approx(80)
    assert r.risk_pct == pytest.approx(0.16)
    assert r.viable is False


def test_fx_micro_fits_small_account():
    r = min_risk("EURUSD", 1.10, 0.02, 500, "micro")
    assert r.risk_usd < 3          # 100 单位 × 0.022
    assert r.risk_pct < 0.01
    assert r.viable is True


def test_cent_account_scales_down_gold():
    r = min_risk("GOLD", 4000, 0.02, 500, "cent")
    # 美分账户 1 手 = 1 盎司，最小 0.01 手 = 0.01 盎司 → 0.8 美元
    assert r.risk_usd < 1.0
    assert r.viable is True


def test_compare_table_covers_all_types():
    rows = compare_table("GOLD", 4000)
    assert len(rows) == len(ACCOUNT_TYPES)
    assert {r["账户"] for r in rows} == set(ACCOUNT_TYPES)


def test_need_equity_matches_half_percent():
    r = min_risk("GOLD", 4000, 0.02, 1000, "standard")
    assert r.need_equity_for_half == pytest.approx(16000)
