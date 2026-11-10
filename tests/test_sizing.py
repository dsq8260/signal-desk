"""小资金可行性测试：200/500/1000 美元账户用黄金 0.01 手的最低风险必须有据可查。"""

from __future__ import annotations

import pytest

from gold.sizing import feasibility, min_equity_for_risk, money_per_lot, plan_table

PRICE = 4000.0


def test_loss_per_lot_gold():
    # 2% 止损 = 80 美元/盎司 × 100 盎司 = 8000 美元/手
    assert money_per_lot("GOLD", PRICE, 0.02) == pytest.approx(8000)


def test_min_lot_loss_for_1000_account():
    # 0.01 手 = 80 美元 → 1000 本金就是 8%
    f = feasibility("GOLD", 1000, PRICE, 0.02, 0.005)
    assert f.actual_risk_pct == pytest.approx(0.08, abs=1e-6)
    assert f.ok is False


def test_min_equity_for_half_percent():
    assert min_equity_for_risk("GOLD", PRICE, 0.02, 0.005) == pytest.approx(16000)


@pytest.mark.parametrize("equity,expect_ok", [(16000, True), (20000, True),
                                             (5000, False), (200, False)])
def test_feasibility_boundary(equity, expect_ok):
    assert feasibility("GOLD", equity, PRICE, 0.02, 0.005).ok is expect_ok


def test_advice_mentions_cent_account_for_small_equity():
    f = feasibility("GOLD", 500, PRICE, 0.02, 0.005)
    assert "美分账户" in f.advice
    assert f.advice.count("所需") or "需本金" in f.advice


def test_tighter_stop_lowers_required_capital():
    wide = min_equity_for_risk("GOLD", PRICE, 0.02, 0.005)
    tight = min_equity_for_risk("GOLD", PRICE, 0.005, 0.005)
    assert tight == pytest.approx(wide / 4)


def test_plan_table_covers_small_accounts():
    rows = plan_table("GOLD", PRICE)
    assert any(r["本金"] == 200 for r in rows)
    assert all("不可行" in r["0.5%风险可行"] for r in rows
               if r["本金"] <= 1000 and r["止损"] == "2.0%")


def test_fx_requires_less_capital_than_gold():
    """EURUSD 0.01 手 = 1000 单位，2% 止损约亏 22 美元 → 0.5% 风险需约 4400 本金。

    比黄金（16000）便宜 3~4 倍，但对 200~1000 美元账户仍然不可行 ——
    这正是"小资金应当用美分账户"的量化依据。
    """
    fx = min_equity_for_risk("EURUSD", 1.10, 0.02, 0.005)
    gold = min_equity_for_risk("GOLD", 4000.0, 0.02, 0.005)
    assert 3000 < fx < 6000, fx
    assert fx < gold / 2
