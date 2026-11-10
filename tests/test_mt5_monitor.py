"""MT5 只读监控的纯函数测试（不需要真实终端）。"""

from __future__ import annotations

import pytest

from market_data.mt5_bridge import (
    curve_drawdown,
    deal_cash,
    deal_stats,
    realized_curve,
    margin_level,
    position_margin,
    position_rows,
    risk_summary,
    safe_float,
    stop_out_distance,
)


def _acct(**kw):
    base = {"balance": 10_000.0, "equity": 11_000.0, "margin": 500.0,
            "margin_free": 10_500.0, "leverage": 100, "profit": 1_000.0}
    base.update(kw)
    return base


def _pos(**kw):
    base = {"ticket": 1, "symbol": "GOLD#", "type": 0, "volume": 0.08,
            "price_open": 3977.82, "price_current": 4059.08, "profit": 650.08,
            "swap": 0.0, "sl": 0.0, "tp": 0.0, "contract_size": 100.0}
    base.update(kw)
    return base


# ------------------------------------------------------------ 基础工具
def test_safe_float_handles_nan_and_garbage():
    assert safe_float(None) == 0.0
    assert safe_float("12.5") == 12.5
    assert safe_float(float("nan")) == 0.0
    assert safe_float("abc", 7.0) == 7.0


def test_margin_level_zero_when_no_margin():
    assert margin_level(11_000, 0) == 0.0
    assert margin_level(11_000, 550) == pytest.approx(2000.0)


def test_position_margin_formula():
    # 0.08 手 × 100 盎司 × 4059 / 100 倍杠杆
    assert position_margin(0.08, 4059.0, 100, 100) == pytest.approx(324.72, rel=1e-6)
    # 杠杆缺失按 1 处理，不应除零
    assert position_margin(0.01, 4000, 100, 0) == pytest.approx(4000.0)


def test_stop_out_distance():
    assert stop_out_distance(4059.0, 3950.0, "BUY") == pytest.approx(109.0)
    assert stop_out_distance(4059.0, 0.0, "BUY") is None


# ------------------------------------------------------------ 持仓整理
def test_position_rows_maps_fields():
    rows = position_rows([_pos()], _acct())
    r = rows[0]
    assert r["side"] == "BUY"
    assert r["symbol"] == "GOLD#"
    assert r["profit"] == pytest.approx(650.08)
    assert r["margin"] == pytest.approx(324.7264, rel=1e-6)


def test_position_rows_short_side():
    rows = position_rows([_pos(type=1, profit=-120.0)], _acct())
    assert rows[0]["side"] == "SELL"


def test_position_rows_uses_quote_when_price_missing():
    rows = position_rows([_pos(price_current=0.0)], _acct(),
                         quotes={"GOLD#": {"bid": 4060.0}})
    assert rows[0]["price_now"] == 4060.0


# ------------------------------------------------------------ 风险汇总
def test_risk_summary_basic_numbers():
    rows = position_rows([_pos()], _acct())
    s = risk_summary(rows, _acct())
    assert s["positions"] == 1
    assert s["profit"] == pytest.approx(650.08)
    assert s["long_volume"] == pytest.approx(0.08)
    assert s["short_volume"] == 0.0
    assert s["margin_level"] > 0


def test_risk_summary_net_volume_counts_both_sides():
    rows = position_rows([_pos(), _pos(ticket=2, type=1, volume=0.03)], _acct())
    s = risk_summary(rows, _acct())
    assert s["long_volume"] == pytest.approx(0.08)
    assert s["short_volume"] == pytest.approx(0.03)
    assert s["net_volume"] == pytest.approx(0.05)


def test_risk_summary_empty_positions():
    s = risk_summary([], _acct())
    assert s["positions"] == 0
    assert s["profit"] == 0.0
    assert any(a["level"] == "info" for a in s["alerts"])


def test_alert_on_low_margin_level():
    acct = _acct(equity=11_000, leverage=100)
    big = position_rows([_pos(volume=3.0)], acct)   # 3 手 × 100 × 4059 / 100
    s = risk_summary(big, acct)
    assert any("保证金率" in a["text"] for a in s["alerts"])
    assert any(a["level"] == "danger" for a in s["alerts"])


def test_alert_on_negative_swap():
    rows = position_rows([_pos(swap=-78.59)], _acct())
    s = risk_summary(rows, _acct())
    assert s["swap"] == pytest.approx(-78.59)
    assert any("库存费" in a["text"] for a in s["alerts"])


def test_alert_on_floating_loss():
    acct = _acct(equity=10_000)
    rows = position_rows([_pos(profit=-900.0)], acct)
    s = risk_summary(rows, acct)
    assert any("浮动亏损" in a["text"] for a in s["alerts"])


def test_alert_when_price_near_stop():
    rows = position_rows([_pos(price_current=4000.0, sl=3990.0)], _acct())
    s = risk_summary(rows, _acct())
    assert any("距止损" in a["text"] for a in s["alerts"])


def test_no_margin_alert_for_healthy_account():
    rows = position_rows([_pos()], _acct())
    s = risk_summary(rows, _acct())
    assert not any(a["level"] == "danger" for a in s["alerts"])


# ------------------------------------------------------------ 成交历史与曲线
def _deal(**kw):
    base = {"ticket": 1, "time": "2026-10-27 11:03:00", "symbol": "GOLD#",
            "type": 0, "entry": 1, "volume": 0.04, "price": 4000.0,
            "profit": 0.0, "commission": 0.0, "swap": 0.0, "fee": 0.0,
            "comment": ""}
    base.update(kw)
    return base


def test_deal_cash_sums_all_cost_fields():
    d = _deal(profit=17.16, commission=-1.0, swap=-0.5, fee=-0.2)
    assert deal_cash(d) == pytest.approx(15.46)


def test_realized_curve_ends_at_current_balance():
    deals = [_deal(ticket=1, profit=0.0), _deal(ticket=2, profit=50.0),
             _deal(ticket=3, profit=-20.0)]
    curve = realized_curve(deals, current_balance=1_030.0)
    assert len(curve) == 3
    assert curve[-1]["balance"] == pytest.approx(1_030.0)
    assert curve[0]["balance"] == pytest.approx(1_000.0)


def test_realized_curve_empty():
    assert realized_curve([], 100.0) == []


def test_deal_stats_excludes_deposits():
    deals = [
        _deal(ticket=1, type=2, entry=0, profit=10_000.0),
        _deal(ticket=2, type=0, entry=1, profit=30.0),
        _deal(ticket=3, type=0, entry=1, profit=-10.0),
    ]
    st = deal_stats(deals)
    assert st["deposits"] == pytest.approx(10_000.0)
    assert st["net_profit"] == pytest.approx(20.0)
    assert st["trades"] == 2
    assert st["win_rate"] == pytest.approx(0.5)
    assert st["profit_factor"] == pytest.approx(3.0)
    assert st["best"] == pytest.approx(30.0)
    assert st["worst"] == pytest.approx(-10.0)


def test_deal_stats_by_symbol():
    deals = [_deal(symbol="GOLD#", profit=30.0),
             _deal(ticket=2, symbol="SILVER#", profit=-5.0)]
    st = deal_stats(deals)
    assert st["by_symbol"]["GOLD#"]["profit"] == pytest.approx(30.0)
    assert st["by_symbol"]["SILVER#"]["wins"] == 0


def test_deal_stats_empty():
    st = deal_stats([])
    assert st["trades"] == 0
    assert st["net_profit"] == 0.0


def test_curve_drawdown_tracks_peak():
    curve = [{"balance": 100}, {"balance": 120}, {"balance": 90}, {"balance": 130}]
    dd = curve_drawdown(curve)
    assert dd[0] == pytest.approx(0.0)
    assert dd[2] == pytest.approx(90 / 120 - 1)
    assert dd[3] == pytest.approx(0.0)
