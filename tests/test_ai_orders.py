"""半自动工单：价位校验 / 手数反算 / 盯盘状态机。

这里不测 LLM、不测真实行情——只钉住三件会直接影响钱的事：
  1. AI 给的止损方向反了 / 入场区离谱时，必须用 ATR 兜底重算；
  2. 手数只能由「净值×风险%÷(止损×合约)」反算，算出不够就警告并禁止一键执行；
  3. 观望不出单、风控否决出「只能看」的单、进区/破位/止损/目标/过期各自只报一次。
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "signal_desk"))

import ai_orders  # noqa: E402


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """每个用例独立的工单文件 + 可控行情。"""
    monkeypatch.setattr(ai_orders, "FILE", tmp_path / "ai_orders.json")
    box = {"f": {"bid": 4000.0, "ask": 4000.3, "price": 4000.0, "age": 2.0,
                 "fresh": True, "atr": 8.0, "equity": 10000.0, "balance": 10000.0,
                 "positions": [], "h1": "up", "d1": "up"}}
    monkeypatch.setattr(ai_orders, "facts", lambda: box["f"])
    return box


def _a1(**kw):
    """默认给一张「回踩做多」的计划：入场区在现价 4000 下方，需要等回踩才触发。"""
    base = {"direction": "buy", "confidence": 62, "entry_zone": [3990.0, 3994.0],
            "stop": 3980.0, "targets": [4012.0, 4020.0], "horizon": "intraday",
            "reasons": ["H1 上行", "回踩支撑"], "invalid": "跌破 3980"}
    base.update(kw)
    return base


# ---------------------------------------------------------------- 价位校验


def test_prices_keep_sane_ai_levels(iso):
    p, err = ai_orders._prices(_a1(), iso["f"])
    assert err == ""
    assert p["entry_zone"] == [3990.0, 3994.0]
    assert p["stop"] < p["entry_zone"][0]          # 多头止损在区间下方
    assert p["tp1"] > p["entry"] and p["tp2"] > p["tp1"]


def test_prices_rebuild_reversed_stop(iso):
    """AI 把多头止损写到入场区上方（方向反了）→ 必须重算到下方。"""
    p, _ = ai_orders._prices(_a1(stop=4010.0), iso["f"])
    assert p["stop"] < p["entry_zone"][0]
    assert 0.3 * iso["f"]["atr"] <= p["entry_zone"][0] - p["stop"] <= 3 * iso["f"]["atr"]


def test_prices_shrink_absurd_zone(iso):
    """入场区宽于 2×ATR（等于整段行情都叫入场区）→ 按现价 ±0.25 ATR 重算。"""
    p, _ = ai_orders._prices(_a1(entry_zone=[3940.0, 4060.0]), iso["f"])
    assert p["entry_zone"][1] - p["entry_zone"][0] <= 2 * iso["f"]["atr"]


def test_prices_reject_without_atr(iso):
    iso["f"]["atr"] = None
    p, err = ai_orders._prices(_a1(), iso["f"])
    assert p is None and "ATR" in err


# ---------------------------------------------------------------- 手数


def test_lots_from_risk_budget():
    """净值 1 万 × 0.5% = 50 USD；12 美元止损 × 100 盎司 → 0.04 手（风险 48 USD）。"""
    lots, risk, warn = ai_orders._lots({"sl_dist": 12.0}, 10000.0)
    assert lots == 0.04
    assert risk == pytest.approx(48.0)
    assert warn == ""


def test_lots_floor_down_to_step():
    """不放大风险：0.083 手要向下取到 0.08，不能四舍五入成 0.09。"""
    lots, _, _ = ai_orders._lots({"sl_dist": 12.0}, 12000.0)
    assert lots == 0.05          # 0.1 → 上限 0.05


def test_lots_small_equity_warns():
    """净值 300、止损 12 美元：1% = 3 USD，连 0.01 手（12 USD）都不够 → 给最小手数 + 警告。"""
    lots, risk, warn = ai_orders._lots({"sl_dist": 12.0}, 300.0)
    assert lots == 0.01
    assert risk == pytest.approx(12.0)
    assert "不够开最小手数" in warn


def test_lots_without_equity_is_conservative():
    lots, _, warn = ai_orders._lots({"sl_dist": 12.0}, None)
    assert lots == 0.01 and "净值" in warn


# ---------------------------------------------------------------- 生成


@pytest.fixture()
def stub_council(monkeypatch):
    def _set(a1, gates=None, a2=None, tf="M30"):
        stub = {"last": {"a1": a1, "gates": gates or [], "a2": a2 or {"verdict": "allow"},
                         "tf": tf}}
        monkeypatch.setattr(ai_orders, "_state_stub", lambda: stub, raising=False)
        import ai_council
        monkeypatch.setattr(ai_council, "state", lambda: stub)
    return _set


def test_create_rejects_wait(iso, stub_council):
    stub_council(_a1(direction="wait", confidence=30))
    r = ai_orders.create()
    assert not r["ok"] and "观望" in r["error"]


def test_create_pending_order(iso, stub_council):
    stub_council(_a1())
    r = ai_orders.create()
    assert r["ok"], r
    o = r["order"]
    assert o["status"] == "pending"           # 回踩位在现价下方 → 等触发
    assert o["volume"] == 0.04                # 净值 1 万 × 0.5% ÷ (12 美金 × 100)
    assert o["expires"] > o["at"]


def test_create_immediate_armed(iso, stub_council):
    """入场区本来就含现价 → 直接 armed，不用白等一次轮询。"""
    stub_council(_a1(entry_zone=[3998.0, 4002.0], stop=3988.0, targets=[4016.0, 4024.0]))
    r = ai_orders.create()
    assert r["ok"] and r["order"]["status"] == "armed"
    assert r["order"]["fired"] == ["armed"]


def test_create_blocked_by_gate(iso, stub_council):
    """硬闸门命中 → 工单照出，但状态 blocked，页面只能看。"""
    stub_council(_a1(), gates=[{"gate": "高影响数据窗口", "hit": True, "why": "非农 12 分钟后"}])
    r = ai_orders.create()
    assert r["ok"] and r["order"]["status"] == "blocked"
    assert "非农" in " ".join(r["order"]["blocked_by"])


def test_create_refuses_second_active(iso, stub_council):
    stub_council(_a1())
    assert ai_orders.create()["ok"]
    r2 = ai_orders.create()
    assert not r2["ok"] and "进行中" in r2["error"]


# ---------------------------------------------------------------- 盯盘状态机


def _mk_pending(iso, stub_council):
    stub_council(_a1())
    ai_orders.create()
    d = ai_orders._store()
    return d["orders"][-1]


def test_tick_arm_then_void(iso, stub_council):
    _mk_pending(iso, stub_council)
    iso["f"]["bid"] = 3992.0                       # 回踩进入 3990–3994
    evs = ai_orders.tick()
    assert [e["type"] for e in evs] == ["armed"]
    assert ai_orders._store()["orders"][-1]["status"] == "armed"

    assert ai_orders.tick() == []                  # 同一条不重复报
    iso["f"]["bid"] = 3978.0                       # 跌破止损 3980
    evs = ai_orders.tick()
    assert [e["type"] for e in evs] == ["void"]
    assert ai_orders._store()["orders"][-1]["status"] == "void"


def test_tick_entered_tp_then_stop(iso, stub_council):
    o = _mk_pending(iso, stub_council)
    ai_orders.ack(o["id"], "enter", 4001.0)
    assert ai_orders._store()["orders"][-1]["status"] == "entered"

    iso["f"]["bid"] = 4013.0                       # 越过 tp1=4012
    evs = ai_orders.tick()
    assert [e["type"] for e in evs] == ["tp1"]
    assert ai_orders._store()["orders"][-1]["status"] == "entered"

    iso["f"]["bid"] = 4013.5
    assert [e["type"] for e in ai_orders.tick()] == []   # tp1 只报一次

    iso["f"]["bid"] = 3979.0                       # 触及止损 3980 下方
    evs = ai_orders.tick()
    assert [e["type"] for e in evs] == ["stop"]
    o2 = ai_orders._store()["orders"][-1]
    assert o2["status"] == "done" and o2["done_reason"] == "stop"


def test_tick_expires(iso, stub_council):
    o = _mk_pending(iso, stub_council)
    d = ai_orders._store()
    d["orders"][-1]["expires"] = (dt.datetime.now() - dt.timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M")
    ai_orders._save(d)
    evs = ai_orders.tick()
    assert [e["type"] for e in evs] == ["expired"]
    assert ai_orders._store()["orders"][-1]["status"] == "expired"


def test_tick_ignores_stale_quote(iso, stub_council):
    _mk_pending(iso, stub_council)
    iso["f"]["fresh"] = False
    iso["f"]["bid"] = 4001.0
    assert ai_orders.tick() == []


# ---------------------------------------------------------------- 一键执行闸门


def test_exec_off_by_default(iso, stub_council):
    o = _mk_pending(iso, stub_council)
    r = ai_orders.exec_order(o["id"])
    assert not r["ok"] and "默认关闭" in r["error"]


def test_exec_refuses_over_risk(iso, stub_council):
    iso["f"]["equity"] = 300.0                     # 触发「超预算」警告
    o = _mk_pending(iso, stub_council)
    ai_orders.set_exec(True)
    r = ai_orders.exec_order(o["id"])
    assert not r["ok"] and "超预算" in r["error"]


def test_exec_refuses_live_account(iso, stub_council, monkeypatch):
    import ai_trader
    monkeypatch.setattr(ai_trader, "account", lambda: {"type": "real", "ok": True})
    monkeypatch.setattr(ai_trader, "exec_open", lambda pl: {"ok": True, "ticket": 1})
    o = _mk_pending(iso, stub_council)
    ai_orders.set_exec(True)
    r = ai_orders.exec_order(o["id"])
    assert not r["ok"] and "模拟盘" in r["error"]


def test_listing_shape(iso, stub_council):
    _mk_pending(iso, stub_council)
    out = ai_orders.listing()
    assert out["ok"] and out["active"] and out["orders"]
    assert out["exec_enabled"] is False
