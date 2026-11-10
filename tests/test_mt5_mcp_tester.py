"""给 MT5 MCP 新方法补测试（不联网）。"""

from __future__ import annotations

import pathlib

import pytest

from gold import mt5_mcp as M


class FakeMcp(M.Mt5Mcp):
    def __init__(self, responses):
        super().__init__(url="http://fake/mcp", key="k")
        self.responses = responses
        self.calls = []

    def handshake(self):
        self.handshaked = True
        return True

    def call(self, name, args=None, timeout=None):
        self.calls.append((name, args or {}))
        r = self.responses.get(name)
        return r(args or {}) if callable(r) else (r or {})


def test_wait_indexed_returns_true_when_present(monkeypatch):
    c = FakeMcp({"list_available_mql5_programs": {"programs": [
        {"name": "GoldEA_X", "path": "p"}]}})
    assert c.wait_indexed("GoldEA_X", timeout=3, interval=0.1) is True


def test_wait_indexed_times_out(monkeypatch):
    c = FakeMcp({"list_available_mql5_programs": {"programs": []}})
    assert c.wait_indexed("Nope", timeout=1.2, interval=0.3) is False


def test_empty_chart_prefers_chart_without_expert():
    c = FakeMcp({"list_open_charts": {"charts": [
        {"chart_id": "1", "symbol": "GOLD#", "period": "H1",
         "expert": {"name": "GoldBridge"}},
        {"chart_id": "2", "symbol": "GOLD#", "period": "H1", "expert": None},
    ]}})
    assert c._empty_chart("GOLD#", "H1") == "2"


def test_empty_chart_opens_new_when_all_occupied():
    c = FakeMcp({
        "list_open_charts": {"charts": [
            {"chart_id": "1", "symbol": "GOLD#", "period": "H1",
             "expert": {"name": "X"}}]},
        "chart_open": {"chart_id": "99"},
    })
    assert c._empty_chart("GOLD#", "H1") == "99"
    assert any(n == "chart_open" for n, _ in c.calls)


def test_run_tester_reports_not_indexed():
    c = FakeMcp({"list_available_mql5_programs": {"programs": []}})
    out = c.run_tester("GoldEA_New", "GOLD#", "H1",
                       "2025-01-01T00:00:00", "2025-02-01T00:00:00")
    assert out["ok"] is False and out["reason"] == "not_indexed"


def test_run_tester_prepares_runs_and_returns_report():
    c = FakeMcp({
        "list_available_mql5_programs": {"programs": [
            {"name": "GoldEA_OK", "path": "C:/x/GoldEA_OK.ex5"}]},
        "tester_prepare_config": {"ok": True},
        "tester_run_backtest": {"run_id": "run-7"},
        "tester_get_report": {"net_profit": 123.4, "profit_factor": 1.6,
                              "trades": 42},
    })
    out = c.run_tester("GoldEA_OK", "GOLD#", "H1",
                       "2025-01-01T00:00:00", "2025-02-01T00:00:00",
                       model="Every tick")
    assert out["ok"] and out["run_id"] == "run-7"
    assert out["report"]["profit_factor"] == 1.6
    names = [n for n, _ in c.calls]
    assert "tester_prepare_config" in names
    assert "tester_run_backtest" in names
    assert "tester_get_report" in names
    prep = dict(c.calls)["tester_prepare_config"]
    assert prep["model"] == "Every tick"
    assert prep["symbol"] == "GOLD#"


def test_run_tester_retries_when_tester_busy(monkeypatch):
    calls = {"n": 0}

    def run(args):
        calls["n"] += 1
        if calls["n"] < 2:
            return {"ok": False, "error": "Tool 'x' tester still started"}
        return {"run_id": "r2"}

    c = FakeMcp({
        "list_available_mql5_programs": {"programs": [
            {"name": "GoldEA_OK", "path": "p"}]},
        "tester_prepare_config": {"ok": True},
        "tester_run_backtest": run,
        "tester_get_report": {"net_profit": 1.0},
    })
    monkeypatch.setattr("time.sleep", lambda *_: None)
    out = c.run_tester("GoldEA_OK", "GOLD#", "H1", "2025-01-01T00:00:00",
                       "2025-02-01T00:00:00")
    assert out["ok"] and out["run_id"] == "r2"
    assert calls["n"] == 2


def test_templates_dir_derives_from_mql5_folder():
    c = FakeMcp({"get_workspace_info": {"workspace": {
        "mql5_folder": "C:/MT5/MQL5"}}})
    assert c.templates_dir().endswith("Profiles\\Templates")
