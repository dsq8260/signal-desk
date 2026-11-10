"""MT5 MCP 客户端测试：配置解析、索引预检、干跑参数注入、失败原因。

这些测试**不联网**：用一个假的 Mt5Mcp 覆盖 call()，验证逻辑分支。
另有一个真实集成测试，MCP 不可达时自动跳过。
"""

from __future__ import annotations

import json
import pathlib

import pytest

from gold import mt5_mcp as M


# ---------------------------------------------------------------- 配置
def test_load_config_from_file(monkeypatch, tmp_path):
    cfg = tmp_path / ".mt5_mcp.json"
    cfg.write_text(json.dumps({"url": "http://x:1/mcp", "key": "abc123"}),
                   encoding="utf-8")
    monkeypatch.setattr(M, "CONFIG_FILE", cfg)
    monkeypatch.delenv("MT5_MCP_URL", raising=False)
    monkeypatch.delenv("MT5_MCP_KEY", raising=False)
    got = M.load_config()
    assert got["url"] == "http://x:1/mcp"
    assert got["key"] == "abc123"
    assert got["source"] == ".mt5_mcp.json"


def test_env_beats_file(monkeypatch, tmp_path):
    cfg = tmp_path / ".mt5_mcp.json"
    cfg.write_text(json.dumps({"url": "http://file", "key": "filekey"}),
                   encoding="utf-8")
    monkeypatch.setattr(M, "CONFIG_FILE", cfg)
    monkeypatch.setenv("MT5_MCP_URL", "http://env/mcp")
    monkeypatch.setenv("MT5_MCP_KEY", "envkey")
    got = M.load_config()
    assert got["url"] == "http://env/mcp" and got["source"] == "env"


def test_load_config_falls_back_to_default_url(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "CONFIG_FILE", tmp_path / "nope.json")
    monkeypatch.delenv("MT5_MCP_URL", raising=False)
    monkeypatch.delenv("MT5_MCP_KEY", raising=False)
    monkeypatch.setattr(M, "_read_assistant_endpoint", lambda: None)
    got = M.load_config()
    assert got["url"] == M.DEFAULT_URL and got["key"] == ""


def test_guess_mt5_symbol_keeps_known_names():
    assert M.guess_mt5_symbol("GOLD") == "GOLD"
    assert M.guess_mt5_symbol("GOLD#") == "GOLD#"
    assert M.guess_mt5_symbol("eurusd") == "EURUSD"


# ---------------------------------------------------------------- 假客户端
class FakeMcp(M.Mt5Mcp):
    """不联网：把 call() 换成查表。"""

    def __init__(self, responses):
        super().__init__(url="http://fake/mcp", key="k")
        self.responses = responses
        self.calls = []

    def handshake(self):  # noqa: D102
        self.handshaked = True
        return True

    def call(self, name, args=None):  # noqa: D102
        self.calls.append((name, args or {}))
        r = self.responses.get(name)
        return r(args or {}) if callable(r) else (r or {})


def test_send_ea_rejects_missing_file(tmp_path):
    c = FakeMcp({})
    out = c.send_ea(str(tmp_path / "nope.ex5"), "GOLD#", "H1")
    assert out["ok"] is False and "找不到" in out["error"]


def test_send_ea_injects_dry_run_and_magic():
    ex5 = pathlib.Path("mql5/GoldEA_MA.ex5")
    if not ex5.exists():
        pytest.skip("没有可用的示例 .ex5")
    seen = {}

    def add_expert(args):
        seen.update(args)
        return {"ok": True}

    c = FakeMcp({
        "list_open_charts": {"charts": [
            {"chart_id": "111", "symbol": "GOLD#", "period": "H1"}]},
        "chart_add_expert": add_expert,
    })
    out = c.send_ea(str(ex5), "GOLD#", "H1", magic=20261101, dry_run=True)
    assert out["ok"] and out["chart_id"] == "111"
    assert "InpDryRun=true" in seen["expert_parameters"]
    assert "InpMagic=20261101" in seen["expert_parameters"]
    assert " " not in seen["expert_parameters"], "参数串不能带空格"


def test_send_ea_opens_chart_when_none_matches():
    ex5 = pathlib.Path("mql5/GoldEA_MA.ex5")
    if not ex5.exists():
        pytest.skip("没有可用的示例 .ex5")
    c = FakeMcp({
        "list_open_charts": {"charts": []},
        "chart_open": {"chart_id": "999"},
        "chart_add_expert": {"ok": True},
    })
    out = c.send_ea(str(ex5), "GOLD#", "M15")
    assert out["chart_id"] == "999"
    names = [n for n, _ in c.calls]
    assert "chart_open" in names


def test_deploy_reports_not_indexed():
    """文件就位但终端索引没刷新 → 必须给出 not_indexed 与刷新指引。"""
    src = pathlib.Path("mql5/GoldEA_MA.mq5")
    if not src.exists():
        pytest.skip("没有示例 .mq5")
    c = FakeMcp({
        "write_file": {"ok": True},
        "list_available_mql5_programs": {"programs": []},
        "get_workspace_info": {"workspace": {"mql5_folder": "C:/MT5/MQL5"}},
    })
    # 避免真的调 MetaEditor：mock 掉
    import subprocess
    import types

    real_run = subprocess.run

    def fake_run(*a, **kw):
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    subprocess.run = fake_run
    try:
        out = c.deploy_and_attach(str(src), "GOLD#", "H1")
    finally:
        subprocess.run = real_run
    assert out["ok"] is False
    assert out["reason"] == "not_indexed"
    assert "刷新" in out["error"]


def test_expert_indexed_matches_name_case_insensitively():
    c = FakeMcp({"list_available_mql5_programs": {"programs": [
        {"name": "GoldEA_MA", "path": "x"}]}})
    assert c.expert_indexed("goldEA_ma") is not None
    assert c.expert_indexed("nothing") is None


# ---------------------------------------------------------------- 真实集成
def _live() -> M.Mt5Mcp | None:
    try:
        c = M.Mt5Mcp(timeout=6)
        return c if c.handshake() else None
    except Exception:  # noqa: BLE001
        return None


def test_live_terminal_reachable_or_skip():
    c = _live()
    if c is None:
        pytest.skip("MT5 MCP 不可达（终端没开或密钥失效）")
    info = c.terminal()
    assert info, "terminal_get_info 返回空"


def test_live_attach_and_detach_known_expert():
    """真挂一次**已在索引里**的 EA，验证机制通，然后卸下。"""
    c = _live()
    if c is None:
        pytest.skip("MT5 MCP 不可达")
    idx = None
    for p in c.programs():
        if str(p.get("name", "")).startswith("GoldEA"):
            idx = p
            break
    if idx is None:
        pytest.skip("终端索引里没有 GoldEA_* 可测")
    charts = c.charts()
    target = None
    for ch in charts:
        if not ch.get("expert"):
            target = str(ch.get("chart_id"))
            sym = ch.get("symbol")
            tf = ch.get("period")
            break
    if target is None:
        pytest.skip("没有空闲图表可测（避免覆盖已有 EA）")
    r = c.call("chart_add_expert", {"chart_id": target,
                                    "expert_path": idx["path"],
                                    "expert_parameters": "InpDryRun=true"})
    try:
        assert not r.get("error"), r
    finally:
        c.detach(target)
