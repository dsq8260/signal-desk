"""`gold.mt5_batch` 与 `run_tester_batch` 的离线测试。

原则：
- 正常用例全部用假客户端，**不联网、不碰 MT5**；
- 只有最后一个集成用例真去握手，连不上就 `pytest.skip`（不算失败）；
- 断言「不编数字」：拿不到的指标必须是 None，而不是 0 或猜测值。
"""

from __future__ import annotations

import json
import pathlib

import pytest

import run_tester_batch as R
from gold import mt5_batch as B

FROM = "2025-01-01T00:00:00"
TO = "2025-02-01T00:00:00"


class FakeClient:
    """假 MCP 客户端：只实现批量验证用到的那几个方法。"""

    def __init__(self, indexed=(), raises=(), reports=None, connected=True):
        self.indexed = set(indexed)
        self.raises = set(raises)
        self.reports = reports or {}
        self.connected = connected
        self.calls = []
        self.last_error = "" if connected else "假客户端：连不上"
        self.url = "http://fake/mcp"

    def handshake(self):
        self.calls.append(("handshake", {}))
        return self.connected

    def expert_indexed(self, name):
        self.calls.append(("expert_indexed", {"name": name}))
        if name in self.raises:
            raise RuntimeError(f"索引查询炸了：{name}")
        if name in self.indexed:
            return {"name": name, "path": f"C:/MQL5/Experts/{name}.ex5"}
        return None

    def run_tester(self, ex5_name, symbol, timeframe, from_date, to_date, *,
                   model="Every tick", deposit=1000, leverage=100,
                   inputs_path=None, timeout_sec=900):
        self.calls.append(("run_tester", {
            "name": ex5_name, "symbol": symbol, "timeframe": timeframe,
            "from_date": from_date, "to_date": to_date, "model": model,
            "deposit": deposit, "leverage": leverage,
            "timeout_sec": timeout_sec}))
        if ex5_name in self.raises:
            raise RuntimeError("测试器崩了")
        if ex5_name not in self.indexed:
            return {"ok": False, "reason": "not_indexed",
                    "error": f"{ex5_name} 不在终端程序索引里"}
        return {"ok": True, "run_id": f"run-{ex5_name}", "ini": "x.ini",
                "report": self.reports.get(ex5_name, {"net_profit": 1.0}),
                "error": ""}


# ---------------------------------------------------------------- candidates
def _write_run(dirpath: pathlib.Path, name: str, payload: dict) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / name).write_text(json.dumps(payload, ensure_ascii=False),
                                encoding="utf-8")


def test_candidates_from_runs_reads_winners_and_genome(tmp_path):
    _write_run(tmp_path, "260101_000000_a.json", {
        "run_id": "r-new", "started_at": "2026-01-01T00:00:00",
        "outcome": {"winners": [{"name": "GoldEA_W1"},
                                {"mql5_name": "GoldEA_W2.ex5"}]},
    })
    _write_run(tmp_path, "250101_000000_b.json", {
        "run_id": "r-old", "started_at": "2025-01-01T00:00:00",
        "outcome": {"top": [{"_genome": {
            "entry": "SMA金叉", "params": {"fast": 10}, "filter": "无过滤",
            "allow_short": False}}]},
    })

    got = B.candidates_from_runs(limit=5, run_dir=tmp_path)
    names = [c["name"] for c in got]
    assert names[:2] == ["GoldEA_W1", "GoldEA_W2"]  # 最近的在最前、去 .ex5
    assert any(n.startswith("GoldEA_MA_fast10") for n in names)
    assert got[0]["run_id"] == "r-new"


def test_candidates_from_runs_returns_empty_with_reason(tmp_path):
    _write_run(tmp_path, "260101_000000_a.json",
               {"run_id": "r1", "outcome": {"final_sharpe": 1.2}})
    assert B.candidates_from_runs(limit=3, run_dir=tmp_path) == []
    assert "winners" in B.LAST_REASON or "top" in B.LAST_REASON

    empty = tmp_path / "empty"
    empty.mkdir()
    assert B.candidates_from_runs(run_dir=empty) == []
    assert B.LAST_REASON  # 空目录也要说明原因


# ---------------------------------------------------------------- index_status
def test_index_status_groups_indexed_and_missing():
    c = FakeClient(indexed=["GoldEA_A"])
    got = B.index_status(c, ["GoldEA_A", "GoldEA_B"])
    assert got == {"indexed": ["GoldEA_A"], "missing": ["GoldEA_B"]}


def test_index_status_treats_query_error_as_missing():
    c = FakeClient(indexed=["GoldEA_A"], raises=["GoldEA_Boom"])
    got = B.index_status(c, ["GoldEA_A", "GoldEA_Boom", ""])
    assert got["indexed"] == ["GoldEA_A"]
    assert got["missing"] == ["GoldEA_Boom"]


# ---------------------------------------------------------------- run_batch
def test_run_batch_writes_results_and_counts(tmp_path):
    c = FakeClient(indexed=["GoldEA_A", "GoldEA_B"], reports={
        "GoldEA_A": {"net_profit": 123.4, "profit_factor": 1.6,
                     "trades": 42, "max_drawdown": 8.5, "sharpe": 1.1},
        "GoldEA_B": {"net_profit": -10.0},
    })
    out = B.run_batch([{"name": "GoldEA_A"}, "GoldEA_B"],
                      from_date=FROM, to_date=TO, symbol="GOLD#",
                      timeframe="H1", deposit=5000, leverage=200,
                      client=c, out_dir=tmp_path)

    assert out["ok"] is True
    assert out["ran"] == 2
    assert out["skipped"] == [] and out["failed"] == []
    assert [r["name"] for r in out["results"]] == ["GoldEA_A", "GoldEA_B"]

    for name in ("GoldEA_A", "GoldEA_B"):
        path = tmp_path / f"{name}.json"
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["name"] == name
        assert payload["symbol"] == "GOLD#" and payload["deposit"] == 5000
        assert payload["leverage"] == 200
        assert payload["from_date"] == FROM and payload["to_date"] == TO
        assert payload["report"]["net_profit"] in (123.4, -10.0)

    run_calls = [args for meth, args in c.calls if meth == "run_tester"]
    assert [a["name"] for a in run_calls] == ["GoldEA_A", "GoldEA_B"]
    assert run_calls[0]["model"] == "Every tick"


def test_run_batch_missing_index_skips_without_writing(tmp_path):
    c = FakeClient(indexed=["GoldEA_A"])
    out = B.run_batch(["GoldEA_A", "GoldEA_Ghost"],
                      from_date=FROM, to_date=TO, client=c, out_dir=tmp_path)

    assert out["ran"] == 1
    assert out["ok"] is False
    assert [s["name"] for s in out["skipped"]] == ["GoldEA_Ghost"]
    skipped = out["skipped"][0]
    assert skipped["reason"] == "not_indexed"
    assert "刷新" in skipped["hint"] and "F5" in skipped["hint"]
    assert (tmp_path / "GoldEA_A.json").exists()
    assert not (tmp_path / "GoldEA_Ghost.json").exists()


def test_run_batch_single_exception_does_not_stop_batch(tmp_path):
    c = FakeClient(indexed=["GoldEA_A", "GoldEA_Boom", "GoldEA_C"],
                   raises=["GoldEA_Boom"])
    out = B.run_batch(["GoldEA_A", "GoldEA_Boom", "GoldEA_C"],
                      from_date=FROM, to_date=TO, client=c, out_dir=tmp_path)

    assert out["ran"] == 2
    assert [r["name"] for r in out["results"]] == ["GoldEA_A", "GoldEA_C"]
    assert [f["name"] for f in out["failed"]] == ["GoldEA_Boom"]
    assert out["failed"][0]["reason"] == "exception"
    assert "测试器崩了" in out["failed"][0]["error"]
    assert (tmp_path / "GoldEA_C.json").exists()
    assert not (tmp_path / "GoldEA_Boom.json").exists()


def test_run_batch_without_mcp_reports_unreachable(tmp_path):
    c = FakeClient(indexed=["GoldEA_A"], connected=False)
    out = B.run_batch(["GoldEA_A"], from_date=FROM, to_date=TO,
                      client=c, out_dir=tmp_path)
    assert out["connected"] is False
    assert out["reason"] == "no_mcp"
    assert out["ran"] == 0 and out["results"] == []
    assert not list(tmp_path.glob("*.json"))


def test_run_batch_default_client_is_constructed(monkeypatch, tmp_path):
    fake = FakeClient(indexed=["GoldEA_A"])
    monkeypatch.setattr(B.mt5_mcp, "Mt5Mcp",
                        lambda timeout=25: fake)
    out = B.run_batch(["GoldEA_A"], from_date=FROM, to_date=TO,
                      out_dir=tmp_path)
    assert out["ran"] == 1
    assert out["results"][0]["path"].endswith("GoldEA_A.json")


def test_batch_module_never_calls_trade_tools():
    src = pathlib.Path(B.__file__).read_text(encoding="utf-8")
    # 只允许测试器/只读接口；源码里不应出现 trade_* 工具调用
    for tool in ("trade_buy", "trade_sell", "trade_close", "trade_modify",
                 "trade_delete"):
        assert tool not in src
    assert 'call("trade_' not in src and "call('trade_" not in src


# ---------------------------------------------------------------- summary_table
def test_summary_table_tolerates_missing_fields():
    batch = {"results": [
        {"name": "GoldEA_A", "run_id": "r1",
         "report": {"profit_factor": 1.5, "max_drawdown": 12.3}},
        {"name": "GoldEA_B", "run_id": "r2", "report": {}},
        {"name": "GoldEA_C", "run_id": "r3", "report": None},
    ]}
    rows = B.summary_table(batch)
    assert [r["EA"] for r in rows] == ["GoldEA_A", "GoldEA_B", "GoldEA_C"]

    a, b, c = rows
    assert a["盈利因子"] == 1.5 and a["最大回撤"] == 12.3
    # 缺的字段必须是 None：0 也是编数字
    assert a["净利润"] is None and a["成交数"] is None and a["夏普"] is None
    for row in (b, c):
        for col in ("净利润", "盈利因子", "成交数", "最大回撤", "夏普"):
            assert row[col] is None


def test_summary_table_reads_nested_metrics_and_empty_batch():
    batch = {"results": [{"name": "GoldEA_A", "run_id": "r1",
                          "report": {"metrics": {"net_profit": 42.0,
                                                 "trades": 7}}}]}
    rows = B.summary_table(batch)
    assert rows[0]["净利润"] == 42.0 and rows[0]["成交数"] == 7
    assert B.summary_table({}) == []
    assert B.summary_table(None) == []


# ---------------------------------------------------------------- CLI 退出码
def _patch_cli(monkeypatch, fake, names):
    monkeypatch.setattr(B, "candidates_from_runs",
                        lambda limit=5, run_dir=None:
                        [{"name": n, "run_id": "r", "source": "test"}
                         for n in names[:limit]])
    monkeypatch.setattr(B.mt5_mcp, "Mt5Mcp", lambda timeout=25: fake)
    return fake


def test_cli_exit_0_when_all_candidates_succeed(tmp_path, monkeypatch):
    fake = _patch_cli(monkeypatch,
                      FakeClient(indexed=["GoldEA_A"]), ["GoldEA_A"])
    code = R.main(["--top", "1", "--from", FROM, "--to", TO,
                   "--out", str(tmp_path)])
    assert code == 0
    assert fake.calls and (tmp_path / "GoldEA_A.json").exists()


def test_cli_exit_2_when_a_candidate_is_skipped(tmp_path, monkeypatch):
    _patch_cli(monkeypatch, FakeClient(indexed=[]), ["GoldEA_Ghost"])
    code = R.main(["--top", "1", "--from", FROM, "--to", TO,
                   "--out", str(tmp_path)])
    assert code == 2
    assert not list(tmp_path.glob("*.json"))


def test_cli_exit_3_when_mcp_unreachable(tmp_path, monkeypatch):
    _patch_cli(monkeypatch, FakeClient(indexed=["GoldEA_A"],
                                       connected=False), ["GoldEA_A"])
    code = R.main(["--top", "1", "--from", FROM, "--to", TO,
                   "--out", str(tmp_path)])
    assert code == 3


# ---------------------------------------------------------------- 真实 MCP（可达才跑）
def test_real_mcp_reachable_or_skip():
    try:
        from gold.mt5_mcp import Mt5Mcp

        client = Mt5Mcp(timeout=3)
        ok = client.handshake()
    except Exception as exc:  # noqa: BLE001 —— 任何连接问题都算环境不可用
        pytest.skip(f"MT5 MCP 不可达：{type(exc).__name__}: {exc}")
    if not ok:
        pytest.skip(f"MT5 MCP 不可达：{client.last_error or '握手失败'}")

    status = B.index_status(client, ["__mt5_batch_probe__"])
    assert status["missing"] == ["__mt5_batch_probe__"]
    assert status["indexed"] == []
