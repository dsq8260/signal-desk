"""多品种检验测试：共同区间、逐品种口径、60% 门槛、与 make_gates 的接线。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gold import multisymbol as ms
from gold.forge import Genome
from gold.verify import make_gates


def _bars(n=800, seed=1, drift=0.0004, vol=0.002, price=100.0):
    rng = np.random.default_rng(seed)
    close = price * np.cumprod(1 + rng.normal(drift, vol, n))
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close,
                         "tick_volume": 100.0, "spread": 10.0}, index=idx)


@pytest.fixture()
def market(tmp_path, monkeypatch):
    """三个品种：重叠区间从 2024-06 开始（模拟真实的不齐历史）。"""
    d = tmp_path / "market"
    d.mkdir()
    spec = {}
    for name, start_row, rows in (("AAA", 0, 800), ("BBB", 200, 600),
                                  ("CCC", 400, 400)):
        df = _bars(800)
        sub = df.iloc[start_row:start_row + rows]
        p = d / f"{name}_H1.csv"
        sub.to_csv(p)
        spec[name] = {"file": p.name, "rows": int(len(sub)),
                      "start": str(sub.index[0]), "end": str(sub.index[-1]),
                      "sha256": "x", "hash_scope": "file"}
    # 再放一个"太短"的品种，必须被过滤掉
    tiny = d / "TINY_H1.csv"
    _bars(100).to_csv(tiny)
    spec["TINY"] = {"file": tiny.name, "rows": 100,
                    "start": str(_bars(100).index[0]),
                    "end": str(_bars(100).index[-1]), "sha256": "y"}
    (d / "index.json").write_text(json.dumps({"symbols": spec}), encoding="utf-8")
    monkeypatch.setattr(ms, "MARKET_DIR", d)
    monkeypatch.setattr(ms, "MIN_BARS", 200)   # TINY 只有 100 根，必须被滤掉
    monkeypatch.setattr(ms, "INDEX_FILE", d / "index.json")
    return d


def test_available_filters_short_history(market):
    got = ms.available()
    assert set(got) == {"AAA", "BBB", "CCC"}
    assert "TINY" not in got


def test_common_window_is_intersection(market):
    start, end, syms = ms.common_window()
    assert syms == ["AAA", "BBB", "CCC"] or set(syms) == {"AAA", "BBB", "CCC"}
    # 交集起点必须是三者里最晚的那个起点
    assert start == pd.Timestamp(ms.available()["CCC"]["start"])
    assert start < end


def test_load_bars_shape(market):
    df = ms.load_bars("AAA")
    assert len(df) == 800
    assert {"open", "high", "low", "close"}.issubset(df.columns)


def test_check_returns_per_symbol_detail(market):
    g = Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False)
    r = ms.check(g)
    assert r["ok"] is True
    assert r["n_symbols_tested"] == 3
    assert len(r["detail"]) == 3
    for d in r["detail"]:
        assert {"品种", "通过", "方向", "超额", "盈利因子"}.issubset(d)
    assert r["window"][0] < r["window"][1]


def test_check_ratio_and_60_percent(market):
    g = Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False)
    r = ms.check(g)
    assert 0.0 <= r["ratio"] <= 1.0
    assert r["reach_60"] == (r["n_symbols_tested"] >= 2 and r["ratio"] >= 0.6)


def test_check_handles_no_window(tmp_path, monkeypatch):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "index.json").write_text(json.dumps({"symbols": {}}), encoding="utf-8")
    monkeypatch.setattr(ms, "MARKET_DIR", d)
    monkeypatch.setattr(ms, "INDEX_FILE", d / "index.json")
    r = ms.check(Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False))
    assert r["ok"] is False and r["reason"] == "no_common_window"
    assert r["n_symbols_tested"] == 0


def test_gate_uses_real_multi_symbol_counts():
    """多品种门槛：0/3 → fail；3/3 → pass；未测 → unmeasured。"""
    base = {"方向": 0.60, "超额": 0.05, "回撤": -0.05, "盈利因子": 1.9,
            "每年回合": 60, "块置换p": 0.01, "q": 0.02, "deflated_p": 0.01}
    g_bad = make_gates(base, n_symbols_passed=0, n_symbols_tested=3,
                       wf_positive=3, wf_total=3, same_sign_rate=0.9)
    gate = [x for x in g_bad if "多品种" in x.name][0]
    assert gate.status == "fail"

    g_ok = make_gates(base, n_symbols_passed=3, n_symbols_tested=3,
                      wf_positive=3, wf_total=3, same_sign_rate=0.9)
    gate_ok = [x for x in g_ok if "多品种" in x.name][0]
    assert gate_ok.status == "pass"

    g_na = make_gates(base, n_symbols_passed=None, n_symbols_tested=None,
                      wf_positive=3, wf_total=3, same_sign_rate=0.9)
    gate_na = [x for x in g_na if "多品种" in x.name][0]
    assert gate_na.status == "unmeasured"
