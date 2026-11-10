"""点差模型的测试：实测优先、静态兜底、未知品种回落。"""

from __future__ import annotations

import json

import pytest

from gold import symbols as S


@pytest.fixture()
def with_market(tmp_path, monkeypatch):
    idx = tmp_path / "index.json"
    idx.write_text(json.dumps({"symbols": {
        "GOLD#": {"spread_points_median": 16.0, "rows": 33545},
        "EURUSD#": {"spread_points_median": 10.0, "rows": 15778},
        "USDJPY#": {"spread_points_median": 8.0, "rows": 13402},
    }}), encoding="utf-8")
    monkeypatch.setattr(S, "MARKET_INDEX", idx)
    return idx


def test_measured_gold_is_016(with_market):
    assert S.spread_for("GOLD") == pytest.approx(0.16)
    assert S.spread_for("GOLD#") == pytest.approx(0.16)
    assert S.spread_source("GOLD") == "measured"


def test_measured_fx_is_scaled_by_point(with_market):
    assert S.spread_for("EURUSD") == pytest.approx(0.0001)
    assert S.spread_for("USDJPY") == pytest.approx(0.008)


def test_measured_beats_static(with_market):
    # 静态表里 GOLD 是 0.33，实测 0.16 必须优先
    assert S.SPREADS["GOLD"][0] == 0.33
    assert S.spread_for("GOLD") != S.SPREADS["GOLD"][0]


def test_symbol_without_measurement_falls_back_to_static(with_market):
    assert S.spread_for("XAGUSD") == pytest.approx(0.030)
    assert S.spread_source("XAGUSD") == "static"


def test_unknown_symbol_falls_back_to_default(with_market):
    assert S.spread_for("NOPE") == pytest.approx(S.DEFAULT_SPREAD)
    assert S.spread_source("NOPE") == "default"


def test_missing_index_file_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "MARKET_INDEX", tmp_path / "nope.json")
    assert S.spread_for("GOLD") == pytest.approx(0.33)
    assert S.spread_source("GOLD") == "static"


def test_broken_json_does_not_raise(tmp_path, monkeypatch):
    bad = tmp_path / "index.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(S, "MARKET_INDEX", bad)
    assert S.spread_for("GOLD") == pytest.approx(0.33)


def test_point_values_match_mt5_spec():
    assert S.point_value("GOLD") == pytest.approx(0.01)
    assert S.point_value("EURUSD") == pytest.approx(1e-05)
    assert S.point_value("USDJPY") == pytest.approx(0.001)
    assert S.point_value("NOPE") is None


def test_none_and_empty_symbol():
    assert S.spread_for(None) == S.DEFAULT_SPREAD
    assert S.spread_for("") == S.DEFAULT_SPREAD
    assert S.unit_note(None) == ""
