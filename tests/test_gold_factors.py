"""因子层、挖矿台账、跨策略全局校正的测试（全部离线）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.factors import (
    atr_pct,
    build_features,
    drawdown_state,
    factor_ic,
    gate_positions,
    realized_vol,
    regime_mask,
    trend_strength,
)
from gold.ledger import Record, append, clear, global_deflated_p, load, summary


def _bars(n: int = 600, seed: int = 9, drift: float = 0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.01, size=n))
    idx = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.005,
                         "low": close * 0.995, "close": close,
                         "volume": 0.0}, index=idx)


# ---------------------------------------------------------------- 因子
def test_features_have_expected_columns():
    f = build_features(_bars())
    for col in ("vol", "atr_pct", "trend", "dd", "vol_pct", "atr_pct_pct", "trend_pct"):
        assert col in f.columns
    assert len(f) == 600


def test_realized_vol_positive():
    v = realized_vol(_bars(), 20).dropna()
    assert len(v) > 0 and (v > 0).all()


def test_atr_pct_in_reasonable_range():
    a = atr_pct(_bars(), 14).dropna()
    assert len(a) > 0
    assert a.max() < 1.0          # ATR 不应超过价格本身


def test_drawdown_state_non_positive():
    d = drawdown_state(_bars(), 250).dropna()
    assert (d <= 1e-12).all()


def test_percentile_rank_is_bounded():
    f = build_features(_bars())
    for col in ("vol_pct", "atr_pct_pct", "trend_pct"):
        v = f[col].dropna()
        assert len(v) > 0
        assert v.min() >= 0.0 and v.max() <= 1.0


def test_regime_mask_thresholds():
    f = build_features(_bars())
    low = regime_mask(f, "vol_pct", 0.3, "below")
    high = regime_mask(f, "vol_pct", 0.7, "above")
    assert low.dtype == bool and high.dtype == bool
    # below 0.3 与 above 0.7 不应同时为真
    both = (low & high).sum()
    assert both == 0


def test_regime_mask_unknown_column_raises():
    with pytest.raises(KeyError):
        regime_mask(build_features(_bars()), "不存在", 0.5)


def test_gate_positions_zeroes_outside_mask():
    idx = pd.bdate_range("2024-01-01", periods=10)
    pos = pd.Series([1.0] * 10, index=idx)
    mask = pd.Series([True, False] * 5, index=idx)
    g = gate_positions(pos, mask)
    assert g.iloc[0] == 1.0 and g.iloc[1] == 0.0
    assert set(np.unique(g)).issubset({0.0, 1.0})


def test_factor_ic_detects_perfect_signal():
    idx = pd.bdate_range("2020-01-01", periods=200)
    fwd = pd.Series(np.linspace(-0.05, 0.05, 200), index=idx)
    feat = pd.DataFrame({"x": fwd * 2}, index=idx)      # 完全同向
    r = factor_ic(feat, fwd, "x")
    assert r["ic"] == pytest.approx(1.0, abs=1e-9)
    assert r["n"] == 200


def test_factor_ic_returns_nan_when_too_few():
    idx = pd.bdate_range("2020-01-01", periods=10)
    feat = pd.DataFrame({"x": np.arange(10.0)}, index=idx)
    fwd = pd.Series(np.arange(10.0), index=idx)
    assert factor_ic(feat, fwd, "x")["ic"] != factor_ic(feat, fwd, "x")["ic"]


# ---------------------------------------------------------------- 台账
def test_ledger_append_and_summary(tmp_path):
    p = tmp_path / "ledger.json"
    clear(p)
    assert summary(p)["runs"] == 0

    append(Record(ts="t1", strategy="双均线", n_trials=16, oos_sharpe=1.2,
                  verdict="未通过"), p)
    append(Record(ts="t2", strategy="LLT", n_trials=7, oos_sharpe=0.95,
                  verdict="未通过"), p)
    s = summary(p)
    assert s["runs"] == 2
    assert s["total_trials"] == 23
    assert s["best_oos_sharpe"] == pytest.approx(1.2)
    assert set(s["strategies"]) == {"双均线", "LLT"}
    assert load(p)[0]["strategy"] == "双均线"


def test_ledger_corrupt_file_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert load(p) == []


def test_global_deflated_penalises_total_trials():
    """同样的夏普，累计试验越多，全局 p 越大。"""
    few = global_deflated_p(1.5, 20)
    many = global_deflated_p(1.5, 2000)
    assert many > few
    assert 0.0 <= few <= 1.0 and 0.0 <= many <= 1.0


def test_global_p_better_with_high_sharpe():
    assert global_deflated_p(6.0, 100) < 0.05


def test_summary_global_p_uses_total_trials(tmp_path):
    p = tmp_path / "l.json"
    clear(p)
    for i in range(3):
        append(Record(ts=f"t{i}", strategy="A", n_trials=100, oos_sharpe=1.1), p)
    s = summary(p)
    assert s["total_trials"] == 300
    # 300 次的期望最大夏普 ≈ 3.38，1.1 远低于它 -> 不显著
    assert s["global_p"] > 0.05
