"""组合与暴露限制测试：相关性去重 / 权重 / 暴露上限 / 压力测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.portfolio_risk import (
    Exposure,
    build_plan,
    combine,
    correlation,
    pick_uncorrelated,
    risk_contribution,
    stress_test,
    weights,
)


def _series(n=500, seed=1, vol=0.01):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, vol, n))


# ---------------------------------------------------------------- 相关性
def test_correlation_matrix_shape_and_diagonal():
    r = {"A": _series(seed=1), "B": _series(seed=2)}
    c = correlation(r)
    assert c.shape == (2, 2)
    assert c.loc["A", "A"] == pytest.approx(1.0)
    assert abs(c.loc["A", "B"]) < 0.3


def test_correlation_single_ea():
    c = correlation({"A": _series()})
    assert c.shape == (1, 1) and c.iloc[0, 0] == 1.0


def test_correlation_empty():
    assert correlation({}).empty


def test_pick_uncorrelated_drops_duplicates():
    base = _series(seed=5)
    r = {"A": base, "A2": base * 1.0001, "B": _series(seed=6)}
    c = correlation(r)
    picked = pick_uncorrelated(c, thresh=0.7, keep_order=["A", "A2", "B"])
    assert "A" in picked
    assert "A2" not in picked          # 和 A 几乎完全相关，必须去掉
    assert "B" in picked


def test_pick_uncorrelated_respects_priority():
    base = _series(seed=7)
    r = {"差": base, "好": base * 1.0001}
    c = correlation(r)
    assert pick_uncorrelated(c, 0.7, ["好", "差"]) == ["好"]


def test_pick_uncorrelated_empty():
    assert pick_uncorrelated(pd.DataFrame()) == []


# ---------------------------------------------------------------- 权重
def test_weights_sum_to_one_for_all_kinds():
    vol = {"A": 0.01, "B": 0.02, "C": 0.04}
    for kind in ("equal", "inverse_vol", "risk_parity"):
        w = weights(kind, vol, max_weight=1.0)
        assert float(w.sum()) == pytest.approx(1.0)
        assert (w >= 0).all()


def test_equal_weights_are_uniform():
    w = weights("equal", {"A": 0.01, "B": 0.05}, max_weight=1.0)
    assert float(w["A"]) == pytest.approx(0.5)


def test_inverse_vol_gives_lower_weight_to_volatile():
    w = weights("inverse_vol", {"稳": 0.01, "抖": 0.04}, max_weight=1.0)
    assert w["稳"] > w["抖"]


def test_risk_parity_equalises_risk_contribution():
    vol = {"A": 0.01, "B": 0.04}
    w = weights("risk_parity", vol, max_weight=1.0)
    rc = risk_contribution(w, vol)
    assert abs(float(rc["A"]) - float(rc["B"])) < 0.05


def test_max_weight_cap_is_enforced():
    # 3 个 EA、上限 40% → 可分配（3×40% > 100%）
    w = weights("inverse_vol", {"A": 0.01, "B": 0.02, "C": 0.05},
                max_weight=0.4)
    assert float(w.max()) <= 0.4 + 1e-9
    assert float(w.sum()) == pytest.approx(1.0)


def test_infeasible_cap_raises():
    """2 个 EA、上限 30% 根本凑不满 100% → 必须报错而不是悄悄违反。"""
    with pytest.raises(ValueError):
        weights("equal", {"A": 0.01, "B": 0.01}, max_weight=0.3)


def test_weights_requires_vol():
    with pytest.raises(ValueError):
        weights("equal", None)


def test_weights_rejects_unknown_kind():
    with pytest.raises(ValueError):
        weights("神秘方法", {"A": 0.01})


# ---------------------------------------------------------------- 暴露
def test_exposure_summary_and_no_breach():
    e = Exposure(equity=10000.0, positions={
        "A": ("GOLD", "多", 2000.0), "B": ("EURUSD", "空", 1000.0)},
        leverage=100.0)
    s = e.summary()
    assert s["品种最大占比"] == pytest.approx(0.2)
    assert s["方向最大占比"] == pytest.approx(0.2)
    assert s["总保证金占比"] == pytest.approx(0.003)
    assert e.breaches() == []


def test_exposure_flags_same_direction_concentration():
    e = Exposure(equity=10000.0, positions={
        "A": ("GOLD", "多", 5000.0), "B": ("GOLD", "多", 4000.0)},
        leverage=100.0)
    bad = e.breaches()
    assert any("单方向" in b for b in bad)
    assert any("单品种" in b for b in bad)


def test_exposure_flags_margin_overuse():
    e = Exposure(equity=10000.0, positions={
        "A": ("GOLD", "多", 400000.0)}, leverage=10.0,
        max_margin_pct=0.5)
    assert any("保证金" in b for b in e.breaches())


def test_exposure_detail_table():
    e = Exposure(equity=1000.0, positions={"A": ("GOLD", "多", 100.0)})
    d = e.detail()
    assert list(d.columns) == ["EA", "品种", "方向", "名义价值", "占权益", "保证金"]
    assert float(d.iloc[0]["保证金"]) == pytest.approx(1.0)


# ---------------------------------------------------------------- 压力
def test_stress_test_monotonic_in_cost():
    r = _series(n=200, seed=3, vol=0.005)
    st = stress_test(r, spread_mult=2.0, extra_slip_bps=10.0)
    base = float(st[st["情景"] == "基准"]["组合收益"].iloc[0])
    both = float(st[st["情景"] == "两者叠加"]["组合收益"].iloc[0])
    assert both < base, "成本叠加必须让收益变差"
    sp = float(st[st["情景"].str.contains("点差")]["组合收益"].iloc[0])
    assert sp < base


def test_stress_test_includes_gap_and_streak():
    r = _series(n=200, seed=4, vol=0.01)
    st = stress_test(r, gap_pct=0.05, n_loss_streak=3)
    assert any("跳空" in x for x in st["情景"])
    assert any("连亏" in x for x in st["情景"])
    gap_row = st[st["情景"].str.contains("跳空")].iloc[0]
    assert float(gap_row["组合收益"]) < float(
        st[st["情景"] == "基准"]["组合收益"].iloc[0])


def test_stress_test_zero_returns_still_charges_cost():
    st = stress_test(pd.Series(np.zeros(100)), trades_per_year=200.0)
    base = float(st[st["情景"] == "基准"]["组合收益"].iloc[0])
    assert base < 0


# ---------------------------------------------------------------- 组合
def test_combine_weighted_sum():
    a = pd.Series([0.01, 0.02])
    b = pd.Series([0.02, 0.00])
    w = pd.Series({"a": 0.5, "b": 0.5})
    got = combine({"a": a, "b": b}, w)
    assert got.iloc[0] == pytest.approx(0.015)
    assert got.iloc[1] == pytest.approx(0.01)


def test_combine_empty():
    assert combine({}, pd.Series(dtype=float)).empty


def test_build_plan_end_to_end():
    base = _series(seed=9)
    returns = {"A": base, "A2": base * 1.0001, "B": _series(seed=10)}
    vols = {"A": 0.01, "A2": 0.011, "B": 0.02}
    pos = {"A": ("GOLD", "多", 2000.0), "A2": ("GOLD", "多", 2000.0),
           "B": ("EURUSD", "空", 1000.0)}
    plan = build_plan(returns, vols, 20000.0, pos,
                      kind="risk_parity", priority=["A", "A2", "B"])
    assert len(plan.table) == 2                 # A2 被去重
    assert float(plan.weights.sum()) == pytest.approx(1.0)
    assert "相关性去重 3 → 2" in plan.note
    assert not plan.stress.empty
    assert plan.exposure["总保证金占比"] > 0


def test_build_plan_without_candidates():
    plan = build_plan({}, {}, 10000.0, {})
    assert plan.note == "没有可用 EA"
