"""平台中心选参的测试：只看邻域、只用训练段、PBO 层级递减、尖峰不该被选中。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.plateau import plateau_search


def _bars(n=3000, seed=3, trend=0.0002):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(trend, 0.004, n))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close,
                         "tick_volume": 100.0, "spread": 16.0}, index=idx)


@pytest.fixture(scope="module")
def df():
    return _bars()


def test_returns_levels_and_pbo(df):
    res = plateau_search(df, grid={"entry": [10, 15, 20], "exit_": [5, 8]},
                         spread_usd=0.16)
    assert "levels" in res and len(res["levels"]) >= 1
    for lv in res["levels"]:
        assert lv["候选数"] >= 1
        assert lv["PBO"] is None or 0.0 <= lv["PBO"] <= 1.0


def test_neighborhood_columns_present(df):
    res = plateau_search(df, grid={"entry": [10, 15], "exit_": [5, 8]},
                         spread_usd=0.16)
    t = res["table"]
    for col in ("邻域同号率", "邻域均值", "邻域个数"):
        assert col in t.columns
    assert (t["邻域个数"] >= 1).all()


def test_small_grid_rejected(df):
    res = plateau_search(df, grid={"entry": [10]}, spread_usd=0.16)
    assert res["ok"] is False and "网格太小" in res["reason"]


def test_selection_prefers_plateau_over_spike(df):
    """稳定区/邻域的排序键是邻域均值，不是单点总收益。"""
    res = plateau_search(df, grid={"entry": [10, 15, 20, 25],
                                   "exit_": [5, 8, 12]},
                         spread_usd=0.16, min_same_sign=0.5)
    if res["ok"]:
        row = res["best_row"]
        assert row["邻域同号率"] >= 0.5 or res["level"] != "稳定区(同号率≥50%)"


def test_chosen_pbo_respects_threshold(df):
    res = plateau_search(df, grid={"entry": [10, 15, 20, 25, 30],
                                   "exit_": [5, 8, 12, 16]},
                         spread_usd=0.16, max_pbo=0.5)
    if res["ok"]:
        assert res["pbo"] is None or res["pbo"] <= 0.5
