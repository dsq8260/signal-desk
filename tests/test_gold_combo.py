"""组合挖掘测试：候选枚举、排行榜逻辑（离线）。

干支维度已移除，这里同时守住"不能再冒出干支"的回归。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gold.combo as combo
from gold.combo import ENTRIES, FILTERS, build_candidates, rank


def _bars(n: int = 700, seed: int = 13, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.01, size=n))
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.005,
                         "low": close * 0.995, "close": close,
                         "volume": 0.0}, index=idx)


# ---------------------------------------------------------------- 候选
def test_candidate_count_is_entries_x_filters():
    # 6 种进场 × 3 种过滤 = 18（2026-11-02 新增「均线偏离回归」）
    assert len(build_candidates()) == len(ENTRIES) * len(FILTERS) == 21


def test_candidates_only_cover_entry_and_filter():
    c = build_candidates()
    assert {x["entry"] for x in c} == set(ENTRIES)
    assert {x["filter"] for x in c} == set(FILTERS)
    # 干支维度必须彻底消失：候选里不许再出现 zhi 键
    assert all("zhi" not in x for x in c)


def test_zhi_api_removed_from_combo():
    for gone in ("zhi_series", "day_zhi_index", "ZHI_NAMES"):
        assert not hasattr(combo, gone), f"{gone} 应该已经从 combo 移除"


def test_entries_and_filters_registry():
    assert len(ENTRIES) == 7   # 2026-11-02 新增「均线偏离回归」「ATR通道突破」
    assert len(FILTERS) == 3


# ---------------------------------------------------------------- 排行榜
def test_rank_returns_table_and_best():
    lb = rank(_bars(800), n_perm=20, max_candidates=40)
    if lb.table.empty:
        pytest.skip("合成数据未产生有效候选")
    assert len(lb.table) <= 40
    assert set(lb.table.columns) >= {"进场", "过滤", "IS收益", "OOS收益", "q", "p"}
    assert "干支" not in lb.table.columns
    assert lb.best["spec"]["entry"] in ENTRIES
    assert lb.is_range[0] < lb.oos_range[0]


def test_rank_sorted_by_is_return():
    lb = rank(_bars(800), n_perm=20, max_candidates=40)
    if len(lb.table) < 2:
        pytest.skip("候选太少")
    v = lb.table["IS收益"].to_numpy()
    assert np.all(np.diff(v) <= 1e-12)


def test_rank_best_is_is_best_not_oos():
    """最优必须是 IS 收益最大那行（不能偷看 OOS）。"""
    lb = rank(_bars(800), n_perm=20, max_candidates=40)
    if lb.table.empty:
        pytest.skip("无候选")
    assert lb.best["IS收益"] == pytest.approx(lb.table["IS收益"].max())


def test_rank_rejects_short_history():
    lb = rank(_bars(120), n_perm=10)
    assert "不足" in lb.note
    assert lb.table.empty


def test_global_p_increases_with_candidates():
    """候选越多，同样夏普下全局 p 越大。"""
    from gold.mining import deflated_p

    assert deflated_p(1.5, 200) > deflated_p(1.5, 20)
