"""分层搜索测试：剪枝有效性 + 试验次数如实记录 + 选参不看 OOS。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.strata import DEFAULT_KEEP, PARAM_AXES, search


def _bars(n: int = 900, seed: int = 17, drift: float = 0.0006) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.01, size=n))
    idx = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.005,
                         "low": close * 0.995, "close": close,
                         "volume": 0.0}, index=idx)


def test_param_axes_not_empty():
    assert set(PARAM_AXES) == {"SMA金叉", "唐奇安突破", "RSI反转", "MACD金叉", "LLT上穿"}
    for name, grid in PARAM_AXES.items():
        assert len(grid) >= 3, name
        assert all(isinstance(p, dict) for p in grid)


def test_keep_defaults_ordered():
    assert DEFAULT_KEEP["l1"] >= DEFAULT_KEEP["l2"] >= DEFAULT_KEEP["l3"] >= 1
    # 干支层已移除，旧的 l4 键不该再留在默认值里
    assert "l4" not in DEFAULT_KEEP


def test_search_evaluates_far_fewer_than_naive():
    r = search(_bars(), n_perm=20)
    if r.table.empty:
        pytest.skip("合成数据未产生候选")
    assert r.trials_evaluated < r.trials_naive * 0.75, (
        f"分层没省下多少：{r.trials_evaluated} vs {r.trials_naive}")


def test_trials_counted_honestly_not_just_kept():
    """评估数必须远大于最终保留数（说明剪掉的那些也计入了试验次数）。"""
    r = search(_bars(), n_perm=20)
    if r.table.empty:
        pytest.skip("无候选")
    assert r.trials_evaluated > len(r.table)


def test_layer_log_covers_three_layers():
    r = search(_bars(), n_perm=20)
    if r.table.empty:
        pytest.skip("无候选")
    assert len(r.layer_log) == 3
    assert [l["层"][:2] for l in r.layer_log] == ["L1", "L2", "L3"]
    for l in r.layer_log:
        assert l["评估"] >= l["保留"]


def test_best_is_is_best():
    r = search(_bars(), n_perm=20)
    if r.table.empty:
        pytest.skip("无候选")
    assert r.best["IS收益"] == pytest.approx(r.table["IS收益"].max())


def test_table_sorted_by_is_return():
    r = search(_bars(), n_perm=20)
    if len(r.table) < 2:
        pytest.skip("候选太少")
    v = r.table["IS收益"].to_numpy()
    assert np.all(np.diff(v) <= 1e-12)


def test_q_values_present_and_bounded():
    r = search(_bars(), n_perm=20)
    if r.table.empty:
        pytest.skip("无候选")
    q = r.table["q"].to_numpy()
    assert np.all(q >= 0) and np.all(q <= 1)


def test_short_history_rejected():
    r = search(_bars(200), n_perm=10)
    assert "不足" in r.note
    assert r.table.empty


def test_global_p_uses_trials_not_kept():
    """同一份数据下，评估次数越多 E[max SR] 越大。"""
    from gold.mining import deflated_p

    assert deflated_p(1.2, 60) > deflated_p(1.2, 6)
