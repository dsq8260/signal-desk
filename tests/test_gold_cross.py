"""跨品种横截面策略测试（离线合成面板）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.cross import (
    SCORERS,
    build_panel,
    cross_permutation,
    cross_section_weights,
    momentum_score,
    run_cross,
    zscore_score,
)


def _series(n: int, drift: float, seed: int, vol: float = 0.01,
            start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start * np.cumprod(1 + rng.normal(drift, vol, n))
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close,
                         "volume": 0.0}, index=idx)


def _panel(n: int = 600) -> dict[str, pd.DataFrame]:
    """四个品种：一个持续强势、一个持续弱势、两个随机。

    强弱差距必须明显大于噪声，否则 20 期动量排不出版序——
    那是数据问题，不是策略问题。
    """
    return {
        "STRONG": _series(n, 0.004, 1, vol=0.004),
        "WEAK": _series(n, -0.003, 2, vol=0.004),
        "RAND1": _series(n, 0.0, 3, vol=0.01),
        "RAND2": _series(n, 0.0, 4, vol=0.01),
    }


# ---------------------------------------------------------------- 打分
def test_momentum_score_shape():
    s = momentum_score(_series(100, 0.001, 1), 20).dropna()
    assert len(s) > 50


def test_zscore_score_bounded_mostly():
    s = zscore_score(_series(300, 0.001, 1), 20).dropna()
    assert len(s) > 100
    assert abs(float(s.mean())) < 1.0


def test_scorers_registry():
    assert set(SCORERS) == {"动量", "动量标准分"}


# ---------------------------------------------------------------- 面板
def test_build_panel_aligns():
    close = build_panel(_panel(200))
    assert close.shape[1] == 4
    assert close.index.is_monotonic_increasing
    assert close.notna().all().all()


def test_build_panel_empty():
    assert build_panel({}).empty


# ---------------------------------------------------------------- 权重
def test_weights_long_short_sums_to_zero():
    close = build_panel(_panel(300))
    w = cross_section_weights(close, "动量", 20, top_k=1, bottom_k=1)
    bad = w.dropna(how="all").sum(axis=1).abs()
    live = w.iloc[100:]                       # 跳过预热期
    sums = live.sum(axis=1).abs()
    assert (sums[sums > 0] < 1e-9).all(), "多空应该等权抵消"


def test_weights_long_only_never_negative():
    close = build_panel(_panel(300))
    w = cross_section_weights(close, "动量", 20, top_k=2, long_only=True)
    assert float(w.min().min()) >= -1e-12


def test_weights_picks_the_strong_symbol():
    """持续强势的品种应被明显更多地选为多头（多数时间在场即可）。"""
    close = build_panel(_panel(600))
    w = cross_section_weights(close, "动量", 20, top_k=1, long_only=True)
    tail = w.iloc[-100:]
    strong_share = float(tail["STRONG"].mean())
    others = tail[["WEAK", "RAND1", "RAND2"]].mean().max()
    assert strong_share > 0.6, f"强势品种只被选中 {strong_share:.2f}"
    assert strong_share > others, "强势品种的选中率应高于任何其他品种"


def test_weights_shorts_the_weak_symbol():
    close = build_panel(_panel(600))
    w = cross_section_weights(close, "动量", 20, top_k=1, bottom_k=1)
    tail = w.iloc[-100:]
    assert tail["WEAK"].mean() < -0.3


def test_weights_zero_before_enough_history():
    close = build_panel(_panel(300))
    w = cross_section_weights(close, "动量", 20, top_k=1)
    assert float(w.iloc[:20].abs().sum().sum()) == 0.0


def test_rebalance_reduces_turnover():
    close = build_panel(_panel(400))
    w1 = cross_section_weights(close, "动量", 20, top_k=1, rebalance=1)
    w5 = cross_section_weights(close, "动量", 20, top_k=1, rebalance=5)
    t1 = (w1 - w1.shift(1)).abs().sum().sum()
    t5 = (w5 - w5.shift(1)).abs().sum().sum()
    assert t5 < t1


# ---------------------------------------------------------------- 回测
def test_run_cross_returns_result():
    r = run_cross(_panel(600), n_perm=20)
    if r.equity.empty:
        pytest.skip(f"面板不足：{r.note}")
    assert len(r.equity) > 50
    assert "total_return" in r.stats
    assert 0.0 < r.perm["p_value"] <= 1.0


def test_run_cross_benchmark_present():
    r = run_cross(_panel(600), n_perm=20)
    if r.benchmark.empty:
        pytest.skip("面板不足")
    assert len(r.benchmark) == len(r.equity)


def test_run_cross_rejects_single_symbol():
    one = {"ONLY": _series(600, 0.001, 1)}
    r = run_cross(one, n_perm=10)
    assert "至少需要" in r.note
    assert r.equity.empty


def test_run_cross_rejects_short_panel():
    r = run_cross(_panel(100), n_perm=10)
    assert "至少需要" in r.note


# ---------------------------------------------------------------- 置换
def test_cross_permutation_bounds():
    close = build_panel(_panel(300))
    w = cross_section_weights(close, "动量", 20, top_k=1, bottom_k=1)
    rets = close.pct_change()
    turnover = (w - w.shift(1)).abs().sum(axis=1).fillna(0.0)
    out = cross_permutation(rets, w, turnover, 0.0001, n_perm=30)
    assert 0.0 < out["p_value"] <= 1.0
    assert np.isfinite(out["observed"])


def test_cross_permutation_detects_real_signal():
    """权重重排后收益应明显变差，说明排序确实带信息。"""
    close = build_panel(_panel(800))
    w = cross_section_weights(close, "动量", 20, top_k=1, bottom_k=1)
    rets = close.pct_change()
    turnover = (w - w.shift(1)).abs().sum(axis=1).fillna(0.0)
    out = cross_permutation(rets.iloc[-300:], w.iloc[-300:],
                            turnover.iloc[-300:], 0.0001, n_perm=200)
    assert out["p_value"] < 0.05, f"没测出信号，p={out['p_value']:.3f}"
