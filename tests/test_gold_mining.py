"""参数挖掘与多重检验校正的测试（离线）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.mining import (
    bh_qvalues,
    deflated_p,
    emax_sharpe,
    make_grid,
    mine,
    split_is_oos,
)


def _bars(n: int = 800, seed: int = 4, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.01, size=n))
    idx = pd.bdate_range("2018-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.004,
                         "low": close * 0.996, "close": close,
                         "volume": 0.0}, index=idx)


# ---------------------------------------------------------------- 网格
def test_make_grid_cartesian_product():
    g = make_grid("双均线", {"fast": [10, 20], "slow": [50, 60, 100]})
    assert len(g) == 6
    assert {"fast": 10, "slow": 50} in g


def test_make_grid_default_single_point():
    g = make_grid("低延迟趋势线LLT")
    assert len(g) == 1 and "n" in g[0]


# ---------------------------------------------------------------- IS/OOS
def test_split_is_oos_ordered_and_nonempty():
    df = _bars(400)
    a, b = split_is_oos(df, 0.7)
    # 默认会切掉 purge gap（400 根 → 自动 40 根），所以两段加起来不再等于总长
    assert len(a) + len(b) + 40 == len(df)
    assert a.index[-1] < b.index[0]
    assert len(b) == 120
    # 显式 purge=0 时才是严格 70/30 全用满
    a0, b0 = split_is_oos(df, 0.7, purge_bars=0)
    assert len(a0) + len(b0) == len(df)


# ---------------------------------------------------------------- 多重检验
def test_emax_sharpe_grows_with_trials():
    vals = [emax_sharpe(k) for k in (1, 10, 100, 1000)]
    assert vals[0] == 0.0
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    # 1000 次试验的期望最大夏普约 sqrt(2 ln 1000) ≈ 3.7
    assert 3.0 < emax_sharpe(1000) < 4.0


def test_deflated_p_penalises_more_trials():
    """同样 1.5 的 OOS 夏普，试验次数越多 p 越大（越不显著）。"""
    p_few = deflated_p(1.5, 5)
    p_many = deflated_p(1.5, 500)
    assert p_many > p_few
    assert 0.0 <= p_few <= 1.0 and 0.0 <= p_many <= 1.0


def test_deflated_p_high_sharpe_is_significant():
    assert deflated_p(5.0, 10) < 0.05


def test_bh_qvalues_known_case():
    # 教科书例子：p = [0.01, 0.02, 0.03, 0.04, 0.05] -> q 单调不减
    q = bh_qvalues([0.01, 0.02, 0.03, 0.04, 0.05])
    assert all(0 <= v <= 1 for v in q)
    assert q == sorted(q)
    assert q[0] == pytest.approx(0.05, abs=1e-9)


def test_bh_keeps_order():
    p = [0.5, 0.001, 0.2]
    q = bh_qvalues(p)
    assert q[1] < q[2] < q[0]


def test_bh_empty():
    assert bh_qvalues([]) == []


# ---------------------------------------------------------------- 端到端
def test_mine_returns_full_table():
    df = _bars(900)
    r = mine(df, "双均线", {"fast": [10, 20], "slow": [40, 60]},
             n_perm=30)
    assert r.n_trials == 4
    assert len(r.table) == 4
    assert set(r.table.columns) >= {"IS收益", "OOS收益", "OOS夏普", "p", "q"}
    assert r.verdict in ("通过", "部分通过", "未通过")
    assert r.best and "fast" in r.best
    assert r.is_range[0] < r.oos_range[0]


def test_mine_selects_best_on_is_not_oos():
    """选参必须只看 IS：最优组的 IS 收益应为全表最大。"""
    df = _bars(900)
    r = mine(df, "双均线", {"fast": [10, 20], "slow": [40, 60]}, n_perm=20)
    best_is = r.table["IS收益"].max()
    chosen = r.table[r.table["IS排名"] == 1]["IS收益"].iloc[0]
    assert chosen == pytest.approx(best_is)


def test_mine_rejects_short_history():
    r = mine(_bars(120), "双均线", {"fast": [10], "slow": [40]}, n_perm=10)
    assert "不足" in r.note
    assert r.table.empty


def test_mine_heatmap_for_two_axes():
    df = _bars(900)
    r = mine(df, "双均线", {"fast": [10, 20], "slow": [40, 60]}, n_perm=20)
    assert r.heatmap.get("x") == "fast"
    assert r.heatmap.get("y") == "slow"
    assert r.heatmap["z"].shape == (2, 2)
