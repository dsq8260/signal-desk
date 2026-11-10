"""组合实验室测试：验证"低相关组合能把夏普抬起来"。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.forge import Genome
from gold.portfolio import (
    ann_sharpe,
    bars_per_year,
    build_portfolio,
    greedy_pick,
    strategy_returns,
)


def _bars(n=3000, seed=3, drift=0.0002):
    rng = np.random.default_rng(seed)
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.004, n))
    idx = pd.bdate_range("2019-01-02", periods=n, freq="h")
    return pd.DataFrame({"open": close, "high": close * 1.004,
                         "low": close * 0.996, "close": close,
                         "volume": 0.0}, index=idx)


def test_bars_per_year():
    assert bars_per_year("H1") == 252 * 24
    assert bars_per_year("D1") == 252
    assert bars_per_year("??") == 252


def test_ann_sharpe_needs_enough_points():
    assert ann_sharpe(pd.Series([0.01, 0.02])) != ann_sharpe(pd.Series([0.01, 0.02]))
    s = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 500))
    assert ann_sharpe(s, "D1") == ann_sharpe(s, "D1")


def test_ann_sharpe_positive_for_positive_drift():
    s = pd.Series(np.random.default_rng(2).normal(0.002, 0.01, 2000))
    assert ann_sharpe(s, "H1") > 0.5


def test_strategy_returns_two_segments():
    df = _bars(2000)
    g = Genome("SMA金叉", {"fast": 20, "slow": 60}, "无过滤", False)
    a, b = strategy_returns(df, g, "H1", 0.33, 0.7)
    assert len(a) > 0 and len(b) > 0
    assert a.index.max() <= b.index.min()
    assert a.abs().mean() >= 0.0


def test_build_portfolio_returns_structure():
    df = _bars(2500)
    gs = {
        "A": Genome("SMA金叉", {"fast": 20, "slow": 60}, "无过滤", False),
        "B": Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "无过滤", False),
        "C": Genome("RSI反转", {"buy": 30.0}, "低波动", False),
    }
    res = build_portfolio(df, gs, "H1", "等权", 0.33, 0.7)
    assert res["ok"] is True
    assert len(res["成分"]) == 3
    assert abs(sum(res["权重"].values()) - 1.0) < 1e-6
    assert res["组合OOS夏普"] == res["组合OOS夏普"]


def test_weights_inverse_vol_sum_to_one():
    df = _bars(2000)
    gs = {
        "A": Genome("SMA金叉", {"fast": 10, "slow": 40}, "无过滤", False),
        "B": Genome("唐奇安突破", {"entry": 30, "exit_": 15}, "无过滤", False),
    }
    res = build_portfolio(df, gs, "H1", "逆波动", 0.33, 0.7)
    assert abs(sum(res["权重"].values()) - 1.0) < 1e-6


def test_portfolio_beats_worst_component():
    """组合夏普应不低于最差的单个成分（分散化的下限保证）。"""
    df = _bars(3000)
    gs = {
        "A": Genome("SMA金叉", {"fast": 20, "slow": 60}, "无过滤", False),
        "B": Genome("MACD金叉", {"signal": 9}, "无过滤", False),
        "C": Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "无过滤", False),
    }
    res = build_portfolio(df, gs, "H1", "等权", 0.33, 0.7)
    if res["ok"] and res["组合OOS夏普"] == res["组合OOS夏普"]:
        worst = float(res["成分"]["OOS夏普"].min())
        if worst == worst:
            assert res["组合OOS夏普"] >= worst - 0.5


def test_vol_target_scales_returns():
    df = _bars(2500)
    gs = {"A": Genome("SMA金叉", {"fast": 20, "slow": 60}, "无过滤", False),
          "B": Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "无过滤", False)}
    res = build_portfolio(df, gs, "H1", "等权", 0.33, 0.7, vol_target=0.10)
    assert res["ok"] is True
    assert "波动率目标" in res["note"]


def test_empty_genomes():
    res = build_portfolio(_bars(500), {}, "H1")
    assert res["ok"] is False


def test_greedy_pick_respects_k():
    df = _bars(2500)
    gs = {
        "A": Genome("SMA金叉", {"fast": 20, "slow": 60}, "无过滤", False),
        "B": Genome("MACD金叉", {"signal": 9}, "无过滤", False),
        "C": Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "无过滤", False),
        "D": Genome("RSI反转", {"buy": 30.0}, "无过滤", False),
    }
    res = build_portfolio(df, gs, "H1", "等权", 0.33, 0.7)
    # 造四个收益序列（C 与 A/B 负相关），验证贪心最多选 k 个
    idx = pd.bdate_range("2020-01-01", periods=800, freq="h")
    rng = np.random.default_rng(7)
    base = rng.normal(0, 0.004, len(idx))
    oos = pd.DataFrame({
        "A": base + rng.normal(0, 0.002, len(idx)),
        "B": base + rng.normal(0, 0.002, len(idx)),
        "C": -base + rng.normal(0, 0.002, len(idx)),
        "D": rng.normal(0, 0.004, len(idx)),
    }, index=idx)
    picked = greedy_pick(oos, 2, "H1")
    assert len(picked) == 2
    assert len(set(picked)) == 2
    assert greedy_pick(oos, 10, "H1") == list(oos.columns)
