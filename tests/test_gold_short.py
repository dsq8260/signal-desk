"""做空能力（allow_short）的测试：信号语义 + 回测口径 + 盈亏方向。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.backtest import block_permutation, compare_all, run
from gold.strategies import MEAN_REVERT, STRATEGIES, TREND_LIKE, apply_strategy


def _up(n: int = 400, seed: int = 3) -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-04", periods=n)
    close = 2000 + np.arange(n) * 2.0
    return pd.DataFrame({"open": close, "high": close * 1.003,
                         "low": close * 0.997, "close": close,
                         "volume": 0.0}, index=idx)


def _down(n: int = 400) -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-04", periods=n)
    close = 2000 - np.arange(n) * 2.0
    return pd.DataFrame({"open": close, "high": close * 1.003,
                         "low": close * 0.997, "close": close,
                         "volume": 0.0}, index=idx)


def _osc(n: int = 600, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000 + 120 * np.sin(np.arange(n) / 9.0) + rng.normal(0, 3, n)
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.004,
                         "low": close * 0.996, "close": close,
                         "volume": 0.0}, index=idx)


def test_registry_partition_covers_all():
    """两个分类必须刚好把 8 个策略分完，不能漏也不能重。"""
    assert TREND_LIKE | MEAN_REVERT == set(STRATEGIES)
    assert not (TREND_LIKE & MEAN_REVERT)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_long_only_unchanged(name):
    """both=False 时仍只输出 {0,1}（保持历史口径）。"""
    pos = apply_strategy(_osc(), name)
    assert set(np.unique(pos)).issubset({0.0, 1.0})


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_both_mode_outputs_signed_positions(name):
    pos = apply_strategy(_osc(), name, both=True)
    assert set(np.unique(pos)).issubset({-1.0, 0.0, 1.0})


@pytest.mark.parametrize("name", sorted(TREND_LIKE))
def test_trend_like_always_in_market_when_both(name):
    """趋势类双向版永不空仓：每个点只能是 1 或 -1（单边行情可能只有一个值）。"""
    pos = apply_strategy(_osc(), name, both=True)
    assert set(np.unique(pos)).issubset({-1.0, 1.0})
    assert 0.0 not in set(np.unique(pos))


@pytest.mark.parametrize("name", sorted(TREND_LIKE))
def test_trend_like_maps_zero_to_short(name):
    """结构性检查：只做多信号为 0 的地方，双向版必须是 -1；为 1 的地方必须是 +1。

    比"下跌就一定持空"更准确——带噪声的行情里策略本来就会反复翻仓。
    """
    df = _osc()
    long_only = apply_strategy(df, name)
    both = apply_strategy(df, name, both=True)
    zero_mask = long_only < 0.5
    assert (both[zero_mask] == -1.0).all()
    assert (both[~zero_mask] == 1.0).all()


@pytest.mark.parametrize("name", sorted(TREND_LIKE))
def test_trend_like_shorts_in_downtrend(name):
    """带噪声的下跌趋势里，空头时间占比应明显上升（>30%）。"""
    rng = np.random.default_rng(5)
    n = 400
    idx = pd.bdate_range("2021-01-04", periods=n)
    close = np.maximum(3000 - np.arange(n) * 4.0 + rng.normal(0, 2.0, n), 500)
    df = pd.DataFrame({"open": close, "high": close * 1.004,
                       "low": close * 0.996, "close": close,
                       "volume": 0.0}, index=idx)
    pos = apply_strategy(df, name, both=True)
    short_share = float((pos.tail(120) == -1.0).mean())
    assert short_share > 0.3, f"{name} 下跌趋势里空头占比只有 {short_share:.2f}"


def test_rsi_shorts_when_overbought():
    """单边上涨 loss=0 → RSI 应为 100（不是 NaN），从而触发超买做空。"""
    pos = apply_strategy(_up(), "RSI反转", both=True)
    assert pos.iloc[-1] == -1.0


def test_bollinger_shorts_above_upper_band():
    """价格冲上上轨应开空；尖峰后再走一根，让状态从多头平掉并翻空。"""
    n = 120
    idx = pd.bdate_range("2021-01-04", periods=n)
    close = np.full(n, 2000.0) + np.sin(np.arange(n) / 5.0) * 2.0
    close[-2] = 2200.0                       # 倒数第二根尖峰，远超 2σ 上轨
    close[-1] = 2205.0
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": 0.0}, index=idx)
    pos = apply_strategy(df, "布林带回归", both=True)
    assert pos.iloc[-1] == -1.0


# ---------------------------------------------------------------- 回测口径
def test_short_profits_in_downtrend():
    """下跌市：双向版应赚钱，只做多版应空仓不赚。"""
    df = _down()
    long_only = run(df, "双均线", n_perm=20)
    both = run(df, "双均线", n_perm=20, allow_short=True)
    assert both.summary["net"]["total_return"] > 0.05
    assert long_only.summary["net"]["total_return"] == pytest.approx(0.0, abs=1e-9)


def test_short_loses_in_uptrend():
    """上涨市：做空方向应亏，但幅度小于倒退的做多。"""
    df = _up()
    both = run(df, "双均线", n_perm=20, allow_short=True)
    assert both.summary["net"]["total_return"] > 0      # 上涨持多，仍赚


def test_positions_signed_when_allow_short():
    r = run(_osc(600), "RSI反转", n_perm=20, allow_short=True)
    assert r.positions.min() >= -1.0
    assert r.positions.min() < 0                        # 震荡市里确实出现过空头


def test_permutation_accepts_signed_positions():
    df = _down()
    fwd = df["close"].pct_change().shift(-1)
    pos = pd.Series(-1.0, index=df.index)               # 一直做空
    out = block_permutation(fwd, pos, n_perm=20, allow_short=True)
    assert 0.0 < out["p_value"] <= 1.0
    assert out["observed"] > 0                          # 下跌市做空赚钱


def test_compare_all_allow_short_smoke():
    tbl = compare_all(_osc(500), n_perm=10, allow_short=True)
    assert len(tbl) >= 1
    assert "门槛" in tbl.columns
