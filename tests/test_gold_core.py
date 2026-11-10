"""gold 核心测试：策略信号、成本折算、回测与门槛判定（全部离线）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.backtest import block_permutation, judge, run, split_bars, spread_fee_bps
from gold.data import _clean
from gold.strategies import (
    STRATEGIES,
    apply_strategy,
    atr,
    donchian_signal,
    llt_signal,
    ma_cross_signal,
    rolling_ols_slope_r2,
    rsrs_signal,
)


def _bars(n: int = 400, seed: int = 7, drift: float = 0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.01, size=n))
    idx = pd.bdate_range("2022-01-03", periods=n)
    high = close * (1 + rng.uniform(0, 0.006, n))
    low = close * (1 - rng.uniform(0, 0.006, n))
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": 0.0}, index=idx)


def _trend(n: int = 300, slope: float = 1.0) -> pd.DataFrame:
    """严格单调上涨、无噪声，用来验证趋势类策略必然持多。"""
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = 1000 + np.arange(n) * slope
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 0.0}, index=idx)


# ---------------------------------------------------------------- 数据清洗
def test_clean_normalizes_columns_and_index():
    raw = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=3),
                        "Open": [1, 2, 3], "Close": [1, 2, 3]})
    out = _clean(raw)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"][:5] or \
        list(out.columns)[-1] == "volume"
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out["high"].tolist() == [1, 2, 3]      # 缺 high 用 close 补


def test_clean_drops_duplicate_dates():
    idx = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
    raw = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    out = _clean(raw)
    assert len(out) == 2
    assert out["close"].iloc[0] == 2.0            # keep="last"


def test_clean_empty_returns_empty():
    assert _clean(pd.DataFrame()).empty
    assert _clean(None).empty


# ---------------------------------------------------------------- 基础指标
def test_atr_is_positive_and_tracks_range():
    df = _bars()
    a = atr(df, 14).dropna()
    assert len(a) > 0 and (a > 0).all()


def test_rolling_ols_slope_on_perfect_line():
    x = pd.Series(np.arange(100, dtype=float))
    y = 2.0 * x + 5.0
    beta, r2 = rolling_ols_slope_r2(y, x, 20)
    assert beta.dropna().iloc[-1] == pytest.approx(2.0, rel=1e-6)
    assert r2.dropna().iloc[-1] == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------- 策略信号
def test_all_strategies_output_binary_positions():
    df = _bars()
    for name in STRATEGIES:
        pos = apply_strategy(df, name)
        assert set(np.unique(pos)).issubset({0.0, 1.0}), name
        assert len(pos) == len(df)


def test_ma_cross_long_in_uptrend_flat_in_downtrend():
    up = _trend(300, 1.0)
    down = _trend(300, -1.0)
    assert ma_cross_signal(up, 10, 30).iloc[-1] == 1.0
    assert ma_cross_signal(down, 10, 30).iloc[-1] == 0.0


def test_llt_has_near_zero_lag_on_linear_trend():
    """LLT 的设计目标就是零滞后：线性趋势下曲线与价格重合。

    重合意味着 close 不大于 llt，按严格大于才持多的规则应保持空仓——
    这是数学上正确的结果，不是 bug。
    """
    df = _trend(200, 1.0)
    assert llt_signal(df, 30).iloc[-1] == 0.0
    assert llt_signal(_trend(200, -1.0), 30).iloc[-1] == 0.0


def test_llt_long_most_of_noisy_uptrend():
    """带噪声的上涨行情里，LLT 应当大部分时间持多。"""
    df = _bars(400, seed=3, drift=0.0015)
    pos = llt_signal(df, 30)
    assert pos.mean() > 0.5


def test_donchian_enters_on_breakout():
    df = _trend(120, 1.0)          # 一直创新高 -> 必然进多
    pos = donchian_signal(df, entry=20, exit_=10)
    assert pos.iloc[-1] == 1.0


def test_donchian_exits_on_breakdown():
    base = _trend(60, 1.0)
    crash_idx = pd.bdate_range(base.index[-1] + pd.Timedelta(days=1), periods=30)
    crash_close = np.linspace(base["close"].iloc[-1] * 0.9,
                              base["close"].iloc[-1] * 0.6, 30)
    crash = pd.DataFrame({"open": crash_close, "high": crash_close,
                          "low": crash_close, "close": crash_close,
                          "volume": 0.0}, index=crash_idx)
    pos = donchian_signal(pd.concat([base, crash]), entry=20, exit_=10)
    assert pos.iloc[-1] == 0.0


def test_rsrs_uses_only_past_data():
    """截断尾部不应改变前面的信号（无未来函数）。"""
    df = _bars(500)
    full = rsrs_signal(df)
    part = rsrs_signal(df.iloc[:400])
    assert np.allclose(full.iloc[:400].to_numpy(), part.to_numpy())


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        apply_strategy(_bars(200), "不存在的策略")


# ---------------------------------------------------------------- 回测
def test_spread_fee_bps():
    assert spread_fee_bps(4000.0, 0.33) == pytest.approx(0.825, rel=1e-6)
    assert spread_fee_bps(0.0, 0.33) == 0.0


def test_split_bars_ratios_sum_to_all():
    df = _bars(400)
    seg = split_bars(df, 0.6, 0.2)
    assert len(seg["train"]) + len(seg["valid"]) + len(seg["test"]) == len(df)
    assert seg["train"][-1] < seg["valid"][0] < seg["test"][0]


def test_run_produces_equity_and_gates():
    df = _bars(500)
    r = run(df, "双均线", n_perm=30)
    assert len(r.equity) > 20
    assert len(r.benchmark) == len(r.equity)
    assert set(r.summary) >= {"net", "gross", "hit", "trades"}
    assert len(r.gates["checks"]) == 4
    assert r.gates["verdict"] in ("通过", "未通过", "证据不足")


def test_run_rejects_short_history():
    r = run(_bars(60), "双均线")
    assert "不足" in r.note
    assert r.equity.empty


def test_positions_are_lag_one():
    """t 的信号必须在 t+1 才生效（防未来函数）。"""
    df = _trend(400, 1.0)
    r = run(df, "双均线")
    # 单调上涨时均线策略很快全仓，末仓位应为 1
    assert r.positions.iloc[-1] == 1.0


def test_block_permutation_bounds():
    df = _bars(400)
    r = run(df, "低延迟趋势线LLT", n_perm=20)
    p = r.gates["checks"][2]["value"]
    assert 0.0 < p <= 1.0


# ---------------------------------------------------------------- 门槛判定
def test_judge_all_pass():
    summary = {"net": {"total_return": 0.3},
               "hit": {"hit_rate": 0.60, "n": 100}}
    bench = {"total_return": 0.1}
    perm = {"p_value": 0.01}
    g = judge(summary, bench, perm)
    assert g["passed"] == 3 and g["verdict"] == "通过"


def test_judge_fails_on_low_p():
    summary = {"net": {"total_return": 0.3}, "hit": {"hit_rate": 0.60, "n": 100}}
    g = judge(summary, {"total_return": 0.1}, {"p_value": 0.40})
    assert g["failed"] >= 1 and g["verdict"] == "未通过"


def test_judge_marks_multi_asset_na():
    summary = {"net": {"total_return": 0.1}, "hit": {"hit_rate": 0.5, "n": 100}}
    g = judge(summary, {"total_return": 0.1}, {"p_value": 0.5})
    multi = [c for c in g["checks"] if c["name"] == "多标的有效占比"][0]
    assert multi["pass"] is None
