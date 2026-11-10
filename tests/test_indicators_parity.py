"""指标一致性测试：我们的实现必须和第三方库/MT5 口径逐根一致。

这组测试的价值：指标是回测的地基。指标偏一点，所有结论都不可信。
2026-11 就是靠这个机制发现「ATR 用 TR 简单均值、而 MT5 的 iATR 是 Wilder」
——两边的强趋势过滤会不一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.factors import atr_pct
from gold.indicator_check import _bars, compare, run_report
from gold.strategies import atr, donchian_signal, ma_cross_signal

pytest.importorskip("pandas_ta_classic", reason="需要 pandas-ta-classic 做交叉校验")

import pandas_ta_classic as pta  # noqa: E402


def test_atr_is_wilder_same_as_mt5_iatr():
    """ATR 必须是 Wilder 平滑（MT5 的 iATR 口径），不是 TR 的简单均值。"""
    df = _bars(400)
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    wilder = tr.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
    ours = atr(df, 14)
    assert np.allclose(ours.dropna(), wilder.dropna(), atol=1e-10)
    # 简单均值必须**不**相等（否则说明又退回旧口径了）
    sma = tr.rolling(14, min_periods=2).mean()
    assert not np.allclose(ours.iloc[300:], sma.iloc[300:], atol=1e-6)


def test_atr_pct_reuses_same_atr_definition():
    """atr_pct 必须复用 strategies.atr，不能自己另写一份。"""
    df = _bars(300)
    expect = atr(df, 14) / df["close"].astype(float)
    got = atr_pct(df, 14)
    assert np.allclose(got.dropna(), expect.dropna(), atol=1e-12)


def test_sma_exact_match():
    df = _bars()
    ours = df["close"].rolling(10, min_periods=10).mean()
    assert np.allclose(ours.dropna(), pta.sma(df["close"], length=10).dropna())


def test_donchian_exact_match_with_correct_columns():
    """pandas-ta 的列序是 DCL / DCM / DCU —— 上轨是 DCU。"""
    df = _bars()
    h, l = df["high"], df["low"]
    don = pta.donchian(h, l, lower_length=10, upper_length=20)
    hi = h.rolling(20, min_periods=20).max()
    lo = l.rolling(10, min_periods=10).min()
    assert np.allclose(hi.dropna(), don["DCU_10_20"].dropna())
    assert np.allclose(lo.dropna(), don["DCL_10_20"].dropna())


def test_bollinger_uses_ddof0():
    df = _bars()
    c = df["close"]
    bb = pta.bbands(c, length=20, std=2.0)
    mid_ta = bb[[x for x in bb.columns if x.startswith("BBM")][0]]
    up_ta = bb[[x for x in bb.columns if x.startswith("BBU")][0]]
    mid = c.rolling(20, min_periods=20).mean()
    up = mid + 2 * c.rolling(20, min_periods=20).std(ddof=0)
    assert np.allclose(mid.dropna(), mid_ta.dropna())
    assert np.allclose(up.dropna(), up_ta.dropna(), atol=1e-8)


def test_ema_macd_rsi_converge_after_warmup():
    """指数平滑类指标只在预热期有种子差异，250 根后必须收敛。"""
    df = _bars()
    c = df["close"]
    ema_ours = c.ewm(span=12, adjust=False).mean()
    ema_ta = pta.ema(c, length=12)
    r = compare("EMA", ema_ours, ema_ta, warmup=250)
    assert r["一致率"] == 1.0, r

    rsi_ours = (lambda n=14: (
        100 - 100 / (1 + (c.diff().clip(lower=0).ewm(alpha=1 / n, adjust=False)
                          .mean()
                          / (-c.diff().clip(upper=0)).ewm(alpha=1 / n,
                                                          adjust=False)
                          .mean().replace(0, np.nan)))))()
    r2 = compare("RSI", rsi_ours, pta.rsi(c, length=14), tol=1e-6, warmup=250)
    assert r2["一致率"] == 1.0, r2


def test_full_report_all_consistent():
    """整份报告必须全绿（否则说明某个指标偏离了第三方实现）。"""
    r = run_report()
    bad = r[~r["结论"].isin(["一致", "设计如此（防未来函数）"])]
    assert bad.empty, bad.to_string()


def test_compare_skips_warmup_and_reports_it():
    a = pd.Series(np.arange(10, dtype=float))
    b = pd.Series(np.arange(10, dtype=float))
    b.iloc[:3] = 999.0                     # 前 3 根故意不同
    r = compare("x", a, b, warmup=3)
    assert r["预热跳过"] == 3
    assert r["一致率"] == 1.0
    r0 = compare("x", a, b, warmup=0)
    assert r0["结论"] == "偏离"


def test_compare_detects_real_deviation():
    a = pd.Series(np.arange(100, dtype=float))
    b = a.copy()
    b.iloc[-1] += 1.0
    assert compare("x", a, b, warmup=0)["结论"] == "偏离"
