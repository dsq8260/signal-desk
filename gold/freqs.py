"""按周期正确年化：把 H1/M30 等频率注册进 metrics.PERIODS_PER_YEAR。

背景（重要）
-----------
`validation.metrics.performance_stats` 用 `PERIODS_PER_YEAR.get(freq, 252)`
做夏普年化。而主干代码一直传 `freq="D"`，于是**小时线也被按 252 年化**，
夏普被低估 √(6048/252) ≈ 4.9 倍。

本模块提供 `freq_key(tf)`：把周期名注册进那张表并返回可用的 key，
让 H1/M30 等按真实频率年化。
"""

from __future__ import annotations

import pandas as pd

from validation import metrics as M

# 每年 bar 数（24h × 252 交易日近似；黄金日均约 23 小时，取整不影响量级）
BARS_PER_YEAR = {
    "M1": 252 * 24 * 60,
    "M5": 252 * 24 * 12,
    "M15": 252 * 24 * 4,
    "M30": 252 * 24 * 2,
    "H1": 252 * 24,
    "H4": 252 * 6,
    "D1": 252,
    "W1": 52,
}


def freq_key(tf: str | None) -> str:
    """返回可传给 performance_stats 的 freq key（必要时先注册）。"""
    if not tf:
        return "D"
    key = str(tf).upper()
    if key in ("D", "DAY", "DAILY"):
        return "D"
    if key not in BARS_PER_YEAR:
        return "D"
    if key not in M.PERIODS_PER_YEAR:
        M.PERIODS_PER_YEAR[key] = BARS_PER_YEAR[key]
    return key


def ann_factor(tf: str | None) -> float:
    """该周期每年的 bar 数。"""
    return float(BARS_PER_YEAR.get(str(tf).upper() if tf else "", 252))


def years_between(rng) -> float:
    """(起始, 结束) 日期区间 → 年数；给不出来时返回 NaN。

    用于把"区间内交易了几回合"换算成"每年几回合"——这是能和 MT5 测试器
    对上的单位（MT5 报的是回合，不是一个回合的两条腿）。
    """
    try:
        a, b = rng[0], rng[1]
    except (TypeError, IndexError, KeyError):
        return float("nan")
    try:
        a = pd.Timestamp(str(a))
        b = pd.Timestamp(str(b))
    except (TypeError, ValueError):
        return float("nan")
    if pd.isna(a) or pd.isna(b) or b <= a:
        return float("nan")
    return float((b - a).days / 365.25)


def per_year(count, years) -> float:
    """把区间内的次数换算成"每年几次"；年数无效/为 0 时返回 NaN。"""
    try:
        c = float(count)
        y = float(years)
    except (TypeError, ValueError):
        return float("nan")
    if c != c or not (y > 0):
        return float("nan")
    return c / y
