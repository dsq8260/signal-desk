"""各品种的点差（价格单位）与合约特性。

为什么必须分开
--------------
黄金的点差是「美元/盎司」（实测约 0.16），而 EURUSD 的点差是「报价货币单位」
（实测约 0.00010）。如果给 EURUSD 传 0.33，费率会变成 0.33/1.11 ≈ 30%/笔，
回测直接亏到 -100%。

点差来源（优先级从高到低）
------------------------
1. `data/market/index.json` 里 MT5 实测的 `spread_points_median`（points × 点值）；
2. 本文件的静态表（保守估计，用于没有实测数据时兜底）。
`spread_source()` 会告诉你当前用的是哪一种。
"""

from __future__ import annotations

import json
import pathlib

# 品种 -> (静态点差, 单位说明)；实测数据可用时会被覆盖
SPREADS: dict[str, tuple[float, str]] = {
    "GOLD": (0.33, "美元/盎司"),
    "XAUUSD": (0.33, "美元/盎司"),
    "XAGUSD": (0.030, "美元/盎司"),
    "SILVER": (0.030, "美元/盎司"),
    "EURUSD": (0.00012, "美元（报价货币）"),
    "GBPUSD": (0.00015, "美元（报价货币）"),
    "USDCHF": (0.00015, "瑞郎（报价货币）"),
    "USDJPY": (0.015, "日元（报价货币）"),
}

# 1 点 = 多少价格单位（来自 MT5 品种规格，2026-11-01 实测）
POINTS: dict[str, float] = {
    "GOLD": 0.01, "XAUUSD": 0.01, "XAGUSD": 0.001, "SILVER": 0.001,
    "EURUSD": 1e-05, "GBPUSD": 1e-05, "USDCHF": 1e-05, "USDJPY": 0.001,
}

DIGITS: dict[str, int] = {
    "GOLD": 2, "XAUUSD": 2, "XAGUSD": 3, "SILVER": 3,
    "EURUSD": 5, "GBPUSD": 5, "USDCHF": 5, "USDJPY": 3,
}

DEFAULT_SPREAD = 0.33
MARKET_INDEX = pathlib.Path("data/market/index.json")


def _key(symbol: str | None) -> str:
    return str(symbol or "").upper().replace("#", "").strip()


def _measured() -> dict:
    """读实测点差（缺文件/坏 JSON 都退回空字典，不抛）。"""
    try:
        if not MARKET_INDEX.exists():
            return {}
        data = json.loads(MARKET_INDEX.read_text(encoding="utf-8"))
        return data.get("symbols") or {}
    except Exception:  # noqa: BLE001
        return {}


def measured_spread(symbol: str | None) -> float | None:
    """实测点差（价格单位）；没有实测数据返回 None。"""
    k = _key(symbol)
    if not k:
        return None
    meta = _measured().get(f"{k}#") or _measured().get(k)
    if not meta:
        return None
    pts = meta.get("spread_points_median")
    point = POINTS.get(k)
    if pts is None or point is None:
        return None
    try:
        val = float(pts) * float(point)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def spread_for(symbol: str | None) -> float:
    """取该品种的点差（价格单位）。优先实测，其次静态表，未知品种回落黄金。"""
    hit = measured_spread(symbol)
    if hit is not None:
        return hit
    if not symbol:
        return DEFAULT_SPREAD
    return SPREADS.get(_key(symbol), (DEFAULT_SPREAD, ""))[0]


def spread_source(symbol: str | None) -> str:
    """`measured`（MT5 实测）或 `static`（保守兜底）+ 单位说明。"""
    if measured_spread(symbol) is not None:
        return "measured"
    return "static" if _key(symbol) in SPREADS else "default"


def unit_note(symbol: str | None) -> str:
    if not symbol:
        return ""
    return SPREADS.get(_key(symbol), (DEFAULT_SPREAD, ""))[1]


def point_value(symbol: str | None) -> float | None:
    return POINTS.get(_key(symbol))
