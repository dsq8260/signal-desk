"""周期支持：8 个周期 + 本地聚合。

现实约束（必须说清）：
- MT5 的 `copy_rates_*` 目前仍 `Call failed`，所以 M1/M5/M15/M30/H1/H4 **没有历史**；
- COMEX 兜底源只有日线；
- 因此：日线走 COMEX；**周线由日线聚合**；分钟级由 `gold.ticker` 从 tick 累积的 M1 聚合。

也就是说：分钟级数据**从你开始跑收集器的那一刻起**才有，之前没有。
"""

from __future__ import annotations

import pathlib
import sqlite3

import pandas as pd

PAPER_DB = "data/paper.db"

# 8 个周期 → MT5 常量名 + 分钟数（W1 用周聚合）
TF_NAMES = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1", "W1": "TIMEFRAME_W1",
}
TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
              "H1": 60, "H4": 240, "D1": 1440, "W1": 10080}
ALL_TFS = list(TF_NAMES)
INTRADAY_TFS = ["M1", "M5", "M15", "M30", "H1", "H4"]

# pandas 聚合规则（W1 用 W-FRI，贴合外汇周线）
_PANDAS_RULE = {"M5": "5min", "M15": "15min", "M30": "30min",
                "H1": "1h", "H4": "4h", "D1": "1D", "W1": "W-FRI"}


def resample_bars(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """把细粒度 bar 聚合到更粗的周期（O=首, H=最高, L=最低, C=末）。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if tf not in _PANDAS_RULE:
        return df
    d = df.copy()
    if "volume" not in d.columns:
        d["volume"] = 0.0
    out = d.resample(_PANDAS_RULE[tf]).agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna(subset=["close"])
    out.index.name = d.index.name
    return out


def load_local_m1(db: str | pathlib.Path = PAPER_DB) -> pd.DataFrame:
    """读收集器攒下的 M1 bar（没有就返回空）。"""
    p = pathlib.Path(db)
    if not p.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    try:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        df = pd.read_sql_query(
            "SELECT ts,open,high,low,close FROM bars WHERE tf='M1' ORDER BY ts", con)
        con.close()
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if len(df) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts"), unit="s")
    df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def intraday_bars(tf: str, db: str | pathlib.Path = PAPER_DB) -> pd.DataFrame:
    """把本地 M1 聚合成目标周期（M1 直接返回）。"""
    m1 = load_local_m1(db)
    if len(m1) == 0:
        return m1
    return m1 if tf == "M1" else resample_bars(m1, tf)
