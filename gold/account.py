"""读数据桥写出的账户 / 持仓 / 成交 CSV（绕开 MT5 Python IPC）。

文件（MT5 公共文件夹）：
    mt5_account.csv    账户快照，每 InpAccountSecs 秒整份重写
    mt5_positions.csv  当前持仓
    mt5_deals.csv      近 InpDealDays 天成交（含 magic，可按 EA 区分）
"""

from __future__ import annotations

import pathlib

import pandas as pd

from gold.bridge import common_dir

ACCOUNT_FILE = "mt5_account.csv"
POSITIONS_FILE = "mt5_positions.csv"
DEALS_FILE = "mt5_deals.csv"


def _read(d: pathlib.Path, name: str) -> pd.DataFrame:
    p = d / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def account(d: pathlib.Path | str | None = None) -> dict | None:
    """账户快照；没有文件返回 None。"""
    d = pathlib.Path(d) if d else common_dir()
    df = _read(d, ACCOUNT_FILE)
    if len(df) == 0:
        return None
    row = df.iloc[-1].to_dict()
    for k in ("balance", "equity", "margin", "margin_free",
              "margin_level", "profit"):
        if k in row:
            try:
                row[k] = float(row[k])
            except (TypeError, ValueError):
                row[k] = 0.0
    if "leverage" in row:
        try:
            row["leverage"] = int(float(row["leverage"]))
        except (TypeError, ValueError):
            row["leverage"] = 0
    return row


def positions(d: pathlib.Path | str | None = None) -> pd.DataFrame:
    d = pathlib.Path(d) if d else common_dir()
    df = _read(d, POSITIONS_FILE)
    if len(df) == 0:
        return df
    for c in ("volume", "price_open", "price_current", "profit", "swap"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "type" in df.columns:
        df["side"] = df["type"].map({0: "BUY", 1: "SELL"}).fillna("?")
    return df


def deals(d: pathlib.Path | str | None = None) -> pd.DataFrame:
    d = pathlib.Path(d) if d else common_dir()
    df = _read(d, DEALS_FILE)
    if len(df) == 0:
        return df
    for c in ("volume", "price", "profit", "swap", "commission", "magic"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
    return df


def by_magic(d: pathlib.Path | str | None = None) -> pd.DataFrame:
    """按 magic（每个 EA 一个）汇总成交：净收益、笔数、手续费。"""
    df = deals(d)
    if len(df) == 0 or "magic" not in df.columns:
        return pd.DataFrame(columns=["magic", "净收益", "笔数", "手续费", "库存费"])
    trade = df[df.get("profit", pd.Series(dtype=float)).fillna(0) != 0]
    if len(trade) == 0:
        trade = df
    g = trade.groupby("magic").agg(
        净收益=("profit", "sum"),
        笔数=("profit", "size"),
        手续费=("commission", "sum"),
        库存费=("swap", "sum"),
    ).reset_index()
    return g.sort_values("净收益", ascending=False).reset_index(drop=True)


def status(d: pathlib.Path | str | None = None) -> dict:
    d = pathlib.Path(d) if d else common_dir()
    return {
        "account": (d / ACCOUNT_FILE).exists(),
        "positions": (d / POSITIONS_FILE).exists(),
        "deals": (d / DEALS_FILE).exists(),
        "dir": str(d),
    }
