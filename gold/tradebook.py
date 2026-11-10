"""给模拟盘加「逐笔交易明细」：像 MT5 测试器的「交易」页签。

数据来源
--------
`trades` 表记录 BUY/SELL 两条腿。这里把同一笔的两条腿配对，
算出真实盈亏（含双边手续费）与持有时长。

注意：`replay_window` 写入的 pnl 字段是 0（因为平仓时才算得出），
所以这里用「买价 vs 卖价 × 数量 − 双边费用」重算，不依赖那列。
"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd


def trade_list(con: sqlite3.Connection, sid: int) -> pd.DataFrame:
    """把某策略的成交配对成「一笔一行的交易明细」。"""
    rows = con.execute(
        "SELECT ts, side, price, units, fee FROM trades "
        "WHERE strategy_id=? ORDER BY ts, id", (int(sid),)).fetchall()
    if not rows:
        return pd.DataFrame(columns=["开仓时间", "平仓时间", "方向", "开仓价",
                                     "平仓价", "数量", "手续费", "盈亏",
                                     "持有小时", "备注"])

    out = []
    open_leg = None
    for ts, side, price, units, fee in rows:
        side = str(side).upper()
        price = float(price or 0.0)
        units = float(units or 0.0)
        fee = float(fee or 0.0)
        if side == "BUY" and open_leg is None:
            open_leg = {"ts": int(ts), "price": price, "units": units, "fee": fee}
        elif side == "SELL":
            if open_leg is None:            # 没有对应开仓（历史残留），跳过
                continue
            pnl = (price - open_leg["price"]) * open_leg["units"] \
                - open_leg["fee"] - fee
            hours = (int(ts) - open_leg["ts"]) / 3600.0
            out.append({
                "开仓时间": pd.to_datetime(open_leg["ts"], unit="s"),
                "平仓时间": pd.to_datetime(int(ts), unit="s"),
                "方向": "BUY",
                "开仓价": open_leg["price"],
                "平仓价": price,
                "数量": open_leg["units"],
                "手续费": open_leg["fee"] + fee,
                "盈亏": pnl,
                "持有小时": round(hours, 1),
                "备注": "",
            })
            open_leg = None

    df = pd.DataFrame(out)
    if len(df) == 0:
        return df
    df["累计盈亏"] = df["盈亏"].cumsum()
    df = df.sort_values("平仓时间").reset_index(drop=True)
    return df


def strategy_meta(con: sqlite3.Connection, sid: int) -> dict:
    row = con.execute(
        "SELECT id, name, kind, params, tf, units, cash0 FROM strategies WHERE id=?",
        (int(sid),)).fetchone()
    if not row:
        return {}
    return {"id": row[0], "name": row[1], "kind": row[2],
            "params": json.loads(row[3] or "{}"), "tf": row[4],
            "units": row[5], "cash0": row[6]}
