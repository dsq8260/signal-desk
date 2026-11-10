"""前向验证（C）：把系统设计的 EA 的真实前向表现，与它的回测表现对照。

为什么这是最关键的一环
--------------------
自主设计能产出"OOS 夏普 0.93"的候选，但全局 Deflated p 仍是 0.99——
统计上分不清是本事还是运气。**唯一能分辨的办法是让它继续往前走**：

    回测夏普（设计时的 OOS）
        vs
    前向夏普（登记进模拟盘后、真正逐日推进的表现）

衰减率 = 前向夏普 / 回测夏普。接近 1 说明不是过拟合；接近 0 或负数说明回测骗人。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from gold.backtest import spread_fee_bps
from gold.freqs import freq_key
from gold.paper import list_strategies, load_bars, signal_series
from gold.mt5_report import _annual_factor
from validation import metrics as M


@dataclass
class ForwardCheck:
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = ""
    window: tuple = ()


def _forward_metrics(con, sid: int, tf: str = "D1") -> dict | None:
    """从 snapshots 读出前向净值，算表现。"""
    rows = con.execute(
        "SELECT ts, equity FROM snapshots WHERE strategy_id=? ORDER BY ts",
        (sid,)).fetchall()
    if len(rows) < 5:
        return None
    idx = pd.to_datetime([r[0] for r in rows], unit="s")
    eq = pd.Series([float(r[1]) for r in rows], index=idx)
    base = float(eq.iloc[0]) or 1.0
    peak = eq.cummax()
    rets = eq.pct_change().dropna()
    sharpe = float("nan")
    if len(rets) > 2 and float(rets.std(ddof=1)) > 0:
        # 必须和回测同口径：回测夏普是年化的，这里也要乘 √年化因子，
        # 否则衰减率会恒等于 1/√252≈0.063（单位不一致，不是真衰减）。
        import math as _m
        sharpe = float(rets.mean() / rets.std(ddof=1)
                       * _m.sqrt(_annual_factor(tf)))
    return {
        "前向收益": float(eq.iloc[-1] / base - 1.0),
        "前向夏普": sharpe,
        "前向回撤": float((eq / peak - 1.0).min()),
        "前向bar数": int(len(eq)),
        "起点": idx[0], "终点": idx[-1],
    }


def _backtest_on_same_window(df: pd.DataFrame, kind: str, params: dict,
                             start, end, fee_bps: float,
                             tf: str = "D1") -> dict | None:
    """在**同一个窗口**上用回测口径重算，作对照。"""
    win = df[(df.index >= start) & (df.index <= end)]
    if len(win) < 5:
        return None
    sig = signal_series(win, kind, params)
    pos = sig.shift(1).fillna(0.0)
    fwd = win["close"].astype(float).pct_change().shift(-1)
    st = M.summarize_strategy(fwd, pos, fee_bps=fee_bps, freq=freq_key(tf),
                              allow_short=False)
    return {
        "回测收益": st["net"].get("total_return", float("nan")),
        "回测夏普": st["net"].get("sharpe", float("nan")),
        "回测回撤": st["net"].get("max_drawdown", float("nan")),
        "回测笔数": int(st["trades"]),
    }


def compare(con, df: pd.DataFrame, tf: str = "D1",
            spread_usd: float = 0.33) -> ForwardCheck:
    """逐个策略对比「前向」与「同日历窗口回测」。"""
    strats = list_strategies(con)
    if len(strats) == 0:
        return ForwardCheck(note="模拟盘里还没有登记策略")

    fee = spread_fee_bps(float(df["close"].mean()) if len(df) else 4000.0,
                         spread_usd)
    rows = []
    for r in strats.to_dict("records"):
        sid = int(r["id"])
        # 用对照数据本身的周期年化（前向曲线就是用这份数据推出来的），
        # 不能用策略登记时的 tf——两者不一致会算出错误的衰减率。
        fw = _forward_metrics(con, sid, tf)
        if fw is None:
            continue
        params = json.loads(r["params"] or "{}")
        bt = _backtest_on_same_window(df, r["kind"], params,
                                      fw["起点"], fw["终点"], fee, tf)
        if bt is None:
            continue
        bt_sr = float(bt["回测夏普"])
        fw_sr = float(fw["前向夏普"])
        decay = (fw_sr / bt_sr) if bt_sr == bt_sr and abs(bt_sr) > 1e-9 else float("nan")
        rows.append({
            "策略": r["name"], "周期": str(r.get("tf") or tf),
            "回测收益": bt["回测收益"], "回测夏普": bt_sr,
            "前向收益": fw["前向收益"], "前向夏普": fw_sr,
            "衰减率": decay,
            "前向bar数": fw["前向bar数"],
            "笔数(回测)": bt["回测笔数"],
        })

    if not rows:
        return ForwardCheck(note="没有足够的前向数据（至少 5 个快照）")

    tbl = pd.DataFrame(rows).sort_values("前向夏普", ascending=False)
    return ForwardCheck(
        table=tbl.reset_index(drop=True),
        window=(str(tbl.iloc[0]["前向bar数"]),),
        note=(f"{len(tbl)} 个策略 · 前向数据来自 snapshots（逐日推进）· "
              f"回测在同一日历窗口重算，两者可比"),
    )
