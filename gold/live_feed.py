"""实盘监测：读 EA 自己写的成绩单，做偏差分析。

数据来源（最可靠的那条路）
------------------------
EA 里 `InpReport=true` 时，每根 bar 追加一行到
`<终端公共目录>/Files/ea_<tag>_equity.csv`：

    time,equity,balance,position,lots,price

这是**唯一由 EA 自己产生的实盘记录**，不依赖经纪商接口，也不怕 MCP 掉线。
本模块负责：取回、解析、算偏差（回测预期 vs 实盘实际）。
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd

COLS = ["time", "equity", "balance", "position", "lots", "price"]


def parse_equity_csv(text: str) -> pd.DataFrame:
    """解析成绩单文本（容忍空文件、坏行、缺表头）。"""
    if not text or not text.strip():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip().lower() for c in df.columns]
    keep = [c for c in COLS if c in df.columns]
    if "time" not in keep:
        return pd.DataFrame(columns=COLS)
    df = df[keep].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    for c in ("equity", "balance", "price", "lots", "position"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


@dataclass
class LiveStats:
    bars: int = 0
    start: str = ""
    end: str = ""
    days: float = 0.0
    equity_start: float = 0.0
    equity_end: float = 0.0
    peak: float = 0.0
    max_dd: float = 0.0
    in_position_pct: float = 0.0
    open_now: int = 0
    trades_hint: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {"bar数": self.bars, "区间": f"{self.start} → {self.end}",
                "天数": round(self.days, 1), "净值": round(self.equity_end, 2),
                "最大回撤": f"{self.max_dd:.2%}",
                "在场时间": f"{self.in_position_pct:.0%}",
                "当前有仓": self.open_now, "估计回合": self.trades_hint}


def live_stats(df: pd.DataFrame) -> LiveStats:
    """从成绩单算出实盘概况。"""
    if df is None or not len(df):
        return LiveStats(warnings=["没有成绩单数据（EA 未开 InpReport，或还没跑过）"])
    eq = df["equity"].astype(float) if "equity" in df else df["balance"]
    eq = eq.dropna()
    if eq.empty:
        return LiveStats(warnings=["成绩单里没有有效净值"])
    peak = eq.cummax()
    dd = float((eq / peak - 1).min())
    days = max(0.0, (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds()
               / 86400.0)
    pos = df["position"] if "position" in df else pd.Series([0] * len(df))
    in_pos = float((pos.fillna(0) != 0).mean())
    # 回合估计：仓位从 0 变成非 0 的次数
    turns = int(((pos.fillna(0) != 0) & (pos.fillna(0).shift(1).fillna(0) == 0))
                .sum())
    return LiveStats(
        bars=int(len(df)),
        start=str(df["time"].iloc[0])[:16],
        end=str(df["time"].iloc[-1])[:16],
        days=days,
        equity_start=float(eq.iloc[0]),
        equity_end=float(eq.iloc[-1]),
        peak=float(peak.iloc[-1]),
        max_dd=dd,
        in_position_pct=in_pos,
        open_now=int(pos.iloc[-1] or 0),
        trades_hint=turns,
    )


def deviation(backtest: dict, live: LiveStats) -> list[dict]:
    """把"回测预期"和"实盘实际"并排放，标出偏差是否超阈值。"""
    rows = []
    bt_dd = backtest.get("回撤")
    if bt_dd is not None and live.max_dd:
        rows.append({
            "指标": "最大回撤",
            "回测": f"{bt_dd:.1%}",
            "实盘": f"{live.max_dd:.1%}",
            "结论": "偏差大（实盘更差）" if live.max_dd < bt_dd * 1.5 else "在预期内",
        })
    bt_days = backtest.get("天数")
    if bt_days:
        rows.append({
            "指标": "样本天数", "回测": f"{bt_days:.0f}",
            "实盘": f"{live.days:.0f}",
            "结论": "够用" if live.days >= 30 else "样本太短（少于 30 天）",
        })
    bt_turns = backtest.get("回合")
    if bt_turns and live.trades_hint:
        rows.append({
            "指标": "回合数", "回测": str(bt_turns), "实盘": str(live.trades_hint),
            "结论": "成交率异常" if live.trades_hint < bt_turns * 0.5 else "正常",
        })
    return rows


def read_equity_csv(client, tag: str, common: str | None = None) -> str:
    """通过 MCP 读回 EA 的成绩单文本（读不到返回空串，不抛异常）。"""
    try:
        if not common:
            ws = client.call("get_workspace_info", {}).get("workspace") or {}
            base = ws.get("common_folder") or ws.get("common_files") or ""
            if base:
                common = f"{base}\\Files"
        if not common:
            return ""
        resp = client.call("read_file",
                           {"path": f"{common}\\ea_{tag}_equity.csv"})
        return str(resp.get("text") or resp.get("content") or "")
    except Exception:  # noqa: BLE001
        return ""
