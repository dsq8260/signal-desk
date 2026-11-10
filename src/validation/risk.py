"""风险与回撤指标（纯计算，不参与任何策略决策）。

为什么单独成模块：
- 回测摘要原来只有 6 个数字，看不出"怎么亏的"；
- 这里把常用风险指标和**回撤解剖**（每一段回撤的起止、深度、持续天数、是否修复）
  做成可复用的纯函数，界面只负责显示。

所有输入都是"已经跑完的"权益曲线与成交表，函数不读取未来数据、
不修改策略参数，因此可以放心接在任何回测结果上。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DRAWDOWN_COLUMNS = ["开始", "谷底", "修复", "深度", "下跌天数", "修复天数",
                    "持续天数", "是否修复"]


def _to_series(x) -> pd.Series:
    s = pd.Series(x, dtype=float)
    return s.dropna()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """水下曲线：当前净值距历史高点的百分比（<=0）。"""
    eq = _to_series(equity)
    if len(eq) == 0:
        return eq
    peak = eq.cummax()
    return eq / peak - 1.0


def _positions(index: pd.Index) -> dict:
    return {ts: i for i, ts in enumerate(index)}


def drawdown_table(equity: pd.Series) -> pd.DataFrame:
    """把权益曲线切成一段段回撤，逐段给出深度、耗时、是否修复。

    列：开始 / 谷底 / 修复 / 深度 / 下跌天数 / 修复天数 / 持续天数 / 是否修复
    按深度从深到浅排序。
    """
    eq = _to_series(equity)
    if len(eq) < 2:
        return pd.DataFrame(columns=DRAWDOWN_COLUMNS)

    pos = _positions(eq.index)
    last_ts = eq.index[-1]
    rows: list[dict] = []

    peak_val = float(eq.iloc[0])
    peak_ts = eq.index[0]
    trough_val = peak_val
    trough_ts = peak_ts
    in_dd = False

    for ts, value in eq.items():
        value = float(value)
        if value >= peak_val:                      # 创新高 = 上一段回撤修复
            if in_dd:
                rows.append({
                    "开始": peak_ts, "谷底": trough_ts, "修复": ts,
                    "深度": trough_val / peak_val - 1.0,
                    "下跌天数": pos[trough_ts] - pos[peak_ts],
                    "修复天数": pos[ts] - pos[trough_ts],
                    "持续天数": pos[ts] - pos[peak_ts],
                    "是否修复": True,
                })
                in_dd = False
            peak_val, peak_ts = value, ts
            trough_val, trough_ts = value, ts
        else:
            if not in_dd:
                in_dd = True
                trough_val, trough_ts = value, ts
            elif value < trough_val:
                trough_val, trough_ts = value, ts

    if in_dd:                                       # 还没修复的那一段
        rows.append({
            "开始": peak_ts, "谷底": trough_ts, "修复": None,
            "深度": trough_val / peak_val - 1.0,
            "下跌天数": pos[trough_ts] - pos[peak_ts],
            "修复天数": None,
            "持续天数": pos[last_ts] - pos[peak_ts],
            "是否修复": False,
        })

    if not rows:
        return pd.DataFrame(columns=DRAWDOWN_COLUMNS)
    df = pd.DataFrame(rows)
    return df.sort_values("深度").reset_index(drop=True)


def trade_stats(trades: pd.DataFrame | None) -> dict:
    """成交层面的统计：胜率、盈亏比、单笔期望、最大连亏。"""
    empty = {"平仓笔数": 0, "胜率": float("nan"), "盈亏比": float("nan"),
             "单笔期望": float("nan"), "平均盈利": float("nan"),
             "平均亏损": float("nan"), "最大连续亏损": 0,
             "最大单笔盈利": float("nan"), "最大单笔亏损": float("nan")}
    if trades is None or len(trades) == 0 or "动作" not in trades.columns:
        return empty

    closed = trades[trades["动作"] == "卖出"]
    if len(closed) == 0:
        return empty

    pnl = pd.to_numeric(closed.get("本笔实现盈亏"), errors="coerce").dropna()
    if len(pnl) == 0:
        return empty

    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    streak = worst = 0
    for v in pnl:
        streak = streak + 1 if v < 0 else 0
        worst = max(worst, streak)

    return {
        "平仓笔数": int(len(pnl)),
        "胜率": float(len(wins) / len(pnl)),
        "盈亏比": (float(wins.mean() / abs(losses.mean()))
                   if len(wins) and len(losses) else float("nan")),
        "单笔期望": float(pnl.mean()),
        "平均盈利": float(wins.mean()) if len(wins) else float("nan"),
        "平均亏损": float(losses.mean()) if len(losses) else float("nan"),
        "最大连续亏损": int(worst),
        "最大单笔盈利": float(pnl.max()),
        "最大单笔亏损": float(pnl.min()),
    }


def _core_stats(equity: pd.Series, ppy: int = 252) -> dict:
    eq = _to_series(equity)
    if len(eq) < 2:
        return {"总收益率": float("nan"), "年化收益率": float("nan"),
                "年化波动率": float("nan"), "夏普": float("nan"),
                "索提诺": float("nan"), "卡玛": float("nan"),
                "最大回撤": float("nan"), "当前回撤": float("nan"),
                "最长回撤持续": 0, "月度胜率": float("nan")}

    rets = eq.pct_change().dropna()
    first, last = float(eq.iloc[0]), float(eq.iloc[-1])
    total = last / first - 1.0 if first else float("nan")
    years = len(rets) / ppy
    annual = ((last / first) ** (1 / years) - 1.0
              if years > 0 and last > 0 and first > 0 else float("nan"))
    vol = float(rets.std(ddof=1) * math.sqrt(ppy)) if len(rets) > 1 else float("nan")
    sharpe = (float(rets.mean() * ppy) / vol) if vol and vol > 0 else float("nan")

    downside = rets[rets < 0]
    dvol = (float(downside.std(ddof=1) * math.sqrt(ppy))
            if len(downside) > 1 else float("nan"))
    sortino = (float(rets.mean() * ppy) / dvol) if dvol and dvol > 0 else float("nan")

    dd = drawdown_series(eq)
    mdd = float(dd.min()) if len(dd) else float("nan")
    calmar = (annual / abs(mdd)) if mdd and mdd < 0 and annual == annual else float("nan")

    table = drawdown_table(eq)
    longest = int(table["持续天数"].max()) if len(table) else 0

    monthly = eq.resample("ME").last().pct_change().dropna()
    month_win = float((monthly > 0).mean()) if len(monthly) else float("nan")

    return {"总收益率": total, "年化收益率": annual, "年化波动率": vol,
            "夏普": sharpe, "索提诺": sortino, "卡玛": calmar,
            "最大回撤": mdd, "当前回撤": float(dd.iloc[-1]),
            "最长回撤持续": longest, "月度胜率": month_win}


def extended_stats(equity: pd.Series, trades: pd.DataFrame | None = None,
                   benchmark: pd.Series | None = None,
                   periods_per_year: int = 252) -> dict:
    """一次性给出完整指标字典（界面按需取用）。"""
    out = _core_stats(equity, periods_per_year)
    out.update(trade_stats(trades))

    table = drawdown_table(equity)
    out["回撤段数"] = int(len(table))
    if len(table):
        worst = table.iloc[0]
        out["最深回撤开始"] = worst["开始"]
        out["最深回撤谷底"] = worst["谷底"]
        out["最深回撤是否修复"] = bool(worst["是否修复"])

    if benchmark is not None:
        bh = _to_series(benchmark)
        if len(bh) >= 2:
            b = _core_stats(bh, periods_per_year)
            out["基准收益"] = b["总收益率"]
            out["基准夏普"] = b["夏普"]
            out["基准最大回撤"] = b["最大回撤"]
            out["超额收益"] = (out["总收益率"] - b["总收益率"]
                               if out["总收益率"] == out["总收益率"] else float("nan"))
            out["超额回撤"] = (out["最大回撤"] - b["最大回撤"]
                               if out["最大回撤"] == out["最大回撤"] else float("nan"))

    return out


def format_metrics(stats: dict) -> pd.DataFrame:
    """把指标字典整理成"指标 / 数值 / 说明"三列表格，便于界面直接显示。"""
    def pct(x):
        return "—" if x is None or x != x else f"{x:.2%}"

    def num(x, d=2):
        return "—" if x is None or x != x else f"{x:.{d}f}"

    rows = [
        ("总收益率", pct(stats.get("总收益率")), "整段赚了多少"),
        ("年化收益率", pct(stats.get("年化收益率")), "按年折算"),
        ("年化波动率", pct(stats.get("年化波动率")), "越大越颠簸"),
        ("夏普比率", num(stats.get("夏普")), "每承担 1 份波动赚多少（>1 才算好）"),
        ("索提诺比率", num(stats.get("索提诺")), "只惩罚下跌波动"),
        ("卡玛比率", num(stats.get("卡玛")), "年化收益 / 最大回撤"),
        ("最大回撤", pct(stats.get("最大回撤")), "最惨时距高点跌多少"),
        ("当前回撤", pct(stats.get("当前回撤")), "现在距历史高点还差多少"),
        ("最长回撤持续", f"{stats.get('最长回撤持续', 0)} 个交易日",
         "连续多久没回到高点（比深度更能说明煎熬）"),
        ("月度胜率", pct(stats.get("月度胜率")), "有几个月是赚的"),
        ("平仓笔数", str(stats.get("平仓笔数", 0)), "已经卖出的次数"),
        ("胜率", pct(stats.get("胜率")), "卖出单里赚钱的比例"),
        ("盈亏比", num(stats.get("盈亏比")), "平均赚 / 平均亏（>1 才划算）"),
        ("单笔期望", f"{num(stats.get('单笔期望'))} 元", "每交易一次平均赚亏多少"),
        ("平均盈利 / 平均亏损",
         f"{num(stats.get('平均盈利'), 0)} / {num(stats.get('平均亏损'), 0)} 元", "赚一笔 vs 亏一笔"),
        ("最大连续亏损", f"{stats.get('最大连续亏损', 0)} 笔", "连亏最多的一次"),
        ("基准收益", pct(stats.get("基准收益")), "同期买了不动的收益"),
        ("基准夏普", num(stats.get("基准夏普")), "躺着不动的性价比"),
        ("基准最大回撤", pct(stats.get("基准最大回撤")), "躺着不动最惨时"),
        ("超额收益", pct(stats.get("超额收益")), "策略减去基准（负数=不如躺着）"),
        ("超额回撤", pct(stats.get("超额回撤")), "回撤比基准好多少（正数=更抗跌）"),
    ]
    return pd.DataFrame(rows, columns=["指标", "数值", "说明"])
