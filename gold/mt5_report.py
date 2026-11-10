"""MT5 策略测试器风格的回测报告。

为什么要这个
------------
我们原来的「四道门槛」是**统计显著性**视角（p 值、方向准确率），
而 MT5 测试器给的是**交易员**视角：毛利/毛损、盈利因子、预期收益、
恢复因子、最大结余亏损、连胜连败、按小时/周/月的分布。

两者都需要，但交易员先看后者。本模块把回测净值曲线拆成"一笔笔交易"，
算出与 MT5「质量历史」面板同名的指标。

口径说明（与 MT5 的差异必须说清楚）
---------------------------------
- MT5 用固定手数下单；我们用「全额仓位」（pos=1 时满仓），
  所以金额是相对初始资金的比例，不是固定 0.01 手的美元数。
- 一笔交易 = 一段连续持仓（从进场到回场），期间的净值变化就是这笔的盈亏。
- 夏普用 bar 收益年化（D1=252，H1=252*24 等近似）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# 年化因子（按周期）
_BARS_PER_YEAR = {
    "M1": 252 * 24 * 60, "M5": 252 * 24 * 12, "M15": 252 * 24 * 4,
    "M30": 252 * 24 * 2, "H1": 252 * 24, "H4": 252 * 6,
    "D1": 252, "W1": 52,
}


def _annual_factor(tf: str) -> float:
    return float(_BARS_PER_YEAR.get(str(tf).upper(), 252))


def trades_from_curve(equity: pd.Series, pos: pd.Series,
                      initial: float = 10000.0) -> pd.DataFrame:
    """把净值曲线拆成逐笔交易。

    持仓从有到无（或方向翻转）算一笔结束；空仓期间的净值不变。
    返回列：open_time, close_time, side, bars, pnl, ret。
    """
    if equity is None or len(equity) == 0:
        return pd.DataFrame(columns=["open_time", "close_time", "side", "bars", "pnl", "ret"])

    eq = pd.Series(equity).astype(float)
    p = pd.Series(pos, dtype=float).reindex(eq.index).fillna(0.0)

    rows = []
    holding = 0.0
    entry_t = None
    entry_eq = None
    bars = 0

    def _close(t, cur_eq):
        nonlocal holding, entry_t, entry_eq, bars
        pnl = float(cur_eq - entry_eq)
        rows.append({
            "open_time": entry_t, "close_time": t,
            "side": int(holding), "bars": bars,
            "pnl": pnl, "ret": float(pnl / entry_eq) if entry_eq else 0.0,
        })
        holding, entry_t, entry_eq, bars = 0.0, None, None, 0

    for t, cur_pos in p.items():
        new = float(cur_pos)
        cur_eq = float(eq.loc[t])
        if holding == 0.0 and new != 0.0:
            holding, entry_t, entry_eq, bars = new, t, cur_eq, 0
        elif holding != 0.0 and new == 0.0:
            _close(t, cur_eq)
        elif holding != 0.0 and new != 0.0 and np.sign(new) != np.sign(holding):
            _close(t, cur_eq)
            holding, entry_t, entry_eq, bars = new, t, cur_eq, 0
        elif holding != 0.0:
            bars += 1

    if holding != 0.0 and len(eq):
        _close(eq.index[-1], float(eq.iloc[-1]))

    return pd.DataFrame(rows)


def _max_dd(curve: pd.Series) -> tuple[float, float]:
    """返回 (绝对最大回撤, 相对最大回撤%)。"""
    if curve is None or len(curve) == 0:
        return 0.0, 0.0
    c = pd.Series(curve).astype(float)
    peak = c.cummax()
    dd = c - peak
    worst_abs = float(dd.min())
    rel = ((c / peak) - 1.0).min()
    return abs(worst_abs), abs(float(rel))


def mt5_metrics(equity: pd.Series, pos: pd.Series, tf: str = "D1",
                initial: float = 10000.0, fee_bps: float = 0.0) -> dict:
    """算出 MT5「质量历史」面板同名的指标。"""
    nan = float("nan")
    out = {
        "初始入金": float(initial), "总净盈利": nan, "毛利": nan, "毛损": nan,
        "盈利因子": nan, "预期收益": nan, "恢复因子": nan, "夏普比率": nan,
        "LR 相关性": nan, "AHPR": nan, "GHPR": nan,
        "交易总计": 0, "盈利交易": 0, "亏损交易": 0,
        "盈利交易%": nan, "亏损交易%": nan,
        "最大获利交易": nan, "最大亏损交易": nan,
        "平均获利交易": nan, "平均亏损交易": nan,
        "最大结余亏损": nan, "最大结余亏损%": nan,
        "最大净值亏损": nan, "最大净值亏损%": nan,
        "连胜(金额)": nan, "连败(金额)": nan,
        "最大连胜(笔)": 0, "最大连败(笔)": 0,
    }
    if equity is None or len(equity) < 3:
        return out

    eq = pd.Series(equity).astype(float)
    p = pd.Series(pos, dtype=float).reindex(eq.index).fillna(0.0)

    trades = trades_from_curve(eq, p, initial)
    n = int(len(trades))
    net = float(eq.iloc[-1] - initial)
    out["总净盈利"] = net

    # 毛利 / 毛损 / 盈利因子 / 预期收益
    if n:
        pnl = trades["pnl"].astype(float)
        gp = float(pnl[pnl > 0].sum())
        gl = float(pnl[pnl < 0].sum())
        out["毛利"] = gp
        out["毛损"] = gl
        out["盈利因子"] = (gp / abs(gl)) if abs(gl) > 1e-12 else nan
        out["预期收益"] = net / n
        out["交易总计"] = n
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        out["盈利交易"] = int(len(wins))
        out["亏损交易"] = int(len(losses))
        out["盈利交易%"] = float(len(wins) / n)
        out["亏损交易%"] = float(len(losses) / n)
        out["最大获利交易"] = float(wins.max()) if len(wins) else nan
        out["最大亏损交易"] = float(losses.min()) if len(losses) else nan
        out["平均获利交易"] = float(wins.mean()) if len(wins) else nan
        out["平均亏损交易"] = float(losses.mean()) if len(losses) else nan

        # AHPR / GHPR（逐笔收益率）
        rets = trades["ret"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(rets):
            out["AHPR"] = float((1.0 + rets).mean())
            prod = float(np.prod(1.0 + rets))
            out["GHPR"] = float(prod ** (1.0 / len(rets))) if prod > 0 else nan

        # 连胜 / 连败
        best_w = best_l = cur_w = cur_l = 0
        sum_w = sum_l = 0.0
        for v in pnl:
            if v > 0:
                cur_w += 1
                cur_l = 0
                sum_w += float(v)
            elif v < 0:
                cur_l += 1
                cur_w = 0
                sum_l += float(v)
            else:
                cur_w = cur_l = 0
            best_w = max(best_w, cur_w)
            best_l = max(best_l, cur_l)
        out["最大连胜(笔)"] = int(best_w)
        out["最大连败(笔)"] = int(best_l)
        out["连胜(金额)"] = sum_w
        out["连败(金额)"] = sum_l
    # 回撤（结余曲线 = 净值，因为我们空仓时无浮动）
    dd_abs, dd_pct = _max_dd(eq)
    out["最大结余亏损"] = -dd_abs
    out["最大结余亏损%"] = -dd_pct
    out["最大净值亏损"] = -dd_abs
    out["最大净值亏损%"] = -dd_pct
    out["恢复因子"] = (net / dd_abs) if dd_abs > 1e-12 else nan

    # 夏普（bar 收益年化）
    rets = eq.pct_change().dropna()
    if len(rets) > 2:
        sd = float(rets.std(ddof=1))
        if sd > 0:
            out["夏普比率"] = float(rets.mean() / sd * math.sqrt(_annual_factor(tf)))

    # LR 相关性：累计盈亏与交易序号的相关性
    if n >= 3:
        cum = trades["pnl"].astype(float).cumsum().to_numpy()
        idx = np.arange(1, n + 1, dtype=float)
        if np.std(cum) > 0 and np.std(idx) > 0:
            out["LR 相关性"] = float(np.corrcoef(idx, cum)[0, 1])

    return out


def period_breakdown(equity: pd.Series, pos: pd.Series, initial: float = 10000.0
                     ) -> dict[str, pd.DataFrame]:
    """按 小时 / 周 / 月 汇总每笔交易（对应 MT5 的三张柱状图）。"""
    trades = trades_from_curve(equity, pos, initial)
    empty = pd.DataFrame(columns=["笔数", "盈亏"])
    if len(trades) == 0:
        return {"每小时": empty, "每周": empty, "每月": empty}

    t = trades.copy()
    t["close_time"] = pd.to_datetime(t["close_time"])
    t = t.set_index("close_time")

    out = {}
    for label, group in (("每小时", lambda d: d.index.hour),
                         ("每周", lambda d: d.index.dayofweek),
                         ("每月", lambda d: d.index.month)):
        g = t.groupby(group(t)).agg(笔数=("pnl", "size"), 盈亏=("pnl", "sum"))
        out[label] = g
    return out
