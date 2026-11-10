"""事件研究法（阶段 2）。

问的问题：某个"事件标签"出现之后，未来 1/3/5/10 期是否出现异常收益？

（早期这一层是给干支组合用的；干支维度已从本系统移除，这里退化为通用工具：
把任意一列当事件标签即可。）

口径：
- 基准收益 = 样本期内所有期的平均收益（作为"正常收益"的朴素估计）
- 异常收益 AR_h = r_{t+h} - 基准
- 累计异常收益 CAR_h = sum_{k=1..h} AR_{t+k}
- t 值 = mean(CAR_h) / (std(CAR_h)/sqrt(n))
- n < 10 一律标注"仅探索"，不得当作结论
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SAMPLE = 10


def event_study(
    table: pd.DataFrame,
    *,
    combo_col: str = "事件",
    combos: list[str] | None = None,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    returns: pd.Series | None = None,
) -> pd.DataFrame:
    """对指定事件标签做事件研究，返回逐 (组合, 期数) 的结果表。

    :param table: 含 close 与 combo_col 的行情表（索引为日期）
    :param combos: 只研究这些组合；None 表示对出现过的组合全部研究
    :param returns: 显式指定收益序列；默认用 close.pct_change()
    """
    if combo_col not in table.columns:
        raise ValueError(f"表里没有组合列: {combo_col}")

    if returns is None:
        close = pd.to_numeric(table["close"], errors="coerce").astype(float)
        returns = close.pct_change()
    returns = pd.Series(returns, dtype=float).reindex(table.index)

    baseline = float(returns.mean(skipna=True))
    if not np.isfinite(baseline):
        baseline = 0.0

    combos = list(combos) if combos is not None else list(pd.unique(table[combo_col]))
    r_arr = returns.to_numpy(dtype=float)
    labels = table[combo_col].to_numpy()

    rows = []
    for combo in combos:
        event_idx = np.where(labels == combo)[0]
        event_idx = event_idx[event_idx < len(r_arr) - 1]
        n_events = len(event_idx)
        for h in horizons:
            cars = []
            for i in event_idx:
                window = r_arr[i + 1 : i + 1 + h]
                window = window[np.isfinite(window)]
                if len(window) == 0:
                    continue
                cars.append(float(np.sum(window - baseline)))
            n = len(cars)
            if n == 0:
                rows.append({"组合": combo, "期数": h, "样本数": 0, "CAR": np.nan,
                             "t值": np.nan, "结论": "样本不足"})
                continue
            cars_arr = np.array(cars, dtype=float)
            car = float(cars_arr.mean())
            if n >= 2 and cars_arr.std(ddof=1) > 0:
                t_stat = float(car / (cars_arr.std(ddof=1) / np.sqrt(n)))
            else:
                t_stat = float("nan")
            verdict = "仅探索" if n < MIN_SAMPLE else ("显著" if abs(t_stat) > 2 else "不显著")
            rows.append(
                {
                    "组合": combo,
                    "期数": h,
                    "样本数": n,
                    "CAR": car,
                    "t值": t_stat,
                    "结论": verdict,
                }
            )
    return pd.DataFrame(rows)


def top_events(results: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """按 |CAR| 挑最显眼的事件行，仅用于展示。"""
    if results is None or len(results) == 0:
        return pd.DataFrame()
    d = results.dropna(subset=["CAR"]).copy()
    if d.empty:
        return d
    d["abs_CAR"] = d["CAR"].abs()
    return d.sort_values("abs_CAR", ascending=False).head(n).drop(columns="abs_CAR")
