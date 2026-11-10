"""平台中心选参：只从"稳定区域"里挑，用更小的候选集把 PBO 压下去。

为什么这样能降 PBO
------------------
PBO 衡量的是"从一批配置里挑样本内最优，它在样本外还能排进前半吗"。
它有两个来源：
1. **候选集太大** —— 挑得越多，运气成分越大；
2. **挑在尖峰上** —— 邻域不稳，样本外立刻塌。

所以做法是：
1. 在参数网格上算每个点的**邻域稳定性**（3×3 窗口内收益同号率）与**邻域均值**；
2. 只在"同号率 ≥ 阈值"的稳定区域里选；
3. 逐级缩小候选集（全部 → 稳定区 → 稳定区前 k），重算 PBO，**取第一个 PBO ≤ 0.5 的层级**；
4. 在该层级里挑邻域均值最高的点作为部署参数（平台中心，而不是尖峰）。

选参只用训练段（by construction: 传进来的 segment 必须由调用方切好）。
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from gold.backtest import spread_fee_bps
from gold.factors import build_features
from gold.forge import Genome, build_position
from gold.pbo import pbo_from_returns_by_config


def _returns(seg: pd.DataFrame, feat: pd.DataFrame, g: Genome,
             fee_frac: float) -> pd.Series:
    pos = build_position(seg, feat.loc[seg.index], g).shift(1).fillna(0.0)
    fwd = seg["close"].astype(float).pct_change().shift(-1)
    return pos * fwd.fillna(0.0) - fee_frac * pos.diff().abs().fillna(0.0)


def _sharpe(r: pd.Series) -> float:
    sd = float(r.std())
    return float(r.mean() / sd * np.sqrt(6048)) if sd > 0 else 0.0


def plateau_search(df: pd.DataFrame, *, grid: dict, base: dict | None = None,
                   entry: str = "唐奇安突破", filt: str = "强趋势",
                   allow_short: bool = False, sl: float | None = 0.02,
                   tp: float | None = 0.05, tf: str = "H1",
                   spread_usd: float = 0.16, min_same_sign: float = 0.6,
                   max_pbo: float = 0.5) -> dict:
    """在参数网格上找"平台中心"，并给出逐级缩小候选集后的 PBO。

    ``grid``：{参数名: [取值,...]}，例如 {"entry": [15,20,25], "exit_": [8,10,15]}
    返回 dict：best（Genome）、table（逐点指标）、levels（逐级 PBO）、pbo、level。
    """
    base = dict(base or {})
    names = list(grid)
    combos = list(itertools.product(*(grid[k] for k in names)))
    if len(combos) < 2:
        return {"ok": False, "reason": "网格太小，至少 2 个点"}

    feat = build_features(df)
    fee_frac = spread_fee_bps(float(df["close"].mean()), float(spread_usd)) / 10000.0

    rets: dict[str, pd.Series] = {}
    info: list[dict] = []
    for combo in combos:
        params = {**base, **dict(zip(names, combo))}
        g = Genome(entry, dict(params), filt, allow_short, sl, tp)
        r = _returns(df, feat, g, fee_frac)
        key = "|".join(f"{k}={v}" for k, v in params.items())
        rets[key] = r
        info.append({"key": key, "params": dict(params), "genome": g,
                     "总收益": float((1 + r).prod() - 1), "夏普": _sharpe(r)})

    table = pd.DataFrame(info)
    # 邻域：每个坐标轴上取相邻取值，笛卡尔积即 3^k 窗口
    def neighbors(row) -> list[int]:
        idx = []
        for i, rec in table.iterrows():
            close = True
            for k in names:
                vals = list(grid[k])
                a, b = vals.index(row["params"][k]), vals.index(rec["params"][k])
                if abs(a - b) > 1:
                    close = False
                    break
            if close:
                idx.append(i)
        return idx

    same, nb_mean, nb_n = [], [], []
    for _, row in table.iterrows():
        idx = neighbors(row)
        signs = np.sign(table.loc[idx, "总收益"].to_numpy())
        same.append(float((signs == np.sign(row["总收益"])).mean()))
        nb_mean.append(float(table.loc[idx, "总收益"].mean()))
        nb_n.append(len(idx))
    table["邻域同号率"] = same
    table["邻域均值"] = nb_mean
    table["邻域个数"] = nb_n

    levels = []
    def _pbo(sub: pd.DataFrame) -> float | None:
        if len(sub) < 3:
            return None
        res = pbo_from_returns_by_config(
            {r["key"]: rets[r["key"]] for _, r in sub.iterrows()}, n_blocks=12)
        return res.get("pbo") if res.get("ok") else None

    all_pbo = _pbo(table)
    levels.append({"层级": "全部", "候选数": len(table), "PBO": all_pbo})

    stable = table[table["邻域同号率"] >= min_same_sign]
    if len(stable) >= 3:
        levels.append({"层级": f"稳定区(同号率≥{min_same_sign:.0%})",
                       "候选数": len(stable), "PBO": _pbo(stable)})

    chosen_level = None
    for lv in levels:
        if lv["PBO"] is not None and lv["PBO"] <= max_pbo:
            chosen_level = lv
    picked = None
    if chosen_level is not None:
        pool = (stable if chosen_level["层级"].startswith("稳定区") else table)
        k = int(min(9, len(pool)))
        top = pool.sort_values("邻域均值", ascending=False).head(k)
        lv_k = {"层级": f"{chosen_level['层级']} 前 {k}", "候选数": k,
                "PBO": _pbo(top) if k >= 3 else None}
        levels.append(lv_k)
        if lv_k["PBO"] is not None and lv_k["PBO"] <= max_pbo:
            chosen_level = lv_k
            pool = top
        picked = pool.sort_values("邻域均值", ascending=False).iloc[0]

    best = picked["genome"] if picked is not None else None
    return {"ok": best is not None,
            "best": best,
            "best_row": (picked.to_dict() if picked is not None else None),
            "table": table, "levels": levels,
            "pbo": (chosen_level or {}).get("PBO"),
            "level": (chosen_level or {}).get("层级"),
            "reason": "" if best is not None else
                      "没有任何层级能把 PBO 压到阈值以下"}
