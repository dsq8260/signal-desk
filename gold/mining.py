"""参数挖掘：网格搜索 + 样本内/外走查 + 多重检验校正。

为什么挖参要校正：
在同一份数据上试 K 组参数，最好看的那组几乎必然"显著"——因为你在 K 次里挑最大值。
所以判定不能再用单次 p<0.05，必须按"试了多少次"来收紧阈值：
- **Deflated / E[max] 法**：K 次独立试验下，最大夏普的期望约 sqrt(2·ln K)，
  把观测到的 OOS 夏普跟这个基准比，超出部分才算真本事。
- **BH-FDR**：对 K 个置换 p 值做 Benjamini-Hochberg 校正，用 q 值判，不用原始 p。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gold.backtest import block_permutation, spread_fee_bps
from gold.freqs import freq_key
from gold.strategies import STRATEGIES, apply_strategy
from validation import metrics as M


# ---------------------------------------------------------------- 工具
def make_grid(strategy: str, axes: dict | None = None) -> list[dict]:
    """由参数轴生成笛卡尔网格。axes 缺省时用策略自带默认值的单点。"""
    spec = STRATEGIES[strategy]
    axes = axes or {k: [v] for k, v in spec["params"].items()}
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]


def split_is_oos(df: pd.DataFrame, is_ratio: float = 0.7,
                 purge_bars: int | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """训练/样本外切分的**唯一实现**（委托 gold/oos.py）。

    其它模块禁止自己按比例截断 DataFrame —— 切分口径只能有一处，
    否则"训练段/样本外段"的定义会各说各话。需要滚动切分请用
    `gold.oos.rolling_folds` / `outward_fold`。
    """
    from gold import oos as _oos

    fold = _oos.fixed_fold(df, is_ratio, purge_bars=purge_bars)
    return _oos.slice_fold(df, fold)


def _segment_metrics(seg: pd.DataFrame, strategy: str, params: dict,
                     spread_usd: float, tf: str = "D1") -> dict:
    """一段行情上的策略表现（次日执行 + 点差）。"""
    close = seg["close"].astype(float)
    pos = apply_strategy(seg, strategy, params).shift(1).fillna(0.0)
    price = float(close.mean())
    fee = spread_fee_bps(price, spread_usd)
    fwd = close.pct_change().shift(-1)
    # 必须按周期年化（H1=√6048），否则夏普被低估 4.9 倍
    st = M.summarize_strategy(fwd, pos, fee_bps=fee, freq=freq_key(tf),
                              allow_short=False)
    return {
        "return": float(st["net"].get("total_return", float("nan"))),
        "sharpe": float(st["net"].get("sharpe", float("nan"))),
        "mdd": float(st["net"].get("max_drawdown", float("nan"))),
        "hit": float(st["hit"].get("hit_rate", float("nan"))),
        "trades": int(st["trades"]),
        "fee_bps": fee,
        "_fwd": fwd, "_pos": pos,
    }


def emax_sharpe(n_trials: int) -> float:
    """K 次试验下最大夏普的期望（标准正态近似）。K<=1 时为 0。"""
    k = max(1, int(n_trials))
    if k == 1:
        return 0.0
    return math.sqrt(2.0 * math.log(k))


def deflated_p(oos_sharpe: float, n_trials: int,
               sr_std: float = 1.0) -> float:
    """把 OOS 夏普对"挖 K 次的期望最大值"做标准化，返回单侧 p 值。

    p 越小，说明这组夏普越不像是"从 K 次里挑出来的最大值"。
    用正态近似：z = (SR - E[max]) / sr_std。
    """
    if oos_sharpe != oos_sharpe:
        return 1.0
    z = (float(oos_sharpe) - emax_sharpe(n_trials)) / max(1e-9, float(sr_std))
    # 单侧：P(Z > z)
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def bh_qvalues(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg 校正，返回 q 值（与输入同序）。"""
    n = len(pvals)
    if n == 0:
        return []
    idx = sorted(range(n), key=lambda i: (pvals[i] if pvals[i] == pvals[i] else 1.0))
    q = [1.0] * n
    prev = 1.0
    for rank in range(n, 0, -1):
        i = idx[rank - 1]
        p = pvals[i] if pvals[i] == pvals[i] else 1.0
        val = min(prev, p * n / rank)
        q[i] = val
        prev = val
    return q


@dataclass
class MiningResult:
    strategy: str = ""
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    best: dict = field(default_factory=dict)
    verdict: str = ""
    reason: str = ""
    n_trials: int = 0
    is_range: tuple = ()
    oos_range: tuple = ()
    note: str = ""
    heatmap: dict = field(default_factory=dict)


def mine(df: pd.DataFrame, strategy: str, axes: dict | None = None,
         spread_usd: float = 0.33, is_ratio: float = 0.7,
         n_perm: int = 200, seed: int = 7, tf: str = "D1",
         fold=None) -> MiningResult:
    """网格挖掘 + IS 选优 + OOS 验证 + 多重检验校正。"""
    grid = make_grid(strategy, axes)
    if df is None or len(df) < 240:
        return MiningResult(strategy=strategy, note="K 线不足 240 根，无法挖掘")
    is_df, oos_df = split_is_oos(df, is_ratio)
    if len(is_df) < 150 or len(oos_df) < 60:
        return MiningResult(strategy=strategy, note="IS/OOS 段太短")

    rows = []
    for params in grid:
        a = _segment_metrics(is_df, strategy, params, spread_usd, tf)
        b = _segment_metrics(oos_df, strategy, params, spread_usd, tf)
        perm = block_permutation(b["_fwd"], b["_pos"], n_perm=n_perm, block=20,
                                 fee_bps=b["fee_bps"], seed=seed)
        rows.append({
            "params": params,
            "IS收益": a["return"], "IS夏普": a["sharpe"],
            "OOS收益": b["return"], "OOS夏普": b["sharpe"], "OOS回撤": b["mdd"],
            "OOS方向": b["hit"], "OOS笔数": b["trades"],
            "p": perm["p_value"],
        })

    tbl = pd.DataFrame(rows)
    # 把参数字典展开成独立列，供热力图 pivot 使用
    if len(tbl):
        expanded = pd.DataFrame(tbl["params"].tolist(), index=tbl.index)
        for col in expanded.columns:
            tbl[col] = expanded[col]
    # 用 IS 段选参（绝不能看 OOS）
    tbl["IS排名"] = tbl["IS收益"].rank(ascending=False, method="min")
    best_row = tbl.sort_values("IS收益", ascending=False).iloc[0]
    n_trials = len(tbl)

    # 多重检验：BH 对全部 K 个 p；Deflated 对最优那组的 OOS 夏普
    tbl["q"] = bh_qvalues(tbl["p"].tolist())
    oos_sharpe_std = float(tbl["OOS夏普"].std(ddof=1)) if n_trials > 2 else 1.0
    dp = deflated_p(float(best_row["OOS夏普"]), n_trials,
                    sr_std=oos_sharpe_std if oos_sharpe_std > 0 else 1.0)
    best_q = float(tbl.loc[best_row.name, "q"])

    bench_oos = M.performance_stats(
        oos_df["close"].astype(float).pct_change().shift(-1).dropna(),
        freq_key(tf))
    bench_ret = float(bench_oos.get("total_return", float("nan")))

    checks = {
        "OOS 夏普 > 0": float(best_row["OOS夏普"]) > 0,
        "OOS 跑赢基准": float(best_row["OOS收益"]) > bench_ret,
        "BH 校正后 q < 0.05": best_q < 0.05,
        "Deflated p < 0.05": dp < 0.05,
        "IS→OOS 未崩（≥50%）": (float(best_row["OOS夏普"]) >=
                               0.5 * float(best_row["IS夏普"])
                               if float(best_row["IS夏普"]) > 0 else False),
    }
    passed = [k for k, v in checks.items() if v]

    if len(passed) == len(checks):
        verdict, reason = "通过", "五道全过"
    elif len(passed) >= 3:
        verdict, reason = "部分通过", "过 " + "、".join(passed)
    else:
        verdict, reason = "未通过", "仅过 " + ("、".join(passed) if passed else "0 项")

    heatmap = {}
    keys = list((axes or {}).keys())
    if len(keys) == 2:
        heatmap = {"x": keys[0], "y": keys[1],
                   "z": tbl.pivot_table(index=keys[1], columns=keys[0],
                                        values="OOS收益", aggfunc="mean")}

    return MiningResult(
        strategy=strategy, table=tbl, best=dict(best_row["params"]),
        verdict=verdict, reason=reason, n_trials=n_trials,
        is_range=(str(is_df.index[0].date()), str(is_df.index[-1].date())),
        oos_range=(str(oos_df.index[0].date()), str(oos_df.index[-1].date())),
        note=(f"网格 {n_trials} 组 · E[max SR]={emax_sharpe(n_trials):.2f} · "
              f"最优组 q={best_q:.3f} · Deflated p={dp:.3f} · "
              f"OOS 基准 {bench_ret:+.2%}"),
        heatmap=heatmap,
    )
