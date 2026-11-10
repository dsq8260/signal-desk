"""遗传进化（B）：把过关的 EA 当种子，在其附近变异继续搜索。

为什么需要进化而不是一直随机采样
------------------------------
随机采样在 2106 的空间里撒点，命中好区域后就走了；进化会**围着好的区域反复微调**：
改一个参数、换一个过滤条件、切一下方向——相当于自动做"局部精调"。

变异算子（每次只动一处，保持可解释）
--------------------------------
1. 参数抖动：在参数网格里换一个同族值（如 fast 20 → 30）
2. 换过滤：无过滤 / 低波动 / 强趋势 三选一
3. 切方向：只多 ↔ 双向

统计纪律
--------
进化会产生大量评估，**每代都记录累计评估次数**，
最终仍按累计次数算 E[max SR]，不允许"分批跑就不算试验"。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gold.backtest import spread_fee_bps
from gold.combo import FILTERS
from gold.factors import build_features
from gold.forge import (
    STOP_GRID,
    Genome,
    build_position,
    evaluate_genome,
    genome_space,
    sample_genomes,
)
from gold.mining import deflated_p, emax_sharpe, split_is_oos
from gold.forge import _param_grid
from gold.strata import PARAM_AXES


@dataclass
class EvolveResult:
    best: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_eval: int = 0
    generations: int = 0
    best_sharpe: float = float("nan")
    global_p: float = 1.0
    emax: float = 0.0
    note: str = ""
    is_range: tuple = ()
    oos_range: tuple = ()


def _apply_op(g: Genome, op: str, rng: random.Random) -> Genome:
    """施加一个算子，返回子代（可能无变化，由 _mutate 负责兜底）。"""
    child = Genome(g.entry, dict(g.params), g.filter, g.allow_short,
                   g.sl, g.tp)
    if op == "param":
        grid = _param_grid(child.entry) or [{}]
        alt = [x for x in grid if dict(x) != child.params] or list(grid)
        child.params = dict(rng.choice(alt))
    elif op == "filter":
        alt = [f for f in FILTERS if f != child.filter] or list(FILTERS)
        child.filter = rng.choice(alt)
    elif op == "risk":
        alt = [x for x in STOP_GRID if x != (child.sl, child.tp)] or STOP_GRID
        child.sl, child.tp = rng.choice(alt)
    else:
        child.allow_short = not child.allow_short
    return child


def _mutate(g: Genome, rng: random.Random) -> Genome:
    """随机施加一个变异，**保证子代与父代不同**（否则白耗一次评估）。"""
    ops = ["param", "filter", "side", "risk"]
    rng.shuffle(ops)
    for op in ops:
        child = _apply_op(g, op, rng)
        if child != g:
            return child
    # 理论上到不了这；兜底强制换方向
    child = Genome(g.entry, dict(g.params), g.filter,
                   not g.allow_short, g.sl, g.tp)
    return child


def evolve(df: pd.DataFrame, generations: int = 5, population: int = 60,
           elite: int = 10, seed: int = 7, spread_usd: float = 0.33,
           is_ratio: float = 0.7, allow_short: bool = True,
           min_trades: int = 5, tf: str = "D1", fold=None) -> EvolveResult:
    """进化搜索：种子随机 → 逐代择优变异。"""
    if df is None or len(df) < 400:
        return EvolveResult(note="K 线不足 400 根，无法进化")

    rng = random.Random(seed)
    feat = build_features(df)
    if fold is not None:
        from gold import oos as _oos

        is_df, oos_df = _oos.slice_fold(df, fold)
    else:
        is_df, oos_df = split_is_oos(df, is_ratio)
    fee = spread_fee_bps(float(df["close"].mean()), spread_usd)
    space = len(genome_space((False, True) if allow_short else (False,)))

    mask_cache: dict = {}
    seen: set[str] = set()
    n_eval = 0
    pop: list[dict] = []

    def evaluate(genomes: list[Genome]) -> list[dict]:
        nonlocal n_eval
        out = []
        for g in genomes:
            key = g.label()
            n_eval += 1
            if key in seen:
                continue
            seen.add(key)
            ev = evaluate_genome(df, feat, is_df, oos_df, fee, g,
                                 min_trades, mask_cache, tf)
            if ev is not None:
                ev["genome"] = g
                out.append(ev)
        return out

    # 第 0 代：随机种子
    seeds = sample_genomes(population, seed=seed, allow_short=allow_short)
    pop = evaluate(seeds)
    hist = []
    if pop:
        pop.sort(key=lambda x: -float(x["OOS夏普"]))
        hist.append({"代": 0, "评估": n_eval, "有效": len(pop),
                     "最优夏普": float(pop[0]["OOS夏普"]),
                     "最优EA": pop[0]["label"]})

    for gen in range(1, int(generations) + 1):
        if not pop:
            break
        parents = pop[: max(1, int(elite))]
        children: list[Genome] = []
        for _ in range(int(population)):
            p = rng.choice(parents)
            children.append(_mutate(p["genome"], rng))
        new = evaluate(children)
        pop = pop + new
        pop.sort(key=lambda x: -float(x["OOS夏普"]))
        # 只保留前 elite 名，防止无限膨胀
        pop = pop[: max(1, int(elite))]
        hist.append({"代": gen, "评估": n_eval, "有效": len(new),
                     "最优夏普": float(pop[0]["OOS夏普"]),
                     "最优EA": pop[0]["label"]})

    rows = []
    for ev in pop:
        g = ev["genome"]
        rows.append({
            "EA": ev["label"],
            "进场": g.entry,
            "过滤": g.filter,
            "方向": "双向" if g.allow_short else "只多",
            "IS收益": ev["IS收益"],
            "OOS收益": ev["OOS收益"],
            "OOS夏普": ev["OOS夏普"],
            "OOS回撤": ev["OOS回撤"],
            "OOS方向": ev["OOS方向"],
            "笔数": ev["笔数"],
            "回合": ev.get("回合", float("nan")),
            "在场": ev.get("在场", float("nan")),
            "OOS根数": ev.get("OOS根数", 0),
            "盈利因子": ev.get("盈利因子", float("nan")),
            "每笔期望": ev.get("每笔期望", float("nan")),
            "_genome": g.as_dict(),
        })

    best = pd.DataFrame(rows)
    if len(best):
        best = best.sort_values("OOS夏普", ascending=False).reset_index(drop=True)

    bsr = float(best["OOS夏普"].max()) if len(best) else float("nan")
    gp = deflated_p(bsr, n_eval) if bsr == bsr else 1.0
    return EvolveResult(
        best=best, history=pd.DataFrame(hist), n_eval=n_eval,
        generations=int(generations), best_sharpe=bsr, global_p=gp,
        emax=emax_sharpe(n_eval),
        is_range=(str(is_df.index[0].date()), str(is_df.index[-1].date())),
        oos_range=(str(oos_df.index[0].date()), str(oos_df.index[-1].date())),
        note=(f"进化 {generations} 代 × 种群 {population} · 累计评估 {n_eval} 次"
              f"（空间 {space}）· E[max SR]={emax_sharpe(n_eval):.2f} · "
              f"最优夏普 {bsr:.2f} · 全局 Deflated p={gp:.3f}"),
    )
