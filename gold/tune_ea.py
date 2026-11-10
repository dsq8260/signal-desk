"""EA 微调：把 EA池 里的自研 EA 沿它自己的参数做小步邻域搜索。

与相邻模块的分工
----------------
- `mining.py`：内置策略的参数网格（按策略名，走 `apply_strategy`）
- `forge.py`：从零生成候选（整个基因组空间随机采样）
- **`tune_ea.py`（本模块）**：**基因组已经定了**，只挪动它的 `params`；
  其余基因——过滤 / 方向 / 止损止盈——一律冻结。这才是"微调"，
  不是重新设计一个 EA。

为什么直接复用 `forge.evaluate_genome`
--------------------------------------
微调的结果要和 EA池 里的原版指标**同口径可比**。IS/OOS 切分、点差换算、
按周期年化夏普（`freq_key`）、次日执行 `shift(1)` 全部由 forge 那一条路径
保证，这里不重写任何一行回测或信号逻辑，避免出现两套算法。

口径提醒：微调也是"试了 K 组"，所以 `deflated_p` 按实际试过的组数算，
不能用单次 p<0.05 判"有效"。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from gold.backtest import spread_fee_bps
from gold.combo import FILTERS
from gold.factors import build_features
from gold.forge import Genome, _all_entries, evaluate_genome
from gold.freqs import per_year, years_between
from gold.mining import deflated_p, emax_sharpe, split_is_oos


def _num(v) -> float | None:
    """尽量转成 float；转不了返回 None（用于"值是否相同"判断）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _same(a, b) -> bool:
    fa, fb = _num(a), _num(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) <= 1e-12 * max(1.0, abs(fa), abs(fb))
    return str(a) == str(b)


def _sort_values(vals: list) -> list:
    nums = [_num(v) for v in vals]
    if all(n is not None for n in nums):
        return sorted(vals, key=lambda v: float(v))
    return sorted(vals, key=str)


def ladder(entry: str, key: str) -> list:
    """该进场的这个参数，系统里**允许取过**的所有值（来自 forge 的网格）。"""
    spec = _all_entries().get(entry)
    vals: list = []
    if spec is not None:
        _fn, defaults, grid = spec
        if key in (defaults or {}):
            vals.append(defaults[key])
        for g in grid:
            if key in g:
                vals.append(g[key])
    out: list = []
    for v in _sort_values(vals):
        if not any(_same(v, o) for o in out):
            out.append(v)
    return out


def neighbor_values(values: list, cur, steps: int = 1) -> list:
    """在排序好的梯子上取 cur 及其左右各 steps 个邻居。

    cur 不在梯子上时，把它插进去再取——保证"原值"永远在候选里。
    """
    vals = list(values)
    if not any(_same(cur, v) for v in vals):
        vals = _sort_values(vals + [cur])
    i = next(i for i, v in enumerate(vals) if _same(cur, v))
    lo = max(0, i - int(steps))
    hi = min(len(vals), i + int(steps) + 1)
    return vals[lo:hi]


def variant_params(entry: str, params: dict, steps: int = 1,
                   max_combos: int = 24) -> list[dict]:
    """围绕原参数生成邻域组合；原参数一定排在最前面。

    组合数不超过 max_combos 时做完整笛卡尔积（双参数 EA 通常是 3×3=9 组）；
    超出时退化成"一次只动一个参数"（one-at-a-time），保证耗时可预期。
    """
    base = {k: v for k, v in dict(params or {}).items()}
    if not base:
        return [{}]

    axes = {k: neighbor_values(ladder(entry, k), cur, steps)
            for k, cur in base.items()}
    keys = list(base)
    total = 1
    for k in keys:
        total *= max(1, len(axes[k]))

    combos: list[dict] = []
    if total <= int(max_combos):
        raw = [dict(zip(keys, c))
               for c in itertools.product(*(axes[k] for k in keys))]
        combos = [base] + [c for c in raw if c != base]
    else:
        combos = [base]
        for k in keys:
            for v in axes[k]:
                if _same(v, base[k]):
                    continue
                d = dict(base)
                d[k] = v
                combos.append(d)

    out: list[dict] = []
    for c in combos:
        if not any(_same_dict(c, o) for o in out):
            out.append(c)
    return out


def _same_dict(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    return all(_same(a[k], b[k]) for k in a)


def param_text(params: dict) -> str:
    """参数写成人话：`fast=10,slow=60`；空参数写「默认」。"""
    if not params:
        return "默认"
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))


def coerce_genome(obj) -> Genome | None:
    """把 EA池 里的 `_genome`（dict 或 JSON 字符串）还原成 Genome。"""
    import json

    if isinstance(obj, Genome):
        return obj
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(obj, dict):
        return None
    entry = obj.get("entry")
    if not entry:
        return None
    sl, tp = obj.get("sl"), obj.get("tp")
    # 旧记录里的 `zhi`（已移除的干支维度）忽略：干支已从基因组删除，
    # 这类记录可以继续微调它剩下的参数，但不会再带上地支过滤。
    return Genome(
        entry=str(entry),
        params=dict(obj.get("params") or {}),
        filter=str(obj.get("filter") or "无过滤"),
        allow_short=bool(obj.get("allow_short")),
        sl=None if sl is None else float(sl),
        tp=None if tp is None else float(tp),
    )


def frozen_text(g: Genome) -> str:
    """被冻结、本次不动的基因，给界面显示用。"""
    risk = []
    if g.sl is not None:
        risk.append(f"SL{g.sl * 100:.0f}%")
    if g.tp is not None:
        risk.append(f"TP{g.tp * 100:.0f}%")
    return (f"过滤={g.filter} · "
            f"{'双向' if g.allow_short else '只多'} · "
            f"风控={('+'.join(risk) or '无')}")


@dataclass
class TuneResult:
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    best: dict = field(default_factory=dict)
    best_label: str = ""
    best_genome: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    baseline_label: str = ""
    n_trials: int = 0
    n_valid: int = 0
    emax: float = 0.0
    deflated_p: float = 1.0
    improved: bool = False
    note: str = ""
    is_range: tuple = ()
    oos_range: tuple = ()
    tf: str = ""


def tune(df: pd.DataFrame, genome, tf: str = "H1", spread_usd: float = 0.33,
         is_ratio: float = 0.7, steps: int = 1, max_combos: int = 24,
         min_trades: int = 5, fold=None) -> TuneResult:
    """围绕一个 EA池 基因组的参数做小步邻域微调（IS 选参 → OOS 验证）。"""
    g0 = coerce_genome(genome)
    if g0 is None:
        return TuneResult(tf=tf, note="这一行没有基因组信息，无法微调")
    if df is None or len(df) < 400:
        return TuneResult(tf=tf, note="K 线不足 400 根，无法微调")

    if fold is not None:
        from gold import oos as _oos

        is_df, oos_df = _oos.slice_fold(df, fold)
    else:
        is_df, oos_df = split_is_oos(df, is_ratio)
    if len(is_df) < 150 or len(oos_df) < 60:
        return TuneResult(tf=tf, note="IS/OOS 段太短，微调结论不可信")

    feat = build_features(df)
    fee = spread_fee_bps(float(df["close"].mean()), spread_usd)
    oos_range = (str(oos_df.index[0].date()), str(oos_df.index[-1].date()))
    oos_years = years_between(oos_range)

    variants = variant_params(g0.entry, g0.params, steps, max_combos)
    rows = []
    mask_cache: dict = {}
    for p in variants:
        g = Genome(g0.entry, dict(p), g0.filter, g0.allow_short,
                   g0.sl, g0.tp)
        ev = evaluate_genome(df, feat, is_df, oos_df, fee, g, min_trades,
                             mask_cache, tf)
        if ev is None:
            continue
        rows.append({
            "params": dict(p),
            "参数": param_text(p),
            "EA": ev["label"],
            "IS收益": ev["IS收益"],
            "OOS收益": ev["OOS收益"],
            "OOS夏普": ev["OOS夏普"],
            "OOS回撤": ev["OOS回撤"],
            "OOS方向": ev["OOS方向"],
            "笔数": ev["笔数"],
            "回合": ev["回合"],
            "在场": ev["在场"],
            "盈利因子": ev["盈利因子"],
            "每笔期望": ev["每笔期望"],
            "原版": _same_dict(dict(p), g0.params),
        })

    if not rows:
        return TuneResult(tf=tf, note="邻域里所有组合都没达到最少笔数要求，"
                                      "建议放宽步长或换周期")

    tbl = pd.DataFrame(rows)
    tbl["IS排名"] = tbl["IS收益"].rank(ascending=False, method="min")
    # 每年回合：判断"笔数够不够"要按年，跨周期才可比（也是 MT5 报告的口径）
    tbl["每年回合"] = [per_year(c, oos_years) for c in tbl["回合"]]

    base_rows = tbl[tbl["原版"]]
    baseline = base_rows.iloc[0].to_dict() if len(base_rows) else {}
    best_row = tbl.sort_values("IS收益", ascending=False).iloc[0]
    best = dict(best_row["params"])
    # 多重检验的 K 是"真正试过多少组"，不是"通过最少笔数后剩下多少组"——
    # 用后者会把没成交的组白送掉，等于偷偷放宽显著性。
    n_trials = int(len(variants))
    n_valid = int(len(tbl))

    sr_std = float(tbl["OOS夏普"].std(ddof=1)) if n_valid > 2 else 1.0
    dp = deflated_p(float(best_row["OOS夏普"]), n_trials,
                    sr_std=sr_std if sr_std and sr_std > 0 else 1.0)

    g_best = Genome(g0.entry, best, g0.filter, g0.allow_short,
                    g0.sl, g0.tp)

    b_sh = float(baseline.get("OOS夏普", float("nan"))) if baseline else float("nan")
    n_sh = float(best_row["OOS夏普"])
    improved = bool(b_sh == b_sh and n_sh > b_sh)

    note = (f"邻域试了 {n_trials} 组（{n_valid} 组够笔数）· "
            f"E[max SR]={emax_sharpe(n_trials):.2f} · "
            f"微调版 Deflated p={dp:.3f} · 原版 OOS 夏普 "
            f"{(f'{b_sh:.2f}' if b_sh == b_sh else '—')} → 微调版 {n_sh:.2f}")

    return TuneResult(
        table=tbl,
        best=best,
        best_label=g_best.label(),
        best_genome=g_best.as_dict(),
        baseline={k: v for k, v in baseline.items() if k != "params"},
        baseline_label=g0.label(),
        n_trials=n_trials,
        n_valid=n_valid,
        emax=emax_sharpe(n_trials),
        deflated_p=dp,
        improved=improved,
        note=note,
        is_range=(str(is_df.index[0].date()), str(is_df.index[-1].date())),
        oos_range=oos_range,
        tf=tf,
    )


def as_pool_row(res: TuneResult, tf: str | None = None) -> pd.DataFrame:
    """把微调结果整理成一行"累计榜格式"，可直接交给 register_top 登记前向。"""
    if not res.best_genome or res.table is None or len(res.table) == 0:
        return pd.DataFrame()
    row = res.table.sort_values("IS收益", ascending=False).iloc[0]
    return pd.DataFrame([{
        "EA": res.best_label,
        "周期": tf or res.tf,
        "最优夏普": float(row["OOS夏普"]),
        "最优收益": float(row["OOS收益"]),
        "最优回撤": float(row["OOS回撤"]),
        "平均笔数": float(row["笔数"]),
        "出现次数": 1,
        "_genome": res.best_genome,
    }])
