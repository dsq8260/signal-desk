"""分层搜索：用贪心前向选择替代全枚举，但要如实记录"试过多少组"。

为什么不能只是缩短名单：
统计上的试验次数是**评估过的候选数**，不是最终保留数。剪枝省的是时间，
不是校验负担——所以 `trials_evaluated` 会累计每一层真正算过的组合，
最终 BH / Deflated 仍按这个数算。这样"省时间"不会顺手把门槛也放低。

三层结构（每层只保留前 K 名）：
    L1 进场（5）           → 留 top 3
    L2 进场 × 过滤（3）     → 留 top 3
    L3 × 参数网格（3-9 组） → 留 top 2

（早期版本还有一层「干支」，那是给股票日频择日用的；本系统只做外汇/黄金，
已移除，见 `combo.py` 顶部说明。）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gold.backtest import block_permutation
from gold.combo import ENTRIES, FILTERS, _metrics
from gold.factors import build_features
from gold.freqs import freq_key
from gold.mining import bh_qvalues, deflated_p, emax_sharpe, split_is_oos
from validation import metrics as M

# 每个进场的参数网格（小尺寸，避免第三层爆炸）
PARAM_AXES = {
    # 2026-11-04：新进场的参数网格。没有它，生成器只能产生 1 个默认参数，
    # 「ATR通道突破」永远出不来 k=1.8 这类已验证过的好变体。
    "ATR通道突破": [
        {"n": 20, "m": 14, "k": 1.5, "confirm": 1},
        {"n": 20, "m": 14, "k": 1.8, "confirm": 1},
        {"n": 20, "m": 14, "k": 2.0, "confirm": 1},
        {"n": 20, "m": 14, "k": 2.5, "confirm": 1},
        {"n": 20, "m": 14, "k": 2.0, "confirm": 2},
        {"n": 30, "m": 14, "k": 2.0, "confirm": 1},
    ],
    "均线偏离回归": [
        {"ma_n": 10, "entry_thr": 100.0, "exit_thr": 100.0},
        {"ma_n": 20, "entry_thr": 130.0, "exit_thr": 100.0},
        {"ma_n": 20, "entry_thr": 160.0, "exit_thr": 150.0},
        {"ma_n": 30, "entry_thr": 160.0, "exit_thr": 150.0},
        {"ma_n": 30, "entry_thr": 190.0, "exit_thr": 150.0},
    ],

    "SMA金叉": [{"fast": f, "slow": s} for f in (10, 20, 30) for s in (60, 90, 120)],
    "唐奇安突破": [{"entry": e, "exit_": x} for e in (10, 20, 30) for x in (5, 10, 20)],
    "RSI反转": [{"buy": b} for b in (25.0, 30.0, 35.0)],
    "MACD金叉": [{"signal": s} for s in (7, 9, 12)],
    "LLT上穿": [{"n": n} for n in (30, 60, 90)],
}

DEFAULT_KEEP = {"l1": 3, "l2": 3, "l3": 2}


@dataclass
class StratResult:
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    layer_log: list[dict] = field(default_factory=list)
    trials_evaluated: int = 0
    trials_naive: int = 0
    emax: float = 0.0
    global_p: float = 1.0
    best: dict = field(default_factory=dict)
    note: str = ""
    is_range: tuple = ()
    oos_range: tuple = ()


def _pos_with_params(df, feat, spec, params) -> pd.Series:
    """按候选 + 自定义参数生成仓位（不改 combo.py，这里自己算）。"""
    fn, defaults = ENTRIES[spec["entry"]]
    kw = {**defaults, **(params or {})}
    pos = pd.Series(fn(df, **kw), index=df.index).fillna(0.0)

    flt = FILTERS[spec["filter"]]
    if flt is not None:
        from gold.factors import regime_mask

        name, q, mode = flt
        pos = pos * regime_mask(feat, name, q, mode).astype(float)

    return pos.clip(0.0, 1.0)


def search(df: pd.DataFrame, spread_usd: float = 0.33, is_ratio: float = 0.7,
           n_perm: int = 100, seed: int = 7,
           keep: dict | None = None, tf: str = "D1",
           fold=None) -> StratResult:
    if df is None or len(df) < 400:
        return StratResult(note="K 线不足 400 根，无法分层搜索")

    keep = {**DEFAULT_KEEP, **(keep or {})}
    feat = build_features(df)
    if fold is not None:
        from gold import oos as _oos

        is_df, oos_df = _oos.slice_fold(df, fold)
    else:
        is_df, oos_df = split_is_oos(df, is_ratio)

    trials = 0
    log: list[dict] = []

    def evaluate(spec, params=None) -> dict | None:
        nonlocal trials
        pos_full = _pos_with_params(df, feat, spec, params or {})
        a = _metrics(is_df, pos_full, spread_usd, tf)
        trials += 1
        if not np.isfinite(a["ret"]) or a["trades"] < 3:
            return None
        return {"spec": spec, "params": params or {},
                "IS收益": a["ret"], "IS夏普": a["sharpe"], "笔数": a["trades"],
                "_pos": pos_full}

    # ---- L1：只测进场 ----
    l1 = [r for r in (evaluate({"entry": e, "filter": "无过滤"})
                      for e in ENTRIES) if r]
    l1 = sorted(l1, key=lambda r: -r["IS收益"])[: keep["l1"]]
    log.append({"层": "L1 进场", "评估": len(ENTRIES), "保留": len(l1),
                "名单": [r["spec"]["entry"] for r in l1]})

    # ---- L2：× 过滤 ----
    l2 = []
    for base in l1:
        for f in FILTERS:
            r = evaluate({"entry": base["spec"]["entry"], "filter": f})
            if r:
                l2.append(r)
    l2 = sorted(l2, key=lambda r: -r["IS收益"])[: keep["l2"]]
    log.append({"层": "L2 ×过滤", "评估": len(l1) * len(FILTERS), "保留": len(l2),
                "名单": [f"{r['spec']['entry']}/{r['spec']['filter']}" for r in l2]})

    # ---- L3：× 参数网格 ----
    l3 = []
    for base in l2:
        for params in PARAM_AXES.get(base["spec"]["entry"], [{}]):
            r = evaluate(base["spec"], params)
            if r:
                l3.append(r)
    l3 = sorted(l3, key=lambda r: -r["IS收益"])[: keep["l3"]]
    log.append({"层": "L3 ×参数", "评估": sum(len(PARAM_AXES.get(b["spec"]["entry"], [{}]))
                                            for b in l2),
                "保留": len(l3),
                "名单": [f"{r['spec']['entry']} {r['params']}" for r in l3]})

    if not l3:
        return StratResult(note=f"分层后无有效候选（评估 {trials} 组）",
                           trials_evaluated=trials, layer_log=log)

    # ---- 最终名单：算 OOS + 置换 + 校正 ----
    rows = []
    for r in l3:
        b = _metrics(oos_df, r["_pos"], spread_usd, tf)
        perm = block_permutation(b["_fwd"], b["_pos"], n_perm=n_perm, block=20,
                                 fee_bps=b["_fee"], seed=seed)
        rows.append({
            "进场": r["spec"]["entry"], "过滤": r["spec"]["filter"],
            "参数": str(r["params"]) if r["params"] else "默认",
            "IS收益": r["IS收益"], "IS夏普": r["IS夏普"],
            "OOS收益": b["ret"], "OOS夏普": b["sharpe"], "OOS回撤": b["mdd"],
            "笔数": r["笔数"], "p": perm["p_value"],
        })

    tbl = pd.DataFrame(rows).sort_values("IS收益", ascending=False).reset_index(drop=True)
    tbl["q"] = bh_qvalues(tbl["p"].tolist())
    n = len(tbl)
    best = tbl.iloc[0]
    sr_std = float(tbl["OOS夏普"].std(ddof=1)) if n > 2 else 1.0
    gp = deflated_p(float(best["OOS夏普"]), trials,
                    sr_std=sr_std if sr_std > 0 else 1.0)

    # 全枚举 = 每个进场自己的参数网格 × 过滤数（干支层已移除）
    naive = sum(len(v) for v in PARAM_AXES.values()) * len(FILTERS)
    ref = M.performance_stats(
        oos_df["close"].astype(float).pct_change().shift(-1).dropna(),
        freq_key(tf))

    return StratResult(
        table=tbl,
        layer_log=log,
        trials_evaluated=trials,
        trials_naive=naive,
        emax=emax_sharpe(trials),
        global_p=gp,
        best={"spec": {"entry": best["进场"], "filter": best["过滤"]},
              "params": best["参数"], "IS收益": float(best["IS收益"]),
              "OOS收益": float(best["OOS收益"]), "q": float(best["q"]),
              "global_p": gp},
        is_range=(str(is_df.index[0].date()), str(is_df.index[-1].date())),
        oos_range=(str(oos_df.index[0].date()), str(oos_df.index[-1].date())),
        note=(f"分层评估 {trials} 组（全枚举需 {naive} 组）· "
              f"E[max SR]={emax_sharpe(trials):.2f} · 全局 Deflated p={gp:.3f} · "
              f"OOS 基准 {float(ref.get('total_return', float('nan'))):+.2%}"),
    )
