"""自主设计 EA：系统自己生成候选、自己回测、自己筛选。

与现有模块的分工
----------------
- `combo.py`：固定 5 进场 × 3 过滤 × 13 干支 = 195，全枚举
- `strata.py`：贪心分层，省时间
- **`forge.py`（本模块）**：**程序化生成候选**——把进场、过滤、干支、参数、
  方向都当基因组，随机组合出上千个候选，逐个回测，再按统计门槛筛选、
  按相关性去重，最终吐出"系统自己设计出来的优秀 EA"。

为什么不是纯穷举
----------------
基因组空间约 8 进场 × 3 过滤 × 13 干支 × 6 参数变体 × 2 方向 ≈ 3700。
穷举能跑但慢；随机采样 + 去重能在 400 个候选里覆盖大部分有价值区域，
同时**如实记录试过多少次**（用于多重检验校正）。
"""

from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gold.backtest import block_permutation, spread_fee_bps
from gold.combo import ENTRIES, FILTERS, ALL_FILTERS
from gold.entries_ext import EXT_ENTRIES, apply_stops
from gold.freqs import freq_key
from gold.factors import build_features, regime_mask
from gold.mining import bh_qvalues, deflated_p, emax_sharpe, split_is_oos
from gold.strata import PARAM_AXES
from validation import metrics as M


def _all_entries() -> dict:
    """内置进场 + 扩展进场统一成 {名称: (fn, 默认参数, 参数网格)}。"""
    out = {}
    for k, (fn, defaults) in ENTRIES.items():
        out[k] = (fn, dict(defaults), list(PARAM_AXES.get(k) or [dict(defaults)]))
    for k, (fn, defaults, grid, _minlen) in EXT_ENTRIES.items():
        out[k] = (fn, dict(defaults), list(grid))
    return out


def _param_grid(entry: str) -> list[dict]:
    e = _all_entries().get(entry)
    return list(e[2]) if e else [{}]


@dataclass
class Genome:
    """一个 EA 的"基因"。

    sl / tp 是**止损/止盈比例**（0.02 = 2%），None 表示不启用。

    历史上的「干支（日柱地支）」维度已移除：那是给股票日频择日用的，
    在外汇/黄金 H1 上会把在场占比压到 5% 左右、每年只剩十几笔。
    旧设计记录里带干支的基因组仍能被 `coerce` 读出来（字段忽略）。
    """
    entry: str = ""
    params: dict = field(default_factory=dict)
    filter: str = "无过滤"
    allow_short: bool = False
    sl: float | None = None
    tp: float | None = None

    def label(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in sorted(self.params.items())) or "默认"
        d = "双向" if self.allow_short else "只多"
        risk = []
        if self.sl is not None:
            risk.append(f"SL{self.sl * 100:.0f}%")
        if self.tp is not None:
            risk.append(f"TP{self.tp * 100:.0f}%")
        r = ("/" + "+".join(risk)) if risk else ""
        return f"{self.entry}[{p}]/{self.filter}/{d}{r}"

    def as_dict(self) -> dict:
        return {"entry": self.entry, "params": self.params,
                "filter": self.filter,
                "allow_short": self.allow_short,
                "sl": self.sl, "tp": self.tp}


STOP_GRID = [(None, None), (0.02, None), (None, 0.05), (0.02, 0.05)]


def genome_space(allow_short_options=(False, True),
                 with_rules: bool = True) -> list[Genome]:
    """完整基因空间（用于统计"可尝试总数"）。

    ``with_rules=False`` 时不含止损止盈维度（回到旧口径）。
    """
    out = []
    stops = STOP_GRID if with_rules else [(None, None)]
    for entry, (_fn, _def, grid) in _all_entries().items():
        for params in grid:
            for flt in FILTERS:
                for sh in allow_short_options:
                    for sl, tp in stops:
                        out.append(Genome(entry, dict(params), flt,
                                          bool(sh), sl, tp))
    return out


def sample_genomes(n: int, seed: int = 7,
                   allow_short: bool = False) -> list[Genome]:
    """随机采样候选基因组（确定性，可复现）。"""
    options = (False, True) if allow_short else (False,)
    space = genome_space(options)
    if n >= len(space):
        return space
    rng = random.Random(seed)
    return rng.sample(space, int(n))


def build_position(df: pd.DataFrame, feat: pd.DataFrame,
                   g: Genome, mask_cache: dict | None = None) -> pd.Series:
    """把一个基因组翻译成仓位序列（0/1 或 -1/0/1）。"""
    entries = _all_entries()
    if g.entry not in entries:
        raise KeyError(g.entry)
    fn, defaults, _grid = entries[g.entry]
    kw = {**defaults, **(g.params or {})}
    pos = pd.Series(fn(df, **kw), index=df.index).fillna(0.0)

    flt = ALL_FILTERS.get(g.filter) if g.filter in ALL_FILTERS else FILTERS.get(g.filter)
    # 新增过滤：大周期确认 / 欧美时段（不走分位机制，直接算布尔掩码）
    if g.filter.startswith("大周期确认"):
        import re as _re

        from gold.factors import htf_trend as _htf

        _m = _re.search(r"(\d+)$", g.filter)
        _span = int(_m.group(1)) if _m else 20
        mask = _htf(df, mult=4, span=_span).astype(float)
        pos = pos * mask
        flt = None
    elif g.filter == "欧美时段":
        from gold.factors import session_ok as _sess

        mask = _sess(df).astype(float)
        pos = pos * mask
        flt = None
    if flt is not None:
        name, q, mode = flt
        key = (name, float(q), str(mode), len(feat))
        mask = None
        if mask_cache is not None:
            mask = mask_cache.get(key)
        if mask is None:
            mask = regime_mask(feat, name, q, mode).astype(float)
            if mask_cache is not None:
                mask_cache[key] = mask
        pos = pos * mask

    if g.allow_short:
        pos = pos.apply(lambda x: 1.0 if x > 0.5 else -1.0)
    else:
        pos = pos.clip(0.0, 1.0)

    if g.sl is not None or g.tp is not None:
        pos = apply_stops(df, pos, g.sl, g.tp)
    return pos


@dataclass
class ForgeResult:
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    survivors: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_sampled: int = 0
    n_valid: int = 0
    n_dedup: int = 0
    n_pass: int = 0
    space_size: int = 0
    emax: float = 0.0
    global_p: float = 1.0
    note: str = ""
    is_range: tuple = ()
    oos_range: tuple = ()


def _corr_dedup(returns: pd.DataFrame, thresh: float = 0.9) -> list[str]:
    """按收益相关性去重：保留**列顺序在前**的，丢弃与已保留高度相关的。

    调用方应先把列按"谁更强"排好（见 `_dedup_order`），这样每个相关簇留下的是
    最强代表，而不是碰巧先出现的那个。

    性能：之前逐个 `corrwith` 是 O(n²) 次 pandas 调用，300 个候选要跑几分钟；
    改成**一次算完整相关矩阵**再贪心，复杂度降为一次矩阵乘法。
    """
    cols = list(returns.columns)
    if not cols:
        return []
    if len(cols) == 1:
        return cols
    corr = returns.corr().abs()
    keep: list[str] = []
    for c in cols:
        if not keep:
            keep.append(c)
            continue
        if float(corr.loc[keep, c].max()) <= thresh:
            keep.append(c)
    return keep


def _dedup_order(tbl: pd.DataFrame, ret_map: dict) -> list[str]:
    """去重顺序：按 OOS 夏普**从强到弱**排，保证每个相关簇留下最强代表。

    为什么必须先排序：`_corr_dedup` 是"保留先出现的那个"，如果按基因组顺序传进去，
    每个相关簇保留的就是碰巧先出现的弱版本——实测把全空间最优夏普从 1.98 压到 1.87，
    甚至整条丢掉更强的基因组。
    """
    if tbl is None or len(tbl) == 0:
        return []
    s = tbl.sort_values("OOS夏普", ascending=False,
                        kind="mergesort")["EA"].tolist()
    return [e for e in s if e in ret_map]


def evaluate_genome(df: pd.DataFrame, feat: pd.DataFrame,
                    is_df: pd.DataFrame, oos_df: pd.DataFrame, fee_bps: float,
                    g: "Genome", min_trades: int = 5,
                    mask_cache: dict | None = None,
                    tf: str = "D1", n_perm: int = 0, block: int = 20,
                    seed: int = 7) -> dict | None:
    """评测一个基因组：IS 段够不够交易、OOS 段表现如何。不合格返回 None。

    ``n_perm > 0`` 时在 OOS 段跑**真实块置换**，返回「块置换p」与「随机检验」。
    以前这里是用收益反推一个 p（0.5-0.5·tanh(net·4)），那是伪造，已删除。
    """
    from validation import metrics as M

    try:
        pos_full = build_position(df, feat, g, mask_cache)
    except Exception:  # noqa: BLE001
        return None
    pos = pos_full.shift(1).fillna(0.0)

    fq = freq_key(tf)          # 关键：按周期年化，而不是一律按日
    is_fwd = is_df["close"].astype(float).pct_change().shift(-1)
    a = M.summarize_strategy(is_fwd, pos.reindex(is_df.index).fillna(0.0),
                             fee_bps=fee_bps, freq=fq,
                             allow_short=g.allow_short)
    if int(a["trades"]) < min_trades:
        return None

    pos_oos = pos.reindex(oos_df.index).fillna(0.0)
    oos_fwd = oos_df["close"].astype(float).pct_change().shift(-1)
    b = M.summarize_strategy(oos_fwd, pos_oos, fee_bps=fee_bps, freq=fq,
                             allow_short=g.allow_short)
    if int(b["trades"]) < min_trades:
        return None

    net = b["net"].get("total_return", float("nan"))
    if net != net:
        return None

    # 笔数口径：`trades` 是"调仓次数"（一进一出各算一次），MT5 测试器报的是
    # "回合"（一进一出算一笔）。两个都留着，界面才能和 MT5 对上。
    prev = pos_oos.shift(1).fillna(0.0)
    rounds = int(((pos_oos != 0) & (prev == 0)).sum())
    in_pos = float((pos_oos != 0).mean()) if len(pos_oos) else float("nan")
    rt = b.get("roundtrip", {}) or {}

    perm = None
    if int(n_perm) > 0:
        perm = block_permutation(oos_fwd, pos_oos, n_perm=int(n_perm),
                                 block=int(block), fee_bps=fee_bps, seed=seed)
    bench = float(oos_df["close"].iloc[-1] / oos_df["close"].iloc[0] - 1.0) \
        if len(oos_df) > 1 else float("nan")

    return {
        "genome": g,
        "label": g.label(),
        "IS收益": a["net"].get("total_return", float("nan")),
        "IS夏普": a["net"].get("sharpe", float("nan")),
        "IS方向": a["hit"].get("hit_rate", float("nan")),
        "OOS收益": net,
        "OOS夏普": b["net"].get("sharpe", float("nan")),
        "OOS回撤": b["net"].get("max_drawdown", float("nan")),
        "OOS方向": b["hit"].get("hit_rate", float("nan")),
        "笔数": int(b["trades"]),
        "回合": rounds,
        "在场": in_pos,
        "OOS根数": int(len(oos_df)),
        # 每笔质量：盈利因子 = 毛利/毛损（回合口径、扣点差）
        "盈利因子": float(rt.get("pf", float("nan"))),
        "每笔期望": float(rt.get("avg", float("nan"))),
        # 随机检验（真实块置换）与基准对照
        "块置换p": float(perm["p_value"]) if perm else float("nan"),
        "随机检验": bool(perm["better_than_random"]) if perm else False,
        "基准收益": bench,
        "跑赢基准": bool(net > bench) if bench == bench else False,
        "_pos_oos": pos_oos,
        "_fwd_oos": oos_fwd,
    }


def forge(df: pd.DataFrame, n_candidates: int = 300, spread_usd: float = 0.33,
          is_ratio: float = 0.7, n_perm: int | str = 50,
          allow_short: bool = False,
          dedup_thresh: float = 0.9, min_trades: int = 5,
          seed: int = 7, tf: str = "D1",
          genomes: list | None = None, alpha: float = 0.05,
          fold=None, run_id: str | None = None,
          symbol: str = "GOLD") -> ForgeResult:
    """自主设计：采样 → 回测 → 筛门槛 → 去重 → 排名。

    ``genomes``：**预注册**的候选空间。给了就用它（K = len(genomes)），不再随机采样。
    ``n_perm="auto"``：按 BH 可达性自动定置换次数（每候选 ≥ K/alpha 次），
    这样「BH q<0.05」才有可能达到；K 大时会被时间成本挡住，界面要如实提示。
    """
    if df is None or len(df) < 400:
        return ForgeResult(note="K 线不足 400 根，无法自主设计")

    feat = build_features(df)
    if fold is not None:
        # 切分由调用方（验证器）给定，这里不再自己切
        from gold import oos as _oos

        is_df, oos_df = _oos.slice_fold(df, fold)
    else:
        is_df, oos_df = split_is_oos(df, is_ratio)
    fee = spread_fee_bps(float(df["close"].mean()), spread_usd)

    if genomes:
        cands = list(genomes)
        prereg = True
    else:
        cands = sample_genomes(n_candidates, seed, allow_short)
        prereg = False
    k_tests = len(cands)
    if isinstance(n_perm, str) and n_perm == "auto":
        from gold.permutation_budget import perms_for_bh
        n_perm = perms_for_bh(k_tests, alpha)
    n_perm = int(n_perm)

    # 记账：用了这段 OOS 做选参，就必须留下痕迹（失败要出声，不能静默）
    if fold is not None and run_id:
        try:
            from gold import oos as _oos

            _oos.record_use(run_id=str(run_id), symbol=str(symbol),
                            timeframe=str(tf),
                            scheme="verifier",
                            fold=int(getattr(fold, "fold", 1)),
                            oos_start=fold.oos_start, oos_end=fold.oos_end,
                            purpose="selection", n_candidates=int(k_tests),
                            seed=int(seed))
        except (OSError, ValueError) as _exc:  # noqa: BLE001
            print(f"[ledger] 记账失败：{type(_exc).__name__}: {_exc}")

    oos_fwd = oos_df["close"].astype(float).pct_change().shift(-1)

    rows = []
    ret_map = {}
    mask_cache: dict = {}
    for g in cands:
        ev = evaluate_genome(df, feat, is_df, oos_df, fee, g, min_trades,
                             mask_cache, tf, n_perm=n_perm,
                             block=int(max(2, len(oos_df) // 60)), seed=seed)
        if ev is None:
            continue
        label = ev["label"]
        rows.append({
            "EA": label,
            "进场": g.entry,
            "参数": ",".join(f"{k}={v}" for k, v in sorted(g.params.items())) or "默认",
            "过滤": g.filter,
            "方向": "双向" if g.allow_short else "只多",
            "IS收益": ev["IS收益"],
            "OOS收益": ev["OOS收益"],
            "OOS夏普": ev["OOS夏普"],
            "OOS回撤": ev["OOS回撤"],
            "OOS方向": ev["OOS方向"],
            "笔数": ev["笔数"],
            "回合": ev["回合"],
            "在场": ev["在场"],
            "OOS根数": ev["OOS根数"],
            "盈利因子": ev["盈利因子"],
            "每笔期望": ev["每笔期望"],
            "块置换p": ev["块置换p"],
            "随机检验": ev["随机检验"],
            "基准收益": ev["基准收益"],
            "跑赢基准": ev["跑赢基准"],
            "_genome": g.as_dict(),
        })
        ret_map[label] = (ev["_pos_oos"] * ev["_fwd_oos"]).fillna(0.0)

    if not rows:
        return ForgeResult(note=f"采样 {len(genomes)} 个候选，没有有效结果")

    tbl = pd.DataFrame(rows)
    n_valid = len(tbl)

    # 按 OOS 夏普去重（保留相关性低的多样组合）。
    # 关键：先把列按夏普从强到弱排，否则每个相关簇会保留"先出现"的弱版本。
    if len(ret_map) > 1:
        order = _dedup_order(tbl, ret_map)
        ret_df = pd.DataFrame(ret_map)[order] if order else pd.DataFrame(ret_map)
        keep_labels = _corr_dedup(ret_df, dedup_thresh)
        tbl = tbl[tbl["EA"].isin(keep_labels)].copy()
        ret_df = ret_df[keep_labels]
    else:
        ret_df = pd.DataFrame(ret_map)
    n_dedup = len(tbl)

    # 多重检验：用**真实块置换**得到的 p 做 BH 校正（以前按收益反推 p，属于伪造）
    if "块置换p" in tbl.columns and tbl["块置换p"].notna().any():
        tbl["q"] = bh_qvalues([float(v) if v == v else 1.0
                               for v in tbl["块置换p"]])
    else:
        tbl["q"] = float("nan")

    # 筛选门槛：OOS 夏普 > 0.3、回撤可控、方向 ≥ 0.5、块置换 p<0.05
    surv = tbl[(tbl["OOS夏普"] > 0.3) & (tbl["OOS方向"] >= 0.5)
               & (tbl["OOS回撤"] > -0.35)].copy()
    if "块置换p" in surv.columns and surv["块置换p"].notna().any():
        surv = surv[surv["块置换p"] < 0.05]
    surv = surv.sort_values("OOS夏普", ascending=False).reset_index(drop=True)
    n_trials = k_tests
    best_sr = float(tbl["OOS夏普"].max()) if len(tbl) else float("nan")
    gp = deflated_p(best_sr, n_trials) if best_sr == best_sr else 1.0

    out = tbl.sort_values("OOS夏普", ascending=False).reset_index(drop=True)
    return ForgeResult(
        table=out, survivors=surv,
        n_sampled=k_tests, n_valid=n_valid, n_dedup=n_dedup,
        n_pass=len(surv), space_size=len(genome_space(
            (False, True) if allow_short else (False,))),
        emax=emax_sharpe(n_trials), global_p=gp,
        is_range=(str(is_df.index[0].date()), str(is_df.index[-1].date())),
        oos_range=(str(oos_df.index[0].date()), str(oos_df.index[-1].date())),
        note=(f"{'预注册' if prereg else '采样'} {k_tests} / 空间 {len(genome_space((False, True) if allow_short else (False,)))} · "
              f"有效 {n_valid} · 去重后 {n_dedup} · 过门槛 {len(surv)} · "
              f"块置换 {int(n_perm)} 次/候选 · "
              f"E[max SR]={emax_sharpe(n_trials):.2f} · 最优夏普 {best_sr:.2f} · "
              f"全局 Deflated p={gp:.3f}"),
    )


def register_survivors(con, res: ForgeResult, top: int = 5, tf: str = "D1",
                       name_prefix: str = "[自研]", symbol: str = "GOLD") -> int:
    """把筛选出的优秀 EA 登记进模拟盘，供系统自行前向跑。"""
    from gold.paper import add_strategy

    added = 0
    for _, r in res.survivors.head(int(top)).iterrows():
        g = r["_genome"]
        if isinstance(g, str):
            g = json.loads(g)
        _params = dict(g.get("params") or {})
        _params["_symbol"] = str(symbol)
        add_strategy(con, f"{name_prefix}{symbol} {r['EA']}"[:60], g["entry"],
                     _params, tf=tf)
        added += 1
    return added

# ---------------------------------------------------------------- 结果落盘
RUNS_DIR = pathlib.Path("data/forge_runs")


def _rows_of(res) -> pd.DataFrame:
    """取"候选明细"：ForgeResult 是 `.table`，EvolveResult 是 `.best`。"""
    t = getattr(res, "table", None)
    if isinstance(t, pd.DataFrame) and len(t):
        return t
    b = getattr(res, "best", None)
    return b if isinstance(b, pd.DataFrame) else pd.DataFrame()


def _survivors_of(res) -> pd.DataFrame:
    s = getattr(res, "survivors", None)
    if isinstance(s, pd.DataFrame) and len(s):
        return s
    return _rows_of(res)


def _num_attr(res, names, default: float = 0.0) -> float:
    """依次尝试若干字段名，取到就用，都取不到给默认值（两种结果类型字段名不同）。"""
    for n in names:
        v = getattr(res, n, None)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:
            return f
    return float(default)


def save_run(res, meta: dict | None = None,
             out_dir: str | pathlib.Path = RUNS_DIR) -> pathlib.Path:
    """把一次设计结果存成 JSON，供界面读取（CLI 跑的结果也能在界面看到）。

    同时支持两种结果对象：
    - `ForgeResult`（随机采样 / 全枚举）：有 `n_sampled/n_valid/n_dedup/n_pass/space_size`
    - `EvolveResult`（遗传进化）：只有 `n_eval/generations`，明细在 `.best` 里

    之前的实现直接取 `res.n_sampled` 与 `res.table`，导致「遗传进化」跑完必崩
    （AttributeError: 'EvolveResult' object has no attribute 'n_sampled'）。
    """
    import datetime as _dt

    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = _rows_of(res)
    surv = _survivors_of(res)
    n_rows = int(len(rows))
    payload = {
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "meta": meta or {},
        "note": getattr(res, "note", ""),
        # 计数：进化结果用 n_eval 顶替采样数，并如实标注来源
        "n_sampled": int(_num_attr(res, ("n_sampled", "n_eval"), n_rows)),
        "n_valid": int(_num_attr(res, ("n_valid",), n_rows)),
        "n_dedup": int(_num_attr(res, ("n_dedup",), n_rows)),
        "n_pass": int(_num_attr(res, ("n_pass",), len(surv))),
        "space_size": int(_num_attr(res, ("space_size", "n_eval"), n_rows)),
        "emax": _num_attr(res, ("emax",), 0.0),
        "global_p": _num_attr(res, ("global_p",), 1.0),
        "is_range": list(getattr(res, "is_range", ()) or ()),
        "oos_range": list(getattr(res, "oos_range", ()) or ()),
        "rows": rows.head(200).to_dict("records") if n_rows else [],
        "survivors": (surv.head(50).to_dict("records")
                      if len(surv) else []),
    }
    path = d / f"run_{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return path


def list_runs(out_dir: str | pathlib.Path = RUNS_DIR) -> list[pathlib.Path]:
    d = pathlib.Path(out_dir)
    if not d.exists():
        return []
    return sorted(d.glob("run_*.json"), reverse=True)


def latest_run(out_dir: str | pathlib.Path = RUNS_DIR) -> dict | None:
    runs = list_runs(out_dir)
    if not runs:
        return None
    try:
        return json.loads(runs[0].read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def load_run(path: str | pathlib.Path) -> dict | None:
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
