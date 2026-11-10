"""累计设计榜：把多次设计结果汇总成一个总榜。

解决什么问题
------------
每次「自主设计」只产出一份排行榜，跑 10 次就有 10 份，无法横向比较。
本模块把所有历史运行（data/forge_runs/*.json）合并去重：

- 同一个 EA（基因组完全相同）在多次运行里出现 → 合并成一行，记「出现次数」
- 出现次数多 = 在不同的随机采样中都被挖到 → **稳健性信号**
- 排序默认按「最优夏普」，也可按出现次数看"常胜将军"

注意：多次运行用的种子不同，采样到的组合不同；出现次数高说明这个组合
在空间里"容易被随机采到"，不等于它更强，但说明它不是采样偶然。
"""

from __future__ import annotations

import json
import pathlib
import re

import pandas as pd

from gold.forge import RUNS_DIR, load_run
from gold.freqs import per_year, years_between

# 旧口径（已移除的干支维度）：标签形如 `…/低波动/亥/只多`，不用干支时是 `…/—/只多`
_ZHI_IN_LABEL = re.compile(r"/[子丑寅卯辰巳午未申酉戌亥]/")
_ZHI_SEG = re.compile(r"/(?:[子丑寅卯辰巳午未申酉戌亥]|—)(?=/)")


def _row_has_zhi(genome, label) -> bool:
    """这条记录是不是"带干支"的旧口径（干支已从基因组里移除）。"""
    g = genome
    if isinstance(g, str):
        try:
            g = json.loads(g)
        except Exception:  # noqa: BLE001
            g = None
    if isinstance(g, dict) and g.get("zhi") is not None:
        return True
    # 老记录的 EA 标签里还留着地支字样兜底判断
    return bool(_ZHI_IN_LABEL.search(str(label or "")))


def display_label(label) -> str:
    """把旧标签里的干支段去掉，只用于**显示**（不改变去重用的 EA 键）。

    旧格式：`SMA金叉[fast=20]/低波动/亥/只多/TP5%`、不用干支时是 `…/—/只多`；
    新格式：`SMA金叉[fast=20]/低波动/只多/TP5%`。
    """
    s = str(label or "")
    return _ZHI_SEG.sub("", s)


def load_all(out_dir: str | pathlib.Path = RUNS_DIR) -> pd.DataFrame:
    """把所有运行的候选行拼成一张长表。"""
    d = pathlib.Path(out_dir)
    if not d.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(d.glob("run_*.json")):
        run = load_run(p)
        if not run:
            continue
        rows = run.get("rows") or []
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["_run"] = run.get("ts", "")
        df["_tf"] = (run.get("meta") or {}).get("tf", "")
        df["_n"] = (run.get("meta") or {}).get("n", 0)
        # 每次运行的样本外区间 → 年数（用于把笔数换算成"每年回合"）
        df["_oos_years"] = years_between(run.get("oos_range") or ())
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def cumulative(out_dir: str | pathlib.Path = RUNS_DIR,
               min_runs: int = 1, passed_fn=None) -> pd.DataFrame:
    """跨运行汇总（按 EA + 周期去重）。

    禁止跨运行拼接（2026-10-30 修正）
    --------------------------------
    以前的实现把「A 次运行的最大夏普 + B 次的最大盈利因子 + C 次的最大回合数」
    拼成一行 —— 那不是任何一个真实 EA 的成绩。现在的规则：

    - `最新*` 开头的列 = **最近一次运行**里这一条 EA 的完整成绩（同区间、同参数）；
    - 跨运行只保留可加的统计：`出现次数`、`通过次数`、以及 `中位` / `最差`。
    - 需要判断"能不能用"时，请用 `最新*` 那组，它才是自洽的一份成绩。

    `passed_fn` 可选：给定某次运行的该 EA 行（DataFrame 或 dict），返回是否通过，
    用于统计 `通过次数`。不给就留空（NaN），不猜。
    """
    df = load_all(out_dir)
    if len(df) == 0:
        return pd.DataFrame(columns=["EA", "周期", "出现次数", "运行次数",
                                     "最近出现", "通过次数",
                                     "夏普中位", "夏普最差", "收益中位",
                                     "回撤最差", "盈利因子中位", "每年回合中位",
                                     "最新夏普", "最新收益", "最新回撤",
                                     "最新盈利因子", "最新每笔期望",
                                     "最新笔数", "最新回合", "最新每年回合",
                                     "最新在场", "最新块置换p", "最新q",
                                     "含干支", "EA显示", "_genome"])

    df = df.copy()
    df["EA"] = df["EA"].astype(str)
    df["_tf"] = df["_tf"].fillna("").astype(str)
    for c in ("OOS夏普", "OOS收益", "OOS回撤", "OOS方向", "基准收益",
              "笔数", "回合", "在场", "盈利因子", "每笔期望",
              "块置换p", "q", "_oos_years"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = float("nan")

    # 每年回合：每次运行单独算（用该次运行自己的 OOS 年数），不能事后拼
    df["_每年回合"] = [per_year(c, y) for c, y in zip(df["回合"], df["_oos_years"])]

    genomes = df["_genome"] if "_genome" in df.columns else [None] * len(df)
    df["_has_zhi"] = [_row_has_zhi(g, ea)
                      for g, ea in zip(genomes, df["EA"])]

    keys = ["EA", "_tf"]
    g = df.groupby(keys, sort=False)
    agg = g.agg(
        出现次数=("EA", "size"),
        运行次数=("_run", "nunique"),
        最近出现=("_run", "max"),
        夏普中位=("OOS夏普", "median"),
        夏普最差=("OOS夏普", "min"),
        收益中位=("OOS收益", "median"),
        回撤最差=("OOS回撤", "min"),
        盈利因子中位=("盈利因子", "median"),
        每年回合中位=("_每年回合", "median"),
        含干支=("_has_zhi", "max"),
    ).reset_index()

    # 最近一次运行的完整记录（同一次运行、同一区间、同一参数）
    df["_run_key"] = df["最近出现"] if False else df["_run"]
    latest_idx = g["_run"].idxmax()
    _want = ["_run", "OOS夏普", "OOS收益", "OOS回撤", "OOS方向",
             "基准收益", "盈利因子", "每笔期望", "笔数", "回合",
             "_每年回合", "在场", "块置换p", "q", "_genome"]
    _cols = [c for c in _want if c in df.columns]
    latest = df.loc[latest_idx, keys + _cols].copy()
    latest = latest.rename(columns={
        "_run": "最近出现",
        "OOS夏普": "最新夏普", "OOS收益": "最新收益", "OOS回撤": "最新回撤",
        "OOS方向": "最新方向", "基准收益": "最新基准",
        "盈利因子": "最新盈利因子", "每笔期望": "最新每笔期望",
        "笔数": "最新笔数", "回合": "最新回合", "每年回合": "最新每年回合",
        "_每年回合": "最新每年回合", "在场": "最新在场",
        "块置换p": "最新块置换p", "q": "最新q",
    })

    out = agg.merge(latest, on=keys, how="left", suffixes=("", "_dup"))
    if "最近出现_dup" in out.columns:
        out = out.drop(columns=["最近出现_dup"])
    out = out.rename(columns={"_tf": "周期"})
    if "_genome" not in out.columns:
        out["_genome"] = None

    # 通过次数：交给调用方判定（口径只能有一处定义）
    out["通过次数"] = float("nan")
    if passed_fn is not None:
        counts = {}
        for (ea, tf), sub in df.groupby(keys, sort=False):
            n = 0
            for _, row in sub.iterrows():
                try:
                    if passed_fn(row.to_dict()):
                        n += 1
                except Exception:  # noqa: BLE001
                    continue
            counts[(ea, tf)] = n
        out["通过次数"] = [counts.get((e, t), 0) for e, t in
                          zip(out["EA"], out["周期"])]

    out["EA显示"] = [display_label(e) for e in out["EA"]]
    out["回合来源"] = "实测"
    out = out[out["出现次数"] >= int(min_runs)]
    return out.sort_values(["夏普中位", "出现次数"],
                           ascending=False).reset_index(drop=True)


def register_top(con, board: pd.DataFrame, top: int = 5,
                 tf: str | None = None, name_prefix: str = "[累计]",
                 symbol: str = "GOLD"):
    """把累计榜前 N 名登记进模拟盘。

    基因组里的 filter / sl / tp 会转成模拟盘引擎认识的
    `_filter` / `_sl` / `_tp` 元参数，保证前向跑的是同一个 EA。

    旧基因组里若有 `zhi`（已移除的干支维度）一律忽略：本系统只做外汇/黄金，
    干支维度已删除，带干支的老记录只能在历史榜里留档，不能再登记跑前向。
    """
    import json

    from gold.paper import add_strategy, list_strategies

    have = set(list_strategies(con)["name"])
    added = 0
    for _, r in board.head(int(top)).iterrows():
        g = r.get("_genome")
        if isinstance(g, str):
            try:
                g = json.loads(g)
            except Exception:  # noqa: BLE001
                g = None
        if not isinstance(g, dict):
            continue
        params = dict(g.get("params") or {})
        flt = g.get("filter") or "无过滤"
        if flt and flt != "无过滤":
            params["_filter"] = flt
        if g.get("sl") is not None:
            params["_sl"] = float(g["sl"])
        if g.get("tp") is not None:
            params["_tp"] = float(g["tp"])

        params["_symbol"] = str(symbol)
        name = f"{name_prefix}{symbol} {r['EA']}"[:60]
        if name in have:
            continue
        add_strategy(con, name, str(g.get("entry")), params,
                     tf=(tf or r.get("周期") or "H1"))
        added += 1
    return added
