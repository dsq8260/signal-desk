"""按 magic 归因 EA 的真实成绩（读 mt5_deals.csv）。

为什么这是最准的一路
------------------
- **EA 自报净值**：EA 重启会丢历史，且只覆盖自己那一段；
- **模拟盘回放**：是历史模拟，不是真实成交；
- **成交归因（本模块）**：直接读券商成交记录，按 magic 分组——
  盈亏、手续费、库存费都是真实扣过的，EA 重启也不影响。

口径
----
净收益 = profit + swap + commission（三者都算进账户）
胜率  = 净收益为正的成交笔数 / 总笔数
盈亏比 = 总盈利 / |总亏损|
曲线  = 按时间累计净收益，用于算最大回撤
综合分 = MAR × (0.5+0.5×一致性) × (0.5+0.5×胜率)
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from gold.account import deals as load_deals

MIN_DEALS = 5


def _net(df: pd.DataFrame) -> pd.Series:
    """每笔的净贡献：profit + swap + commission。"""
    out = pd.Series(0.0, index=df.index)
    for c in ("profit", "swap", "commission"):
        if c in df.columns:
            out = out + pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return out


def _streak(series: pd.Series) -> int:
    best = cur = 0
    for v in series:
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def magic_stats(d: pathlib.Path | str | None = None) -> pd.DataFrame:
    """按 magic 汇总成交。没有成交文件时返回空表。"""
    df = load_deals(d)
    cols = ["magic", "净收益", "笔数", "胜率", "盈亏比", "手续费", "库存费",
            "最大回撤", "MAR", "一致性", "连亏", "综合分", "评级",
            "首次成交", "末次成交"]
    if len(df) == 0 or "magic" not in df.columns:
        return pd.DataFrame(columns=cols)

    df = df.copy()
    # MQL5 DEAL_TYPE: 0=buy 1=sell 2=balance(入金/出金) 3=credit 4=charge
    # 5=correction 6=bonus 7=commission。只有 0/1 才是交易，
    # 入金绝不能算进盈亏（这是之前踩过的坑）。
    if "type" in df.columns:
        df = df[pd.to_numeric(df["type"], errors="coerce").isin([0, 1])]
    if len(df) == 0:
        return pd.DataFrame(columns=cols)
    df["净贡献"] = _net(df)
    df = df.sort_values("time") if "time" in df.columns else df

    rows = []
    for magic, g in df.groupby("magic", sort=False):
        contrib = g["净贡献"].astype(float)
        total = float(contrib.sum())
        n = int(len(contrib))
        wins = contrib[contrib > 0]
        losses = contrib[contrib < 0]
        win_rate = float(len(wins) / n) if n else float("nan")
        pf = (float(wins.sum()) / abs(float(losses.sum()))
              if len(losses) and abs(float(losses.sum())) > 1e-12 else float("nan"))

        curve = contrib.cumsum()
        peak = curve.cummax()
        dd = (curve - peak).min() if len(curve) else 0.0
        base = abs(float(dd)) if dd != 0 else 0.0
        # 无回撤时用 1% 做分母；上限压到 5（成交归因只有逐笔盈亏，
        # 没法像净值曲线那样精确衡量回撤，上限太高会让"全胜"虚高到 A）
        mar = (total / base) if base > 1e-9 else (total / 0.01 if total > 0 else float("nan"))
        if mar == mar:
            mar = float(min(mar, 5.0))

        consistency = float("nan")
        if n >= 4:
            segs = np.array_split(np.arange(n), 4)
            pos = sum(1 for s in segs if len(s) and contrib.iloc[s].sum() > 0)
            consistency = pos / 4.0

        score = float("nan")
        if mar == mar:
            c = consistency if consistency == consistency else 0.5
            w = win_rate if win_rate == win_rate else 0.5
            # 样本充分度：50 笔才给满分。10 笔的"完美"不该和 200 笔的并列。
            ready = min(1.0, n / 50.0)
            score = mar * (0.5 + 0.5 * c) * (0.5 + 0.5 * w) * ready

        if n < MIN_DEALS or score != score:
            rating = "样本不足"
        elif score >= 1.5:
            rating = "A"
        elif score >= 0.8:
            rating = "B"
        elif score >= 0.3:
            rating = "C"
        else:
            rating = "D"

        rows.append({
            "magic": int(magic) if magic == magic else 0,
            "净收益": total,
            "笔数": n,
            "胜率": win_rate,
            "盈亏比": pf,
            "手续费": float(g["commission"].sum()) if "commission" in g else 0.0,
            "库存费": float(g["swap"].sum()) if "swap" in g else 0.0,
            "最大回撤": float(dd),
            "MAR": mar,
            "一致性": consistency,
            "连亏": _streak(contrib),
            "综合分": score,
            "评级": rating,
            "首次成交": str(g["time"].iloc[0]) if "time" in g else "",
            "末次成交": str(g["time"].iloc[-1]) if "time" in g else "",
        })

    out = pd.DataFrame(rows)
    return out.sort_values("综合分", ascending=False, na_position="last").reset_index(drop=True)


def magic_names(path: str | pathlib.Path = "data/magic_names.json") -> dict:
    """magic -> 自定义名字（可选，方便识别是哪个 EA）。"""
    import json

    p = pathlib.Path(path)
    if not p.exists():
        return {}
    try:
        return {int(k): str(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    except Exception:  # noqa: BLE001
        return {}


def save_magic_names(mapping: dict,
                     path: str | pathlib.Path = "data/magic_names.json") -> None:
    import json

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({str(k): v for k, v in mapping.items()},
                            ensure_ascii=False, indent=2), encoding="utf-8")
