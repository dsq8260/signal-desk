"""读 EA 自报的成绩单，做多 EA 评比。

EA 每 InpReportSecs 秒往 MT5 公共文件夹追加一行：
    ea_<tag>_equity.csv:  time,equity,balance,position,lots,price

本模块把这些文件读成 DataFrame，算每个 EA 的实际表现并排名——
**完全不依赖 MT5 Python IPC**，所以在你现在的终端上也能用。

评分体系（借鉴 Darwinex DARWIN 的思路，但用自报净值近似）：

    收益 R      = 末净值/首净值 - 1
    最大回撤    = 净值/历史最高 - 1 的最小值（永远 ≤ 0）
    回撤持续    = 从峰值跌下去、直到重新创新高，最长用了多久（小时）
    MAR         = R / |最大回撤|          ← 风险调整（无回撤用 1% 下限，上限 20）
    夏普近似    = 每快照收益均值 / 标准差  ← 不年化，只用于横向比较
    一致性      = 把净值切成 4 段，正收益段数 / 4
    日胜率      = 正收益天数 / 总天数
    综合分      = MAR × (0.5+0.5×一致性) × (0.5+0.5×日胜率)

为什么不是只看收益：
一次大赚能盖住长期阴跌，所以要用 MAR 惩罚回撤、用一致性惩罚"靠一波行情"。
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from gold.bridge import common_dir

PREFIX = "ea_"
SUFFIX = "_equity.csv"
MIN_SNAPSHOTS = 10          # 少于此样本不评分（避免拿几行数据排第一）


def list_ea_files(d: pathlib.Path | str | None = None) -> list[pathlib.Path]:
    d = pathlib.Path(d) if d else common_dir()
    if not d.exists():
        return []
    return sorted(d.glob(f"{PREFIX}*{SUFFIX}"))


def ea_name(path: pathlib.Path) -> str:
    n = path.name
    if n.startswith(PREFIX) and n.endswith(SUFFIX):
        return n[len(PREFIX):-len(SUFFIX)]
    return path.stem


def load_ea(path: pathlib.Path | str) -> pd.DataFrame:
    """读一个 EA 的成绩单；坏行跳过，返回按时间排序的 DataFrame。"""
    p = pathlib.Path(path)
    empty = pd.DataFrame(columns=["time", "equity", "balance", "position", "lots", "price"])
    if not p.exists():
        return empty
    try:
        df = pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return empty
    if len(df) == 0 or "equity" not in df.columns:
        return empty

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).set_index("time").sort_index()
    for c in ("equity", "balance", "position", "lots", "price"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="last")]
    return df


def _drawdown_duration_hours(eq: pd.Series) -> float:
    """从峰值跌下去到重新创新高，最长用了多少小时（没有恢复则算到末尾）。"""
    if len(eq) < 3 or not isinstance(eq.index, pd.DatetimeIndex):
        return 0.0
    peak = eq.cummax()
    under = eq < peak
    if not under.any():
        return 0.0

    longest = 0.0
    start = None
    for ts, flag in under.items():
        if flag and start is None:
            start = ts
        elif not flag and start is not None:
            longest = max(longest, (ts - start).total_seconds() / 3600.0)
            start = None
    if start is not None:
        longest = max(longest, (eq.index[-1] - start).total_seconds() / 3600.0)
    return float(longest)


def ea_score(df: pd.DataFrame, initial_balance: float | None = None,
             segments: int = 4) -> dict:
    """算一个 EA 的全部评比指标（含综合分与评级）。"""
    nan = float("nan")
    out = {
        "样本数": 0, "起始": None, "结束": None, "运行小时": 0.0,
        "净值": nan, "收益": nan, "最大回撤": nan, "回撤持续": 0.0,
        "MAR": nan, "夏普近似": nan, "一致性": nan, "日胜率": nan,
        "最差单日": nan, "连亏": 0, "换手": 0, "当前仓位": 0,
        "综合分": nan, "评级": "样本不足",
    }
    if df is None or len(df) == 0 or "equity" not in df.columns:
        return out

    eq = df["equity"].dropna()
    if len(eq) < 2:
        return out

    base = float(initial_balance) if initial_balance else float(eq.iloc[0])
    if base <= 0:
        base = float(eq.iloc[0]) or 1.0

    total = float(eq.iloc[-1] / base - 1.0)
    peak = eq.cummax()
    mdd = float((eq / peak - 1.0).min())

    rets = eq.pct_change().dropna()
    sd = float(rets.std(ddof=1)) if len(rets) > 2 else nan
    mean_r = float(rets.mean()) if len(rets) else nan
    sharpe_like = (mean_r / sd) if (sd and sd > 0 and sd == sd) else nan
    # MAR = 收益 / |最大回撤|。若全程没有回撤（一路新高），
    # 分母用 0.1% 做下限——否则会变成 NaN，把"最稳的 EA"挤出排行榜。
    if abs(mdd) > 1e-9:
        mar = total / abs(mdd)
    elif total > 0:
        mar = total / 0.01            # 1% 下限，避免无回撤时 MAR 被放大到几十
    else:
        mar = nan
    if mar == mar:
        mar = float(min(mar, 20.0))   # 上限 20，防止一条直线把榜首焊死

    # 一致性：切成 N 段，每段末/段初 - 1 > 0 的比例
    consistency = nan
    if len(eq) >= segments * 3:
        idx = np.array_split(np.arange(len(eq)), segments)
        pos = 0
        for chunk in idx:
            if len(chunk) < 2:
                continue
            seg_ret = eq.iloc[chunk[-1]] / eq.iloc[chunk[0]] - 1.0
            if seg_ret > 0:
                pos += 1
        consistency = pos / float(segments)

    # 日胜率 / 最差单日 / 连亏
    day_win = nan
    worst_day = nan
    lose_streak = 0

    def _max_losing_streak(series: pd.Series) -> int:
        streak = best = 0
        for v in series:
            if v < 0:
                streak += 1
                best = max(best, streak)
            else:
                streak = 0
        return int(best)

    if isinstance(eq.index, pd.DatetimeIndex):
        daily = eq.resample("1D").last().dropna()
        dret = daily.pct_change().dropna()
        if len(dret):
            day_win = float((dret > 0).mean())
            worst_day = float(dret.min())
        # 样本跨不足两天时，日线只有 0-1 个点，连亏永远算成 0；
        # 这时退回用快照收益，至少能反映"连跌多少根"。
        if len(dret) >= 2:
            lose_streak = _max_losing_streak(dret)
        else:
            lose_streak = _max_losing_streak(rets)

    pos_series = (df["position"].fillna(0)
                  if "position" in df.columns else pd.Series(0, index=df.index))
    changes = int((pos_series.diff().fillna(0) != 0).sum())

    # 综合分：MAR 为主，一致性与日胜率做乘数（各占 0.5~1.0）
    score = nan
    if mar == mar:
        c = consistency if consistency == consistency else 0.5
        w = day_win if day_win == day_win else 0.5
        score = float(mar * (0.5 + 0.5 * c) * (0.5 + 0.5 * w))

    n = int(len(eq))
    if n < MIN_SNAPSHOTS:
        rating = "样本不足"
    elif score != score:
        rating = "样本不足"
    elif score >= 1.5:
        rating = "A"
    elif score >= 0.8:
        rating = "B"
    elif score >= 0.3:
        rating = "C"
    else:
        rating = "D"

    out.update({
        "样本数": n,
        "起始": str(df.index[0]) if isinstance(df.index, pd.DatetimeIndex) else None,
        "结束": str(df.index[-1]) if isinstance(df.index, pd.DatetimeIndex) else None,
        "运行小时": round((df.index[-1] - df.index[0]).total_seconds() / 3600.0, 1)
                    if isinstance(df.index, pd.DatetimeIndex) else 0.0,
        "净值": float(eq.iloc[-1]),
        "收益": total,
        "最大回撤": mdd,
        "回撤持续": round(_drawdown_duration_hours(eq), 1),
        "MAR": mar if mar == mar else nan,
        "夏普近似": sharpe_like,
        "一致性": consistency,
        "日胜率": day_win,
        "最差单日": worst_day,
        "连亏": lose_streak,
        "换手": changes,
        "当前仓位": int(pos_series.iloc[-1]) if len(pos_series) else 0,
        "综合分": score,
        "评级": rating,
    })
    return out


def ea_stats(df: pd.DataFrame, initial_balance: float | None = None) -> dict:
    """兼容旧接口：等价于 ea_score()。"""
    return ea_score(df, initial_balance)


def rank_all(d: pathlib.Path | str | None = None,
             initial_balance: float | None = None,
             sort_by: str = "综合分") -> pd.DataFrame:
    """把所有 EA 的成绩单读出来排名（默认按综合分）。"""
    rows = []
    for f in list_ea_files(d):
        df = load_ea(f)
        rows.append({"EA": ea_name(f), **ea_score(df, initial_balance)})
    if not rows:
        return pd.DataFrame(columns=["EA", "样本数", "净值", "收益", "最大回撤",
                                     "综合分", "评级"])
    out = pd.DataFrame(rows)
    key = sort_by if sort_by in out.columns else "收益"
    return out.sort_values(key, ascending=False, na_position="last").reset_index(drop=True)


def detail(name: str, d: pathlib.Path | str | None = None) -> pd.DataFrame:
    d = pathlib.Path(d) if d else common_dir()
    return load_ea(d / f"{PREFIX}{name}{SUFFIX}")
