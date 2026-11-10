"""组合挖掘：进场条件 × 过滤条件 → 排行榜。

两层结构（每层都是独立开关，组合起来才是"策略"）：
1. **进场**：SMA 金叉 / 唐奇安突破 / RSI<30 / MACD 金叉 / LLT 上穿
2. **过滤**：波动率分位低 / 趋势强度高 / 无过滤

候选数 = 5 × 3 = 15，远小于盲目网格（避免 E[max SR] 被抬到天上）。
选参只用 IS 段，OOS 只做验证；最后用 BH 校正 + 全局 Deflated 判真假。

历史说明：早期版本还有第三层「干支（日柱地支 12 选 1）」，那是给股票市场
日频择日用的。本系统只做外汇/现货黄金（H1 为主），干支会把在场占比压到
5% 左右、每年只剩十几笔，已经移除。旧设计记录里的干支只作历史留档。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gold.backtest import block_permutation, spread_fee_bps
from gold.factors import build_features, regime_mask
from gold.freqs import freq_key
from gold.mining import bh_qvalues, deflated_p, emax_sharpe, split_is_oos
from gold.strategies import (
    donchian_signal,
    llt_signal,
    ma_cross_signal,
    macd_momentum_signal,
    rsi_reversion_signal,
)
from validation import metrics as M

def ma_dev_reversion_signal(df: pd.DataFrame, n: int = 10,
                            entry_thr: float = 100.0,
                            exit_thr: float = 100.0,
                            ma_n: int = 10) -> pd.Series:
    """均线偏离度均值回归：偏离超过阈值后反向做，回到均值了结。

    ``n`` 参数保留（与其它策略的网格一致），实际用 ``ma_n`` 做均线周期。
    返回 0/1 序列；空头由 build_position 的 allow_short 翻转。
    """
    import numpy as np

    close = df["close"].astype(float)
    ma = close.rolling(int(ma_n)).mean()
    sd = close.rolling(int(ma_n)).std(ddof=0)
    # 偏离度按**标准差**归一化：dev=100 表示偏离 1 个标准差（≈ 简易量化 的 ±100 口径）
    dev = (close - ma) / sd.replace(0.0, float("nan")) * 100.0

    out = np.zeros(len(close))
    pos = 0.0
    dev_v = dev.to_numpy()
    for i in range(1, len(close)):
        d, prev = dev_v[i], dev_v[i - 1]
        if np.isnan(d) or np.isnan(prev):
            out[i] = pos
            continue
        if pos == 0.0 and prev < -entry_thr <= d:      # 从超卖区上穿 → 做多
            pos = 1.0
        elif pos == 1.0 and d >= exit_thr:             # 回到均值上方 → 了结
            pos = 0.0
        out[i] = pos
    return pd.Series(out, index=df.index)


def atr_channel_signal(df: pd.DataFrame, n: int = 20, m: int = 14,
                       k: float = 2.0, confirm: int = 1) -> pd.Series:
    """ATR 自适应通道突破：波动大→通道宽（滤噪），波动小→通道窄（抓启动）。"""
    import numpy as np

    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df else close
    low = df["low"].astype(float) if "low" in df else close

    mid = close.rolling(int(n)).mean()
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(int(m)).mean()
    upper = mid + float(k) * atr
    lower = mid - float(k) * atr

    out = np.zeros(len(close))
    pos = 0.0
    c = close.to_numpy()
    up = upper.to_numpy()
    dn = lower.to_numpy()
    md = mid.to_numpy()
    conf = max(1, int(confirm))
    for i in range(conf, len(close)):
        if np.isnan(up[i]) or np.isnan(md[i]):
            out[i] = pos
            continue
        closed_above = all(c[i - j] > up[i - j] for j in range(conf))
        closed_below = all(c[i - j] < dn[i - j] for j in range(conf))
        if pos == 0.0 and closed_above:
            pos = 1.0
        elif pos == 1.0 and c[i] < md[i]:       # 回到中轨下方 → 趋势失效
            pos = 0.0
        out[i] = pos
    return pd.Series(out, index=df.index)


ENTRIES = {
    "SMA金叉": (ma_cross_signal, {"fast": 20, "slow": 60}),
    "唐奇安突破": (donchian_signal, {"entry": 20, "exit_": 10}),
    "RSI反转": (rsi_reversion_signal, {"n": 14, "buy": 30.0, "sell": 55.0}),
    "MACD金叉": (macd_momentum_signal, {"fast": 12, "slow": 26, "signal": 9}),
    "LLT上穿": (llt_signal, {"n": 60}),
    # 2026-11-02 新增（借鉴 @简易量化）：均线偏离度均值回归
    "均线偏离回归": (ma_dev_reversion_signal,
                    {"ma_n": 10, "entry_thr": 100.0, "exit_thr": 100.0}),
    # 2026-11-02 新增：ATR 自适应通道突破（顺势）
    "ATR通道突破": (atr_channel_signal,
                   {"n": 20, "m": 14, "k": 2.0, "confirm": 1}),
}

FILTERS = {
    "无过滤": None,
    "低波动": ("vol_pct", 0.7, "below"),
    "强趋势": ("trend_pct", 0.5, "above"),
}

# 额外过滤（2026-11-02）：默认**不参与枚举**，避免搜索空间膨胀与多重检验惩罚；
# 用户显式勾选时才生效（build_position 里特判这两个名字）。
EXTRA_FILTERS = {
    "大周期确认": ("__htf__", 20.0, "above"),      # 默认 H4 20 根
    "大周期确认10": ("__htf__", 10.0, "above"),
    "大周期确认40": ("__htf__", 40.0, "above"),
    "欧美时段": ("__session__", 0.0, "above"),
}

ALL_FILTERS = {**FILTERS, **EXTRA_FILTERS}


def build_candidates() -> list[dict]:
    """生成全部候选（进场 × 过滤）。"""
    out = []
    for (ename, _), (fname, _) in itertools.product(ENTRIES.items(),
                                                   FILTERS.items()):
        out.append({"entry": ename, "filter": fname})
    return out


def _position(df: pd.DataFrame, feat: pd.DataFrame, spec: dict) -> pd.Series:
    fn, default_params = ENTRIES[spec["entry"]]
    pos = pd.Series(fn(df, **default_params), index=df.index).fillna(0.0)

    flt = FILTERS[spec["filter"]]
    if flt is not None:
        name, q, mode = flt
        pos = pos * regime_mask(feat, name, q, mode).astype(float)

    return pos.clip(0.0, 1.0)


def _metrics(seg: pd.DataFrame, pos: pd.Series, spread_usd: float,
             tf: str = "D1") -> dict:
    close = seg["close"].astype(float)
    p = pos.reindex(seg.index).fillna(0.0)
    fee = spread_fee_bps(float(close.mean()), spread_usd)
    fwd = close.pct_change().shift(-1)
    # 必须按周期年化：H1 用 D1 会把夏普低估约 4.9 倍
    st = M.summarize_strategy(fwd, p, fee_bps=fee, freq=freq_key(tf),
                              allow_short=False)
    return {
        "ret": float(st["net"].get("total_return", float("nan"))),
        "sharpe": float(st["net"].get("sharpe", float("nan"))),
        "mdd": float(st["net"].get("max_drawdown", float("nan"))),
        "hit": float(st["hit"].get("hit_rate", float("nan"))),
        "trades": int(st["trades"]),
        "_pos": p, "_fwd": fwd, "_fee": fee,
    }


@dataclass
class Leaderboard:
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    best: dict = field(default_factory=dict)
    n_candidates: int = 0
    emax: float = 0.0
    global_p: float = 1.0
    note: str = ""
    is_range: tuple = ()
    oos_range: tuple = ()


def rank(df: pd.DataFrame, spread_usd: float = 0.33, is_ratio: float = 0.7,
         n_perm: int = 100, seed: int = 7,
         max_candidates: int | None = None, tf: str = "D1",
         fold=None) -> Leaderboard:
    """跑全部候选 → 排行榜（IS 选参、OOS 验证、BH + 全局 Deflated 校正）。"""
    if df is None or len(df) < 300:
        return Leaderboard(note="K 线不足 300 根，无法组合挖掘")

    feat = build_features(df)
    if fold is not None:
        from gold import oos as _oos

        is_df, oos_df = _oos.slice_fold(df, fold)
    else:
        is_df, oos_df = split_is_oos(df, is_ratio)

    cands = build_candidates()
    if max_candidates:
        cands = cands[:int(max_candidates)]

    rows = []
    for spec in cands:
        pos = _position(df, feat, spec)
        a = _metrics(is_df, pos, spread_usd, tf)
        if not np.isfinite(a["ret"]) or a["trades"] < 3:
            continue
        b = _metrics(oos_df, pos, spread_usd, tf)
        perm = block_permutation(b["_fwd"], b["_pos"], n_perm=n_perm, block=20,
                                 fee_bps=b["_fee"], seed=seed)
        rows.append({
            "进场": spec["entry"], "过滤": spec["filter"],
            "IS收益": a["ret"], "IS夏普": a["sharpe"],
            "OOS收益": b["ret"], "OOS夏普": b["sharpe"], "OOS回撤": b["mdd"],
            "OOS方向": b["hit"], "笔数": b["trades"], "p": perm["p_value"],
            "_spec": spec,
        })

    if not rows:
        return Leaderboard(note="没有候选能产生有效交易")

    tbl = pd.DataFrame(rows)
    tbl["q"] = bh_qvalues(tbl["p"].tolist())
    tbl["IS排名"] = tbl["IS收益"].rank(ascending=False, method="min").astype(int)

    n = len(tbl)
    best_idx = tbl["IS收益"].idxmax()          # 只看 IS 选
    best = tbl.loc[best_idx]
    sr_std = float(tbl["OOS夏普"].std(ddof=1)) if n > 2 else 1.0
    gp = deflated_p(float(best["OOS夏普"]), n, sr_std=sr_std if sr_std > 0 else 1.0)

    ref = M.performance_stats(
        oos_df["close"].astype(float).pct_change().shift(-1).dropna(),
        freq_key(tf))

    return Leaderboard(
        table=tbl.sort_values("IS收益", ascending=False).reset_index(drop=True),
        best={"spec": best["_spec"], "IS收益": float(best["IS收益"]),
              "OOS收益": float(best["OOS收益"]), "q": float(best["q"]),
              "global_p": gp},
        n_candidates=n, emax=emax_sharpe(n), global_p=gp,
        is_range=(str(is_df.index[0].date()), str(is_df.index[-1].date())),
        oos_range=(str(oos_df.index[0].date()), str(oos_df.index[-1].date())),
        note=(f"候选 {n} 组 · E[max SR]={emax_sharpe(n):.2f} · "
              f"IS最优组 OOS 夏普 {float(best['OOS夏普']):.2f} · "
              f"全局 Deflated p={gp:.3f} · OOS 基准 {float(ref.get('total_return', float('nan'))):+.2%}"),
    )
