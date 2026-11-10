"""评价指标（阶段 2）：RankIC、ICIR、分层单调性、扣费后收益。

口径说明：
- forward_returns 指"决策期之后一期"的收益，调用方负责保证它真的是未来一期；
- 所有指标忽略 NaN，样本不足时返回 NaN 而不是编一个数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12}


def _clean(a, b) -> tuple[np.ndarray, np.ndarray]:
    x = pd.Series(a, dtype=float)
    y = pd.Series(b, dtype=float)
    df = pd.concat([x, y], axis=1).dropna()
    return df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()


def rank_ic(scores, forward_returns) -> float:
    """Spearman 秩相关。样本 <3 或任一侧为常数时返回 NaN。"""
    x, y = _clean(scores, forward_returns)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def rank_ic_series(scores: pd.Series, forward_returns: pd.Series, window: int = 60) -> pd.Series:
    """滚动 IC，用于算 ICIR。"""
    s = pd.Series(scores, dtype=float)
    r = pd.Series(forward_returns, dtype=float)
    common = s.index.intersection(r.index)
    out = {}
    s, r = s.loc[common], r.loc[common]
    for i in range(window, len(s) + 1):
        out[s.index[i - 1]] = rank_ic(s.iloc[i - window:i], r.iloc[i - window:i])
    return pd.Series(out, dtype=float)


def icir(ic_values) -> float:
    """ICIR = mean(IC) / std(IC)。std 为 0 或样本 <3 时返回 NaN。"""
    v = pd.Series(ic_values, dtype=float).dropna()
    if len(v) < 3 or v.std(ddof=1) == 0:
        return float("nan")
    return float(v.mean() / v.std(ddof=1))


def layered_monotonicity(scores, forward_returns, n_layers: int = 5) -> dict:
    """按打分分层的平均收益，判断单调性。"""
    x, y = _clean(scores, forward_returns)
    if len(x) < n_layers * 2:
        return {"layer_means": [], "layer_counts": [], "rank_corr": float("nan"),
                "is_monotonic": False, "top_minus_bottom": float("nan")}

    order = np.argsort(x)
    layers = np.array_split(order, n_layers)
    means = [float(np.mean(y[idx])) for idx in layers]
    counts = [int(len(idx)) for idx in layers]

    idx_seq = np.arange(n_layers)
    rank_corr = float(stats.spearmanr(idx_seq, means).statistic) if len(set(means)) > 1 else float("nan")
    monotonic = bool(rank_corr == rank_corr and rank_corr >= 0.9)
    return {
        "layer_means": means,
        "layer_counts": counts,
        "rank_corr": rank_corr,
        "is_monotonic": monotonic,
        "top_minus_bottom": float(means[-1] - means[0]),
    }


def turnover(positions: pd.Series) -> pd.Series:
    """换手：仓位的绝对变化。"""
    p = pd.Series(positions, dtype=float).fillna(0.0)
    return p.diff().abs().fillna(p.abs())


def net_returns(
    forward_returns: pd.Series, positions: pd.Series, fee_bps: float = 5.0,
    allow_short: bool = False,
) -> pd.Series:
    """扣费后收益。

    positions[t] 是在 t 收盘建立、持有到 t+1 的仓位；
    forward_returns[t] 是 t -> t+1 的收益，两者在同一索引上对齐。
    手续费在建立仓位时（t）计提。

    allow_short=False（默认，对应 A 股现金账户）时，负仓位会被裁成空仓：
    看跌信号的含义是"不持有"，而不是"反手做空"。这与 PaperTradingEngine /
    ReplayEngine 的执行语义保持一致。
    """
    r = pd.Series(forward_returns, dtype=float)
    p = pd.Series(positions, dtype=float).reindex(r.index)
    if not allow_short:
        p = p.clip(lower=0.0)
    fee = fee_bps / 10000.0
    cost = turnover(p) * fee
    return (p * r - cost).fillna(0.0)


def position_composition(positions) -> dict:
    """仓位结构：多头 / 空仓 / 空头各占多少天。"""
    p = pd.Series(positions, dtype=float).fillna(0.0)
    n = int(len(p))
    long_n = int((p > 0).sum())
    short_n = int((p < 0).sum())
    flat_n = n - long_n - short_n
    return {
        "bars": n,
        "long": long_n, "flat": flat_n, "short": short_n,
        "long_ratio": (long_n / n) if n else float("nan"),
        "flat_ratio": (flat_n / n) if n else float("nan"),
        "short_ratio": (short_n / n) if n else float("nan"),
    }


def count_trades(positions) -> int:
    """实际调仓次数：仓位发生变化的交易日数（含首次建仓）。"""
    return int((turnover(positions) > 0).sum())


def round_trip_pnls(positions, forward_returns) -> np.ndarray:
    """把仓位序列配对成"一进一出"的每笔盈亏（与 MT5 报告的「交易」同口径）。

    `positions[t]` 是 t 时执行、持有到 t+1 的仓位；`forward_returns[t]` 是 t→t+1
    的收益。连续同向持仓合并成一笔，遇到空仓或反手就平掉。
    """
    p = pd.Series(positions, dtype=float).fillna(0.0).to_numpy()
    r = pd.Series(forward_returns, dtype=float).fillna(0.0).to_numpy()
    n = min(len(p), len(r))
    out = []
    i = 0
    while i < n:
        if p[i] == 0:
            i += 1
            continue
        j = i
        while j < n and p[j] == p[i]:
            j += 1
        # 复利计算这一段持仓的真实盈亏（比求和更贴近实际净值）
        seg = 1.0
        for k in range(i, j):
            seg *= (1.0 + p[i] * r[k])
        out.append(seg - 1.0)
        i = j
    return np.asarray(out, dtype=float)


def round_trip_pnls_from_returns(positions, bar_returns) -> np.ndarray:
    """按持仓段切分**逐根净收益**，返回每笔复利盈亏（用于扣费后的盈利因子）。"""
    p = pd.Series(positions, dtype=float).fillna(0.0).to_numpy()
    r = pd.Series(bar_returns, dtype=float).fillna(0.0).to_numpy()
    n = min(len(p), len(r))
    out = []
    i = 0
    while i < n:
        if p[i] == 0:
            i += 1
            continue
        j = i
        while j < n and p[j] == p[i]:
            j += 1
        seg = 1.0
        for k in range(i, j):
            seg *= (1.0 + r[k])
        out.append(seg - 1.0)
        i = j
    return np.asarray(out, dtype=float)


def _pf_dict(pnls: np.ndarray) -> dict:
    n = int(len(pnls))
    if n == 0:
        return {"trades": 0, "pf": float("nan"), "win_rate": float("nan"),
                "avg": float("nan")}
    gains = float(pnls[pnls > 0].sum())
    losses = float(-pnls[pnls < 0].sum())
    if losses <= 0:
        pf = float("inf") if gains > 0 else float("nan")
    else:
        pf = gains / losses
    return {"trades": n, "pf": pf,
            "win_rate": float((pnls > 0).mean()),
            "avg": float(pnls.mean())}


def profit_factor(positions, forward_returns, min_trades: int = 1) -> dict:
    """按回合配对算盈利因子（毛利 / 毛损），不扣费。"""
    return _pf_dict(round_trip_pnls(positions, forward_returns))


def hit_rate(positions, forward_returns) -> dict:
    """方向准确率：只在有方向（|pos|>0）的样本上统计。"""
    x, y = _clean(positions, forward_returns)
    mask = x != 0
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "hit_rate": float("nan"), "hits": 0}
    hits = int((np.sign(x[mask]) == np.sign(y[mask])).sum())
    return {"n": n, "hit_rate": hits / n, "hits": hits}


def performance_stats(returns, freq: str = "D") -> dict:
    """累计/年化收益、夏普、最大回撤、胜率。"""
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) == 0:
        return {"n": 0, "total_return": float("nan"), "annual_return": float("nan"),
                "sharpe": float("nan"), "max_drawdown": float("nan"), "win_rate": float("nan")}

    equity = (1.0 + r).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    ppy = PERIODS_PER_YEAR.get(str(freq).upper(), 252)
    years = len(r) / ppy
    annual = float(equity.iloc[-1] ** (1 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    std = r.std(ddof=1)
    sharpe = float(r.mean() / std * np.sqrt(ppy)) if std and std > 0 else float("nan")
    peak = equity.cummax()
    mdd = float((equity / peak - 1.0).min())
    return {
        "n": int(len(r)),
        "total_return": total,
        "annual_return": annual,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": float((r > 0).mean()),
    }


def summarize_strategy(
    forward_returns: pd.Series,
    positions: pd.Series,
    *,
    fee_bps: float = 5.0,
    freq: str = "D",
    allow_short: bool = False,
) -> dict:
    """一次给出：扣费后绩效 + 方向准确率 + 换手 + 仓位结构。"""
    # 先统一成"实际执行仓位"，再算收益、方向准确率、仓位结构、换手。
    # 否则会出现"收益按裁过的算、胜率按没裁的算"的口径分裂。
    eff = pd.Series(positions, dtype=float)
    if not allow_short:
        eff = eff.clip(lower=0.0)

    gross = net_returns(forward_returns, eff, fee_bps=0.0, allow_short=allow_short)
    net = net_returns(forward_returns, eff, fee_bps=fee_bps, allow_short=allow_short)
    out = {
        "gross": performance_stats(gross, freq),
        "net": performance_stats(net, freq),
        "hit": hit_rate(eff, forward_returns),
        "avg_turnover": float(turnover(eff).mean()),
        "fee_bps": fee_bps,
        "allow_short": bool(allow_short),
        "composition": position_composition(eff),
        "trades": count_trades(eff),
        # 回合口径：一进一出算一笔，和 MT5 报告的「交易」一致
        "roundtrip": _pf_dict(round_trip_pnls_from_returns(eff, net)),
        "roundtrip_gross": _pf_dict(round_trip_pnls(eff, forward_returns)),
    }
    return out
