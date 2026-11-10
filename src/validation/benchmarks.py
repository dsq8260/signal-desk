"""对照基准（阶段 2）：买入持有、市场指数、随机信号分布。

随机信号基准是这里最关键的"照妖镜"：
如果一条规则的表现落在 100 次随机信号的分布中间，它就没有信息。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

from .metrics import hit_rate, net_returns, performance_stats

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def buy_and_hold_returns(prices: pd.DataFrame) -> pd.Series:
    """买入持有：满仓，不扣费（基准不做交易）。"""
    close = pd.to_numeric(prices["close"], errors="coerce").astype(float).dropna()
    return close.pct_change().dropna()


def market_index_returns(
    code: str = "sh000001", start=None, end=None
) -> pd.Series | None:
    """市场指数收益。取不到时返回 None（调用方需如实报告不可用）。"""
    try:
        import akshare as ak

        df = ak.stock_zh_index_daily(symbol=code)
        if df is None or len(df) == 0:
            return None
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        return close.pct_change().dropna()
    except Exception:  # noqa: BLE001 - 指数取不到不应中断整个流程
        return None


def random_signal_distribution(
    forward_returns: pd.Series,
    *,
    n_runs: int = 100,
    fee_bps: float = 5.0,
    freq: str = "D",
    seed: int = 0,
    same_turnover_as: pd.Series | None = None,
) -> dict:
    """同频率随机信号跑 n_runs 次，给出基准分布。

    若提供 same_turnover_as，则随机信号与真实信号保持相同的换手率水平，
    避免"随机信号几乎不交易，所以省手续费"这种不公平对比。
    """
    r = pd.Series(forward_returns, dtype=float).dropna()
    n = len(r)
    if n < 10:
        return {"n_runs": 0, "note": "样本不足（<10）", "runs": pd.DataFrame()}

    rng = np.random.default_rng(seed)
    target_turn = None
    if same_turnover_as is not None:
        p = pd.Series(same_turnover_as, dtype=float).reindex(r.index).fillna(0.0)
        target_turn = float(p.diff().abs().fillna(0).mean())

    rows = []
    for _ in range(n_runs):
        if target_turn is None:
            pos = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=n), index=r.index)
        else:
            # 生成随机仓位后按目标换手率稀疏化：随机保留一部分调仓
            keep = max(1, int(round(target_turn * n)))
            raw = rng.choice([-1.0, 0.0, 1.0], size=n)
            pos = np.zeros(n)
            idx = np.sort(rng.choice(n, size=min(keep, n), replace=False))
            pos[idx] = raw[idx]
            pos = pd.Series(pos, index=r.index).ffill().fillna(0.0)
        nr = net_returns(r, pos, fee_bps=fee_bps)
        st = performance_stats(nr, freq)
        hit = hit_rate(pos, r)
        rows.append(
            {
                "total_return": st["total_return"],
                "sharpe": st["sharpe"],
                "hit_rate": hit["hit_rate"],
                "n_ticks": hit["n"],
            }
        )

    runs = pd.DataFrame(rows)
    return {
        "n_runs": int(n_runs),
        "runs": runs,
        "total_return_q": {
            "p05": float(runs["total_return"].quantile(0.05)),
            "p50": float(runs["total_return"].quantile(0.50)),
            "p95": float(runs["total_return"].quantile(0.95)),
            "mean": float(runs["total_return"].mean()),
        },
        "sharpe_mean": float(runs["sharpe"].mean()),
        "hit_rate_mean": float(runs["hit_rate"].mean()),
        "note": "",
    }


def beaten_by_random(observed_return: float, distribution: dict) -> dict:
    """观测收益在随机分布中的分位与 p 值。"""
    runs = distribution.get("runs")
    if runs is None or len(runs) == 0:
        return {"percentile": float("nan"), "p_value": float("nan"), "note": "无随机分布"}
    vals = runs["total_return"].to_numpy(dtype=float)
    pct = float((vals < observed_return).mean())
    p_value = float((1 + int((vals >= observed_return).sum())) / (1 + len(vals)))
    return {"percentile": pct, "p_value": p_value, "note": ""}
