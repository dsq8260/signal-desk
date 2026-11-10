"""跨品种横截面策略：不预测"涨不涨"，只赌"谁比谁强"。

为什么换成横截面
----------------
单序列择时（RSRS/LLT/唐奇安…）在 XM 真金价上已用 48 个组合证明：
24 只做多 + 24 双向，**0 个通过四道门槛**。
横截面的信息量不同：黄金涨不涨难猜，但"黄金 vs 欧元 vs 英镑 谁更强"
在同一时刻是可比较的，排序比预测点位稳定得多。

做法
----
每个调仓日：
1. 对全部品种算 N 期动量（或任意打分函数）；
2. 取最强 top_k 做多、最弱 bottom_k 做空；
3. 等权持有到下次调仓；
4. 扣换手手续费。

多空两端都算，所以组合收益可能是正、负或 0，不需要"永远在场"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gold.backtest import block_permutation, spread_fee_bps


def momentum_score(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """N 期动量收益率（可跨品种比较）。"""
    close = df["close"].astype(float)
    return close / close.shift(int(n)) - 1.0


def zscore_score(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """N 期动量做自身滚动标准分，压掉各品种的波动差异。"""
    mom = momentum_score(df, n)
    mu = mom.rolling(120, min_periods=20).mean()
    sd = mom.rolling(120, min_periods=20).std(ddof=0)
    return (mom - mu) / sd.replace(0.0, np.nan)


SCORERS = {"动量": momentum_score, "动量标准分": zscore_score}


def build_panel(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把 {symbol: bars} 合成收盘价宽表（按时间对齐）。"""
    cols = {sym: df["close"].astype(float) for sym, df in panel.items() if len(df)}
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def cross_section_weights(close: pd.DataFrame, scorer: str = "动量",
                          lookback: int = 20, top_k: int = 1,
                          bottom_k: int = 1, long_only: bool = False,
                          rebalance: int = 1) -> pd.DataFrame:
    """每个时间点的横截面权重（+1/top_k 多头，-1/bottom_k 空头）。

    rebalance=1 表示每根 bar 都重排；>1 表示每隔 N 根才调一次（其余沿用）。
    打分只用当根及之前的数据，权重再整体 shift(1) 由回测层处理。
    """
    scorer_fn = SCORERS.get(scorer, momentum_score)
    if close is None or close.shape[1] == 0:
        return pd.DataFrame()

    # 打分：用每个品种自己的 K 线算（保持接口一致）
    scores = {}
    for sym in close.columns:
        s = pd.Series(close[sym].to_numpy(), index=close.index)
        if scorer == "动量":
            sc = s / s.shift(int(lookback)) - 1.0
        else:
            mom = s / s.shift(int(lookback)) - 1.0
            mu = mom.rolling(120, min_periods=20).mean()
            sd = mom.rolling(120, min_periods=20).std(ddof=0)
            sc = (mom - mu) / sd.replace(0.0, np.nan)
        scores[sym] = sc
    rank_df = pd.DataFrame(scores).sort_index()

    w = pd.DataFrame(0.0, index=rank_df.index, columns=rank_df.columns)
    k_long = max(1, int(top_k))
    k_short = 0 if long_only else max(1, int(bottom_k))
    step = max(1, int(rebalance))

    last = pd.Series(dtype=float)
    for i, ts in enumerate(rank_df.index):
        row = rank_df.loc[ts]
        if row.isna().all():
            w.loc[ts] = 0.0
            continue
        if i % step == 0 or last.empty:
            valid = row.dropna().sort_values(ascending=False)
            if valid.empty:
                w.loc[ts] = 0.0
                continue
            cur = pd.Series(0.0, index=rank_df.columns)
            longs = valid.index[: min(k_long, len(valid))]
            cur[longs] = 1.0 / len(longs)
            if k_short and len(valid) > len(longs):
                shorts = valid.index[-min(k_short, len(valid) - len(longs)):]
                shorts = [s for s in shorts if s not in longs]
                if shorts:
                    cur[shorts] = -1.0 / len(shorts)
            last = cur
        w.loc[ts] = last
    return w


@dataclass
class CrossResult:
    equity: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: dict = field(default_factory=dict)
    perm: dict = field(default_factory=dict)
    benchmark: pd.Series = field(default_factory=pd.Series)
    note: str = ""
    params: dict = field(default_factory=dict)


def _stats_from_returns(net: pd.Series, freq: str = "D") -> dict:
    from validation.metrics import performance_stats

    return performance_stats(net.dropna(), freq)


def run_cross(panel: dict[str, pd.DataFrame], scorer: str = "动量",
              lookback: int = 20, top_k: int = 1, bottom_k: int = 1,
              long_only: bool = False, rebalance: int = 1,
              spread_usd: float = 0.33, n_perm: int = 200,
              block: int = 20, test_ratio: float = 0.4,
              seed: int = 7) -> CrossResult:
    """跑一次跨品种横截面回测（只在最后 test_ratio 段评估）。"""
    close = build_panel(panel)
    if close is None or close.shape[1] < 2 or len(close) < 200:
        return CrossResult(note="至少需要 2 个品种、200 根 bar")

    w_all = cross_section_weights(close, scorer, lookback, top_k, bottom_k,
                                  long_only, rebalance)
    if w_all.empty:
        return CrossResult(note="权重计算失败")

    # t 的权重在 t+1 生效
    w = w_all.shift(1).fillna(0.0)
    rets = close.pct_change(fill_method=None)

    n = int(len(close) * (1.0 - test_ratio))
    rets_w = rets.loc[close.index[n:]]
    w_test = w.loc[close.index[n:]]

    # 换手 → 手续费（按平均价的 bps）
    turnover = (w_test - w_test.shift(1)).abs().sum(axis=1).fillna(0.0)
    avg_price = float(close.loc[close.index[n:]].mean().mean())
    fee = spread_fee_bps(avg_price, spread_usd) / 10000.0
    gross = (w_test * rets_w.fillna(0.0)).sum(axis=1)
    net = gross - turnover * fee

    equity = (1.0 + net.fillna(0.0)).cumprod() * 100.0
    stats = _stats_from_returns(net, "D")

    # 基准：等权持有全部品种（不加择时）
    bh_ret = rets_w.mean(axis=1).fillna(0.0)
    benchmark = (1.0 + bh_ret).cumprod() * 100.0

    perm = cross_permutation(rets_w, w_test, turnover, fee, n_perm, seed)
    excess = net - bh_ret

    return CrossResult(
        equity=equity, returns=net, weights=w_test, stats=stats,
        perm={**perm,
              "excess_mean": float(excess.mean()),
              "excess_total": float((1.0 + excess).prod() - 1.0)},
        benchmark=benchmark,
        params={"scorer": scorer, "lookback": lookback, "top_k": top_k,
                "bottom_k": bottom_k, "long_only": long_only,
                "rebalance": rebalance},
        note=(f"{close.shape[1]} 个品种 · {len(net)} 根测试 bar · "
              f"换手均值 {turnover.mean():.3f} · 手续费 {fee * 10000:.2f} bp"),
    )


def cross_permutation(rets_w: pd.DataFrame, w_test: pd.DataFrame,
                      turnover: pd.Series, fee: float,
                      n_perm: int = 200, seed: int = 7) -> dict:
    """横截面置换检验。

    零假设：**排序没有信息**。所以每一行把权重在各品种之间随机重排，
    保留"总暴露相同、换手相同"，只打乱"押在谁身上"。
    这比打乱时间序列更贴合横截面策略的失败模式。
    """
    r = rets_w.fillna(0.0).to_numpy()
    W = w_test.fillna(0.0).to_numpy()
    fee_vec = (turnover.fillna(0.0).to_numpy() * fee)

    def total(Wm: np.ndarray) -> float:
        gross = (Wm * r).sum(axis=1)
        net = gross - fee_vec
        return float((1.0 + net).prod() - 1.0)

    observed = total(W)
    rng = np.random.default_rng(seed)
    sims = np.empty(int(n_perm))
    for k in range(int(n_perm)):
        Ws = W.copy()
        for i in range(Ws.shape[0]):
            rng.shuffle(Ws[i])
        sims[k] = total(Ws)

    p_value = float((sims >= observed).sum() + 1) / float(len(sims) + 1)
    return {"observed": observed, "p_value": p_value, "n_perm": int(n_perm),
            "better_than_random": p_value < 0.05}
