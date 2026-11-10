"""黄金 EA 回测：三段切分 + 次日开盘执行 + 点差成本 + 判真假。

复用现有验证层（validation.metrics / overfit.blocks），不重复造轮子。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gold.freqs import freq_key
from validation import metrics as M
from validation.risk import drawdown_series, drawdown_table
from gold.strategies import STRATEGIES, apply_strategy


@dataclass
class GoldBacktest:
    """一次单策略回测的结果。"""
    strategy: str = ""
    params: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    positions: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    drawdowns: pd.DataFrame = field(default_factory=pd.DataFrame)
    segments: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    note: str = ""


def split_bars(df: pd.DataFrame, train: float = 0.6,
               valid: float = 0.2) -> dict:
    n = len(df)
    a = int(n * train)
    b = int(n * (train + valid))
    return {"train": df.index[:a], "valid": df.index[a:b], "test": df.index[b:]}


def spread_fee_bps(price: float, spread_usd: float = 0.33) -> float:
    """把点差折成 bps（单边）。0.33 美元点差在 4000 美元金价上约 0.8 bp。"""
    if price <= 0:
        return 0.0
    return float(spread_usd / price * 10000.0)


def _stats(close: pd.Series, pos: pd.Series, fee_bps: float,
           freq: str = "D", allow_short: bool = False) -> dict:
    """扣费后统计。pos 已 shift 过（t 的信号在 t+1 生效）。"""
    fwd = close.pct_change().shift(-1)      # t -> t+1
    return M.summarize_strategy(fwd, pos, fee_bps=fee_bps, freq=freq,
                                allow_short=allow_short)


def block_permutation(fwd: pd.Series, pos: pd.Series, n_perm: int = 200,
                      block: int = 20, fee_bps: float = 0.0,
                      seed: int = 7, allow_short: bool = False) -> dict:
    """块置换：打乱收益块，看真实收益是否显著高于随机。

    对单序列策略，这是最关键的"是不是运气"检验。
    """
    r = pd.Series(fwd, dtype=float).dropna()
    p = pd.Series(pos, dtype=float).reindex(r.index).fillna(0.0)
    base = M.net_returns(r, p, fee_bps=fee_bps, allow_short=allow_short)
    observed = float((1.0 + base).prod() - 1.0)

    arr = r.to_numpy()
    n = len(arr)
    block = max(2, int(block))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(int(n_perm)):
        starts = rng.permutation(n_blocks)
        idx = np.concatenate([np.arange(s * block, min((s + 1) * block, n))
                              for s in starts])
        shuffled = pd.Series(arr[idx], index=r.index)
        net = M.net_returns(shuffled, p, fee_bps=fee_bps,
                            allow_short=allow_short)
        sims.append(float((1.0 + net).prod() - 1.0))
    sims = np.asarray(sims)
    p_value = float((sims >= observed).sum() + 1) / float(len(sims) + 1)
    return {"observed": observed, "p_value": p_value, "n_perm": int(n_perm),
            "block": block, "better_than_random": p_value < 0.05}


def judge(summary: dict, benchmark: dict, perm: dict) -> dict:
    """把三条可算的门槛判一遍（多标的门槛单品种不适用，标 N/A）。"""
    net = summary["net"]
    hit = summary["hit"]
    excess = float(net.get("total_return", float("nan"))) - float(
        benchmark.get("total_return", float("nan")))

    checks = [
        {"name": "方向准确率", "value": hit.get("hit_rate"), "n": hit.get("n", 0),
         "threshold": 0.55,
         "pass": bool(hit.get("hit_rate") == hit.get("hit_rate")
                      and hit.get("n", 0) >= 30
                      and hit["hit_rate"] >= 0.55)},
        {"name": "跑赢基准", "value": excess, "threshold": 0.0,
         "pass": bool(excess == excess and excess > 0)},
        {"name": "块置换 p", "value": perm.get("p_value"), "threshold": 0.05,
         "pass": bool(perm.get("p_value", 1.0) < 0.05)},
        {"name": "多标的有效占比", "value": None, "threshold": 0.6,
         "pass": None},
    ]
    passed = sum(1 for c in checks if c["pass"] is True)
    failed = sum(1 for c in checks if c["pass"] is False)
    verdict = ("通过" if passed >= 3 else
               "未通过" if failed >= 1 else "证据不足")
    return {"checks": checks, "passed": passed, "failed": failed,
            "verdict": verdict}


def run(df: pd.DataFrame, strategy: str, params: dict | None = None,
        spread_usd: float = 0.33, n_perm: int = 200, block: int = 20,
        train: float = 0.6, valid: float = 0.2, seed: int = 7,
        allow_short: bool = False, tf: str = "D1") -> GoldBacktest:
    """跑一个策略：全样本信号 → 三段切分 → 测试段评估 + 置换检验。

    allow_short=True 时信号为 {-1,0,1}，空头仓位参与盈亏（趋势类"反向即做空"）。
    """
    if df is None or len(df) < 120:
        return GoldBacktest(strategy=strategy, params=params or {},
                            note="K 线不足 120 根，无法回测")

    seg = split_bars(df, train, valid)
    pos_all = apply_strategy(df, strategy, params, both=allow_short)
    pos = pos_all.shift(1).fillna(0.0)          # t 收盘信号 -> t+1 生效

    test_idx = seg["test"]
    close = df["close"].astype(float)
    price = float(close.loc[test_idx].mean()) if len(test_idx) else float(close.mean())
    fee_bps = spread_fee_bps(price, spread_usd)

    test_close = close.loc[test_idx]
    test_pos = pos.reindex(test_idx).fillna(0.0)
    fq = freq_key(tf)          # 按周期年化（H1 用 √6048，而不是 √252）
    summary = _stats(test_close, test_pos, fee_bps, freq=fq,
                     allow_short=allow_short)
    fwd = test_close.pct_change().shift(-1)

    bh = M.performance_stats(fwd.dropna(), fq)
    perm = block_permutation(fwd, test_pos, n_perm=n_perm, block=block,
                             fee_bps=fee_bps, seed=seed,
                             allow_short=allow_short)
    gates = judge(summary, bh, perm)

    net = M.net_returns(fwd, test_pos, fee_bps=fee_bps,
                        allow_short=allow_short)
    equity = (1.0 + net).cumprod() * 100.0
    bench = (1.0 + fwd.fillna(0.0)).cumprod() * 100.0

    spec = STRATEGIES.get(strategy, {})
    return GoldBacktest(
        strategy=strategy,
        params={**spec.get("params", {}), **(params or {})},
        summary={"net": summary["net"], "gross": summary["gross"],
                 "hit": summary["hit"], "trades": summary["trades"],
                 "avg_turnover": summary["avg_turnover"],
                 "fee_bps": fee_bps},
        equity=equity, positions=test_pos, returns=net,
        benchmark=bench, drawdowns=drawdown_table(equity),
        segments={k: (str(v[0].date()), str(v[-1].date())) if len(v) else ()
                  for k, v in seg.items()},
        gates=gates,
        note=f"点差 {spread_usd} 美元 ≈ {fee_bps:.2f} bp（单边）",
    )


def compare_all(df: pd.DataFrame, tf: str = "D1", **kw) -> pd.DataFrame:
    """把所有策略跑一遍，按测试段收益排序。

    kw 里可传 allow_short=True 做双向版本对比。
    """
    rows = []
    for name in STRATEGIES:
        r = run(df, name, tf=tf, **kw)
        if r.note and "不足" in r.note:
            continue
        rows.append({
            "策略": name,
            "收益": r.summary["net"].get("total_return", float("nan")),
            "最大回撤": r.summary["net"].get("max_drawdown", float("nan")),
            "夏普": r.summary["net"].get("sharpe", float("nan")),
            "方向准确率": r.summary["hit"].get("hit_rate", float("nan")),
            "置换p": r.gates["checks"][2]["value"],
            "门槛": r.gates["verdict"],
        })
    out = pd.DataFrame(rows)
    return out.sort_values("收益", ascending=False) if len(out) else out
