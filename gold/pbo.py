"""CSCV-PBO：用组合对称交叉验证估计"回测过拟合概率"。

为什么需要
----------
DSR（Deflated Sharpe）解决的是"挑最好的那个，它的夏普有多少是运气"；
PBO 回答的是另一个问题：**如果我把这批参数在样本内挑一个最好的，
它在样本外还能排进前半吗？** 排不进的比例就是过拟合概率。

方法（Bailey, Borwein, López de Prado, Zhu 2015）
------------------------------------------------
1. 把时间轴切成 S 个连续块（S 取偶数，默认 12）；
2. 枚举所有"取一半块做样本内、另一半做样本外"的划分（C(12,6)=924 种）；
3. 每个划分里：样本内挑表现最好的策略，看它在样本外的排名分位 ω；
4. λ = logit(ω)，ω = rank/(N+1)（rank 从"最差"数起）；
5. PBO = P(λ ≤ 0)，即"样本内最优在样本外落到后半"的比例。

PBO 越小越好：≤0.5 表示不比随机挑好多少，>0.5 说明样本内挑选基本是噪声。

实现要点：按块预计算 sum / sumsq，划分时只做加法，避免 O(splits×T×N) 的重算。
"""

from __future__ import annotations

import itertools
import math

import numpy as np

METRICS = ("sharpe", "mean")


def _block_stats(ret: np.ndarray, n_blocks: int):
    """把 T×N 收益矩阵按块切成 sum / sumsq / count。"""
    T, N = ret.shape
    size = T // n_blocks
    if size < 2:
        raise ValueError(f"样本太短：{T} 期切 {n_blocks} 块，每块不足 2 期")
    usable = size * n_blocks
    r = ret[:usable].reshape(n_blocks, size, N)
    return r.sum(axis=1), (r ** 2).sum(axis=1), np.full((n_blocks, N), size)


def _metric(sums, sumsqs, counts, metric: str) -> np.ndarray:
    mean = sums / counts
    var = np.maximum(sumsqs / counts - mean ** 2, 0.0)
    if metric == "mean":
        return mean
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, mean / sd, 0.0)
    return out


def cscv_pbo(returns, n_blocks: int = 12, metric: str = "sharpe",
             max_splits: int = 2000) -> dict:
    """估计过拟合概率。

    ``returns``：T×N 的逐期收益（DataFrame 或 ndarray），列 = 一个配置。
    返回 dict：pbo / n_splits / n_blocks / logits / omega / ok / reason。
    """
    if metric not in METRICS:
        return {"ok": False, "reason": f"未知 metric：{metric}", "pbo": float("nan")}
    if hasattr(returns, "to_numpy"):
        returns = returns.to_numpy(dtype=float)
    ret = np.asarray(returns, dtype=float)
    if ret.ndim != 2:
        return {"ok": False, "reason": "returns 必须是二维 T×N", "pbo": float("nan")}
    T, N = ret.shape
    if N < 2:
        return {"ok": False, "reason": "至少需要 2 个配置才能谈过拟合概率",
                "pbo": float("nan"), "n_configs": int(N)}
    if n_blocks % 2:
        n_blocks += 1
    if T < n_blocks * 4:
        return {"ok": False,
                "reason": f"样本太短：{T} 期 < {n_blocks} 块 × 4",
                "pbo": float("nan"), "n_configs": int(N)}

    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    sums, sumsqs, counts = _block_stats(ret, n_blocks)
    half = n_blocks // 2
    combos = list(itertools.combinations(range(n_blocks), half))
    if len(combos) > max_splits:                      # 均匀抽样，保持确定性
        step = len(combos) / max_splits
        combos = [combos[int(i * step)] for i in range(max_splits)]

    logits, omegas = [], []
    for is_blocks in combos:
        oos_blocks = [b for b in range(n_blocks) if b not in is_blocks]
        is_m = _metric(sums[list(is_blocks)].sum(0), sumsqs[list(is_blocks)].sum(0),
                       counts[list(is_blocks)].sum(0), metric)
        oos_m = _metric(sums[oos_blocks].sum(0), sumsqs[oos_blocks].sum(0),
                        counts[oos_blocks].sum(0), metric)
        best = int(np.nanargmax(is_m))               # 样本内挑最优
        # 样本外排名：1 = 最差，N = 最好
        rank = int((oos_m < oos_m[best]).sum()) + 1
        omega = rank / (N + 1.0)                     # 分位
        omegas.append(omega)
        if omega <= 0 or omega >= 1:
            logits.append(float("-inf") if omega <= 0 else float("inf"))
        else:
            logits.append(math.log(omega / (1 - omega)))

    lg = np.array(logits, dtype=float)
    pbo = float((lg <= 0).mean())
    return {"ok": True, "pbo": pbo, "n_splits": len(combos),
            "n_blocks": int(n_blocks), "n_configs": int(N),
            "n_periods": int(T), "metric": metric,
            "logits": [round(float(x), 4) if np.isfinite(x) else None for x in lg],
            "omega_mean": round(float(np.mean(omegas)), 4),
            "reason": "",
            "verdict": ("过拟合嫌疑小" if pbo <= 0.2 else
                        "有一定过拟合" if pbo <= 0.5 else "过拟合嫌疑大")}


def pbo_from_returns_by_config(frames: dict, n_blocks: int = 12,
                               metric: str = "sharpe") -> dict:
    """把 {配置名: 收益 Series} 拼成矩阵后估计 PBO（按时间对齐取交集）。"""
    import pandas as pd

    if not frames:
        return {"ok": False, "reason": "没有可用的收益序列", "pbo": float("nan")}
    df = pd.DataFrame(frames)
    df = df.dropna(how="any")
    if len(df) < n_blocks * 4:
        return {"ok": False, "reason": f"对齐后只剩 {len(df)} 期，太短",
                "pbo": float("nan"), "n_configs": df.shape[1]}
    out = cscv_pbo(df, n_blocks=n_blocks, metric=metric)
    out["configs"] = list(df.columns)[:80]
    return out
