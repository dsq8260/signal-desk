"""防过拟合套件（阶段 2）：样本外衰减、参数敏感性、块置换、多重检验校正。

设计原则：这里所有函数都只做"检验"，不参与拟合，
因此可以被反复调用来质疑策略，而不会污染结果。
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import net_returns


# ------------------------------------------------------------------ 样本外衰减
def oos_decay(train_metric: float, test_metric: float) -> dict:
    """样本外衰减率 = 1 - 测试/训练。

    训练指标 <= 0 时衰减率没有意义，返回 None 并在 note 里说明。
    """
    t, o = float(train_metric), float(test_metric)
    if not np.isfinite(t) or not np.isfinite(o):
        return {"train": t, "test": o, "decay": None, "ratio": None,
                "note": "指标含 NaN，无法计算衰减"}
    if t == 0:
        return {"train": t, "test": o, "decay": None, "ratio": None,
                "note": "训练指标为 0，无法计算比率"}
    ratio = o / t
    return {"train": t, "test": o, "decay": 1.0 - ratio, "ratio": ratio, "note": ""}


# ------------------------------------------------------------------ 参数敏感性
def parameter_sensitivity(results: Mapping[float, float]) -> dict:
    """看同一策略在不同参数下的表现是否剧烈跳动。"""
    vals = np.array([v for v in results.values() if np.isfinite(v)], dtype=float)
    if len(vals) < 2:
        return {"params": dict(results), "mean": float("nan"), "std": float("nan"),
                "cv": float("nan"), "spread": float("nan"), "stable": None}
    mean, std = float(vals.mean()), float(vals.std(ddof=1))
    cv = float(std / abs(mean)) if mean != 0 else float("inf")
    return {
        "params": dict(results),
        "mean": mean,
        "std": std,
        "cv": cv,
        "spread": float(vals.max() - vals.min()),
        "stable": bool(np.isfinite(cv) and cv < 0.5),
    }


# ------------------------------------------------------------------ 块置换
def block_shuffle(arr: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """按块打乱：保持块内自相关结构，只破坏"与信号的对应关系"。"""
    a = np.asarray(arr, dtype=float)
    n = len(a)
    if block <= 1:
        return rng.permutation(a)
    starts = list(range(0, n, block))
    blocks = [a[s:s + block] for s in starts]
    order = rng.permutation(len(blocks))
    out = np.concatenate([blocks[i] for i in order])
    return out[:n]


def block_permutation_test(
    forward_returns: pd.Series,
    positions: pd.Series,
    *,
    metric_fn: Callable[[pd.Series], float] | None = None,
    n_perm: int = 500,
    block: int = 20,
    fee_bps: float = 5.0,
    seed: int = 0,
) -> dict:
    """块置换检验：打乱收益的块顺序，看真实策略表现是否罕见。

    返回 p 值 = (1 + #{置换收益 >= 观测收益}) / (1 + n_perm)。
    默认指标是扣费后累计收益。
    """
    r = pd.Series(forward_returns, dtype=float)
    p = pd.Series(positions, dtype=float).reindex(r.index)
    if len(r) < block * 2:
        return {"observed": float("nan"), "p_value": float("nan"), "n_perm": 0,
                "block": block, "note": "样本不足（<2 个块），无法做块置换"}

    def default_metric(returns: pd.Series) -> float:
        return float(net_returns(returns, p, fee_bps=fee_bps).sum())

    fn = metric_fn or default_metric
    observed = float(fn(r))
    rng = np.random.default_rng(seed)
    arr = r.to_numpy()
    xs = []
    for _ in range(n_perm):
        shuffled = pd.Series(block_shuffle(arr, block, rng), index=r.index)
        xs.append(float(fn(shuffled)))
    xs = np.array(xs, dtype=float)
    p_value = float((1 + int((xs >= observed).sum())) / (1 + len(xs)))
    return {
        "observed": observed,
        "p_value": p_value,
        "n_perm": int(n_perm),
        "block": int(block),
        "null_mean": float(xs.mean()) if len(xs) else float("nan"),
        "null_p95": float(np.quantile(xs, 0.95)) if len(xs) else float("nan"),
        "note": "",
    }


# ------------------------------------------------------------------ 多重检验
def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> pd.DataFrame:
    """BH-FDR 校正。返回每个假设的原始 p、校正后 q 与是否通过。"""
    p = np.asarray([float(x) for x in pvalues], dtype=float)
    n = len(p)
    if n == 0:
        return pd.DataFrame(columns=["index", "p_value", "q_value", "rejected"])
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    # 保证 q 单调不降
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = q
    return pd.DataFrame(
        {
            "index": np.arange(n),
            "p_value": p,
            "q_value": out,
            "rejected": out < alpha,
        }
    )
