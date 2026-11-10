"""解 BH 不可能过的问题：预注册假设空间 + 置换预算 + 决赛精化。

为什么需要这个模块
------------------
BH-FDR 在 K 次检验下，要把最小 q 压到 alpha，需要
    p < alpha / K
而 n_perm 次块置换能给出的最小 p 是 1/(n_perm+1)。于是
    n_perm + 1 > K / alpha   ⇒   n_perm >= K / alpha
K=444、alpha=0.05 时要 8,880 次/候选（约 400 万次置换，几小时）；
K=18 时只要 360 次/候选（约 6,500 次置换，几秒）。

所以正确做法不是"多跑几次让 p 变小"，而是**事前**把一个有经济学依据的
假设空间定下来（K 小），并按 K 反推置换次数。事后挑空间 = p-hacking。

注意：`perms_for_bh(K) = ceil(K/alpha)` 是**保守充分条件**——它保证即使
只有「唯一一条候选显著、其余都很弱」，最小 q 也能压到 alpha 以下。
如果有多条候选同时显著，BH 的阶梯效应会更宽松，需要的置换次数更少；
但预算按保守情形给，才不会在真实情况下过不去。
"""

from __future__ import annotations

import math

import pandas as pd

from gold.backtest import block_permutation
from gold.mining import bh_qvalues


def perms_for_bh(k_tests: int, alpha: float = 0.05) -> int:
    """让 BH-FDR 在 k_tests 次检验下**可能**达到 alpha 的最小置换次数。

    p 的最小可达值为 1/(n_perm+1)；BH 要求 p < alpha/K。
    """
    k = max(1, int(k_tests))
    return int(math.ceil(k / float(alpha)))


def permutation_budget(k_tests: int, alpha: float = 0.05,
                       max_perms: int = 20000,
                       perms_per_sec: float = 850.0,
                       max_minutes: float = 20.0) -> dict:
    """返回该空间的置换预算，并判断在时间上是否划算（供界面提示）。

    ``perms_per_sec`` 是实测吞吐（本机 H1、OOS 约 1 万根 ≈ 850 次/秒）。
    """
    need = perms_for_bh(k_tests, alpha)
    capped = min(need, int(max_perms))
    total = int(capped * max(1, int(k_tests)))
    est_min = total / max(1.0, float(perms_per_sec)) / 60.0
    return {
        "k_tests": int(k_tests),
        "alpha": float(alpha),
        "need": int(need),
        "planned": int(capped),
        "total_permutations": total,
        "est_minutes": round(est_min, 2),
        "feasible": bool(need <= max_perms and est_min <= max_minutes),
        "note": (f"K={k_tests} 时 BH 需要每候选 ≥{need} 次置换"
                 f"（p 最小 1/{need + 1} < {alpha}/{k_tests}）；"
                 f"计划 {capped} 次 × {k_tests} = {total:,} 次，"
                 f"约 {est_min:.1f} 分钟"),
    }


def refine_finalists(df: pd.DataFrame, tbl: pd.DataFrame, pos_of,
                     fwd_oos: pd.Series, fee_bps: float, k_tests: int,
                     n_perm: int, block: int = 20, seed: int = 7,
                     alpha: float = 0.05) -> pd.DataFrame:
    """对已筛出的候选做**更高精度**的置换，再按总试验数 K 重算 BH q。

    参数
    ----
    tbl : 至少含 `EA` 与 `块置换p` 两列的候选表（阶段1的结果）
    pos_of : 可调用对象，给 EA 标签返回该候选的 OOS 执行仓位序列
    k_tests : **总试验次数**（不是决赛个数！否则等于偷放门槛）
    """
    out = tbl.copy()
    if "块置换p" not in out.columns:
        out["块置换p"] = float("nan")

    need = perms_for_bh(k_tests, alpha)
    n_use = int(max(int(n_perm), 1))
    refined = 0
    for i, row in out.iterrows():
        pos = pos_of(row["EA"])
        if pos is None:
            continue
        perm = block_permutation(fwd_oos, pos, n_perm=n_use, block=block,
                                 fee_bps=fee_bps, seed=seed)
        out.at[i, "块置换p"] = float(perm["p_value"])
        out.at[i, "随机检验"] = bool(perm["better_than_random"])
        out.at[i, "置换次数"] = n_use
        refined += 1

    # BH 必须按**总试验数** K 算，不是按决赛个数
    q = bh_qvalues([float(v) if v == v else 1.0 for v in out["块置换p"]])
    out["q"] = q
    out.attrs["k_tests"] = int(k_tests)
    out.attrs["n_perm_used"] = n_use
    out.attrs["perms_needed"] = int(need)
    out.attrs["refined"] = refined
    return out


def preregister(name: str, genomes) -> tuple[str, list]:
    """把"研究假设"显式登记：名字 + 基因组列表（**必须在看结果之前**）。

    这只是把 K 固定下来并留痕，不做任何筛选。
    """
    gens = list(genomes)
    return (f"{name}（K={len(gens)}）", gens)
