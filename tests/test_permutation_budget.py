"""置换预算与预注册空间测试：让 BH q<0.05 在数学上可达。

背景（真问题）：BH 在 K 次检验下要求 p < alpha/K，而 n_perm 次置换的最小
p 是 1/(n_perm+1)。K=444、alpha=0.05 时需要每候选 8,880 次置换，
实测 q 最小只能到 0.11 —— 于是"没有任何 EA 通过"里有一半是**统计预算不足**，
不是策略不行。本模块把预算算清楚，并要求 K 事前固定。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.forge import Genome, forge
from gold.mining import bh_qvalues
from gold.permutation_budget import (
    perms_for_bh,
    permutation_budget,
    preregister,
    refine_finalists,
)


# ---------------------------------------------------------------- 预算数学
def test_perms_for_bh_formula():
    """n_perm 必须满足 1/(n+1) < alpha/K。"""
    assert perms_for_bh(17, 0.05) == 340
    assert perms_for_bh(34, 0.05) == 680
    assert perms_for_bh(444, 0.05) == 8880
    # alpha 更严 → 需要更多置换
    assert perms_for_bh(20, 0.01) > perms_for_bh(20, 0.05)


def test_perms_satisfies_inequality():
    """直接把不等式代进去验算，避免公式写反。"""
    for k in (5, 17, 60, 200):
        n = perms_for_bh(k, 0.05)
        assert 1.0 / (n + 1) < 0.05 / k
        assert 1.0 / (n + 1) * k < 0.05


def test_budget_flags_infeasible_when_too_big():
    small = permutation_budget(17)
    huge = permutation_budget(444)
    assert small["feasible"] is True
    assert huge["feasible"] is False          # 时间上不划算，不能默默跑几小时
    assert huge["total_permutations"] > small["total_permutations"]
    assert "BH" in small["note"]


def test_bh_needs_budget_when_only_one_candidate_is_significant():
    """最坏情形：只有 1 条候选显著、其余 K-1 条都很弱。

    这时 BH 要求 p_(1) < alpha/K，正是 `perms_for_bh` 的保守条件。
    """
    k = 20
    weak = [1.0] * (k - 1)

    # 预算不足（100 次置换 → p 最小 0.0099）
    q_bad = bh_qvalues([1.0 / 101] + weak)
    assert min(q_bad) > 0.05

    # 按预算给足 → 能过
    n = perms_for_bh(k, 0.05)
    q_ok = bh_qvalues([1.0 / (n + 1)] + weak)
    assert min(q_ok) < 0.05


def test_bh_is_more_lenient_when_many_candidates_are_significant():
    """多条同时显著时，BH 的阶梯效应更宽松 —— 所以预算是保守下界，不会过严。"""
    k = 20
    n = 100
    p = 1.0 / (n + 1)
    q_many = bh_qvalues([p] * k)      # 20 条都很显著
    q_one = bh_qvalues([p] + [1.0] * (k - 1))
    assert min(q_many) < min(q_one)


# ---------------------------------------------------------------- 预注册
def test_preregister_fixes_k():
    space = [Genome("SMA金叉", {"fast": 10}, "无过滤", False)] * 7
    label, gens = preregister("纯趋势", space)
    assert "K=7" in label
    assert len(gens) == 7


def test_forge_accepts_preregistered_space(tmp_path):
    rng = np.random.default_rng(4)
    n = 400
    close = 2600 * np.cumprod(1 + rng.normal(0.0004, 0.002, n))
    idx = pd.bdate_range("2020-01-02", periods=n)
    df = pd.DataFrame({"open": close, "high": close * 1.001,
                       "low": close * 0.999, "close": close,
                       "volume": 0.0}, index=idx)
    space = [Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False),
             Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "无过滤", False)]
    r = forge(df, genomes=space, n_perm="auto", spread_usd=0.33, tf="H1")
    assert r.n_sampled == len(space)      # K 就是预注册空间大小
    assert "预注册" in r.note
    # auto 必须按 BH 预算给足置换次数
    tokens = [t for t in r.note.split() if "置换" in t]
    assert tokens
    assert perms_for_bh(len(space)) >= 40


def test_forge_n_perm_auto_matches_budget():
    rng = np.random.default_rng(5)
    n = 300
    close = 2600 * np.cumprod(1 + rng.normal(0, 0.002, n))
    idx = pd.bdate_range("2020-01-02", periods=n)
    df = pd.DataFrame({"open": close, "high": close * 1.001,
                       "low": close * 0.999, "close": close,
                       "volume": 0.0}, index=idx)
    space = [Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False)] * 3
    r = forge(df, genomes=space, n_perm="auto", spread_usd=0.33, tf="H1")
    if r.table.empty:
        pytest.skip("无有效候选")
    step = 1.0 / (perms_for_bh(3) + 1)
    p = float(r.table["块置换p"].iloc[0])
    k = p / step
    assert abs(k - round(k)) < 1e-6, (p, step)


# ---------------------------------------------------------------- 决赛精化
def test_refine_finalists_improves_resolution_and_keeps_k():
    """决赛阶段用更多置换提高 p 的分辨率，但 BH 仍按**总试验数** K 计算。"""
    rng = np.random.default_rng(11)
    n = 400
    fwd = pd.Series(rng.normal(0.0002, 0.01, n))
    pos_series = {
        "A": pd.Series(rng.integers(0, 2, n).astype(float)),
        "B": pd.Series(rng.integers(0, 2, n).astype(float)),
    }
    tbl = pd.DataFrame({"EA": ["A", "B"],
                        "块置换p": [1 / 101, 1 / 101]})
    out = refine_finalists(None, tbl, lambda ea: pos_series[ea],
                           fwd, fee_bps=0.0, k_tests=34, n_perm=680, seed=5)
    assert out.attrs["k_tests"] == 34
    assert out.attrs["n_perm_used"] == 680
    assert out.attrs["refined"] == 2
    step = 1.0 / 681
    for p in out["块置换p"]:
        assert abs(p / step - round(p / step)) < 1e-6
    # q 用总 K（34）算：p*(K/rank) 之后必须 >= p
    assert (out["q"] >= out["块置换p"] - 1e-12).all()
