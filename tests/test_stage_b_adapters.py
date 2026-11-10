"""四模式共用验证器：适配层测试（采样/进化/分层/微调 → 同一套验证）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.evolve import evolve
from gold.forge import Genome, forge
from gold.strata import search
from gold.verify import genomes_from_result, run_verified


def _bars(n=700, seed=5, drift=0.0005):
    rng = np.random.default_rng(seed)
    close = 2600 * np.cumprod(1 + rng.normal(drift, 0.0018, n))
    idx = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close,
                         "volume": 0.0}, index=idx)


def _all_genomes(res):
    gs = genomes_from_result(res)
    assert gs, "适配层没取出任何基因组"
    assert all(isinstance(g, Genome) for g in gs)
    assert all(g.entry for g in gs)
    return gs


def test_adapter_from_forge_result():
    r = forge(_bars(), n_candidates=15, n_perm=10, tf="H1")
    _all_genomes(r)


def test_adapter_from_evolve_result():
    r = evolve(_bars(), generations=2, population=10, elite=4, seed=3)
    if getattr(r, "best", pd.DataFrame()).empty:
        pytest.skip("进化无候选")
    _all_genomes(r)


def test_adapter_from_strata_result():
    r = search(_bars(), n_perm=20, tf="H1")
    if r.table.empty:
        pytest.skip("分层无候选")
    _all_genomes(r)


def test_adapter_returns_empty_for_junk():
    assert genomes_from_result(None) == []
    assert genomes_from_result(object()) == []


def test_all_modes_go_through_the_same_verifier():
    """四种模式的结果都能喂给同一个 run_verified，返回同构表。"""
    df = _bars(900, seed=9)
    gs_forge = genomes_from_result(
        forge(df, n_candidates=10, n_perm=10, tf="H1"))
    gs_evolve = genomes_from_result(
        evolve(df, generations=2, population=8, elite=3, seed=5))
    gs_strata = genomes_from_result(search(df, n_perm=10, tf="H1"))

    assert gs_forge, "采样模式没产出基因组"
    cols = None
    for tag, gs in (("采样", gs_forge), ("进化", gs_evolve),
                    ("分层", gs_strata)):
        if not gs:
            continue
        res = run_verified(df, gs, tf="H1", spread_usd=0.33, n_perm=40,
                           n_windows=2)
        if res.table.empty:
            continue
        # 每个模式都必须产出同一批列与同一套结论词汇
        need = {"训练收益", "验证收益", "测试收益", "块置换p", "q",
                "deflated_p", "结论", "门槛明细", "原因"}
        assert need.issubset(set(res.table.columns)), (tag, res.table.columns)
        assert set(res.table["结论"]).issubset({"通过", "继续观察", "未通过"})
        assert (res.table["结论"] != "通过").all(), f"{tag} 不该出现通过"
        cols = cols or set(res.table.columns)
        assert set(res.table.columns) >= {"结论", "块置换p", "测试方向"}


def test_run_verified_marks_multisymbol_gate_unmeasured():
    """单品种时「多品种≥60%」是 N/A：结论里必须写"缺证据"，不能是通过。"""
    df = _bars(900, seed=13)
    gs = [Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False),
          Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "无过滤", False)]
    res = run_verified(df, gs, tf="H1", spread_usd=0.33, n_perm=40,
                       n_windows=2)
    if res.table.empty:
        pytest.skip("无候选")
    assert (res.table["结论"] != "通过").all()
    gates = res.table.iloc[0]["门槛明细"]
    multi = [g for g in gates if "多品种" in g["name"]][0]
    assert multi["status"] == "unmeasured"
