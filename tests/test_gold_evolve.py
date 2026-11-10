"""遗传进化（B）测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.evolve import _mutate, evolve
from gold.forge import Genome


def _bars(n=800, seed=9, drift=0.0005):
    rng = np.random.default_rng(seed)
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.012, n))
    idx = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.006,
                         "low": close * 0.994, "close": close,
                         "volume": 0.0}, index=idx)


def test_mutate_always_changes_at_least_one_field():
    g = Genome("SMA金叉", {"fast": 20, "slow": 60}, "无过滤", False)
    diffs = 0
    for seed in range(20):
        import random
        c = _mutate(g, random.Random(seed))
        assert c.entry == g.entry          # 进场永不突变
        changed = sum([
            c.params != g.params,
            c.filter != g.filter,
            c.filter != g.filter or c.allow_short != g.allow_short or
        c.params != g.params or (c.sl, c.tp) != (g.sl, g.tp),
            c.allow_short != g.allow_short,
            (c.sl, c.tp) != (g.sl, g.tp),      # 止损止盈也是基因维度
        ])
        diffs += int(changed >= 1)
    assert diffs == 20                      # 每次至少变一处


def test_mutate_keeps_entry_and_valid_params():
    from gold.strata import PARAM_AXES
    import random
    g = Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "低波动", True)
    for seed in range(10):
        c = _mutate(g, random.Random(seed))
        assert c.entry == "唐奇安突破"
        assert c.params in PARAM_AXES["唐奇安突破"] or c.params == g.params


def test_evolve_rejects_short_history():
    r = evolve(_bars(150), generations=2, population=10)
    assert "不足" in r.note
    assert r.best.empty


def test_evolve_runs_and_tracks_evaluations():
    r = evolve(_bars(700), generations=2, population=20, elite=6, seed=3)
    if r.best.empty:
        pytest.skip(f"合成数据未产生有效候选：{r.note}")
    assert r.n_eval > 0
    assert r.generations == 2
    assert len(r.history) >= 1
    assert r.best_sharpe == pytest.approx(float(r.best["OOS夏普"].max()))


def test_evolve_history_monotonic_best():
    """每代记录的最优夏普不应下降（精英保留）。"""
    r = evolve(_bars(800), generations=3, population=20, elite=6, seed=5)
    if len(r.history) < 2:
        pytest.skip("历史不足")
    v = r.history["最优夏普"].to_numpy()
    assert np.all(np.diff(v) >= -1e-9), v


def test_evolve_best_columns():
    r = evolve(_bars(800), generations=2, population=15, elite=5, seed=7)
    if r.best.empty:
        pytest.skip("无候选")
    for c in ("EA", "进场", "过滤", "方向", "OOS收益", "OOS夏普"):
        assert c in r.best.columns


def test_evolve_global_p_bounds():
    r = evolve(_bars(800), generations=2, population=15, elite=5, seed=2)
    if r.best.empty:
        pytest.skip("无候选")
    assert 0.0 <= r.global_p <= 1.0
    assert r.emax > 0


def test_evolve_no_duplicate_labels():
    """同代内不能出现重复基因组（去重靠 seen 集合）。"""
    r = evolve(_bars(800), generations=2, population=20, elite=8, seed=4)
    if r.best.empty:
        pytest.skip("无候选")
    assert r.best["EA"].is_unique


def test_evolve_result_can_be_saved(tmp_path):
    """进化结果必须能落盘。

    回归：过去 `save_run` 直接取 `res.n_sampled` / `res.table`，而 EvolveResult
    只有 `n_eval` / `best`，于是在界面上跑「遗传进化」必定抛
    AttributeError: 'EvolveResult' object has no attribute 'n_sampled'。
    """
    import json

    from gold.forge import save_run

    r = evolve(_bars(800), generations=2, population=12, elite=5, seed=7)
    p = save_run(r, {"mode": "evolve", "tf": "D1"}, out_dir=tmp_path)
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["meta"]["mode"] == "evolve"
    # 进化没有"采样数"，用累计评估次数顶替，且必须对得上
    assert d["n_sampled"] == int(r.n_eval)
    assert isinstance(d["rows"], list)
    if len(r.best):
        assert len(d["rows"]) == min(len(r.best), 200)


def test_save_run_still_works_for_forge_result(tmp_path):
    """随机采样/全枚举的结果落盘不能被这次改动弄坏。"""
    import json

    from gold.forge import forge, save_run

    r = forge(_bars(800), n_candidates=15, n_perm=10, tf="D1")
    p = save_run(r, {"mode": "sample", "n": 15}, out_dir=tmp_path)
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["n_sampled"] == 15
    assert d["space_size"] >= 15
