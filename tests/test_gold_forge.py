"""自主设计（forge）测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.forge import (
    Genome,
    build_position,
    forge,
    genome_space,
    register_survivors,
    sample_genomes,
)
from gold.paper import connect, list_strategies


def _bars(n=700, seed=5, drift=0.0004):
    rng = np.random.default_rng(seed)
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.012, n))
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.006,
                         "low": close * 0.994, "close": close,
                         "volume": 0.0}, index=idx)


def test_genome_label():
    g = Genome("SMA金叉", {"fast": 20}, "低波动", True)
    lab = g.label()
    assert "SMA金叉" in lab and "低波动" in lab
    assert "双向" in lab
    # 干支已从基因组移除，标签里不能再有地支字样
    assert not any(z in lab for z in "子丑寅卯辰巳午未申酉戌亥")


def test_genome_space_size():
    space = genome_space((False,))
    assert len(space) > 100
    assert all(isinstance(g, Genome) for g in space)


def test_sample_is_deterministic():
    a = sample_genomes(20, seed=1)
    b = sample_genomes(20, seed=1)
    assert [g.label() for g in a] == [g.label() for g in b]


def test_sample_respects_allow_short():
    gs = sample_genomes(30, seed=2, allow_short=False)
    assert all(g.allow_short is False for g in gs)


def test_sample_caps_at_space_size():
    space = genome_space((False,))
    gs = sample_genomes(len(space) + 50, allow_short=False)
    assert len(gs) == len(space)


def test_build_position_long_only():
    from gold.factors import build_features

    df = _bars(300)
    feat = build_features(df)
    g = Genome("SMA金叉", {"fast": 10, "slow": 40}, "无过滤", False)
    pos = build_position(df, feat, g)
    assert set(np.unique(pos)).issubset({0.0, 1.0})


def test_build_position_both_sides():
    from gold.factors import build_features

    df = _bars(300)
    feat = build_features(df)
    g = Genome("SMA金叉", {"fast": 10, "slow": 40}, "无过滤", True)
    pos = build_position(df, feat, g)
    assert set(np.unique(pos)).issubset({-1.0, 1.0})


def test_forge_rejects_short_history():
    r = forge(_bars(120), n_candidates=10)
    assert "不足" in r.note
    assert r.table.empty


def test_forge_runs_and_reports_counts():
    r = forge(_bars(800), n_candidates=40, n_perm=10)
    if r.table.empty:
        pytest.skip(f"合成数据未产生有效候选：{r.note}")
    assert r.n_sampled == 40
    assert r.n_valid >= len(r.table) >= 1
    assert r.n_dedup <= r.n_valid
    assert r.n_pass <= r.n_dedup
    assert 0.0 <= r.global_p <= 1.0
    assert r.emax > 0


def test_forge_table_columns():
    r = forge(_bars(800), n_candidates=30, n_perm=10)
    if r.table.empty:
        pytest.skip("无候选")
    for c in ("EA", "进场", "过滤", "方向", "OOS收益", "OOS夏普", "q"):
        assert c in r.table.columns


def test_survivors_sorted_by_sharpe():
    r = forge(_bars(900), n_candidates=60, n_perm=10)
    if len(r.survivors) < 2:
        pytest.skip("过关候选不足")
    v = r.survivors["OOS夏普"].to_numpy()
    assert np.all(np.diff(v) <= 1e-12)


def test_register_survivors_into_paper(tmp_path):
    r = forge(_bars(900), n_candidates=60, n_perm=10)
    if len(r.survivors) == 0:
        pytest.skip("没有过关候选可登记")
    con = connect(tmp_path / "p.db")
    try:
        n = register_survivors(con, r, top=3, tf="D1")
        assert n >= 1
        rows = list_strategies(con)
        assert len(rows) == n
        assert rows["name"].str.startswith("[自研]").all()
    finally:
        con.close()


def test_dedup_reduces_count():
    """去重后数量不应超过有效数量。"""
    r = forge(_bars(1000), n_candidates=80, n_perm=10, dedup_thresh=0.99)
    if r.table.empty:
        pytest.skip("无候选")
    assert r.n_dedup <= r.n_valid
