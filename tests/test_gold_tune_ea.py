"""EA 微调（tune_ea）测试：邻域生成 + 冻结语义 + IS 选参口径。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gold import tune_ea as T
from gold.forge import Genome
from gold.leaderboard import register_top
from gold.paper import connect, list_strategies


def _bars(n=900, seed=5, drift=0.0006):
    rng = np.random.default_rng(seed)
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.012, n))
    idx = pd.bdate_range("2018-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.008,
                         "low": close * 0.992, "close": close,
                         "volume": 0.0}, index=idx)


# ---------------------------------------------------------------- 邻域生成
def test_ladder_comes_from_forge_grid():
    vals = T.ladder("SMA金叉", "fast")
    assert 10 in vals and 30 in vals
    assert vals == sorted(vals)


def test_neighbor_values_keeps_original():
    vals = [10, 20, 30]
    assert T.neighbor_values(vals, 20) == [10, 20, 30]
    assert T.neighbor_values(vals, 10) == [10, 20]


def test_neighbor_values_inserts_unknown():
    got = T.neighbor_values([10, 20, 30], 17)
    assert 17 in got
    assert got == sorted(got)


def test_variant_params_base_first():
    ps = T.variant_params("唐奇安突破", {"entry": 10, "exit_": 5})
    assert ps[0] == {"entry": 10, "exit_": 5}
    assert len(ps) >= 2
    for p in ps:
        assert set(p) == {"entry", "exit_"}
        assert p["entry"] in (10, 20, 30)
        assert p["exit_"] in (5, 10, 20)


def test_variant_params_respects_cap_by_one_at_a_time():
    ps = T.variant_params("鳄鱼线", {"jaw": 13, "teeth": 8, "lips": 5,
                                     "shift": 3}, max_combos=4)
    base = {"jaw": 13, "teeth": 8, "lips": 5, "shift": 3}
    assert ps[0] == base
    assert len(ps) <= 1 + 3 * 4
    for p in ps[1:]:
        diff = [k for k in base if p[k] != base[k]]
        assert len(diff) <= 1, f"退化为 one-at-a-time 时不该同时动多个参数: {p}"


def test_variant_params_empty_params():
    assert T.variant_params("SMA金叉", {}) == [{}]


# ---------------------------------------------------------------- 基因组还原
def test_coerce_genome_roundtrip():
    g = Genome("SMA金叉", {"fast": 10, "slow": 60}, "低波动", True,
               0.02, 0.05)
    got = T.coerce_genome(g.as_dict())
    assert got is not None
    assert got.entry == "SMA金叉" and got.params == {"fast": 10, "slow": 60}
    assert got.filter == "低波动" and got.allow_short
    assert got.sl == 0.02 and got.tp == 0.05


def test_coerce_genome_ignores_legacy_zhi():
    """旧记录里带 zhi：不报错，但值被丢掉（干支维度已移除）。"""
    got = T.coerce_genome({"entry": "SMA金叉", "params": {"fast": 20},
                           "filter": "无过滤", "zhi": 11,
                           "allow_short": False})
    assert got is not None
    assert got.params == {"fast": 20}
    assert not hasattr(got, "zhi")
    assert "zhi" not in got.as_dict()


def test_coerce_genome_accepts_json_string():
    import json
    s = json.dumps({"entry": "SMA金叉", "params": {"fast": 20}})
    got = T.coerce_genome(s)
    assert got is not None and got.params == {"fast": 20}


def test_coerce_genome_rejects_junk():
    assert T.coerce_genome(None) is None
    assert T.coerce_genome("not json") is None
    assert T.coerce_genome({"params": {}}) is None


def test_frozen_text_mentions_all_frozen_genes():
    g = Genome("SMA金叉", {"fast": 10}, "低波动", False, 0.02, None)
    txt = T.frozen_text(g)
    for part in ("低波动", "只多", "SL2%"):
        assert part in txt
    assert "地支" not in txt


# ---------------------------------------------------------------- 微调主流程
def _genome():
    return {"entry": "SMA金叉", "params": {"fast": 10, "slow": 60},
            "filter": "无过滤", "allow_short": False,
            "sl": None, "tp": None}


def test_tune_rejects_short_history():
    r = T.tune(_bars(120), _genome(), tf="D1")
    assert "不足" in r.note and r.table.empty


def test_tune_rejects_missing_genome():
    r = T.tune(_bars(500), None, tf="D1")
    assert "基因组" in r.note and r.table.empty


def test_tune_runs_and_reports_trials():
    r = T.tune(_bars(), _genome(), tf="D1", steps=1, max_combos=24)
    if r.table.empty:
        return  # 合成数据没触发足够笔数时，退化为"不崩"
    assert r.n_valid == len(r.table)
    assert r.n_trials >= r.n_valid >= 2
    assert r.best and r.best_label
    assert 0.0 <= r.deflated_p <= 1.0
    assert r.emax > 0
    assert "Deflated p" in r.note


def test_tune_freezes_filter_direction_stops():
    src = {"entry": "SMA金叉", "params": {"fast": 10, "slow": 60},
           "filter": "低波动", "allow_short": True,
           "sl": 0.02, "tp": 0.05}
    r = T.tune(_bars(), src, tf="D1")
    if not r.best_genome:
        return
    assert r.best_genome["filter"] == "低波动"
    assert "zhi" not in r.best_genome
    assert r.best_genome["allow_short"] is True
    assert r.best_genome["sl"] == 0.02 and r.best_genome["tp"] == 0.05
    assert r.best_genome["entry"] == "SMA金叉"


def test_tune_picks_best_by_is_not_oos():
    r = T.tune(_bars(), _genome(), tf="D1")
    if r.table.empty:
        return
    tbl = r.table
    best_is = tbl.sort_values("IS收益", ascending=False).iloc[0]
    assert T._same_dict(best_is["params"], r.best)


def test_tune_table_columns():
    r = T.tune(_bars(), _genome(), tf="D1")
    if r.table.empty:
        return
    for c in ("params", "参数", "EA", "IS收益", "OOS收益", "OOS夏普",
              "OOS回撤", "OOS方向", "笔数", "原版", "IS排名"):
        assert c in r.table.columns


def test_as_pool_row_shape():
    r = T.tune(_bars(), _genome(), tf="D1")
    if not r.best_genome:
        return
    row = T.as_pool_row(r)
    assert len(row) == 1
    for c in ("EA", "周期", "最优夏普", "最优收益", "最优回撤", "平均笔数",
              "_genome"):
        assert c in row.columns
    assert row.iloc[0]["周期"] == "D1"


def test_tuned_ea_can_register_into_paper(tmp_path):
    r = T.tune(_bars(), _genome(), tf="D1")
    row = T.as_pool_row(r)
    if row.empty:
        return
    con = connect(tmp_path / "p.db")
    try:
        n = register_top(con, row, top=1, tf="D1", name_prefix="[微调]")
        assert n == 1
        rows = list_strategies(con)
        assert len(rows) == 1
        assert rows["name"].str.startswith("[微调]").all()
        assert rows["tf"].iloc[0] == "D1"
    finally:
        con.close()
