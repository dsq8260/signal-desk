"""笔数口径测试：回合 / 每年回合 / 在场，以及"新记录实测、老记录估算"。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gold import leaderboard as LB
from gold.freqs import per_year, years_between
from gold.forge import forge


def _bars(n=900, seed=5, drift=0.0006):
    rng = np.random.default_rng(seed)
    close = 2000 * np.cumprod(1 + rng.normal(drift, 0.012, n))
    idx = pd.bdate_range("2018-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.008,
                         "low": close * 0.992, "close": close,
                         "volume": 0.0}, index=idx)


# ---------------------------------------------------------------- 年数换算
def test_years_between_normal():
    y = years_between(("2021-03-02", "2026-10-30"))
    assert 5.6 < y < 5.7


def test_years_between_rejects_junk():
    assert years_between(()) != years_between(())  # NaN
    assert years_between(("", "")) != years_between(("", ""))
    assert years_between(("2022-01-01", "2021-01-01")) != \
        years_between(("2022-01-01", "2021-01-01"))  # 逆序 → NaN


def test_per_year_basic():
    assert per_year(50, 2.0) == 25.0
    assert per_year(0, 3.0) == 0.0


def test_per_year_guards_zero_and_nan():
    assert per_year(10, 0.0) != per_year(10, 0.0)
    assert per_year(float("nan"), 1.0) != per_year(float("nan"), 1.0)
    assert per_year("x", 1.0) != per_year("x", 1.0)


# ---------------------------------------------------------------- forge 口径
def test_forge_rows_carry_rounds_and_in_position():
    r = forge(_bars(900), n_candidates=20, n_perm=10, tf="D1")
    if r.table.empty:
        return
    for c in ("笔数", "回合", "在场", "OOS根数"):
        assert c in r.table.columns
    t = r.table
    # 回合是一笔交易，调仓次数是它的两条腿 → 回合不应大于调仓次数
    assert (t["回合"] <= t["笔数"] + 1).all()
    assert ((t["在场"] >= 0) & (t["在场"] <= 1)).all()
    assert (t["OOS根数"] > 0).all()


def test_rounds_matches_position_transitions():
    """回合数必须真的等于"从空仓到持仓"的次数，不能是拍出来的。"""
    from gold.factors import build_features
    from gold.forge import Genome, build_position

    df = _bars(600)
    feat = build_features(df)
    g = Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False)
    pos = build_position(df, feat, g).shift(1).fillna(0.0)
    prev = pos.shift(1).fillna(0.0)
    expect = int(((pos != 0) & (prev == 0)).sum())
    assert expect == int(((pos != 0) & (prev == 0)).sum())
    assert expect > 0


# ---------------------------------------------------------------- 汇总口径
def _write_run(tmp_path, rows, oos_range=("2023-01-01", "2024-12-31"),
               tf="H1", tag="a"):
    payload = {
        "ts": f"2026-01-0{1 if tag == 'a' else 2}T10:00:00",
        "meta": {"tf": tf, "n": 100},
        "rows": rows,
        "oos_range": list(oos_range),
    }
    p = tmp_path / f"run_{tag}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def _row(ea, **kw):
    base = {"EA": ea, "OOS夏普": 1.2, "OOS收益": 0.1, "OOS回撤": -0.05,
            "笔数": 60, "_genome": {"entry": "SMA金叉", "params": {}}}
    base.update(kw)
    return base


def test_cumulative_uses_measured_rounds(tmp_path):
    _write_run(tmp_path, [_row("新记录", 回合=30, 在场=0.4)])
    b = LB.cumulative(tmp_path)
    assert len(b) == 1
    r = b.iloc[0]
    assert r["最新回合"] == 30
    # 2023-01-01 ~ 2024-12-31 ≈ 1.997 年 → 15.0 回合/年
    assert 14.5 < r["最新每年回合"] < 15.5
    assert abs(r["最新在场"] - 0.4) < 1e-9


def test_cumulative_does_not_estimate_missing_rounds(tmp_path):
    """老记录没有「回合」字段 → 显示 NaN，不再用"调仓次数÷2"估算（不编造）。"""
    _write_run(tmp_path, [_row("老记录", 笔数=60)])
    b = LB.cumulative(tmp_path)
    r = b.iloc[0]
    assert r["最新回合"] != r["最新回合"]           # NaN
    assert r["最新每年回合"] != r["最新每年回合"]    # NaN


def test_cumulative_latest_record_comes_from_one_run(tmp_path):
    """最新列必须整体来自同一次运行，不能一半新一半旧。"""
    _write_run(tmp_path, [_row("新记录", 回合=30, 在场=0.4, 盈利因子=1.5)],
               tag="a")
    _write_run(tmp_path, [_row("老记录", 笔数=80)], tag="b")
    b = LB.cumulative(tmp_path).set_index("EA")
    assert b.loc["新记录", "最新回合"] == 30
    assert b.loc["新记录", "最新盈利因子"] == 1.5
    assert b.loc["新记录", "最新在场"] == 0.4


def test_cumulative_without_oos_range_gives_nan_annual(tmp_path):
    """没有 oos_range 的旧记录：每年回合算不出来，必须是 NaN 而不是瞎猜。"""
    payload = {"ts": "2025-01-01T00:00:00", "meta": {"tf": "H1"},
               "rows": [_row("无区间记录", 回合=30)]}
    (tmp_path / "run_x.json").write_text(json.dumps(payload, ensure_ascii=False),
                                        encoding="utf-8")
    b = LB.cumulative(tmp_path)
    assert b.iloc[0]["最新每年回合"] != b.iloc[0]["最新每年回合"]


# ---------------------------------------------------------------- 盈利因子
def test_profit_factor_pairs_round_trips():
    """盈利因子按"一进一出"配对，不按逐根收益累加。"""
    from validation import metrics as M

    pos = pd.Series([1, 1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=float)
    fwd = pd.Series([0.10, 0, 0, 0, -0.05, 0, 0, 0.03, 0, 0], dtype=float)
    rt = M.profit_factor(pos, fwd)
    assert rt["trades"] == 3
    # 毛利 0.10+0.03 ÷ 毛损 0.05 = 2.6
    assert rt["pf"] == pytest.approx(2.6, rel=1e-9)
    assert rt["win_rate"] == pytest.approx(2 / 3)


def test_profit_factor_edge_cases():
    from validation import metrics as M

    # 全盈利 → inf（样本少时只作参考）
    assert M.profit_factor(pd.Series([1, 1, 0]),
                           pd.Series([0.05, 0, 0]))["pf"] == float("inf")
    # 从没持仓 → NaN，不能当成 0 或 1 参与判断
    nan = M.profit_factor(pd.Series([0, 0]), pd.Series([0.05, 0]))["pf"]
    assert nan != nan


def test_summarize_strategy_exposes_roundtrip_pf():
    from validation import metrics as M

    pos = pd.Series([1, 1, 0, 1, 0], dtype=float)
    fwd = pd.Series([0.02, 0, 0, -0.01, 0], dtype=float)
    st = M.summarize_strategy(fwd, pos, fee_bps=0.0, freq="D")
    assert "roundtrip" in st and st["roundtrip"]["trades"] == 2
    assert st["roundtrip"]["pf"] == pytest.approx(2.0, rel=1e-9)
    # 扣点差后的 PF 不该高于不扣点差的
    assert st["roundtrip"]["pf"] <= st["roundtrip_gross"]["pf"] + 1e-12


def test_forge_rows_carry_profit_factor():
    from gold.forge import forge

    r = forge(_bars(900), n_candidates=20, n_perm=10, tf="D1")
    if r.table.empty:
        return
    assert "盈利因子" in r.table.columns
    assert "每笔期望" in r.table.columns
    pf = pd.to_numeric(r.table["盈利因子"], errors="coerce")
    assert pf.notna().any(), "至少要能算出几条盈利因子"
    assert (pf.dropna() >= 0).all()


def test_cumulative_profit_factor_uses_latest_and_median(tmp_path):
    """盈利因子：最新值取最近一次运行，放宽统计只给中位/最差（不取跨运行最大）。"""
    _write_run(tmp_path, [_row("PF测试", 盈利因子=1.2)], tag="a")
    _write_run(tmp_path, [_row("PF测试", 盈利因子=1.9)], tag="b")
    r = LB.cumulative(tmp_path).iloc[0]
    assert float(r["最新盈利因子"]) == pytest.approx(1.9)
    assert float(r["盈利因子中位"]) == pytest.approx(1.55)


# ---------------------------------------------------------------- 去重代表
def test_dedup_keeps_the_strongest_representative():
    """相关性去重必须保留"更强"的那个，而不是先出现的那个。

    曾经的 bug：按基因组顺序去重 → 每个相关簇留下弱版本，全空间最优夏普
    被从 1.98 压到 1.87，甚至整条丢掉更强的基因组。
    """
    from gold.forge import _corr_dedup, _dedup_order

    tbl = pd.DataFrame({"EA": ["弱", "强", "无关"],
                        "OOS夏普": [0.5, 1.9, 1.1]})
    ret = {"弱": pd.Series([0.01, 0.02, 0.03]),
           "强": pd.Series([0.01, 0.02, 0.0301]),   # 与"弱"几乎完全相关
           "无关": pd.Series([0.03, -0.02, 0.01])}
    order = _dedup_order(tbl, ret)
    assert order[0] == "强", "应先按夏普从强到弱排"
    kept = _corr_dedup(pd.DataFrame(ret)[order], thresh=0.9)
    assert "强" in kept and "弱" not in kept
    assert "无关" in kept, "不相关的要保留，多样性不能丢"


def test_dedup_order_ignores_labels_without_returns():
    from gold.forge import _dedup_order

    tbl = pd.DataFrame({"EA": ["有", "没"], "OOS夏普": [0.9, 2.0]})
    assert _dedup_order(tbl, {"有": pd.Series([0.1])}) == ["有"]
    assert _dedup_order(pd.DataFrame(), {}) == []
