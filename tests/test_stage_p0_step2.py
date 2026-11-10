"""第二步验收：切分唯一实现 / 入口接受 fold / run_strict 记账与惩罚。"""

from __future__ import annotations

import glob
import io
import re

import numpy as np
import pandas as pd
import pytest

import gold.forge as forge_mod
import gold.oos as oos
from gold.forge import Genome, forge
from gold.verify import run_strict


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    p = tmp_path / "oos_ledger.json"
    monkeypatch.setattr(oos, "LEDGER_PATH", p)
    return p


def _bars(n=2000, seed=5, drift=0.0003, vol=0.002):
    rng = np.random.default_rng(seed)
    close = 2600 * np.cumprod(1 + rng.normal(drift, vol, n))
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close,
                         "volume": 0.0}, index=idx)


# ---------------------------------------------------------------- 切分唯一实现
def test_split_logic_exists_in_one_place_only():
    """`int(len(df) * ratio)` 这种手工切分只允许出现在 gold/oos.py。"""
    offenders = []
    for f in glob.glob("gold/*.py") + ["app.py"]:
        if f.endswith("oos.py"):
            continue
        src = io.open(f, encoding="utf-8").read()
        for i, ln in enumerate(src.split("\n"), 1):
            if re.search(r"int\(\s*len\(df\)\s*\*", ln):
                offenders.append(f"{f}:{i}: {ln.strip()[:80]}")
    assert not offenders, offenders


def test_split_is_oos_delegates_to_oos():
    from gold.mining import split_is_oos

    src = io.open("gold/mining.py", encoding="utf-8").read()
    i = src.find("def split_is_oos")
    body = src[i:i + 1200]
    assert "gold.oos" in body or "_oos" in body, body[:300]


def test_split_is_oos_matches_fixed_fold():
    from gold.mining import split_is_oos

    df = _bars(1000)
    tr, oo = split_is_oos(df, 0.7)          # 默认 purge=None → 自动
    fold = oos.fixed_fold(df, 0.7)          # 同上，两者必须一致
    tr2, oo2 = oos.slice_fold(df, fold)
    assert list(tr.index) == list(tr2.index)
    assert list(oo.index) == list(oo2.index)


def test_split_is_oos_supports_purge():
    from gold.mining import split_is_oos

    df = _bars(1000)
    _, oo_a = split_is_oos(df, 0.7, purge_bars=0)
    tr_b, _ = split_is_oos(df, 0.7, purge_bars=20)
    assert len(tr_b) == len(df) - (len(df) - int(len(df) * 0.7)) - 20 - 1 + 1 - 1 \
        or len(tr_b) < len(df) - (len(df) - int(len(df) * 0.7))
    assert tr_b.index[-1] < oo_a.index[0]


# ---------------------------------------------------------------- 入口接受 fold
def test_all_mining_entries_accept_fold():
    import inspect

    import gold.combo
    import gold.evolve
    import gold.mining
    import gold.strata
    import gold.tune_ea

    for mod, fn in ((forge_mod, "forge"), (gold.evolve, "evolve"),
                    (gold.strata, "search"), (gold.combo, "rank"),
                    (gold.tune_ea, "tune"), (gold.mining, "mine")):
        sig = inspect.signature(getattr(mod, fn))
        assert "fold" in sig.parameters, f"{fn} 缺少 fold 参数"


def test_forge_uses_given_fold_boundaries(monkeypatch):
    """forge(fold=...) 必须把该 fold 的 train/oos 原样传给评测，不能自己再切。"""
    df = _bars(2000)
    fold = oos.rolling_folds(df, n_folds=3)[-1]
    train, expect_oos = oos.slice_fold(df, fold)
    captured = {}
    real = forge_mod.evaluate_genome

    def spy(d, feat, is_df, oos_df, fee, g, *a, **kw):
        captured["is"] = is_df
        captured["oos"] = oos_df
        return real(d, feat, is_df, oos_df, fee, g, *a, **kw)

    monkeypatch.setattr(forge_mod, "evaluate_genome", spy)
    forge(df, n_candidates=6, n_perm=0, fold=fold, tf="H1")
    assert list(captured["is"].index) == list(train.index)
    assert list(captured["oos"].index) == list(expect_oos.index)


def test_forge_records_ledger_when_fold_and_run_id(ledger):
    df = _bars(1200)
    fold = oos.rolling_folds(df, n_folds=2)[-1]
    forge(df, n_candidates=4, n_perm=0, fold=fold, run_id="r-test-1", tf="H1")
    assert oos.count_uses("GOLD", "H1", fold.oos_start, fold.oos_end,
                          path=ledger) == 1
    # 再跑一次 → 复用次数 +1
    forge(df, n_candidates=4, n_perm=0, fold=fold, run_id="r-test-2", tf="H1")
    assert oos.count_uses("GOLD", "H1", fold.oos_start, fold.oos_end,
                          path=ledger) == 2


# ---------------------------------------------------------------- run_strict
def test_run_strict_outputs_fold_rows_and_summary(ledger):
    df = _bars(2400)
    gs = [Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False),
          Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "无过滤", False)]
    t = run_strict(df, gs, tf="H1", run_id="strict-1", n_folds=3, n_perm=10)
    if t.empty:
        pytest.skip("合成数据没有可用候选")
    for c in ("EA", "折数", "正收益折", "OOS夏普均值", "OOS夏普最差",
              "方向均值", "K_eff", "deflated_p_adj", "复用判定"):
        assert c in t.columns, c
    assert t.attrs["note"]
    assert len(t.attrs["folds"]) == 3
    # 跨 fold 只用均值/最差/计数，不存在"最优夏普"这种拼接列
    assert "最优夏普" not in t.columns


def test_run_strict_records_one_use_per_fold(ledger):
    df = _bars(2400)
    gs = [Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False)]
    t = run_strict(df, gs, tf="H1", run_id="strict-2", n_folds=3, n_perm=10)
    folds = t.attrs.get("folds") or oos.rolling_folds(df, n_folds=3)
    total = sum(oos.count_uses("GOLD", "H1", f["oos"][0], f["oos"][1],
                               path=ledger) for f in folds)
    assert total >= 1
    assert oos.verify_ledger(ledger)["ok"]


def test_run_strict_penalty_grows_when_rerun(ledger):
    df = _bars(2400)
    gs = [Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False)]
    t1 = run_strict(df, gs, tf="H1", run_id="p1", n_folds=2, n_perm=5)
    t2 = run_strict(df, gs, tf="H1", run_id="p2", n_folds=2, n_perm=5)
    if t1.empty or t2.empty:
        pytest.skip("无候选")
    assert int(t2["K_eff"].iloc[0]) > int(t1["K_eff"].iloc[0])
    assert "已被使用 2 次" in t2.attrs["note"]


def test_outward_fold_is_last_and_has_constant_train_length():
    """训练窗口长度恒定；只有被数据起点 clamp 的折会短一些。"""
    df = _bars(2000)
    folds = oos.rolling_folds(df, n_folds=4)
    last = oos.outward_fold(df, n_folds=4)
    assert last.oos_start == folds[-1].oos_start
    free = [len(oos.slice_fold(df, f)[0]) for f in folds
            if f.train_start > df.index[0]]
    if len(free) >= 2:
        assert max(free) - min(free) <= 2, free
