"""第四步验收：purge gap 必须真的切出来，且训练/样本外永不重叠。

为什么要 purge：训练段最后几根 bar 的指标（RSRS m=250、强趋势 250、时变夏普 120）
会"看到"样本外开头那段价格；不切掉，样本外就已经被训练段污染了。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold import oos
from gold.forge import Genome
from gold.verify import run_strict


@pytest.fixture()
def df():
    rng = np.random.default_rng(7)
    n = 3000
    close = 2600 * np.cumprod(1 + rng.normal(0.0003, 0.0019, n))
    idx = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close,
                         "volume": 0.0}, index=idx)


def _gap(df, fold):
    tr, oo = oos.slice_fold(df, fold)
    return int(((df.index > tr.index[-1]) & (df.index < oo.index[0])).sum())


# ---------------------------------------------------------------- 自动推导
def test_auto_purge_uses_longest_indicator_window():
    assert oos.PURGE_LOOKBACK == 250
    assert oos.auto_purge_bars(33508) == 250


def test_auto_purge_shrinks_on_short_data():
    assert oos.auto_purge_bars(1000) == 100        # 10%
    assert oos.auto_purge_bars(100) == 10          # 下限
    assert oos.auto_purge_bars(10) == 10


def test_default_purge_is_auto_not_zero(df):
    f = oos.rolling_folds(df, n_folds=3)[0]
    assert f.purge_bars == oos.auto_purge_bars(len(df)) == 250


def test_explicit_zero_still_allowed(df):
    f = oos.rolling_folds(df, n_folds=2, purge_bars=0)[0]
    assert f.purge_bars == 0
    assert _gap(df, f) == 0


def test_fixed_fold_also_gets_purge(df):
    f = oos.fixed_fold(df, 0.7)
    assert f.purge_bars == 250
    assert _gap(df, f) == 250


def test_outward_fold_gets_purge(df):
    f = oos.outward_fold(df, n_folds=4)
    assert f.purge_bars == 250


# ---------------------------------------------------------------- gap 真实性
def test_gap_length_equals_declared_purge_for_every_fold(df):
    for f in oos.rolling_folds(df, n_folds=4):
        assert _gap(df, f) == int(f.purge_bars), f


def test_train_and_oos_never_overlap(df):
    for f in oos.rolling_folds(df, n_folds=4):
        tr, oo = oos.slice_fold(df, f)
        assert not set(tr.index) & set(oo.index)
        assert tr.index[-1] < oo.index[0]


def test_purge_larger_means_shorter_train(df):
    lengths = [len(oos.slice_fold(df, oos.rolling_folds(
        df, n_folds=2, purge_bars=p)[0])[0]) for p in (0, 100, 250)]
    assert lengths[0] > lengths[1] > lengths[2], lengths


def test_slice_fold_rejects_insufficient_gap(df):
    """声明 purge 比实际 gap 大时必须报错，而不是悄悄放过。"""
    f = oos.rolling_folds(df, n_folds=3, purge_bars=250)[0]
    f.purge_bars = f.purge_bars + 10000        # 人为把声明值改大
    with pytest.raises(ValueError, match="purge gap 不足"):
        oos.slice_fold(df, f)


def test_as_dict_records_purge(df):
    d = oos.rolling_folds(df, n_folds=2)[0].as_dict()
    assert d["purge_bars"] == 250


# ---------------------------------------------------------------- 主流程
def test_run_strict_uses_purge_by_default(df, tmp_path, monkeypatch):
    p = tmp_path / "ledger.json"
    monkeypatch.setattr(oos, "LEDGER_PATH", p)
    gs = [Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "无过滤", False)]
    t = run_strict(df, gs, tf="H1", run_id="purge-1", n_folds=3, n_perm=5)
    if t.empty:
        pytest.skip("合成数据没有可用候选")
    folds = t.attrs["folds"]
    assert all(f["purge_bars"] == 250 for f in folds), folds


def test_manifest_folds_record_purge(df, tmp_path, monkeypatch):
    """清单里必须能看到 purge_bars —— 否则审计时无法证明切干净了。"""
    from gold import runlog

    monkeypatch.setattr(runlog, "RUNS_DIR", tmp_path / "runs")
    m = runlog.start_run(entry="t", reproduce_cmd="x",
                         data={"symbol": "GOLD", "timeframe": "H1",
                               "sha256": "abc"},
                         cost={}, search={}, gates={},
                         oos={"scheme": "rolling",
                              "folds": [oos.rolling_folds(df, n_folds=2)[0]
                                        .as_dict()]},
                         seed=1)
    saved = runlog.load_manifest(m["run_id"])
    assert saved["oos"]["folds"][0]["purge_bars"] == 250
