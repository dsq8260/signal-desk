"""OOS 账本测试：只增、按区间分桶计数、report_only 不计、完整性能自检。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gold import oos


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    p = tmp_path / "oos_ledger.json"
    monkeypatch.setattr(oos, "LEDGER_PATH", p)
    return p


def _bars(n=400):
    rng = np.random.default_rng(1)
    close = 2600 * np.cumprod(1 + rng.normal(0, 0.002, n))
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close,
                         "volume": 0.0}, index=idx)


def _rec(ledger, run_id="r1", fold=1, purpose="selection",
         start="2024-01-01", end="2024-06-30"):
    return oos.record_use(run_id=run_id, symbol="GOLD", timeframe="H1",
                          scheme="rolling", fold=fold, oos_start=start,
                          oos_end=end, purpose=purpose, n_candidates=100,
                          seed=7, data_sha256="abc",
                          selected={"EA": "x"}, path=ledger)


# ---------------------------------------------------------------- 切分
def test_rolling_folds_are_ordered_and_oos_after_train():
    df = _bars()
    folds = oos.rolling_folds(df, n_folds=4)
    assert len(folds) == 4
    for f in folds:
        tr, oo = oos.slice_fold(df, f)
        assert len(tr) > 0 and len(oo) > 0
        assert tr.index[-1] < oo.index[0], "训练段必须在样本外之前"
    # fold 依次后移
    starts = [f.oos_start for f in folds]
    assert starts == sorted(starts)


def test_rolling_folds_do_not_overlap_their_own_oos():
    df = _bars()
    for f in oos.rolling_folds(df, n_folds=3):
        tr, oo = oos.slice_fold(df, f)
        assert not set(tr.index) & set(oo.index)


def test_fixed_fold_matches_ratio():
    df = _bars(1000)
    # 比例口径：显式 purge=0 时训练段正好 70%
    f0 = oos.fixed_fold(df, 0.7, purge_bars=0)
    tr0, oo0 = oos.slice_fold(df, f0)
    assert abs(len(tr0) / len(df) - 0.7) < 0.01
    assert tr0.index[-1] < oo0.index[0]
    # 默认走自动 purge：训练段会再短一截（这一截就是 purge gap）
    f1 = oos.fixed_fold(df, 0.7)
    tr1, oo1 = oos.slice_fold(df, f1)
    assert len(tr1) == len(tr0) - f1.purge_bars
    assert tr1.index[-1] < oo1.index[0]


def test_short_data_raises():
    with pytest.raises(ValueError):
        oos.rolling_folds(_bars(40), n_folds=4)


def test_fold_dict_is_serialisable():
    f = oos.rolling_folds(_bars(), n_folds=2)[0]
    d = f.as_dict()
    assert json.dumps(d)  # 能进 manifest
    assert set(d) == {"fold", "train", "oos", "purge_bars"}


def test_purge_bars_shrinks_train_end():
    df = _bars()
    a = oos.rolling_folds(df, n_folds=2, purge_bars=0)[0]
    b = oos.rolling_folds(df, n_folds=2, purge_bars=10)[0]
    assert b.train_end < a.train_end


# ---------------------------------------------------------------- 账本
def test_record_use_increments_count(ledger):
    e1 = _rec(ledger, run_id="r1")
    assert e1["n_uses_at_use"] == 1
    assert oos.count_uses("GOLD", "H1", "2024-01-01", "2024-06-30",
                          path=ledger) == 1
    e2 = _rec(ledger, run_id="r2")
    assert e2["n_uses_at_use"] == 2
    assert oos.count_uses("GOLD", "H1", "2024-01-01", "2024-06-30",
                          path=ledger) == 2


def test_different_intervals_are_separate_buckets(ledger):
    _rec(ledger, run_id="r1", start="2024-01-01", end="2024-06-30")
    _rec(ledger, run_id="r2", start="2024-07-01", end="2024-12-31")
    assert oos.count_uses("GOLD", "H1", "2024-01-01", "2024-06-30",
                          path=ledger) == 1
    assert oos.count_uses("GOLD", "H1", "2024-07-01", "2024-12-31",
                          path=ledger) == 1


def test_report_only_does_not_count(ledger):
    _rec(ledger, purpose="report_only")
    assert oos.count_uses("GOLD", "H1", "2024-01-01", "2024-06-30",
                          path=ledger) == 0


def test_duplicate_use_id_rejected(ledger):
    _rec(ledger, run_id="r1", fold=1)
    with pytest.raises(ValueError):
        _rec(ledger, run_id="r1", fold=1)


def test_ledger_is_append_only_no_delete_api():
    for name in dir(oos):
        assert not name.lower().startswith(("delete", "clear", "remove")), name


def test_ledger_survives_corrupt_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError):
        oos.load_ledger(p)


def test_verify_ledger_ok(ledger):
    _rec(ledger, run_id="r1", fold=1)
    _rec(ledger, run_id="r1", fold=2, start="2024-07-01", end="2024-12-31")
    v = oos.verify_ledger(ledger)
    assert v["ok"], v
    assert v["n_entries"] == 2
    assert v["n_buckets"] == 2


def test_verify_ledger_detects_manual_tampering(ledger):
    _rec(ledger, run_id="r1", fold=1)
    data = json.loads(ledger.read_text("utf-8"))
    data["counts"] = {}                    # 人为删掉计数
    ledger.write_text(json.dumps(data), encoding="utf-8")
    v = oos.verify_ledger(ledger)
    assert not v["ok"]
    assert any("计数不一致" in p for p in v["problems"])


def test_verify_ledger_detects_duplicate_use_id(ledger):
    _rec(ledger, run_id="r1", fold=1)
    data = json.loads(ledger.read_text("utf-8"))
    data["entries"].append(dict(data["entries"][0]))
    ledger.write_text(json.dumps(data), encoding="utf-8")
    v = oos.verify_ledger(ledger)
    assert not v["ok"]
    assert any("重复" in p for p in v["problems"])
