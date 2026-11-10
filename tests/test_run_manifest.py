"""运行清单测试：字段齐全、可复现、指纹随数据变化、状态流转正确。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gold import runlog


@pytest.fixture()
def df():
    rng = np.random.default_rng(3)
    close = 2600 * np.cumprod(1 + rng.normal(0.0004, 0.0018, 300))
    idx = pd.bdate_range("2023-01-02", periods=300)
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close,
                         "volume": 0.0}, index=idx)


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    d = tmp_path / "runs"
    monkeypatch.setattr(runlog, "RUNS_DIR", d)
    return d


def _start(**kw):
    base = dict(entry="test", reproduce_cmd="python run_forge.py --n 10",
                data={"symbol": "GOLD", "timeframe": "H1", "sha256": "abc123def"},
                cost={"spread_usd": 0.33, "spread_source": "fixed"},
                search={"n_candidates": 10, "seed": 7},
                gates={"min_dir": 0.55}, oos={"scheme": "rolling"}, seed=7)
    base.update(kw)
    return runlog.start_run(**base)


def test_manifest_has_all_required_fields(runs_dir, df):
    m = _start(data=runlog.data_fingerprint("GOLD", "H1", df=df))
    saved = json.loads((runs_dir / f"{m['run_id']}.json").read_text("utf-8"))
    for key in ("run_id", "status", "started_at", "finished_at", "entry",
                "reproduce_cmd", "seed", "code", "data", "cost", "search",
                "oos", "gates", "versions", "outcome"):
        assert key in saved, key
    assert saved["status"] == "running"
    assert saved["seed"] == 7
    assert saved["data"]["hash_scope"] in ("file", "derived")
    assert saved["data"]["sha256"]


def test_seed_lands_in_run_id(runs_dir):
    m = _start(seed=42)
    assert "_42_" in m["run_id"]


def test_fingerprint_changes_when_data_changes(df):
    a = runlog.data_fingerprint("GOLD", "H1", df=df)
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] *= 1.0001
    b = runlog.data_fingerprint("GOLD", "H1", df=df2)
    assert a["sha256"] != b["sha256"]
    assert a["hash_scope"] == "derived"


def test_fingerprint_same_data_is_stable(df):
    a = runlog.data_fingerprint("GOLD", "H1", df=df)
    b = runlog.data_fingerprint("GOLD", "H1", df=df.copy())
    assert a["sha256"] == b["sha256"]


def test_fingerprint_uses_file_hash_when_path_given(tmp_path, df):
    p = tmp_path / "bars.csv"
    df.to_csv(p)
    fp = runlog.data_fingerprint("GOLD", "H1", path=p, df=df)
    assert fp["hash_scope"] == "file"
    assert fp["sha256"] == runlog.file_sha256(p)
    assert fp["bytes"] > 0


def test_fingerprint_falls_back_when_path_missing(tmp_path, df):
    fp = runlog.data_fingerprint("GOLD", "H1",
                                 path=tmp_path / "nope.csv", df=df)
    assert fp["hash_scope"] == "derived"
    assert fp.get("path_missing") is True


def test_finish_run_marks_done_and_keeps_outcome(runs_dir):
    m = _start()
    runlog.finish_run(m, outcome={"n_folds": 4, "final_sharpe": 1.2},
                      ledger_write="ok")
    saved = runlog.load_manifest(m["run_id"])
    assert saved["status"] == "done"
    assert saved["finished_at"]
    assert saved["outcome"]["final_sharpe"] == 1.2
    assert saved["ledger_write"] == "ok"


def test_fail_run_records_error_not_success(runs_dir):
    m = _start()
    runlog.fail_run(m, error="ValueError: boom")
    saved = runlog.load_manifest(m["run_id"])
    assert saved["status"] == "failed"
    assert "boom" in saved["error"]
    assert saved["outcome"] is None


def test_list_runs_sorted_and_tolerates_broken_file(runs_dir):
    _start()
    (runs_dir / "broken.json").write_text("{not json", encoding="utf-8")
    got = runlog.list_runs()
    assert len(got) == 2
    assert any(r.get("status") == "unreadable" for r in got)


def test_versions_records_numeric_libs():
    v = runlog.versions()
    assert v["python"]
    for mod in ("numpy", "pandas", "scipy"):
        assert mod in v


def test_git_describe_never_raises():
    assert isinstance(runlog.git_describe(), str)


def test_atomic_write_leaves_no_tmp_file(runs_dir):
    _start()
    assert not list(runs_dir.glob("*.tmp"))
