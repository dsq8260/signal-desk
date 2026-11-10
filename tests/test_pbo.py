"""CSCV-PBO 的统计性质测试（合成数据，已知答案）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold import pbo as P


def _noise(T=1200, N=40, seed=7):
    """纯噪声：所有配置均值 0、无预测力 → PBO 应该很高（接近 0.5~1）。"""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.01, size=(T, N))


def _one_winner(T=1200, N=40, seed=11):
    """一个配置真实占优（恒正漂移）→ 样本内挑它，样本外仍最好 → PBO 应很低。"""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, size=(T, N))
    r[:, 0] += 0.004
    return r


def test_pbo_rejects_too_few_configs():
    out = P.cscv_pbo(_noise(N=1))
    assert out["ok"] is False and "至少需要 2 个配置" in out["reason"]


def test_pbo_rejects_short_sample():
    out = P.cscv_pbo(_noise(T=20, N=5))
    assert out["ok"] is False and "样本太短" in out["reason"]


def test_pbo_high_on_pure_noise():
    out = P.cscv_pbo(_noise(), n_blocks=10)
    assert out["ok"]
    assert out["pbo"] >= 0.3, f"纯噪声 PBO 不该很低：{out['pbo']}"
    assert 0.0 <= out["pbo"] <= 1.0


def test_pbo_low_when_one_config_truly_better():
    out = P.cscv_pbo(_one_winner(), n_blocks=10)
    assert out["ok"]
    assert out["pbo"] <= 0.25, f"有真实赢家时 PBO 应低：{out['pbo']}"


def test_pbo_deterministic():
    a = P.cscv_pbo(_noise(), n_blocks=8)
    b = P.cscv_pbo(_noise(), n_blocks=8)
    assert a["pbo"] == b["pbo"]
    assert a["n_splits"] == b["n_splits"]


def test_pbo_accepts_dataframe_and_handles_nan():
    df = pd.DataFrame(_noise(T=600, N=8))
    df.iloc[3, 2] = np.nan
    df.iloc[10, 5] = np.inf
    out = P.cscv_pbo(df, n_blocks=8)
    assert out["ok"] and np.isfinite(out["pbo"])


def test_pbo_from_frames_aligns_and_reports():
    idx = pd.bdate_range("2025-01-01", periods=600)
    frames = {f"cfg{i}": pd.Series(np.random.default_rng(i).normal(0, 0.01, 600),
                                   index=idx) for i in range(6)}
    out = P.pbo_from_returns_by_config(frames, n_blocks=8)
    assert out["ok"] and 0 <= out["pbo"] <= 1
    assert len(out["configs"]) == 6


def test_pbo_from_frames_rejects_empty():
    assert P.pbo_from_returns_by_config({})["ok"] is False


def test_pbo_metric_switch():
    m = P.cscv_pbo(_noise(N=12), n_blocks=8, metric="mean")
    s = P.cscv_pbo(_noise(N=12), n_blocks=8, metric="sharpe")
    assert m["ok"] and s["ok"]
    assert m["metric"] == "mean" and s["metric"] == "sharpe"


def test_odd_block_count_is_rounded_up():
    out = P.cscv_pbo(_noise(N=8), n_blocks=7)
    assert out["ok"] and out["n_blocks"] == 8
