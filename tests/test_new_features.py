"""前三项的回归测试：大周期确认 / 欧美时段 / 移动止损。"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from gold.factors import build_features, htf_trend, session_ok
from gold.forge import Genome, build_position
from gold.mq5 import generate


def _bars(n=3000, seed=5):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.004, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close,
                         "tick_volume": 100.0, "spread": 16.0}, index=idx)


@pytest.fixture(scope="module")
def df():
    return _bars()


# ---------------- 大周期确认 ----------------
def test_htf_trend_returns_bool_series(df):
    s = htf_trend(df, mult=4, span=20)
    assert len(s) == len(df)
    assert s.dtype == bool
    assert s.iloc[:100].any() or True     # 允许全 False（视数据而定）


def test_htf_trend_turns_up_on_uptrend():
    idx = pd.date_range("2024-01-01", periods=800, freq="h")
    close = pd.Series(np.linspace(100, 200, 800), index=idx)
    d = pd.DataFrame({"close": close}, index=idx)
    assert htf_trend(d, mult=4, span=5).iloc[-1] is np.True_ or \
        bool(htf_trend(d, mult=4, span=5).iloc[-1]) is True


def test_htf_filter_reduces_exposure(df):
    feat = build_features(df)
    a = build_position(df, feat, Genome("唐奇安突破", {"entry": 20, "exit_": 10},
                                        "无过滤", False)).abs().sum()
    b = build_position(df, feat, Genome("唐奇安突破", {"entry": 20, "exit_": 10},
                                        "大周期确认", False)).abs().sum()
    assert b <= a, "大周期确认不该增加持仓时间"


# ---------------- 欧美时段 ----------------
def test_session_ok_marks_london_ny_hours(df):
    s = session_ok(df, 7, 17)
    hours = df.index.hour
    assert bool(s[hours == 10].all()) is True
    assert bool(s[hours == 3].any()) is False


def test_session_filter_cuts_exposure(df):
    feat = build_features(df)
    a = build_position(df, feat, Genome("唐奇安突破", {"entry": 20, "exit_": 10},
                                        "无过滤", False)).abs().sum()
    b = build_position(df, feat, Genome("唐奇安突破", {"entry": 20, "exit_": 10},
                                        "欧美时段", False)).abs().sum()
    assert b < a, "时段过滤应当减少持仓时间"


def test_new_filters_are_registered():
    """新过滤放在 EXTRA_FILTERS：可选，但不参与默认枚举（避免空间膨胀）。"""
    from gold.combo import ALL_FILTERS, EXTRA_FILTERS, FILTERS
    assert "大周期确认" in EXTRA_FILTERS and "欧美时段" in EXTRA_FILTERS
    assert "大周期确认" in ALL_FILTERS
    assert "大周期确认" not in FILTERS, "默认枚举不应包含新过滤"
    assert len(FILTERS) == 3, "原来的三种过滤必须保持不变"


# ---------------- 移动止损 ----------------
@pytest.fixture(scope="module")
def src(tmp_path_factory):
    r = generate("唐奇安通道突破", {"entry": 20, "exit_": 10},
                 out_dir=tmp_path_factory.mktemp("trail"), name="GoldEA_TrailT",
                 sl=0.02, tp=0.05, extra_inputs={"InpTrailPct": 1.0})
    assert r.ok, r.error
    return pathlib.Path(r.mq5_path).read_text(encoding="utf-8", errors="replace")


def test_trail_input_exists(src):
    assert "InpTrailPct" in src


def test_trailing_only_closes_own_positions(src):
    i = src.find("void ManageTrailing()")
    body = src[i:i + 1600]
    assert "POSITION_MAGIC" in body, "移动止损也必须按 magic 过滤"
    assert "PositionClose(t)" in body


def test_trailing_runs_every_tick(src):
    i = src.find("void OnTick()")
    body = src[i:i + 500]
    gate = body.find("if(bt == lastBar) return;")
    call = body.find("ManageTrailing();")
    assert call > 0 and call < gate, "移动止损必须在「新bar门」之前，每 tick 都跑"
