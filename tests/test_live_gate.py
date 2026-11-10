"""实盘就绪判定的测试：缺数据不得判"可上"，最差折与滑点必须真参与。"""

from __future__ import annotations

import pytest

from gold.live_gate import DEFAULTS, live_readiness, robust_score

GOOD = dict(oos_months=9, max_drawdown=-0.08, folds_positive=3, folds_total=3,
            worst_fold_return=0.01, return_with_slippage=0.12,
            rounds_per_year=60, pbo=0.3)


def test_all_good_is_live_ready():
    v = live_readiness(**GOOD)
    assert v.ok and v.verdict == "可上模拟盘"
    assert all(c.ok for c in v.checks)
    assert v.reasons == []


def test_missing_data_never_passes():
    v = live_readiness(**{**GOOD, "pbo": None})
    assert v.verdict == "再观察", v.verdict
    assert any("CSCV-PBO" in r for r in v.reasons)


def test_all_missing_is_unmeasured():
    v = live_readiness()
    assert v.verdict == "再观察"
    assert len(v.reasons) == len(v.checks)


def test_drawdown_breach_blocks():
    v = live_readiness(**{**GOOD, "max_drawdown": -0.35})
    assert v.verdict == "不可实盘"
    assert any("最大回撤" in r for r in v.reasons)


def test_worst_fold_breach_blocks():
    """平均为正但最差一折亏 20% → 不可实盘（实盘会先把人熬死）。"""
    v = live_readiness(**{**GOOD, "worst_fold_return": -0.20})
    assert v.verdict == "不可实盘"


def test_slippage_erosion_blocks():
    v = live_readiness(**{**GOOD, "return_with_slippage": -0.02})
    assert v.verdict == "不可实盘"
    assert any("滑点" in r for r in v.reasons)


def test_high_turnover_blocks():
    v = live_readiness(**{**GOOD, "rounds_per_year": 900})
    assert v.verdict == "不可实盘"


def test_too_few_rounds_blocks():
    v = live_readiness(**{**GOOD, "rounds_per_year": 5})
    assert v.verdict == "不可实盘"


def test_custom_standard():
    v = live_readiness(**{**GOOD, "max_drawdown": -0.25},
                       std={"max_drawdown": 0.30})
    assert v.verdict == "可上模拟盘"


def test_robust_score_prefers_consistency():
    steady = robust_score([0.05, 0.06, 0.04])
    swingy = robust_score([0.30, -0.20, 0.05])
    assert steady > swingy, (steady, swingy)


def test_robust_score_empty_is_neg_inf():
    assert robust_score([]) == float("-inf")
    assert robust_score([None, None]) == float("-inf")


def test_defaults_are_sane():
    assert 0 < DEFAULTS["max_drawdown"] < 1
    assert DEFAULTS["min_folds_positive"] >= 2
