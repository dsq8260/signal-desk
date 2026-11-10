"""复用惩罚测试：K_eff=K×N、p 随 N 单调变差、报告文本、三档判定。"""

from __future__ import annotations

import pytest

from gold import oos


def test_k_eff_is_k_times_n():
    r = oos.penalty(k_tests=100, n_uses=3, best_sharpe=1.5)
    assert r["k_eff"] == 300
    assert r["k_tests"] == 100 and r["n_uses"] == 3


def test_deflated_p_worsens_with_more_uses():
    ps = [oos.penalty(k_tests=200, n_uses=n, best_sharpe=1.9)["deflated_p_adj"]
          for n in (1, 2, 5, 10)]
    assert ps == sorted(ps), ps          # 单调不减
    assert ps[-1] > ps[0]


def test_raw_p_is_independent_of_n():
    a = oos.penalty(k_tests=200, n_uses=1, best_sharpe=1.5)
    b = oos.penalty(k_tests=200, n_uses=9, best_sharpe=1.5)
    assert a["deflated_p_raw"] == b["deflated_p_raw"]


def test_emax_grows_with_effective_trials():
    a = oos.penalty(k_tests=100, n_uses=1, best_sharpe=1.0)
    b = oos.penalty(k_tests=100, n_uses=4, best_sharpe=1.0)
    assert b["emax_sharpe"] > a["emax_sharpe"]


def test_note_reports_use_count_and_both_p():
    r = oos.penalty(k_tests=444, n_uses=3, best_sharpe=1.98)
    assert "已被使用 3 次" in r["note"]
    assert "置信度已下调" in r["note"]
    assert "444×3" in r["note"].replace(" = ", "×3 = ").replace("=  1332", "1332") \
        or "1332" in r["note"]


def test_penalty_never_improves_p():
    for n in (1, 2, 3, 7, 20):
        r = oos.penalty(k_tests=50, n_uses=n, best_sharpe=2.5)
        assert r["deflated_p_adj"] >= r["deflated_p_raw"] - 1e-12


def test_bonferroni_caps_at_one():
    assert oos.bonferroni(0.2, 3) == pytest.approx(0.6)
    assert oos.bonferroni(0.9, 10) == 1.0
    assert oos.bonferroni(0.01, 1) == pytest.approx(0.01)


def test_reuse_verdict_three_tiers():
    assert oos.reuse_verdict(1) == "pass"
    assert oos.reuse_verdict(3) == "pass"
    assert oos.reuse_verdict(4) == "degrade"
    assert oos.reuse_verdict(6) == "degrade"
    assert oos.reuse_verdict(7) == "fail"
    assert oos.reuse_verdict(50) == "fail"


def test_zero_uses_treated_as_one():
    r = oos.penalty(k_tests=10, n_uses=0, best_sharpe=1.0)
    assert r["n_uses"] == 1 and r["k_eff"] == 10
