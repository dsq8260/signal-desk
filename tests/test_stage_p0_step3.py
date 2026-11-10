"""第三步验收：参数平台期门槛 + 结构化 reject_reason。

关键场景：
- **孤立尖峰**（中心好、邻居全差）→ 必须被"参数平台期"这条门槛拒掉；
- **平台**（中心好、邻居也大多同号）→ 通过；
- 拒绝原因必须结构化（哪条门槛 / 实际值 / 阈值）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.forge import Genome
from gold.verify import (
    gates_to_reasons,
    make_gates,
    plateau_check,
    reasons_text,
    run_strict,
    verdict_of,
)


@pytest.fixture()
def df():
    rng = np.random.default_rng(11)
    n = 2600
    close = 2600 * np.cumprod(1 + rng.normal(0.00035, 0.0019, n))
    idx = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.0012,
                         "low": close * 0.9988, "close": close,
                         "volume": 0.0}, index=idx)


def test_make_gates_has_plateau_gate():
    rec = {"方向": 0.58, "超额": 0.05, "回撤": -0.08, "盈利因子": 1.8,
           "每年回合": 80, "块置换p": 0.01, "q": 0.02, "deflated_p": 0.03}
    gates = make_gates(rec, n_symbols_passed=3, n_symbols_tested=5,
                       wf_positive=3, wf_total=3, same_sign_rate=0.8,
                       pbo=0.2)          # 第 12 条：CSCV-PBO ≤0.5
    names = [g.name for g in gates]
    assert "参数平台期" in names
    assert "CSCV-PBO" in names
    assert len(gates) == 12
    assert all(g.ok for g in gates)


def test_isolated_peak_is_rejected():
    """同号率 0.4 < 60% → 平台期门槛必须 fail，结论不能是通过。"""
    rec = {"方向": 0.58, "超额": 0.05, "回撤": -0.08, "盈利因子": 1.8,
           "每年回合": 80, "块置换p": 0.01, "q": 0.02, "deflated_p": 0.03}
    gates = make_gates(rec, n_symbols_passed=3, n_symbols_tested=5,
                       wf_positive=3, wf_total=3, same_sign_rate=0.4)
    plateau = [g for g in gates if g.name == "参数平台期"][0]
    assert plateau.status == "fail"
    v, _ = verdict_of(gates)
    assert v == "未通过"


def test_plateau_unmeasured_when_missing():
    rec = {"方向": 0.58}
    gates = make_gates(rec, n_symbols_passed=None, n_symbols_tested=None,
                       wf_positive=None, wf_total=None, same_sign_rate=None)
    plateau = [g for g in gates if g.name == "参数平台期"][0]
    assert plateau.status == "unmeasured"


def test_reject_reason_structure():
    gates = make_gates({"方向": 0.52, "超额": -0.01}, n_symbols_passed=1,
                       n_symbols_tested=1, wf_positive=None, wf_total=None,
                       same_sign_rate=0.3)
    reasons = gates_to_reasons(gates)
    assert reasons, "有问题却没有任何拒绝原因"
    for r in reasons:
        assert set(r) == {"门槛", "要求", "实际", "状态"}
        assert r["状态"] in ("fail", "unmeasured")
    names = [r["门槛"] for r in reasons]
    assert "方向准确率" in names and "参数平台期" in names
    txt = reasons_text(reasons)
    assert "方向准确率" in txt and ("阈值" in txt)


def test_reject_reason_empty_when_all_pass():
    rec = {"方向": 0.6, "超额": 0.1, "回撤": -0.05, "盈利因子": 2.0,
           "每年回合": 60, "块置换p": 0.01, "q": 0.02, "deflated_p": 0.01}
    gates = make_gates(rec, n_symbols_passed=4, n_symbols_tested=5,
                       wf_positive=3, wf_total=3, same_sign_rate=0.9, pbo=0.15)
    assert gates_to_reasons(gates) == []
    assert reasons_text([]) == "无（全部门槛通过）"


# ---------------------------------------------------------------- 真实计算
def test_plateau_check_returns_rate_and_detail(df):
    g = Genome("SMA金叉", {"fast": 20, "slow": 60}, "无过滤", False)
    out = plateau_check(g, df, tf="H1", spread_usd=0.33)
    assert 0.0 <= out["同号率"] <= 1.0
    assert out["n"] >= 1
    assert out["detail"], "必须给出邻域明细"
    assert out["note"]
    assert isinstance(out["ok"], bool)


def test_plateau_uses_train_segment_only(df):
    """平台期判定必须在训练段上做，不能碰样本外（否则又消费一次 OOS）。"""
    import inspect

    src = inspect.getsource(plateau_check)
    assert "slice_fold" in src
    assert "train" in src


def test_run_strict_has_plateau_and_reasons(df):
    gs = [Genome("SMA金叉", {"fast": 20, "slow": 60}, "无过滤", False),
          Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "无过滤", False)]
    t = run_strict(df, gs, tf="H1", n_folds=2, n_perm=5)
    if t.empty:
        pytest.skip("合成数据没有可用候选")
    for c in ("同号率", "结论", "reject_reason"):
        assert c in t.columns, c
    assert set(t["结论"]).issubset({"通过", "继续观察", "未通过"})
    # 单品种 + 缺 q/回合 数据 → 不可能"通过"
    assert (t["结论"] != "通过").all()
    first = t.iloc[0]["reject_reason"]
    assert isinstance(first, list)
    if first:
        assert "门槛" in first[0]
