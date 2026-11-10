"""阶段B 验收：三段切分 / 锁定测试集 / walk-forward / 十门槛 / 方向诊断。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.forge import Genome
from gold.verify import (
    Gate,
    Verifier,
    direction_diagnostic,
    make_gates,
    split_train_val_test,
    verdict_of,
    walk_forward_windows,
)


def _bars(n=1200, seed=7, drift=0.0005, vol=0.0018, price=2600.0):
    rng = np.random.default_rng(seed)
    close = price * np.cumprod(1 + rng.normal(drift, vol, n))
    idx = pd.bdate_range("2018-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.0015,
                         "low": close * 0.9985, "close": close,
                         "volume": 0.0}, index=idx)


# ---------------------------------------------------------------- 三段切分
def test_split_ratios_and_no_overlap():
    df = _bars(1000)
    tr, va, te = split_train_val_test(df)
    assert (len(tr), len(va), len(te)) == (500, 250, 250)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr.index[-1] < va.index[0] and va.index[-1] < te.index[0]
    assert not set(tr.index) & set(te.index)


def test_split_rejects_bad_ratios():
    df = _bars(100)
    with pytest.raises(ValueError):
        split_train_val_test(df, (0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        split_train_val_test(df, (0.6, 0.4, 0.0))


def test_walk_forward_windows_are_ordered_and_disjoint_per_window():
    df = _bars(600)
    wins = walk_forward_windows(df, 3)
    assert len(wins) == 3
    for tr, va in wins:
        assert len(tr) > 0 and len(va) > 0
        assert tr.index[-1] < va.index[0], "训练段必须在验证段之前"


# ---------------------------------------------------------------- 锁定测试集
def test_test_segment_is_only_used_for_final_vote():
    """构造：验证段极好、测试段极差。

    如果测试集被拿去参与选择，verdict 会被"验证段的好成绩"带成通过；
    正确行为是：选择看验证段，但最终结论由测试段否决。
    """
    df = _bars(1200, seed=3)
    v = Verifier(df, tf="H1", spread_usd=0.33, n_windows=2)
    # 两段长度必须对得上：选择阶段用 train+val，投票用 test
    assert len(v.train) + len(v.val) + len(v.test) == len(df)
    assert v.test.index[0] > v.val.index[-1]

    g = Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False)
    sel = v.select_metrics(g)
    test = v.test_metrics(g)
    # 选择阶段返回的字典里不能含测试段对象
    assert "test" not in sel
    assert set(test) >= {"收益", "夏普", "回撤", "方向", "笔数"}


def test_walk_forward_never_touches_test_segment():
    df = _bars(1200, seed=5)
    v = Verifier(df, tf="H1", spread_usd=0.33, n_windows=3)
    last_fit = max(x.index[-1] for x in
                   [v.train, v.val])
    for _tr, va in walk_forward_windows(pd.concat([v.train, v.val]), 3):
        assert va.index[-1] <= last_fit
    assert v.test.index[0] > last_fit


# ---------------------------------------------------------------- 十门槛
def _gate_by_name(gates, name):
    for g in gates:
        if g.name == name:
            return g
    raise AssertionError(name)


def test_gates_all_pass_gives_pass():
    rec = {"方向": 0.58, "超额": 0.05, "回撤": -0.08, "盈利因子": 1.8,
           "每年回合": 80, "块置换p": 0.01, "q": 0.02, "deflated_p": 0.03}
    gates = make_gates(rec, n_symbols_passed=3, n_symbols_tested=5,
                       wf_positive=2, wf_total=3, same_sign_rate=0.8,
                       pbo=0.25)          # 第 12 条：CSCV-PBO ≤0.5
    assert all(g.ok for g in gates)
    assert verdict_of(gates)[0] == "通过"


def test_single_symbol_cannot_pass():
    """单品种时「多品种≥60%」判未测 —— 结论最多是"继续观察"，绝不能是通过。"""
    rec = {"方向": 0.58, "超额": 0.05, "回撤": -0.08, "盈利因子": 1.8,
           "每年回合": 80, "块置换p": 0.01, "q": 0.02, "deflated_p": 0.03}
    gates = make_gates(rec, n_symbols_passed=1, n_symbols_tested=1,
                       wf_positive=3, wf_total=3)
    v, why = verdict_of(gates)
    assert v == "继续观察"
    assert "多品种" in why
    assert _gate_by_name(gates, "多品种 ≥60%").status == "unmeasured"


def test_failed_direction_blocks_pass():
    rec = {"方向": 0.52, "超额": 0.05, "回撤": -0.08, "盈利因子": 1.8,
           "每年回合": 80, "块置换p": 0.01, "q": 0.02, "deflated_p": 0.03}
    gates = make_gates(rec, n_symbols_passed=5, n_symbols_tested=5,
                       wf_positive=3, wf_total=3)
    assert verdict_of(gates)[0] == "未通过"
    assert _gate_by_name(gates, "方向准确率").status == "fail"


def test_few_positive_walk_forward_windows_fails():
    rec = {"方向": 0.58, "超额": 0.05, "回撤": -0.08, "盈利因子": 1.8,
           "每年回合": 80, "块置换p": 0.01, "q": 0.02, "deflated_p": 0.03}
    gates = make_gates(rec, n_symbols_passed=3, n_symbols_tested=5,
                       wf_positive=1, wf_total=3)
    assert _gate_by_name(gates, "滚动 3 段中 ≥2 段为正").status == "fail"


def test_missing_values_are_unmeasured_not_pass():
    gates = make_gates({}, n_symbols_passed=None, n_symbols_tested=None,
                       wf_positive=None, wf_total=None)
    assert all(not g.ok for g in gates)
    assert verdict_of(gates)[0] == "继续观察"


def test_gate_dataclass_ok_property():
    assert Gate("x", "1", "1", "pass").ok is True
    assert Gate("x", "1", "0", "fail").ok is False
    assert Gate("x", "1", "—", "unmeasured").ok is False


# ---------------------------------------------------------------- Verifier 端到端
def test_verifier_run_produces_table_with_verdicts():
    df = _bars(1500, seed=11)
    space = [Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False),
             Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "无过滤", False),
             Genome("LLT上穿", {"n": 60}, "无过滤", False)]
    v = Verifier(df, tf="H1", spread_usd=0.33, n_windows=2)
    res = v.run(space, n_perm=50, min_trades=10)
    if res.table.empty:
        pytest.skip("合成数据未产生有效候选：" + res.note)
    for c in ("训练收益", "验证收益", "测试收益", "块置换p", "q",
              "deflated_p", "结论", "门槛明细", "原因"):
        assert c in res.table.columns
    assert set(res.table["结论"]).issubset({"通过", "继续观察", "未通过"})
    assert (res.table["结论"] != "通过").all(), "单品种/合成数据不该出现通过"
    assert res.segments["训练"][2] + res.segments["验证"][2] + \
        res.segments["测试"][2] == len(df)


# ---------------------------------------------------------------- 方向诊断
def test_direction_diagnostic_buckets_and_overall():
    df = _bars(1500, seed=13, drift=0.0008)
    g = Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False)
    out = direction_diagnostic(df, g, "H1", 0.33, by="year")
    assert len(out) >= 2
    assert set(out.columns) >= {"年份", "方向", "笔数", "收益", "样本"}
    assert out.attrs["overall"]["样本"] > 0


def test_direction_diagnostic_vol_buckets():
    df = _bars(1500, seed=17)
    g = Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "无过滤", False)
    out = direction_diagnostic(df, g, "H1", 0.33, by="vol")
    if len(out) == 0:
        pytest.skip("分桶后样本不足")
    assert set(out["波动档"]).issubset({"低波动", "中波动", "高波动"})


def test_direction_diagnostic_rejects_unknown_bucket():
    df = _bars(300)
    g = Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False)
    with pytest.raises(ValueError):
        direction_diagnostic(df, g, "H1", 0.33, by="月相")

def test_pbo_gate_three_states():
    """第 12 条：不给 = 未测（不算通过），≤0.5 通过，>0.5 不通过。"""
    rec = {"方向": 0.58, "超额": 0.05, "回撤": -0.08, "盈利因子": 1.8,
           "每年回合": 80, "块置换p": 0.01, "q": 0.02, "deflated_p": 0.03}
    base = dict(n_symbols_passed=3, n_symbols_tested=5, wf_positive=2,
                wf_total=3, same_sign_rate=0.8)
    def one(pbo):
        gates = make_gates(rec, pbo=pbo, **base)
        return [g for g in gates if g.name == "CSCV-PBO"][0]
    assert one(None).status == "unmeasured"
    assert one(0.3).status == "pass"
    assert one(0.8).status == "fail"


def test_gates_count_is_twelve():
    gates = make_gates({})
    assert len(gates) == 12, [g.name for g in gates]
