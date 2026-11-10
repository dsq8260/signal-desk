"""策略工作台：三页导航、生成链路、风险覆盖语义、导出与规则文案。"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from gold import workbench as bench
from gold.forge import Genome

APP = pathlib.Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture()
def df():
    rng = np.random.default_rng(5)
    n = 2200
    close = 2600 * np.cumprod(1 + rng.normal(0.0003, 0.0019, n))
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.0012,
                         "low": close * 0.9988, "close": close,
                         "volume": 0.0}, index=idx)


# ---------------------------------------------------------------- 纯逻辑
def test_space_size_respects_entry_and_filter():
    full = bench.space_size(bench.Request(mode="full"))
    one = bench.space_size(bench.Request(entries=["SMA金叉"], filters=["无过滤"],
                                         mode="full"))
    assert full == 576   # 参数网格补齐后的实测值
    # SMA金叉有 9 组参数 × 1 过滤 × 4 风控 × 1 方向 = 36
    assert one == 36


def test_build_genomes_is_deterministic():
    r = bench.Request(mode="sample", n=8, seed=11)
    a = [g.label() for g in bench.build_genomes(r)]
    b = [g.label() for g in bench.build_genomes(r)]
    assert a == b and len(a) == 8


def test_risk_overrides_instead_of_filtering():
    """单笔风险 2% + 回报 1:2 → 止盈 4%（不在 STOP_GRID 里）也必须能生成。"""
    r = bench.Request(entries=["唐奇安突破"], filters=["无过滤"], mode="sample",
                      n=4, sl=0.02, tp=0.04)
    gs = bench.build_genomes(r)
    assert len(gs) == 4
    assert all(abs(g.sl - 0.02) < 1e-12 and abs(g.tp - 0.04) < 1e-12
               for g in gs)


def test_no_risk_keeps_grid_values():
    r = bench.Request(entries=["SMA金叉"], filters=["无过滤"], mode="full")
    gs = bench.build_genomes(r)
    stops = {(g.sl, g.tp) for g in gs}
    assert (None, None) in stops and (0.02, None) in stops


def test_rule_text_translates_parameters():
    g = Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "强趋势", False,
               0.02, 0.04)
    rules = bench.rule_text(g)
    assert len(rules) == 3
    assert "20 根" in rules[0]["内容"]
    assert "2%" in rules[1]["内容"] and "4%" in rules[1]["内容"]
    assert "趋势分位" in rules[0]["内容"]


def test_rule_text_never_injects_strategy_name_into_params():
    """唐奇安参数就叫 entry；曾经把它错误替换成策略名。"""
    g = Genome("唐奇安突破", {"entry": 30, "exit_": 5}, "无过滤", False)
    txt = bench.rule_text(g)[0]["内容"]
    assert "30 根" in txt
    assert "唐奇安突破 根" not in txt


def test_rule_text_handles_unknown_entry():
    g = Genome("不存在的策略", {"x": 1}, "无过滤", False)
    assert bench.rule_text(g)[0]["内容"]


def test_export_json_roundtrips():
    g = Genome("SMA金叉", {"fast": 10, "slow": 60}, "低波动", False, 0.02, None)
    payload = json.loads(bench.export_json(g, {"seed": 7}))
    assert payload["genome"]["entry"] == "SMA金叉"
    assert payload["genome"]["filter"] == "低波动"
    assert payload["meta"]["seed"] == 7
    assert len(payload["rules"]) == 3


def test_equity_curve_is_real_series(df):
    g = Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False)
    eq = bench.equity_curve(df, g, tf="H1", spread_usd=0.33)
    assert len(eq) >= 10
    assert all(v > 0 for v in eq)


def test_equity_curve_empty_on_bad_input():
    assert bench.equity_curve(pd.DataFrame(), Genome("SMA金叉", {}, "无过滤")) == []


def test_evaluate_candidates_returns_verdict_columns(df):
    gs = [Genome("SMA金叉", {"fast": 20, "slow": 60}, "无过滤", False),
          Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "无过滤", False)]
    tb = bench.evaluate_candidates(df, gs, tf="H1", spread_usd=0.33, n_perm=10,
                                   plateau_top=2)
    if tb.empty:
        pytest.skip("合成数据没有可用候选")
    for c in ("EA", "夏普", "收益", "回撤", "方向准确率", "回合", "每年回合",
              "盈利因子", "块置换p", "q", "同号率", "结论", "拒绝原因", "入池"):
        assert c in tb.columns, c
    assert set(tb["结论"]).issubset({"通过", "继续观察", "未通过"})
    assert (tb["结论"] != "通过").all(), "单品种不该出现通过"
    assert tb["每年回合"].max() > 0


def test_evaluate_candidates_never_selects_on_test():
    """评测函数里的切分必须来自 oos（唯一实现），不能自己按比例截断。"""
    import inspect

    src = inspect.getsource(bench.evaluate_candidates)
    assert "slice_fold" in src
    assert "int(len(df)" not in src


def test_summary_line_mentions_key_metrics(df):
    gs = [Genome("SMA金叉", {"fast": 10, "slow": 60}, "无过滤", False)]
    tb = bench.evaluate_candidates(df, gs, tf="H1", spread_usd=0.33, n_perm=5,
                                   plateau_top=0)
    if tb.empty:
        pytest.skip("无候选")
    line = bench.summary_line(tb.iloc[0])
    assert "夏普" in line and "回撤" in line


# ---------------------------------------------------------------- 界面
def test_three_workbench_pages_render():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=900)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    html = " ".join(str(m.value) for m in at.markdown)
    assert "定义你的策略" in html and "策略预览" in html
    assert "未来函数禁用" in html and "OOS 账本" in html

    # 2026-11-02 起默认专注模式只有「生成 EA / 我的 EA / 帮助」
    btn = [b for b in at.sidebar.button if "我的 EA" in str(b.label)]
    assert btn, "侧栏没有 我的 EA"
    btn[0].click()
    at.run()
    assert not at.exception, [str(e.value)[:200] for e in at.exception]
    # 回测记录移入高级模式（关掉专注模式才出现）
    for t in at.toggle:
        if "专注模式" in str(t.label) and t.value:
            t.set_value(False).run()
            break
    # 回测记录在侧栏显示为「运行日志」（内部页名不变）
    btn = [b for b in at.sidebar.button
           if "运行日志" in str(b.label) or "回测记录" in str(b.label)]
    assert btn, "关掉专注模式后应能看到运行日志（回测记录）"
    btn[0].click()
    at.run()
    assert not at.exception, [str(e.value)[:200] for e in at.exception]


def test_focus_mode_shows_only_three_pages():
    src = APP.read_text(encoding="utf-8")
    i = src.find("FOCUS_PAGES = [")
    line = src[i:i + 140].split("\n")[0]
    # 2026-11-02 起专注模式只留三页：生成 EA / 我的 EA / 帮助
    assert "策略生成器" in line and "历史策略" in line and "策略库" in line
    assert "回测记录" not in line
    assert "EA池" not in line


def test_mql5_name_is_ascii_and_readable():
    g = Genome("唐奇安突破", {"entry": 20, "exit_": 10}, "强趋势", False,
               0.02, 0.04)
    name = bench.mql5_name(g)
    assert name.startswith("GoldEA_")
    assert name.isascii()
    assert "Donchian" in name and "Trend" in name and "SL2" in name


def test_mql5_name_never_empty():
    g = Genome("未知策略", {}, "无过滤", False)
    assert bench.mql5_name(g) == "GoldEA_EA"
