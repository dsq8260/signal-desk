"""MQL5 模板覆盖测试：每个能挖的进场都必须导得出，且过滤开关真的写进源码。

为什么要有这组测试
----------------
2026-10-30 的教训：全枚举前三名（时变夏普 / TSMOM / 鳄鱼线）都能挖出来，
但**没有 MQL5 模板**，界面上只能弹「没有 MQL5 模板」；「强趋势」过滤更是
被悄悄丢掉。挖出来导不出 = 白挖，所以这条链路必须有测试守住。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.forge import _all_entries
from gold.mq5 import ENTRY_ALIAS, SPECS, generate, spec_for


def test_every_minable_entry_has_a_template():
    """能挖的进场 == 能导出的进场，不能有缺口。"""
    missing = [e for e in _all_entries() if spec_for(e) is None]
    assert not missing, f"这些进场能挖但没有 MQL5 模板：{missing}"


def test_alias_targets_all_exist():
    for entry, target in ENTRY_ALIAS.items():
        assert target in SPECS, f"{entry} 映射到了不存在的模板 {target}"


def test_new_templates_exist():
    for name in ("时序动量TSMOM", "时变夏普", "鳄鱼线"):
        assert name in SPECS


def test_spec_for_unknown_entry_returns_none():
    assert spec_for("不存在的策略") is None


@pytest.mark.parametrize("entry,params", [
    ("时序动量TSMOM", {"n": 60, "thr": 0.0}),
    ("时变夏普", {"n": 60, "thr": 0.0}),
    ("鳄鱼线", {"jaw": 13, "teeth": 8, "lips": 5, "shift": 3}),
])
def test_new_entry_generates_source(tmp_path, entry, params):
    spec = spec_for(entry)
    g = generate(spec, params, name="GoldEA_test", out_dir=tmp_path)
    assert g.ok, g.error
    src = open(g.mq5_path, encoding="utf-8").read()
    # 模板必须真的设置 wantLong / wantShort（执行层靠这两个变量）
    assert "wantLong" in src and "wantShort" in src
    # 干支已移除，不许再冒出来
    assert "InpZhi" not in src and "DayZhi" not in src
    # 两个过滤开关都存在，且默认关闭
    assert "InpUseVolFilter = false;" in src
    assert "InpUseTrendFilter = false;" in src


def test_trend_filter_can_be_enabled(tmp_path):
    g = generate("唐奇安通道突破", {"entry": 20, "exit_": 10},
                 name="GoldEA_trend", out_dir=tmp_path,
                 trend_filter_pct=0.5)
    assert g.ok, g.error
    src = open(g.mq5_path, encoding="utf-8").read()
    assert "input bool   InpUseTrendFilter = true;" in src
    assert "input double InpTrendMinPct    = 0.50;" in src
    assert "VolFilterOK() && TrendFilterOK()" in src


def test_vol_filter_can_be_enabled(tmp_path):
    g = generate("鳄鱼线", {"jaw": 13, "teeth": 8, "lips": 5, "shift": 3},
                 name="GoldEA_vol", out_dir=tmp_path, vol_filter_pct=0.7)
    assert g.ok, g.error
    src = open(g.mq5_path, encoding="utf-8").read()
    assert "input bool   InpUseVolFilter = true;" in src
    assert "input double InpVolMaxPct    = 0.70;" in src


def test_alligator_long_only():
    """鳄鱼线原版只做多：做空分支必须是 false，不能凭空加空头。"""
    src = SPECS["鳄鱼线"]["ontick"]
    assert "gWantShort = false;" in src


def test_trend_filter_formula_matches_python_percentile():
    """MQL5 的趋势过滤公式与 Python 的 pct_rank 必须算同一个分位。

    这里用纯 numpy 复刻 MQL5 的循环（窗口 LB 根、有效样本 >= LB/4），
    与 factors._pct_rank 对比；抽样检查，避免 O(n×250×60)。
    """
    from gold.factors import _pct_rank

    rng = np.random.default_rng(3)
    trend = pd.Series(rng.normal(size=600).cumsum())
    pct = _pct_rank(trend, 250).to_numpy()
    vals = trend.to_numpy()
    N, LB = 60, 250
    for i in range(400, 600, 25):
        t0 = vals[i]
        cnt = tot = 0
        for k in range(LB):
            v = vals[i - k]
            if not np.isfinite(v):
                continue
            tot += 1
            if v <= t0:
                cnt += 1
        if tot < max(20, LB // 4):
            continue
        assert abs(cnt / tot - pct[i]) < 1e-9, (i, cnt / tot, pct[i])
