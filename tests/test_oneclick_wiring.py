"""一键生成的接线测试：EA 默认值必须被 extra_inputs 写进源码。"""

from __future__ import annotations

import pathlib

from gold.account_detect import auto_plan, ea_defaults
from gold.mq5 import generate


def test_extra_inputs_land_in_source(tmp_path):
    plan = auto_plan(500, account="micro")
    ins = ea_defaults(plan, 500)
    r = generate("唐奇安通道突破", {"entry": 20, "exit_": 10},
                 out_dir=tmp_path, name="GoldEA_Wired",
                 sl=0.02, tp=0.05, extra_inputs=ins)
    assert r.ok, r.error
    src = pathlib.Path(r.mq5_path).read_text(encoding="utf-8", errors="replace")
    assert "LOT_PERCENT" in src
    assert "InpMaxRiskPct        = 2.0" in src
    assert "InpMaxSpreadPoints   = 40" in src
    assert "InpMaxDailyLossPct   = 5.0" in src
    assert "InpOnePerBar         = true" in src


def test_extra_inputs_ignore_unknown_names(tmp_path):
    """未知输入名不能抛异常（否则一次生成就崩）。"""
    r = generate("双均线", {"fast": 20, "slow": 60}, out_dir=tmp_path,
                 name="GoldEA_Unknown", extra_inputs={"NotAnInput": 1})
    assert r.ok, r.error


def test_app_has_one_click_account_input():
    src = pathlib.Path("app.py").read_text(encoding="utf-8", errors="replace")
    assert "wb_equity" in src, "缺少账户资金输入"
    assert "account_detect as adetect" in src
    assert "extra_inputs=" in src, "生成时没把方案传给 EA"
    assert "adetect.ea_defaults(" in src, "没有按账户方案算 EA 默认值"
