"""仓位模式测试：固定手数 / 按权益百分比，两种都要在模板里且被 OpenTrade 使用。"""

from __future__ import annotations

import pathlib

import pytest

from gold.mq5 import generate


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    r = generate("唐奇安通道突破", {"entry": 20, "exit_": 10},
                 out_dir=tmp_path_factory.mktemp("lotmode"),
                 name="GoldEA_LotGuard")
    assert r.ok, r.error
    return pathlib.Path(r.mq5_path).read_text(encoding="utf-8", errors="replace")


def test_lot_mode_inputs_exist(src):
    assert "enum LotMode" in src
    assert "InpLotMode" in src and "InpRiskPct" in src and "InpMaxLots" in src
    assert "LOT_FIXED" in src and "LOT_PERCENT" in src


def test_default_is_fixed_lots(src):
    """默认必须是固定手数：可预测，不随账户规模漂移。"""
    assert "input LotMode InpLotMode   = LOT_FIXED;" in src


def test_lots_helper_handles_both_modes(src):
    assert "double LotsForTrade(double price, double slPct)" in src
    i = src.find("double LotsForTrade")
    body = src[i:i + 900]
    assert "InpLotMode == LOT_FIXED" in body, "固定手数分支缺失"
    assert "InpRiskPct / 100.0" in body, "百分比风险分支缺失"
    assert "SYMBOL_TRADE_TICK_VALUE" in body and "SYMBOL_TRADE_TICK_SIZE" in body


def test_normalize_lots_clamps(src):
    i = src.find("double NormalizeLots(double lots)")
    body = src[i:i + 700]
    for k in ("SYMBOL_VOLUME_MIN", "SYMBOL_VOLUME_MAX", "SYMBOL_VOLUME_STEP",
              "InpMaxLots"):
        assert k in body, k


def test_open_trade_uses_computed_lots(src):
    i = src.find("void OpenTrade(int dir)")
    body = src[i:i + 900]
    body = src[i:i + 1600]
    assert "double lots = LotsForTrade(" in body
    assert "trade.Buy(lots," in body and "trade.Sell(lots," in body
    assert "trade.Buy(InpLots," not in body, "不能再用写死的手数"
