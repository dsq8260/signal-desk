"""回归测试：唐奇安进场窗口必须排除被判断的 K 线（否则永远不开单）。"""

from __future__ import annotations

import pathlib
import re

import pytest

from gold.mq5 import generate


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    r = generate("唐奇安通道突破", {"entry": 20, "exit_": 10},
                 out_dir=tmp_path_factory.mktemp("donchian"),
                 name="GoldEA_DonchianGuard")
    assert r.ok, r.error
    return pathlib.Path(r.mq5_path).read_text(encoding="utf-8", errors="replace")


def test_breakout_window_excludes_current_bar(src):
    """`CopyHigh(..., 1, entry, ...)` 会把 bar1 自己算进窗口，导致 close>max 永假。"""
    highs = re.findall(r"CopyHigh\(_Symbol, _Period, (\d+), InpEntry", src)
    lows = re.findall(r"CopyLow\(_Symbol, _Period, (\d+), InpExit", src)
    assert highs == ["2"], highs
    assert lows == ["2"], lows


def test_compares_close_of_bar1(src):
    assert "CopyClose(_Symbol, _Period, 1, 1, cl)" in src


def test_breakout_condition_present(src):
    assert "cl[0] > hh" in src and "cl[0] < ll" in src


def test_min_bars_uses_entry(src):
    assert "Bars(_Symbol, _Period) < 20 + 5" in src
