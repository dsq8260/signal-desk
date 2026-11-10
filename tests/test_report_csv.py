"""回归测试：成绩单必须"每次运行独立 + 带 magic + 用 bar 时间"。"""

from __future__ import annotations

import pathlib

import pytest

from gold.mq5 import generate


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    r = generate("均线偏离回归", {"ma_n": 20, "entry_thr": 150.0,
                              "exit_thr": 100.0},
                 out_dir=tmp_path_factory.mktemp("csv"), name="GoldEA_CsvT",
                 extra_inputs={"InpMagic": 20261115, "InpReport": True})
    assert r.ok, r.error
    return pathlib.Path(r.mq5_path).read_text(encoding="utf-8",
                                              errors="replace")


def test_filename_includes_magic(src):
    assert 'IntegerToString(InpMagic)' in src, "文件名要带 magic，避免多 EA 混写"


def test_cleared_on_tester_start(src):
    assert "MQLInfoInteger(MQL_TESTER)" in src
    assert "FileDelete(fname)" in src, "测试器里必须先删旧文件（否则跨运行追加成脏数据）"


def test_uses_bar_time_not_current(src):
    assert "iTime(_Symbol, _Period, 0), TIME_DATE" in src
    assert "TimeToString(TimeCurrent()" not in src
