"""安全改文件的单元测试：区间断言必须挡住"吞掉后面几千行"这类错误。"""

from __future__ import annotations

import pathlib

import pytest

from tools import safe_patch


@pytest.fixture()
def f(tmp_path):
    p = tmp_path / "demo.py"
    p.write_text("a\nb\nSTART\nx\ny\nEND\nc\n", encoding="utf-8")
    return p


def test_replace_block_replaces_only_inside(f):
    r = safe_patch.replace_block(str(f), "START\n", "END\n", "START\nNEW\n",
                                 expect_lines=(1, 5))
    assert f.read_text(encoding="utf-8") == "a\nb\nSTART\nNEW\nEND\nc\n"
    assert r["lines"] == 3
    assert pathlib.Path(r["backup"]).exists()


def test_replace_block_refuses_wrong_size(f):
    with pytest.raises(AssertionError):
        safe_patch.replace_block(str(f), "START\n", "END\n", "X\n",
                                 expect_lines=(50, 60))
    # 没有写盘
    assert "x\ny" in f.read_text(encoding="utf-8")


def test_replace_block_missing_anchor(f):
    with pytest.raises(ValueError):
        safe_patch.replace_block(str(f), "NOPE\n", "END\n", "")


def test_backup_keeps_original_content(f):
    before = f.read_text(encoding="utf-8")
    r = safe_patch.replace_block(str(f), "START\n", "END\n", "START\n", 
                                 expect_lines=(1, 5))
    assert pathlib.Path(r["backup"]).read_text(encoding="utf-8") == before


def test_append_text_does_not_truncate(f):
    before = f.read_text(encoding="utf-8")
    r = safe_patch.append_text(str(f), "# tail\n")
    after = f.read_text(encoding="utf-8")
    assert after.startswith(before.rstrip("\n"))
    assert after.endswith("# tail\n")
    assert pathlib.Path(r["backup"]).exists()
