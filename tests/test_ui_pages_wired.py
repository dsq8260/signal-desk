"""页面拆分后的接线测试：每个 ui/page_*.py 都存在、都有 render(ctx)，并能经分发渲染。

这些测试防止"页面模块漏建/忘记接线"——事故后新增的守卫。
"""

from __future__ import annotations

import importlib
import pathlib

import pytest

APP = pathlib.Path(__file__).resolve().parents[1] / "app.py"

MODULES = {
    "黄金看盘": "ui.page_quote",
    "挖掘与设计": "ui.page_mine",
    "策略对比": "ui.page_compare",
    "EA评分": "ui.page_rating",
    "EA回测": "ui.page_backtest",
    "EA池": "ui.page_pool",
    "内置策略体检": "ui.page_checkup",
    "参数微调": "ui.page_tune",
    "策略库": "ui.page_library",
    "账户": "ui.page_account",
}


@pytest.mark.parametrize("page,mod_name", sorted(MODULES.items()))
def test_module_exists_and_has_render(page, mod_name):
    mod = importlib.import_module(mod_name)
    assert callable(getattr(mod, "render", None)), f"{mod_name} 缺少 render(ctx)"


def test_app_declares_dispatch_table():
    src = APP.read_text(encoding="utf-8")
    assert "# ---------------- 页面分发" in src
    for page in MODULES:
        if page == "策略生成器":
            continue
        assert page in src, f"分发表里缺少 {page}"


def test_ctx_reports_missing_name_clearly():
    from ui.ctx import Ctx

    ctx = Ctx({"zh": lambda m: None})
    assert ctx.has("zh") and not ctx.has("nope")
    with pytest.raises(AttributeError) as ei:
        ctx.nope
    assert "nope" in str(ei.value)


def test_ui_modules_do_not_import_app():
    """页面模块不能反向 import app.py（会环形依赖）。

    只匹配真正的 import 语句：文档里写「不要 import app.py」不算违规。
    """
    import re

    pat = re.compile(r"^\s*(?:from\s+app\b|import\s+app\b)", re.M)
    for mod_name in MODULES.values():
        path = pathlib.Path(mod_name.replace(".", "/") + ".py")
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        m = pat.search(src)
        assert not m, f"{mod_name} 不能 import app：{m.group(0) if m else ''}"
