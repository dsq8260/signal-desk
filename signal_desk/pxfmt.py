# -*- coding: utf-8 -*-
"""按品种精度格式化价格（plan / verdict / context / assist 共用）。

问题：这些模块里到处是 `round(x, 2)` 和 `{x:.2f}`——黄金 2650.12 没问题，
EURUSD 1.08473 会被写成 1.08，等于抹掉 5 个点，止损目标全错。

做法：计算开始时 `use(sym)` 设定本线程的小数位（黄金 2、直盘 5），
之后 `pr(v)` 取整、`fmt(v)` 转字符串。用 ContextVar 而不是模块全局变量，
是为了「三个品种同时重算」时互不串味（后台 SWR 线程与请求线程并行）。
默认值 2，任何没设置的地方行为与改造前完全一致。
"""
from __future__ import annotations

import contextvars

import markets

_DIG = contextvars.ContextVar("px_digits", default=2)
_SYM = contextvars.ContextVar("px_sym", default="GOLD")


def use(sym: str) -> int:
    """设定当前线程/上下文的价格精度，返回小数位。"""
    d = markets.digits(sym)
    _DIG.set(d)
    _SYM.set(sym)
    return d


def sym() -> str:
    return _SYM.get()


def px() -> int:
    return _DIG.get()


def pr(v, nd: int | None = None):
    """按当前品种精度取整（None 原样返回）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return round(f, px() if nd is None else nd)


def fmt(v, nd: int | None = None) -> str:
    """按当前品种精度格式化（None → 「—」）。"""
    try:
        return f"{float(v):.{px() if nd is None else nd}f}"
    except (TypeError, ValueError):
        return "—"
