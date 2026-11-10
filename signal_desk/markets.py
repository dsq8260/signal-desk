# -*- coding: utf-8 -*-
"""品种清单与价格刻度（多品种并行的唯一入口）。

为什么要单独一层
----------------
系统原来只有黄金，很多地方默认「价格保留两位小数、1 手 = 100 盎司、
点差阈值 1.8 美金」。这些假设放到 EURUSD（1.08473）上会立刻出问题：
· 价格 round(…, 2) 直接把 1.0847 变成 1.08，等于抹掉 4 个点；
· 1 手 EURUSD = 100,000 欧元，按 100 算手数会放大 1000 倍；
· 点差阈值 1.8 对 1 pips 的品种等于永远不告警。

所以刻度、合约、阈值全部集中在这张表里，其它模块一律 `from markets import …`
取值，不要再写死数字。

品种代号用「桥里的名字」（GOLD / EURUSD / GBPUSD），显示名可以不同
（GOLD 显示成 XAUUSD，因为 MT5 的黄金叫 XAUUSD 或 GOLD#，都是同一品种）。
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
FILE = HERE / "data" / "markets.json"

# 默认清单：黄金 + 两个直盘。数值都是「该品种自己的口径」，不是黄金的换算。
DEFAULT_LIST: list[dict] = [
    {
        "sym": "GOLD", "name": "XAUUSD", "label": "黄金", "enabled": True,
        "digits": 2, "point": 0.01, "contract": 100,
        "near": 3.0, "spread_alert": 1.8,
        "spread_default": 0.30, "slip": 0.05,
        "unit": "美元/盎司", "note": "1 手 = 100 盎司",
    },
    {
        "sym": "EURUSD", "name": "EURUSD", "label": "欧元", "enabled": True,
        "digits": 5, "point": 0.00001, "contract": 100000,
        "near": 0.0012, "spread_alert": 0.00025,
        "spread_default": 0.00012, "slip": 0.00002,
        "unit": "美元（报价货币）", "note": "1 手 = 100,000 欧元",
    },
    {
        "sym": "GBPUSD", "name": "GBPUSD", "label": "英镑", "enabled": True,
        "digits": 5, "point": 0.00001, "contract": 100000,
        "near": 0.0016, "spread_alert": 0.00030,
        "spread_default": 0.00015, "slip": 0.00003,
        "unit": "美元（报价货币）", "note": "1 手 = 100,000 英镑",
    },
]

_cache: dict = {"key": None, "val": None}


def _norm(sym: str) -> str:
    return "".join(ch for ch in str(sym or "").replace("#", "")
                   if ch.isalnum() or ch == "_").upper() or "GOLD"


def load() -> list[dict]:
    """读品种清单；文件缺失/坏掉都退回默认表（绝不因为配置坏了整页报错）。"""
    try:
        st = FILE.stat()
    except OSError:
        return [dict(x) for x in DEFAULT_LIST]
    key = (st.st_mtime_ns, st.st_size)
    if _cache["key"] == key and _cache["val"]:
        return _cache["val"]
    try:
        raw = json.loads(FILE.read_text(encoding="utf-8"))
        items = raw.get("symbols") if isinstance(raw, dict) else raw
        out = []
        for d in DEFAULT_LIST:
            hit = next((x for x in (items or [])
                        if _norm((x or {}).get("sym")) == d["sym"]), None)
            merged = dict(d)
            for k in ("enabled", "near", "spread_alert", "digits", "point",
                      "contract", "spread_default", "slip", "name", "label"):
                if isinstance(hit, dict) and hit.get(k) is not None:
                    merged[k] = hit[k]
            out.append(merged)
        _cache.update({"key": key, "val": out})
        return out
    except Exception:  # noqa: BLE001
        return [dict(x) for x in DEFAULT_LIST]


def save(body: dict) -> dict:
    """保存品种设置（只允许改开关与阈值，代号/合约由代码决定，避免改坏手数）。"""
    cur = {x["sym"]: dict(x) for x in load()}
    for x in (body.get("symbols") or []):
        if not isinstance(x, dict):
            continue
        s = _norm(x.get("sym"))
        if s not in cur:
            continue
        for k in ("enabled", "near", "spread_alert"):
            if x.get(k) is None:
                continue
            cur[s][k] = bool(x[k]) if k == "enabled" else float(x[k])
    FILE.parent.mkdir(exist_ok=True)
    FILE.write_text(json.dumps({"symbols": list(cur.values())},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    _cache.update({"key": None})
    return {"ok": True, "symbols": load()}


def all() -> list[dict]:
    return load()


def enabled() -> list[dict]:
    return [x for x in load() if x.get("enabled")]


def of(sym: str) -> dict:
    s = _norm(sym)
    for x in load():
        if x["sym"] == s:
            return x
    # 未登记的品种：按名字猜刻度（黄金类 2 位、外汇类 5 位），保证不崩
    digits = 2 if any(k in s for k in ("XAU", "GOLD", "SILVER", "XAG")) else 5
    point = 10 ** -digits
    return {"sym": s, "name": s, "label": s, "enabled": True,
            "digits": digits, "point": point,
            "contract": 100 if digits == 2 else 100000,
            "near": 3.0 * point * 100 if digits == 2 else 0.0012,
            "spread_alert": 1.8 if digits == 2 else 0.00025,
            "spread_default": 0.30 if digits == 2 else 0.00012,
            "slip": 0.05 if digits == 2 else 0.00002,
            "unit": "价格单位", "note": ""}


def digits(sym: str) -> int:
    return int(of(sym).get("digits") or 2)


def point(sym: str) -> float:
    return float(of(sym).get("point") or 10 ** -digits(sym))


def contract(sym: str) -> float:
    return float(of(sym).get("contract") or 100)


def name(sym: str) -> str:
    """显示名：GOLD → XAUUSD。"""
    return str(of(sym).get("name") or _norm(sym))


def label(sym: str) -> str:
    o = of(sym)
    return f"{o.get('name') or _norm(sym)}（{o.get('label') or ''}）"


def near(sym: str, cfg: dict | None = None) -> float:
    """靠近关键位的距离（价格单位）。市场清单没写的品种才回落到旧配置值。"""
    o = of(sym)
    if o.get("near"):
        return float(o["near"])
    return float((cfg or {}).get("near_usd") or 3.0)


def spread_alert(sym: str, cfg: dict | None = None) -> float:
    o = of(sym)
    if o.get("spread_alert"):
        return float(o["spread_alert"])
    return float((cfg or {}).get("spread_alert") or 1.8)


def spread_default(sym: str) -> float:
    return float(of(sym).get("spread_default") or 0.30)


def slip(sym: str) -> float:
    return float(of(sym).get("slip") or 0.05)


def rnd(sym: str, v, nd: int | None = None):
    """按品种精度取整；None 原样返回（页面显示「—」）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return round(f, digits(sym) if nd is None else nd)


def fmt(sym: str, v, nd: int | None = None) -> str:
    try:
        return f"{float(v):.{digits(sym) if nd is None else nd}f}"
    except (TypeError, ValueError):
        return "—"
