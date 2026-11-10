"""账户 / 持仓 / 成交：直接读数据桥写出的 CSV，给行情看板做「MT5 式」图表叠加。

为什么走文件而不是 MCP：
  · 桥（GoldBridge EA）每几秒整份重写 mt5_account/positions/deals 三个文件，
    只读、无副作用、不会碰到交易接口；
  · MCP 那条路用于下单和校验，行情页每 5 秒刷新的东西不该占用它。
所有解析都是容错的：EA 正在重写时可能读到半行，解析失败就当没数据，绝不让看板崩。
"""

from __future__ import annotations

import csv
import datetime as dt
import pathlib
import threading
import time

import market as mk

ACCOUNT_FILE = "mt5_account.csv"
POSITIONS_FILE = "mt5_positions.csv"
DEALS_FILE = "mt5_deals.csv"

_cache: dict = {"at": 0.0, "sig": None, "val": None}
_row_cache: dict = {}          # (路径, mtime) → (表头, 行)：文件没重写就复用解析结果
_lock = threading.Lock()


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _rows(path: pathlib.Path) -> tuple[list[str], list[dict]]:
    """读 CSV：容错半行、容错 BOM/空文件。返回 (表头, 行)。

    带按 mtime 的行级缓存：桥每几秒重写一次文件，mtime 没变的重读直接复用，
    避免 assist 一次 payload 里 3 处调用重复解析几万行。
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return [], []
    key = (str(path), mtime)
    hit = _row_cache.get(key)
    if hit is not None:
        return hit
    try:
        raw = path.read_bytes()
        if not raw:
            return [], []
        text = raw.decode("utf-8-sig", errors="replace")
    except OSError:
        return [], []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], []
    head = [h.strip() for h in next(csv.reader([lines[0]]))]
    nhead = len(head)
    out = []
    for ln in lines[1:]:
        try:
            vals = next(csv.reader([ln]))
        except (StopIteration, csv.Error):
            continue
        if len(vals) < 2:
            continue
        # 少列的行（正在写）直接跳过；多余列则按表头截断
        if len(vals) < nhead:
            continue
        out.append(dict(zip(head, [v.strip() for v in vals])))
    result = (head, out)
    if len(_row_cache) > 64:                 # 防止长期运行积累旧 mtime 键
        _row_cache.clear()
    _row_cache[key] = result
    return result


def _age(path: pathlib.Path) -> float | None:
    try:
        return round(time.time() - path.stat().st_mtime, 1)
    except OSError:
        return None


def norm_sym(s: str) -> str:
    """GOLD# / GOLD.a / XAUUSD → 便于比较的裸代码。"""
    t = str(s or "").upper().replace("#", "")
    for suf in (".A", ".M", ".PRO", "M"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
    return t


def parse_time(s: str) -> dt.datetime | None:
    """定长格式的手工解析（strptime 有 locale 开销，几万行时会吃掉 1.5 秒以上）。"""
    t = str(s or "").strip()
    # "%Y.%m.%d %H:%M:%S"（19 字符）与 "%Y.%m.%d %H:%M"（16 字符）
    try:
        if len(t) >= 16 and t[4] == "." and t[7] == "." and t[10] == " " and t[13] == ":":
            return dt.datetime(int(t[0:4]), int(t[5:7]), int(t[8:10]),
                               int(t[11:13]), int(t[14:16]),
                               int(t[17:19]) if len(t) >= 19 else 0)
    except (ValueError, IndexError):
        pass
    try:
        return dt.datetime.strptime(t, "%Y.%m.%d %H:%M:%S")
    except ValueError:
        pass
    try:
        return dt.datetime.strptime(t, "%Y.%m.%d %H:%M")
    except ValueError:
        return None


def snapshot(symbol: str = "GOLD", days: int = 30, max_deals: int = 800) -> dict:
    """账户 + 该品种持仓 + 近 days 天该品种成交。带 2 秒缓存。"""
    d = mk.bridge_dir()
    pa, pp, pd_ = d / ACCOUNT_FILE, d / POSITIONS_FILE, d / DEALS_FILE
    sig = tuple((p.name, _mtime(p)) for p in (pa, pp, pd_)) + (symbol, days, max_deals)
    with _lock:
        if _cache["val"] and _cache["sig"] == sig and time.time() - _cache["at"] < 2.0:
            return _cache["val"]

    want = norm_sym(symbol)
    _, arows = _rows(pa)
    acc = None
    if arows:
        a = arows[-1]
        acc = {
            "time": a.get("time"), "login": a.get("login"), "server": a.get("server"),
            "company": a.get("company"), "currency": a.get("currency") or "USD",
            "balance": _num(a.get("balance"), 0.0), "equity": _num(a.get("equity"), 0.0),
            "margin": _num(a.get("margin"), 0.0), "margin_free": _num(a.get("margin_free"), 0.0),
            "margin_level": _num(a.get("margin_level"), 0.0),
            "profit": _num(a.get("profit"), 0.0), "leverage": int(_num(a.get("leverage"), 0) or 0),
            "age_s": _age(pa),
        }

    _, prows = _rows(pp)
    positions, others = [], []
    for p in prows:
        side = {"0": "buy", "1": "sell"}.get(str(p.get("type")), "?")
        row = {
            "ticket": p.get("ticket"), "symbol": p.get("symbol"), "side": side,
            "volume": _num(p.get("volume"), 0.0),
            "price_open": _num(p.get("price_open")),
            "price_current": _num(p.get("price_current")),
            "profit": _num(p.get("profit"), 0.0), "swap": _num(p.get("swap"), 0.0),
            "sl": _num(p.get("sl")), "tp": _num(p.get("tp")),
            "magic": int(_num(p.get("magic"), 0) or 0),
            "comment": p.get("comment") or "",
        }
        (positions if norm_sym(row["symbol"]) == want else others).append(row)

    _, drows = _rows(pd_)
    cutoff = dt.datetime.now() - dt.timedelta(days=max(1, int(days)))
    deals = []
    for x in drows:
        if norm_sym(x.get("symbol")) != want:
            continue
        t = parse_time(x.get("time"))
        if not t or t < cutoff:
            continue
        dtype = str(x.get("type"))
        if dtype not in ("0", "1"):      # 只留买卖成交，滤掉入金/手续费
            continue
        ent = str(x.get("entry"))
        deals.append({
            "t": t.strftime("%Y-%m-%d %H:%M"),
            "side": "buy" if dtype == "0" else "sell",
            "entry": "in" if ent == "0" else ("out" if ent == "1" else ent),
            "volume": _num(x.get("volume"), 0.0),
            "price": _num(x.get("price")),
            "profit": _num(x.get("profit"), 0.0),
            "magic": int(_num(x.get("magic"), 0) or 0),
            "comment": x.get("comment") or "",
        })
    deals = deals[-max_deals:]

    out = {
        "ok": bool(acc or positions),
        "symbol": symbol, "norm": want,
        "account": acc,
        "positions": positions,
        "others": {"n": len(others),
                   "symbols": sorted({p["symbol"] for p in others if p.get("symbol")})[:8],
                   "float": round(sum(p["profit"] + (p["swap"] or 0) for p in others), 2)},
        "deals": deals,
        "files": {"account": _age(pa), "positions": _age(pp), "deals": _age(pd_)},
        "dir": str(d),
        "ts": time.time(),
    }
    if not out["ok"]:
        out["error"] = ("数据桥没在写账户文件（mt5_account.csv / mt5_positions.csv）。"
                        "挂上 GoldBridge EA 后这里会出现账户与持仓；文件位置：" + str(d))
    with _lock:
        _cache.update({"at": time.time(), "sig": sig, "val": out})
    return out


def _mtime(p: pathlib.Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0
