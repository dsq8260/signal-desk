# -*- coding: utf-8 -*-
"""影子账本（Shadow Mode）：只记录决策，不下单，用后续行情自己结算。

为什么要有它：三审制到底有没有用，靠感觉说不清，只能靠样本。每条「本会开仓」
的计划、以及每条「被本地规则拦下」的计划都记一笔，之后用 M5 K 线向前推进：
先到止损记 -1R，先到第一目标记 +RR，扣掉点差与滑点再算期望。攒满 100 笔
才谈得上「这套规则能不能挣钱」，在此之前任何「更优秀」都是空话。

三条设计原则：
1) 不依赖系统时钟。结算起点记的是「记录当时最后一根 K 线的时间」，之后只在
   K 线序列里往后数——服务器时区跟本机差 6 小时也不会算错持仓时长。
2) 同根 K 线同时触及止损与目标 → 一律按止损计（坏情况优先），宁可低估。
3) 「被拦的计划」单独成一类。它的盈亏是反事实：如果当初没拦会怎样。这正好
   用来检验纪律是救了账户还是挡了利润。
"""
from __future__ import annotations

import bisect
import datetime as dt
import json
import pathlib
import threading

import markets

DATA = pathlib.Path(__file__).resolve().parent / "data"
BOOK = DATA / "shadow_book.json"
LOCK = threading.Lock()

# 结算口径（改这里就改了账本的判定标准）
CFG = {
    "settle_tf": "M5",      # 用 M5 走价格路径，比主周期细，避免「一根 K 线里先损后盈」判错
    "expire_bars": 288,     # 24 小时（M5×288）还没碰到入场价 → 记为错过
    "max_hold_bars": 576,   # 入场后 48 小时仍未到止损/目标 → 按最后收盘价结账
    "max_items": 900,       # 账本上限（三个品种共用一本，600 太紧）
    "target": 100,          # 影子阶段目标样本数（三品种合计）
}


def _digits(sym: str) -> int:
    """品种价格精度：黄金 2、直盘 5。金额/成本都按这个刻度过账。"""
    try:
        return markets.digits(sym)
    except Exception:  # noqa: BLE001
        return 2


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read() -> dict:
    try:
        d = json.loads(BOOK.read_text(encoding="utf-8"))
        if isinstance(d, dict) and isinstance(d.get("items"), list):
            return d
    except Exception:  # noqa: BLE001
        pass
    return {"items": [], "updated": None}


def _write(d: dict) -> None:
    d["updated"] = _now()
    items = d.get("items") or []
    if len(items) > CFG["max_items"]:
        d["items"] = items[-CFG["max_items"]:]
    DATA.mkdir(exist_ok=True)
    tmp = BOOK.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(BOOK)


def _bars(sym: str, tf: str) -> list[list]:
    try:
        import market  # 同目录模块
        rows, _src = market._load(sym, tf)
        return rows or []
    except Exception:  # noqa: BLE001
        return []


def _spread(ctx: dict, sym: str = "GOLD") -> float:
    """取记录当时的点差（价格单位），拿不到用本品种兜底值。"""
    base = float(markets.spread_default(sym))
    for key in ("spread", "spread_usd"):
        v = _num(ctx.get(key))
        if v and v > 0:
            # 黄金的 ctx.spread 可能是「点」（>5 视为点），按 10 点 = 1 美金折算
            return round(v / 10.0, 3) if (v > 5 and base > 0.01) else round(v, _digits(sym))
    q = ctx.get("quote") or {}
    for key in ("spread", "spread_usd"):
        v = _num(q.get(key))
        if v and v > 0:
            return round(v / 10.0, 3) if (v > 5 and base > 0.01) else round(v, _digits(sym))
    return round(base, _digits(sym))


def _plan_parts(plan: dict) -> tuple[float | None, float | None, float | None]:
    """从计划里取 入场价（区间中值）/ 止损 / 第一目标。"""
    if not isinstance(plan, dict):
        return None, None, None
    ez = plan.get("entry_zone") or plan.get("entry") or []
    if isinstance(ez, dict):
        ez = [ez.get("from"), ez.get("to")]
    nums = [v for v in (_num(x) for x in (ez if isinstance(ez, list) else [ez]))
            if v is not None]
    entry = sum(nums) / len(nums) if nums else None
    stop = _num(plan.get("stop") or plan.get("stop_loss"))
    tps = plan.get("targets") or plan.get("take_profit") or []
    tp1 = None
    if isinstance(tps, list) and tps:
        first = tps[0]
        tp1 = _num(first.get("price")) if isinstance(first, dict) else _num(first)
    return entry, stop, tp1


def record(sym: str, cfg: dict, result: dict, ctx: dict) -> dict | None:
    """把一次三审决策记进账本。只记「有明确计划」的决策，观望不占样本位。"""
    a1 = result.get("a1") or {}
    a2 = result.get("a2") or {}
    facts = result.get("facts") or {}
    direction = a1.get("direction")
    plan = a1.get("plan")
    kind = "trade"
    if direction not in ("buy", "sell"):
        # 被本地/风控拦下的计划：留作反事实样本
        direction = a1.get("ai_direction")
        plan = a1.get("ai_plan")
        kind = "blocked"
    if direction not in ("buy", "sell"):
        return None
    entry, stop, tp1 = _plan_parts(plan)
    if entry is None or stop is None or tp1 is None:
        return None
    risk = abs(entry - stop)
    if risk <= 1e-9:
        return None
    rr = abs(tp1 - entry) / risk
    if rr < 0.2:
        return None

    bars = _bars(sym, CFG["settle_tf"])
    bar_start = bars[-1][0] if bars else None
    lots = ((a2.get("local_lots") or {}).get("lots")) if isinstance(a2.get("local_lots"), dict) else None
    risk_usd = ((a2.get("local_lots") or {}).get("risk_usd")
                if isinstance(a2.get("local_lots"), dict) else None)

    item = {
        "id": dt.datetime.now().strftime("%Y%m%d%H%M%S%f")[:18],
        "at": _now(),
        "sym": sym,
        "tf": cfg.get("tf"),
        "kind": kind,
        "direction": direction,
        "ai_direction": a1.get("ai_direction"),
        "verdict": a2.get("verdict") or "unknown",
        "confidence": a1.get("confidence"),
        "market_state": a1.get("market_state") or (facts.get("market_state") or {}).get("state"),
        "entry": round(entry, _digits(sym)),
        "stop": round(stop, _digits(sym)),
        "tp1": round(tp1, _digits(sym)),
        "rr": round(rr, 2),
        "price": _num(ctx.get("price")),
        "spread": _spread(ctx),
        "lots": lots,
        "risk_usd": risk_usd,
        "zone": (facts.get("zone") or {}).get("zone"),
        "sentiment": (facts.get("sentiment") or {}).get("label"),
        "kickoff": (facts.get("kickoff") or {}).get("tf"),
        "bar_start": bar_start,          # 结算起点（K 线序列里的位置，不依赖时钟）
        "status": "open",                # open → entered → won/lost/missed
        "entered_at": None,
        "closed_at": None,
        "exit": None,
        "r": None,
        "cost_r": None,
        "net_r": None,
        "hold_bars": None,
    }
    with LOCK:
        d = _read()
        items = d["items"]
        # 防抖：同一计划在同一轮里被反复记录时只更新，不重复占样本
        for prev in reversed(items[-3:]):
            if (prev.get("sym") == sym and prev.get("tf") == item["tf"]
                    and prev.get("direction") == direction
                    and prev.get("entry") == item["entry"]
                    and prev.get("stop") == item["stop"]
                    and prev.get("status") in ("open", "entered")
                    and prev.get("kind") == kind):
                prev["at"] = item["at"]
                prev["confidence"] = item["confidence"]
                prev["verdict"] = item["verdict"]
                _write(d)
                return prev
        items.append(item)
        _write(d)
    return item


def _close(it: dict, exit_price: float, status: str, bar_time: str) -> None:
    entry, stop = it["entry"], it["stop"]
    risk = abs(entry - stop)
    if risk <= 1e-9:
        return
    r = ((exit_price - entry) / risk) if it["direction"] == "buy" \
        else ((entry - exit_price) / risk)
    sym0 = it.get("sym") or "GOLD"
    cost = (float(it.get("spread") or markets.spread_default(sym0))
            + markets.slip(sym0)) / risk
    it["exit"] = round(exit_price, _digits(it.get("sym")))
    it["r"] = round(r, 3)
    it["cost_r"] = round(cost, 3)
    it["net_r"] = round(r - cost, 3)
    it["status"] = status
    it["closed_at"] = bar_time


def _advance(it: dict, bars: list[list]) -> bool:
    """用 M5 K 线把一条样本往前推进；返回状态是否有变化。"""
    st = it.get("status")
    if st in ("won", "lost", "missed"):
        return False
    if not bars or not it.get("bar_start"):
        return False
    times = [r[0] for r in bars]
    buy = it["direction"] == "buy"
    entry, stop, tp1 = it["entry"], it["stop"], it["tp1"]
    changed = False

    if st == "open":
        pend = bars[bisect.bisect_right(times, it["bar_start"]):]
        if not pend:
            return False
        hit = None
        for j, r in enumerate(pend):
            if (r[3] <= entry) if buy else (r[2] >= entry):
                hit = j
                break
        if hit is None:
            if len(pend) >= CFG["expire_bars"]:
                it["status"] = "missed"
                it["closed_at"] = pend[-1][0]
                it["exit"] = round(pend[-1][4], _digits(it.get("sym")))
                return True
            return False
        it["status"] = "entered"
        it["entered_at"] = pend[hit][0]
        it["entry_bar"] = pend[hit][0]
        changed = True
        work = bars[bisect.bisect_left(times, pend[hit][0]):]
    else:
        base = it.get("entry_bar") or it["bar_start"]
        work = bars[bisect.bisect_left(times, base):]

    stay = 0
    for k, r in enumerate(work):
        hi, lo, close = r[2], r[3], r[4]
        stay = k
        # 同一根 K 线里止损与目标都碰到 → 按止损算（坏情况优先，宁可低估）
        if (lo <= stop) if buy else (hi >= stop):
            _close(it, stop, "lost", r[0])
            it["hold_bars"] = stay
            return True
        if (hi >= tp1) if buy else (lo <= tp1):
            _close(it, tp1, "won", r[0])
            it["hold_bars"] = stay
            return True
    it["hold_bars"] = stay
    if len(work) >= CFG["max_hold_bars"]:
        _close(it, work[-1][4], "expired", work[-1][0])
        return True
    return changed


def settle(force: bool = False) -> dict:
    """推进所有未结样本。行情未更新时直接返回（_load 自身有 mtime 缓存）。"""
    with LOCK:
        live = [dict(x) for x in (_read().get("items") or [])
                if x.get("status") in ("open", "entered")]
    if not live:
        return {"settled": 0}
    # 锁外取行情（有 mtime 缓存，代价低），锁内推进，避免和 record 抢写
    cache: dict[str, list] = {}
    for it in live:
        sym = it.get("sym") or "GOLD"
        if sym not in cache:
            cache[sym] = _bars(sym, CFG["settle_tf"])
    n = 0
    with LOCK:
        d = _read()
        by_id = {x.get("id"): x for x in (d.get("items") or [])}
        for it in live:
            cur = by_id.get(it.get("id"))
            if not cur:
                continue
            if _advance(cur, cache.get(cur.get("sym") or "GOLD") or []):
                n += 1
        if n or force:
            _write(d)
    return {"settled": n}


def _bucket(items: list[dict]) -> dict:
    closed = [x for x in items if x.get("status") in ("won", "lost", "expired")]
    won = [x for x in closed if x.get("status") == "won"]
    lost = [x for x in closed if x.get("status") == "lost"]
    rs = [float(x.get("net_r")) for x in closed if x.get("net_r") is not None]
    risk_usd = [float(x.get("risk_usd")) for x in closed
                if x.get("risk_usd") not in (None, 0)]
    avg_r = round(sum(rs) / len(rs), 3) if rs else None
    return {
        "n": len(items),
        "open": sum(1 for x in items if x.get("status") == "open"),
        "entered": sum(1 for x in items if x.get("status") == "entered"),
        "won": len(won),
        "lost": len(lost),
        "missed": sum(1 for x in items if x.get("status") == "missed"),
        "expired": sum(1 for x in items if x.get("status") == "expired"),
        "closed": len(closed),
        "win_rate": round(len(won) / len(closed) * 100, 1) if closed else None,
        "avg_r": avg_r,
        "sum_r": round(sum(rs), 2) if rs else 0.0,
        "exp_usd": (round(avg_r * (sum(risk_usd) / len(risk_usd)), 2)
                    if (avg_r is not None and risk_usd) else None),
        "avg_risk_usd": round(sum(risk_usd) / len(risk_usd), 2) if risk_usd else None,
    }


def state(settle_now: bool = True) -> dict:
    if settle_now:
        settle()
    d = _read()
    items = d.get("items") or []
    trade = [x for x in items if x.get("kind") == "trade"]
    blocked = [x for x in items if x.get("kind") == "blocked"]
    st = _bucket(trade)
    st["target"] = CFG["target"]
    st["progress"] = round(min(100.0, len(trade) / CFG["target"] * 100), 1)
    by_dir = {}
    for k in ("buy", "sell"):
        by_dir[k] = _bucket([x for x in trade if x.get("direction") == k])
    # 分品种统计：三品种并行时，要能看出「是黄金能干、还是这套规则都能干」
    by_sym = {}
    for x in items:
        k = x.get("sym") or "GOLD"
        by_sym.setdefault(k, []).append(x)
    sym_stats = {k: _bucket([i for i in v if i.get("kind") == "trade"])
                 for k, v in by_sym.items()}
    recent = sorted(items, key=lambda x: x.get("at") or "", reverse=True)[:8]
    a_cfg = {}
    try:
        a_cfg = json.loads((DATA / "assist_config.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    ai_cfg = {}
    try:
        ai_cfg = json.loads((DATA / "ai_council_config.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return {
        "ok": True,
        "updated": d.get("updated"),
        "mode": "shadow",
        "settle_tf": CFG["settle_tf"],
        "target": CFG["target"],
        "trade": st,
        "blocked": _bucket(blocked),
        "by_dir": by_dir,
        "by_sym": sym_stats,
        "symbols": sorted(by_sym.keys()),
        "recent": recent,
        "params": {
            "min_rr": a_cfg.get("min_rr"),
            "require_sentiment": a_cfg.get("require_sentiment"),
            "expire_h": round(CFG["expire_bars"] * 5 / 60, 1),
            # 滑点/点差按品种各自扣（黄金 0.05、直盘 0.2 pips），不再是一个写死的数字
            "slip_note": "黄金 0.05 · 直盘 0.2 pips",
            # 手数是按三审的单笔风险算的，所以这里显示三审口径，别跟 02 面板的 1% 混
            "risk_pct": ai_cfg.get("risk_pct"),
        },
        "note": ("影子模式不下单：只记录「本会开的单」和「被拦下的单」，"
                 "用 M5 后续行情结算，扣点差+滑点后统计期望。"),
    }


def reset() -> dict:
    with LOCK:
        _write({"items": []})
    return {"ok": True, "items": 0}
