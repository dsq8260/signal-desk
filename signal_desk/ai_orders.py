"""半自动交易工单：AI-1 出计划 → AI-2 风控放行 → 人工确认 → 机器盯盘提醒。

半自动的「自动」到哪一步（别改边界）
--------------------------------------
  * 机器负责：算价位、算手数、判断是否进入入场区、判断止损/目标是否触及、
    到点提醒、全程写事件流。
  * 人负责：按不按「已入场」、要不要放弃、真仓下单动作本身。
  * 默认**不发单**。唯一触达执行通道的入口是 exec_order()，必须同时满足
    「配置里 exec_enabled=true」「账户类型是 demo」，缺一条直接拒绝。
    真仓一键下单这条口子本模块不提供——不是做不到，是不该默认给。

数值可信度
----------
  * 手数不由 LLM 决定、也不接受前端输入：净值 × 风险% ÷ (止损距离 × 合约)
    反算，向下取到 0.01，硬上限 0.05；算出小于最小手数时给最小手数 + 明确警告，
    并且该工单**禁止一键执行**，只能手动做。
  * LLM 给的价位必须过本地合理性校验：止损方向错、区间宽于 2×ATR、
    入场区离现价超过 3×ATR，一律用 ATR 兜底重算——模型算数不可靠。

数据：data/ai_orders.json，每单一条，事件内嵌，只留最近 KEEP 单。
"""

from __future__ import annotations

import datetime as dt
import json
import math
import threading

HERE = __import__("pathlib").Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
FILE = DATA / "ai_orders.json"
_LOCK = threading.Lock()

CFG = {
    "risk_pct": 0.5,        # 单笔风险 = 净值 × 0.5%（三审制风控口径）
    "max_lots": 0.05,
    "lot_step": 0.01,
    "lot_min": 0.01,
    "contract": 100.0,      # GOLD 1 手 = 100 盎司
    "atr_sl": 1.5,          # 止损 = 1.5 × ATR（AI 给的止损只在合理范围内才采纳）
    "atr_tp": 3.0,
    "max_sl_atr": 3.0,      # 止损最远 3×ATR，再远就是「计划有问题」
    "zone_atr": 0.25,       # 兜底入场区 = 现价 ± 0.25 ATR
}
EXPIRE_H = {"M15": 2, "M30": 4, "H1": 8, "H4": 24, "D1": 72}
KEEP = 60
FMT = "%Y-%m-%d %H:%M"


# ---------------------------------------------------------------- 存储


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _store() -> dict:
    try:
        d = json.loads(FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        d = {}
    if not isinstance(d, dict):
        d = {}
    d.setdefault("seq", 0)
    d.setdefault("orders", [])
    d.setdefault("exec_enabled", False)
    return d


def _save(d: dict) -> None:
    d["orders"] = d["orders"][-KEEP:]
    try:
        FILE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _now() -> dt.datetime:
    return dt.datetime.now()


def _ts(t: dt.datetime | None = None) -> str:
    return (t or _now()).strftime("%Y-%m-%d %H:%M:%S")


def _ev(o: dict, kind: str, text: str, level: str = "info") -> dict:
    e = {"at": _ts(), "type": kind, "text": text, "level": level}
    o.setdefault("events", []).append(e)
    o["events"] = o["events"][-40:]
    return e


# ---------------------------------------------------------------- 行情事实


def facts() -> dict:
    """当前报价 / ATR / 净值 / 持仓。只读 assist 的计算缓存，不发任何请求。"""
    try:
        import assist
        s = assist.payload("GOLD", None, False) or {}
    except Exception:  # noqa: BLE001
        return {}
    risk = s.get("risk") or {}
    q = risk.get("quote") or {}
    acct = risk.get("account") or {}
    ind = s.get("indicators") or {}
    age = _f(q.get("age_s"))
    return {
        "bid": _f(q.get("bid")),
        "ask": _f(q.get("ask")),
        "price": _f(s.get("price")),
        "age": age,
        "fresh": age is not None and age <= 120 and bool(q.get("ok")),
        "atr": _f((ind.get("atr") or {}).get("value")),
        "equity": _f(acct.get("equity")),
        "balance": _f(acct.get("balance")),
        "positions": (risk.get("positions") or {}).get("list") or [],
        "h1": ((s.get("tf_extra") or {}).get("H1") or {}).get("trend_key"),
        "d1": ((s.get("tf_extra") or {}).get("D1") or {}).get("trend_key"),
    }


# ---------------------------------------------------------------- 价位与手数


def _prices(a1: dict, f: dict) -> tuple[dict | None, str]:
    """把 AI-1 的点位校成一张能执行的计划；不合理就用 ATR 兜底。"""
    atr, side = f.get("atr"), a1.get("direction")
    bid, ask = f.get("bid"), f.get("ask")
    ref = (ask if side == "buy" else bid) or f.get("price")
    if not atr or atr <= 0:
        return None, "拿不到 ATR，行情还没就绪"
    if not ref:
        return None, "拿不到报价"

    zone = [_f(x) for x in (a1.get("entry_zone") or [])]
    zone = [x for x in zone if x]
    if len(zone) >= 2:
        lo, hi = min(zone[0], zone[1]), max(zone[0], zone[1])
    else:
        lo, hi = ref - CFG["zone_atr"] * atr, ref + CFG["zone_atr"] * atr
    # 区间宽于 2 ATR（等于把整个波段都叫入场区）→ 直接按现价重算
    if hi - lo > 2 * atr or hi <= lo:
        lo, hi = ref - CFG["zone_atr"] * atr, ref + CFG["zone_atr"] * atr
    # 入场区离现价超过 3 ATR → 挂上去也基本不会触发，按现价重算
    if (side == "buy" and lo > ref + 3 * atr) or (side == "sell" and hi < ref - 3 * atr):
        lo, hi = ref - CFG["zone_atr"] * atr, ref + CFG["zone_atr"] * atr

    sl_far = CFG["atr_sl"] * atr
    stop = _f(a1.get("stop"))
    if side == "buy":
        # 多头止损必须在入场区下方，且不能贴着区间（<0.2 ATR 等于没设）
        if stop is None or stop >= lo - 0.2 * atr:
            stop = lo - sl_far
        if lo - stop > CFG["max_sl_atr"] * atr:
            stop = lo - sl_far
        if lo - stop < 0.3 * atr:
            stop = lo - 0.3 * atr
    else:
        if stop is None or stop <= hi + 0.2 * atr:
            stop = hi + sl_far
        if stop - hi > CFG["max_sl_atr"] * atr:
            stop = hi + sl_far
        if stop - hi < 0.3 * atr:
            stop = hi + 0.3 * atr

    tps = [_f(x) for x in (a1.get("targets") or [])]
    if side == "buy":
        tps = sorted(x for x in tps if x and x > hi + 0.5 * atr)
    else:
        tps = sorted((x for x in tps if x and x < lo - 0.5 * atr), reverse=True)
    if side == "buy":
        tp1 = tps[0] if tps else hi + CFG["atr_tp"] * atr
        tp2 = tps[1] if len(tps) > 1 else tp1 + (CFG["atr_tp"] + 0.5) * atr
    else:
        tp1 = tps[0] if tps else lo - CFG["atr_tp"] * atr
        tp2 = tps[1] if len(tps) > 1 else tp1 - (CFG["atr_tp"] + 0.5) * atr

    entry = round((lo + hi) / 2, 2)
    return {"side": side, "entry_zone": [round(lo, 2), round(hi, 2)],
            "entry": entry, "stop": round(stop, 2),
            "tp1": round(tp1, 2), "tp2": round(tp2, 2),
            "sl_dist": round(abs(entry - stop), 2),
            "atr": round(atr, 2)}, ""


def _lots(plan: dict, equity: float | None) -> tuple[float, float | None, str]:
    """手数 = 净值×风险% ÷ (止损距离×合约)。返回 (手数, 实际风险USD, 警告)。"""
    sl_dist = plan.get("sl_dist") or 0
    if sl_dist <= 0:
        return CFG["lot_min"], None, "止损距离为 0，无法反算手数"
    if not equity or equity <= 0:
        r = sl_dist * CFG["contract"] * CFG["lot_min"]
        return CFG["lot_min"], round(r, 2), "拿不到账户净值，按最小手数保守给"
    raw = equity * CFG["risk_pct"] / 100.0 / (sl_dist * CFG["contract"])
    lots = math.floor(raw / CFG["lot_step"]) * CFG["lot_step"]
    lots = round(min(lots, CFG["max_lots"]), 2)
    warn = ""
    if lots < CFG["lot_min"]:
        lots = CFG["lot_min"]
        risk = sl_dist * CFG["contract"] * lots
        warn = (f"净值的 {CFG['risk_pct']}% 不够开最小手数，"
                f"{lots:.2f} 手实际风险约 {risk:.0f} USD"
                f"（{risk / equity * 100:.1f}% 净值），只能手动做")
    risk = lots * sl_dist * CFG["contract"]
    return lots, round(risk, 2), warn


# ---------------------------------------------------------------- 生成 / 查询


def _active(d: dict) -> list[dict]:
    return [o for o in d["orders"] if o.get("status") in ("pending", "armed", "entered")]


def create(tf: str | None = None) -> dict:
    """从最近一次 AI-1 结论生成一张待确认工单。"""
    import ai_council
    st = ai_council.state()
    res = st.get("last") or {}
    a1 = res.get("a1") or {}
    if not a1:
        return {"ok": False, "error": "还没有 AI-1 结论，先点「运行三角色分析」"}
    if a1.get("direction") not in ("buy", "sell"):
        return {"ok": False, "error": "AI-1 当前是观望，不生成工单"}

    with _LOCK:
        d = _store()
        act = _active(d)
        if act:
            return {"ok": False, "error": f"已有进行中的工单 {act[-1]['id']}，先处理掉再生成"}

        f = facts()
        plan, err = _prices(a1, f)
        if not plan:
            return {"ok": False, "error": err}
        lots, risk_usd, warn = _lots(plan, f.get("equity"))

        gates = res.get("gates") or []
        a2 = res.get("a2") or {}
        blocks = [f"{g.get('gate')}：{g.get('why')}" for g in gates if g.get("hit")]
        if gates and not blocks:                       # 兼容旧格式
            blocks = [f"{g.get('gate')}：{g.get('why')}" for g in gates]
        if a2.get("verdict") == "veto":
            blocks.append("AI-2 风控否决")

        # 逆势单提示：M30 出多头但 H1/D1 是下行，噪音概率高
        h1 = (f.get("h1") or "").lower()
        tf_warn = ""
        if (plan["side"] == "buy" and h1.startswith("down")) or \
           (plan["side"] == "sell" and h1.startswith("up")):
            tf_warn = f"与 H1 方向相反（H1={h1}），逆势单，噪声概率高"

        use_tf = tf or res.get("tf") or ai_council._cfg().get("tf") or "M30"
        now = _now()
        d["seq"] = int(d.get("seq") or 0) + 1
        o = {
            "id": "O" + now.strftime("%m%d") + f"-{d['seq']}",
            "at": _ts(now), "tf": use_tf, "symbol": "GOLD",
            "side": plan["side"], "entry_zone": plan["entry_zone"],
            "entry": plan["entry"], "stop": plan["stop"],
            "tp1": plan["tp1"], "tp2": plan["tp2"],
            "sl_dist": plan["sl_dist"], "atr": plan["atr"],
            "volume": lots, "risk_usd": risk_usd,
            "risk_pct": round((risk_usd or 0) / f["equity"] * 100, 2) if f.get("equity") else None,
            "risk_warn": warn, "tf_warn": tf_warn,
            "status": "blocked" if blocks else "pending",
            "blocked_by": blocks,
            "fired": [], "events": [], "done_reason": None,
            "expires": (now + dt.timedelta(hours=EXPIRE_H.get(use_tf, 4))).strftime(FMT),
            "a2_verdict": a2.get("verdict"),
            "a1_confidence": a1.get("confidence"),
            "a1_why": (a1.get("reasons") or [])[:3],
            "invalid": a1.get("invalid"),
            "quote_at": _ts(now),
            "price_at_create": f.get("price"),
        }
        if blocks:
            _ev(o, "blocked", "风控未放行，工单只能观察：" + "；".join(blocks), "bad")
        else:
            _ev(o, "created", f"工单生成：{use_tf} {_side_cn(plan['side'])} "
                              f"{lots:.2f} 手，待价格进入 {plan['entry_zone'][0]}–{plan['entry_zone'][1]}")
        # 生成时价格已在区内 → 直接进 armed，省一次等待
        if not blocks:
            px = (f.get("bid") if plan["side"] == "buy" else f.get("ask")) or f.get("price")
            if px and ((plan["side"] == "buy" and px <= plan["entry_zone"][1]) or
                       (plan["side"] == "sell" and px >= plan["entry_zone"][0])):
                o["status"] = "armed"
                o["fired"].append("armed")
                _ev(o, "armed", f"价格 {px:.2f} 已在入场区内，可以直接确认入场", "good")
        d["orders"].append(o)
        _save(d)
    return {"ok": True, "order": o}


def _side_cn(s: str) -> str:
    return "做多" if s == "buy" else "做空"


def ack(oid: str, action: str, price=None) -> dict:
    """人工确认：enter=已入场 / drop=放弃。"""
    with _LOCK:
        d = _store()
        o = next((x for x in d["orders"] if x.get("id") == oid), None)
        if not o:
            return {"ok": False, "error": "工单不存在"}
        st = o.get("status")
        if action == "enter":
            if st not in ("pending", "armed"):
                return {"ok": False, "error": f"当前状态 {st} 不能登记入场"}
            px = _f(price)
            if not px:
                f = facts()
                px = (f.get("ask") if o["side"] == "buy" else f.get("bid")) or f.get("price")
            o["status"] = "entered"
            o["entered"] = {"at": _ts(), "price": round(px, 2) if px else None}
            slip = None
            if px:
                slip = round((px - o["entry"]) * (1 if o["side"] == "buy" else -1), 2)
            _ev(o, "entered", f"人工确认入场 @{px:.2f}" if px else "人工确认入场",
                "good" if slip is None or slip <= 0 else "warn")
            if slip is not None and slip > 0:
                o["events"][-1]["text"] += f"，比计划价差 {slip:.2f} 美元"
        elif action == "drop":
            if st not in ("pending", "armed", "entered", "blocked"):
                return {"ok": False, "error": f"当前状态 {st} 不能放弃"}
            o["status"] = "done"
            o["done_reason"] = "drop"
            o["done_at"] = _ts()
            _ev(o, "drop", "手动放弃工单")
        else:
            return {"ok": False, "error": "未知动作"}
        _save(d)
        return {"ok": True, "order": o, "listing": listing()}


def exec_order(oid: str) -> dict:
    """一键执行（默认关闭，仅模拟盘）。真仓一律拒绝。"""
    d = _store()
    if not d.get("exec_enabled"):
        return {"ok": False, "error": "一键执行默认关闭（刻意不提供页面开关）；需要模拟盘自动下单时在 ai_orders 配置里显式开启"}
    o = next((x for x in d["orders"] if x.get("id") == oid), None)
    if not o:
        return {"ok": False, "error": "工单不存在"}
    if o.get("status") not in ("pending", "armed"):
        return {"ok": False, "error": f"当前状态 {o['status']} 不能执行"}
    if o.get("risk_warn"):
        return {"ok": False, "error": "这单风险超预算，只允许手动执行"}
    if o.get("blocked_by"):
        return {"ok": False, "error": "风控未放行"}
    try:
        import ai_trader
        acct = ai_trader.account() or {}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"拿不到账户：{type(e).__name__}: {e}"}
    if str(acct.get("type") or "").lower() != "demo":
        return {"ok": False, "error": f"账户类型是 {acct.get('type')}，一键执行只在模拟盘开放"}
    pl = {"symbol": "GOLD#", "side": o["side"], "volume": o["volume"],
          "sl": o["stop"], "tp": o["tp1"], "comment": "AIDesk-" + o["id"]}
    try:
        r = ai_trader.exec_open(pl)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if not r.get("ok"):
        return {"ok": False, "error": r.get("error") or "执行失败", "raw": r}
    with _LOCK:
        o["status"] = "entered"
        o["entered"] = {"at": _ts(), "price": _f(r.get("price")) or o["entry"],
                        "ticket": r.get("ticket"), "via": r.get("via")}
        _ev(o, "exec", f"模拟盘一键执行成交 #{r.get('ticket')} @{r.get('price')}", "good")
        _save(d)
    return {"ok": True, "raw": r, "order": o, "listing": listing()}


def set_exec(on: bool) -> dict:
    with _LOCK:
        d = _store()
        d["exec_enabled"] = bool(on)
        _save(d)
    return {"ok": True, "exec_enabled": bool(on)}


# ---------------------------------------------------------------- 盯盘


def tick() -> list[dict]:
    """每 15 秒一次：进区 / 破位 / 止损 / 目标 / 过期，只发提醒，不做动作。"""
    # 没有进行中的工单就完全不碰行情计算（assist.payload 有计算锁，空闲时别抢）
    if not any(o.get("status") in ("pending", "armed", "entered")
               for o in _store().get("orders") or []):
        return []
    f = facts()
    if not f or not f.get("fresh"):
        return []
    bid, ask = f.get("bid"), f.get("ask")
    if not bid:
        return []
    out: list[dict] = []
    with _LOCK:
        d = _store()
        now = _now()
        for o in d["orders"]:
            st = o.get("status")
            if st not in ("pending", "armed", "entered"):
                continue
            side = o.get("side")
            lo, hi = (o.get("entry_zone") or [None, None])[:2]
            stop, tp1, tp2 = o.get("stop"), o.get("tp1"), o.get("tp2")
            px = bid if side == "buy" else ask
            if not px or lo is None or stop is None:
                continue
            fired = o.setdefault("fired", [])

            # 过期：计划的时间窗过了就作废，别留一张两周前的单在页面上
            try:
                exp = dt.datetime.strptime(o.get("expires") or "", FMT)
            except ValueError:
                exp = None
            if exp and now > exp and st in ("pending", "armed"):
                o["status"] = "expired"
                o["done_at"] = _ts(now)
                out.append(_ev(o, "expired", f"计划过期未触发（截止 {o['expires']}）", "warn"))
                continue

            if st in ("pending", "armed"):
                inzone = (px <= hi) if side == "buy" else (px >= lo)
                if inzone and "armed" not in fired:
                    fired.append("armed")
                    o["status"] = "armed"
                    out.append(_ev(o, "armed",
                                   f"价格 {px:.2f} 进入入场区 {lo}–{hi}，可以确认入场", "good"))
                if (side == "buy" and px <= stop) or (side == "sell" and px >= stop):
                    o["status"] = "void"
                    o["done_at"] = _ts(now)
                    out.append(_ev(o, "void",
                                   f"价格 {px:.2f} 已穿越止损位 {stop}，计划前提作废", "bad"))
                continue

            # 已入场：盯止损与目标
            if (side == "buy" and px <= stop) or (side == "sell" and px >= stop):
                o["status"] = "done"
                o["done_reason"] = "stop"
                o["done_at"] = _ts(now)
                out.append(_ev(o, "stop", f"触及止损 {stop}（现价 {px:.2f}），该离场了", "bad"))
                continue
            if tp1 is not None and ((side == "buy" and px >= tp1) or
                                    (side == "sell" and px <= tp1)):
                if "tp1" not in fired:
                    fired.append("tp1")
                    out.append(_ev(o, "tp1", f"触及目标1 {tp1}（现价 {px:.2f}），可减仓/移保本", "good"))
                if tp2 is not None and ((side == "buy" and px >= tp2) or
                                        (side == "sell" and px <= tp2)):
                    if "tp2" not in fired:
                        fired.append("tp2")
                        o["status"] = "done"
                        o["done_reason"] = "tp2"
                        o["done_at"] = _ts(now)
                        out.append(_ev(o, "tp2", f"触及目标2 {tp2}（现价 {px:.2f}），计划完成", "good"))
        if out:
            _save(d)
    return out


# ---------------------------------------------------------------- 给页面


def listing(days: int = 0) -> dict:
    d = _store()
    orders = list(reversed(d["orders"]))
    act = [o for o in orders if o.get("status") in ("pending", "armed", "entered")]
    evs = []
    for o in orders[:12]:
        for e in (o.get("events") or []):
            evs.append({**e, "oid": o["id"]})
    evs.sort(key=lambda x: x.get("at") or "", reverse=True)
    f = facts()
    return {"ok": True, "active": act[:1], "orders": orders[:8],
            "events": evs[:14], "exec_enabled": bool(d.get("exec_enabled")),
            "quote": {"bid": f.get("bid"), "ask": f.get("ask"),
                      "age": f.get("age"), "fresh": f.get("fresh")},
            "cfg": {"risk_pct": CFG["risk_pct"], "max_lots": CFG["max_lots"]}}


def reset_all() -> dict:
    """清空工单（调试/月初用，单独入口，不在页面暴露）。"""
    with _LOCK:
        d = _store()
        d["orders"] = []
        _save(d)
    return {"ok": True}
