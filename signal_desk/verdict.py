"""决策条：把「跨周期方向 / 关键价位 / ATR / 持仓」翻译成一句可执行的话。

设计原则（P0）：
- 只做规则的显性化，不给"胜率"承诺；
- 三票制（D1 / H4 / H1）：3/3 才叫「可开」，2/3 叫「等回踩」，其余「观望」；
- 持仓动作由「方向票数 + R 倍数」决定，止损参考统一取 1.5×ATR(H1)；
- 风险闸只提示、不拦截，平仓永远不受限。
"""
from __future__ import annotations

from pxfmt import pr, px   # 价格按品种精度（黄金 2 位 / 直盘 5 位）

ATR_MULT = 1.5
TICK_STALE_S = 60


def _f(v, d=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _votes(cross: list[dict] | None) -> list[dict]:
    out = []
    for x in cross or []:
        out.append({"tf": x.get("tf") or "?", "dir": str(x.get("dir") or "none")})
    return out


def _dir_label(votes: list[dict]) -> tuple[str, str, str]:
    """返回 (dir, 标签, 说明)。dir ∈ bull / bear / mix。✓/✗ 相对最终方向。"""
    bull = sum(1 for v in votes if v["dir"] == "bull")
    bear = sum(1 for v in votes if v["dir"] == "bear")
    d = ("bull" if bull > bear else "bear" if bear > bull else "mix") if (bull or bear) else "mix"

    def _mk(v: dict) -> str:
        if d == "mix":
            return v["tf"] + ("↑" if v["dir"] == "bull" else "↓" if v["dir"] == "bear" else "—")
        return v["tf"] + ("✓" if v["dir"] == d else "✗")

    mark = "".join(_mk(v) for v in votes)
    if bull == 3:
        return "bull", "偏多", f"三周期共振 · {mark}"
    if bear == 3:
        return "bear", "偏空", f"三周期共振 · {mark}"
    if d == "bull":
        return "bull", "偏多", f"{bull}/3 · {mark}"
    if d == "bear":
        return "bear", "偏空", f"{bear}/3 · {mark}"
    if bull == 0 and bear == 0:
        return "mix", "中性", f"无明确方向 · {mark}"
    return "mix", "中性", f"多空对冲 · {mark}"


def _levels_txt(levels: dict, price: float, d: str, atr: float | None = None) -> dict:
    items = [x for x in (levels or {}).get("items") or [] if _f(x.get("price")) is not None]
    # 太近的价位（<0.2×ATR）没有触发意义，先剔除，避免"离现价 1 美金就喊转空"
    gap = max(0.2 * atr, 1.5) if atr else 3.0
    items = [x for x in items if abs(_f(x["price"]) - price) >= gap] or items
    above = min((x for x in items if _f(x["price"]) > price), key=lambda x: _f(x["price"]), default=None)
    below = max((x for x in items if _f(x["price"]) < price), key=lambda x: _f(x["price"]), default=None)
    a = _f(above["price"]) if above else None
    b = _f(below["price"]) if below else None
    if d == "bull":
        txt = (f"站上 {a:.{px()}f} 追多 · 回踩 {b:.{px()}f} 接多" if a and b
               else f"站上 {a:.{px()}f} 追多" if a else f"回踩 {b:.{px()}f} 接多" if b else "无可用关键位")
        inv = b
    elif d == "bear":
        txt = (f"跌破 {b:.{px()}f} 追空 · 反弹 {a:.{px()}f} 接空" if a and b
               else f"跌破 {b:.{px()}f} 追空" if b else f"反弹 {a:.{px()}f} 接空" if a else "无可用关键位")
        inv = a
    else:
        txt = (f"站上 {a:.{px()}f} 转多 / 跌破 {b:.{px()}f} 转空" if a and b else "等区间边界")
        inv = None
    return {"text": txt, "above": pr(a) if a else None,
            "below": pr(b) if b else None,
            "invalid": pr(inv) if inv else None,
            "above_name": (above or {}).get("name"), "below_name": (below or {}).get("name")}


def _gate(risk: dict) -> dict:
    ev = (risk or {}).get("event") or {}
    q = (risk or {}).get("quote") or {}
    reasons = []
    if ev.get("no_open"):
        reasons.append(f"事件窗口 {ev.get('name')}（{ev.get('countdown')}）")
    if q.get("spread_alert"):
        reasons.append(f"点差 {q.get('spread_price')} 美金 > 阈值 {q.get('threshold')}")
    age = _f(q.get("age_s"))
    if not q.get("ok") or (age is not None and age > TICK_STALE_S):
        reasons.append("实时报价过期")
    for b in ((risk or {}).get("manual") or {}).get("breached") or []:
        reasons.append(str(b))
    return {"blocked": bool(reasons),
            "text": ("禁止新开 · " + "；".join(reasons)) if reasons else "风险闸通过 · 可开",
            "reasons": reasons}


def _pos_block(risk: dict, price: float, atr: float | None, votes: list[dict],
               dir_label: str) -> dict | None:
    rows = ((risk or {}).get("positions") or {}).get("list") or []
    if not rows:
        return None
    risk_ref = ATR_MULT * atr if atr else None
    sides = {}
    for p in rows:
        side = str(p.get("side") or "?")
        vol = _f(p.get("volume"), 0.0) or 0.0
        ent = _f(p.get("price_open"))
        g = sides.setdefault(side, {"side": side, "volume": 0.0, "profit": 0.0,
                                    "entry": 0.0, "w": 0.0, "hold_h": None})
        g["volume"] += vol
        g["profit"] += _f(p.get("profit"), 0.0) or 0.0
        if ent is not None and vol > 0:
            g["entry"] += ent * vol
            g["w"] += vol
        h = _f(p.get("hold_h"))
        if h is not None:
            g["hold_h"] = max(g["hold_h"] or 0.0, h)
    out = []
    for side, g in sides.items():
        entry = g["entry"] / g["w"] if g["w"] else None
        want = "bull" if side == "buy" else "bear"
        votes_side = sum(1 for v in votes if v["dir"] == want)
        votes_opp = sum(1 for v in votes if v["dir"] not in (want, "none"))
        r = None
        stop_ref = dist_stop = None
        if entry and risk_ref:
            if side == "buy":
                r = (price - entry) / risk_ref
                stop_ref = entry - risk_ref
                dist_stop = price - stop_ref
            else:
                r = (entry - price) / risk_ref
                stop_ref = entry + risk_ref
                dist_stop = stop_ref - price
        if votes_side == 0:
            act, cls, note = "平仓", "close", "三周期已全部反向，原入场依据不再成立"
        elif votes_opp > votes_side:
            act, cls, note = "减仓 / 平仓", "close", f"反向票 {votes_opp} > 顺向票 {votes_side}"
        elif r is not None and r >= 1.5 and votes_side < 3:
            act, cls, note = "减半 + 保本", "trim", "已过 1.5R，且三票不再满员"
        elif r is not None and r >= 1.0:
            act, cls, note = "保本", "be", "已过 1R，止损移至入场，让利润奔跑"
        else:
            act, cls, note = "持有", "hold", f"顺向票 {votes_side}/3，结构未破"
        out.append({
            "side": side, "label": "多单" if side == "buy" else "空单",
            "volume": pr(g["volume"]), "entry": pr(entry) if entry else None,
            "profit": pr(g["profit"]), "r": pr(r) if r is not None else None,
            "risk_ref": pr(risk_ref) if risk_ref else None,
            "stop_ref": pr(stop_ref) if stop_ref else None,
            "dist_stop": pr(dist_stop) if dist_stop is not None else None,
            "hold_h": round(g["hold_h"], 1) if g["hold_h"] else None,
            "votes_side": votes_side, "signal_ok": votes_side >= 2,
            "action": act, "action_cls": cls, "note": note,
        })
    if len(out) > 1:
        out.sort(key=lambda x: -abs(x["profit"]))
    return {"rows": out, "multi": len(out) > 1}


def build(cross: list[dict] | None, levels: dict, ind: dict, risk: dict,
          price: float) -> dict:
    votes = _votes(cross)
    d, dir_label, dir_note = _dir_label(votes)
    atr = _f(((ind or {}).get("atr") or {}).get("value"))
    gate = _gate(risk)
    trig = _levels_txt(levels, price, d, atr)
    pos = _pos_block(risk, price, atr, votes, dir_label)

    if pos:
        act = pos["rows"][0]
        action = act["action"]
        cls = act["action_cls"]
        note = act["note"]
        # 风险闸不影响持仓动作，只在提示里保留
    else:
        if d == "bull":
            action, cls = ("可开多", "long") if sum(1 for v in votes if v["dir"] == "bull") == 3 else ("等回踩", "wait")
        elif d == "bear":
            action, cls = ("可开空", "short") if sum(1 for v in votes if v["dir"] == "bear") == 3 else ("等反弹", "wait")
        else:
            action, cls = "观望", "wait"
        note = ""
        if gate["blocked"] and cls in ("long", "short"):
            action, cls, note = "只许观察", "wait", "风险闸拦截，不建议新开"
    return {"dir": d, "dir_label": dir_label, "dir_note": dir_note, "votes": votes,
            "action": action, "action_cls": cls, "action_note": note,
            "trigger": trig, "gate": gate, "atr": pr(atr) if atr else None,
            "risk_ref": pr(ATR_MULT * atr) if atr else None,
            "position": pos, "price": pr(price)}
