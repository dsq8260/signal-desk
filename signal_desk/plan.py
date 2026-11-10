"""决策层：把「跨周期方向 / 关键位 / 指标 / 持仓 / 风控」翻译成三个结论。

页面只回答三个问题：
  1. 现在是什么趋势（五档：强势多头 → 强势空头）
  2. 已有单子怎么办（安全持有 / 警惕减仓 / 建议平仓）
  3. 空仓做不做（观望等待 / 可试做多 / 可试做空）

约束：
  * 纯函数，不做 IO，不碰下单接口，算不出来就返回 None / "—"；
  * 所有价位都来自已计算的关键位（Pivot / 波段高低点 / 中枢边界 / 时段线），
    不随手给价；止损统一按 stop_atr_mult × ATR(H1)；
  * 结论是规则的显性化，不是胜率承诺。
"""
from __future__ import annotations

import math

from pxfmt import pr, px   # 价格按品种精度（黄金 2 位 / 直盘 5 位）

CONTRACT = 100.0          # 兜底：XAUUSD 1 手 = 100 盎司（真实值由 cfg["_contract"] 按品种给）
POINT = 0.01              # 兜底：黄金 1 point = 0.01 美金（FX 由 cfg["_point"] 给）


def _contract(cfg: dict | None) -> float:
    """本品种 1 手多少「价格单位×合约」——黄金 100，直盘 100000。"""
    return _f(((cfg or {}).get("_contract")), CONTRACT) or CONTRACT


def _point(cfg: dict | None) -> float:
    return _f(((cfg or {}).get("_point")), POINT) or POINT

# 主趋势权重（文档：H4 50% / H1 30% / M15 20%）
WEIGHTS = {"H4": 0.50, "H1": 0.30, "M15": 0.20}


def _f(v, d=None):
    try:
        if v is None:
            return d
        x = float(v)
        return x if math.isfinite(x) else d
    except (TypeError, ValueError):
        return d


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def _sgn(x, eps=1e-9):
    return 1 if x > eps else (-1 if x < -eps else 0)


# ------------------------------------------------------------------ 主趋势
def _ma_state(ma: dict, price: float) -> str:
    f, m, s = _f(ma.get("fast")), _f(ma.get("mid")), _f(ma.get("slow"))
    if not (f and m and s) or not price:
        return "—"
    if f > m > s and price > f:
        return "bull"
    if f < m < s and price < f:
        return "bear"
    return "mix"


def _tf_row(t: dict | None, tf: str, price: float) -> dict:
    if not t:
        return {"tf": tf, "dir": "none", "label": "无数据", "strength": 0,
                "ma": "—", "score": 0.0, "trend": "无数据"}
    d = str(t.get("dir") or "none")
    st = _f(t.get("strength"), 0) or 0
    ma = _ma_state(t.get("ma") or {}, price)
    score = 0.0
    if d in ("bull", "bear"):
        s = _sgn(1 if d == "bull" else -1) * (0.45 + 0.55 * min(100.0, st) / 100.0)
        if (d == "bull" and ma == "bull") or (d == "bear" and ma == "bear"):
            s += 0.15
        elif ma in ("bull", "bear"):
            s -= 0.10
        score = _clip(s)
    return {"tf": tf, "dir": d, "label": t.get("trend") or "—", "strength": round(st),
            "ma": ma, "score": round(score, 3), "trend": t.get("trend") or "—"}


TREND_TXT = {
    "bull2": "强势多头", "bull1": "震荡偏多", "mix": "震荡整理",
    "bear1": "震荡偏空", "bear2": "强势空头",
}


def trend_block(cross: list[dict] | None, extra: dict | None, price: float,
                ind: dict) -> dict:
    """五档主趋势：三周期加权分 + 「三周期同向」硬条件。"""
    by_tf = {x.get("tf"): x for x in (cross or [])}
    rows = [_tf_row(by_tf.get("H4"), "H4", price),
            _tf_row(by_tf.get("H1"), "H1", price),
            _tf_row((extra or {}).get("M15"), "M15", price)]
    ok = [r for r in rows if r["dir"] in ("bull", "bear")]
    wsum = sum(WEIGHTS[r["tf"]] for r in ok) or 1.0
    total = sum(WEIGHTS[r["tf"]] * r["score"] for r in ok) / wsum

    hist = _f(((ind or {}).get("macd") or {}).get("hist"))
    if hist is not None:
        total += 0.05 * _sgn(hist)
    total = _clip(total)

    dirs = {r["dir"] for r in rows}
    aligned = (len(dirs) == 1 and dirs.pop() in ("bull", "bear")
               and all(r["ma"] in ("bull", "bear") for r in rows))
    if aligned and abs(total) >= 0.55:
        key = "bull2" if total > 0 else "bear2"
    elif total >= 0.18:
        key = "bull1"
    elif total <= -0.18:
        key = "bear1"
    else:
        key = "mix"

    bias = round(50 + 50 * total)
    parts = []
    for r in rows:
        parts.append({"tf": r["tf"], "w": int(WEIGHTS[r["tf"]] * 100),
                      "dir": r["dir"], "label": r["label"],
                      "score": r["score"], "strength": r["strength"]})
    note = " · ".join(f"{r['tf']} {r['label']}{'↑' if r['dir'] == 'bull' else '↓' if r['dir'] == 'bear' else '—'}"
                      for r in rows)
    side = "bull" if key.startswith("bull") else ("bear" if key.startswith("bear") else "mix")
    all_up = all(r["dir"] == "bull" for r in rows)
    all_dn = all(r["dir"] == "bear" for r in rows)
    reason = {
        "bull2": "H4 / H1 / M15 同向向上，均线多排，回踩做多为主",
        "bull1": ("三个周期都向上但强度不高（均线未完全排开），等回踩再接，不追"
                  if all_up else "大周期向上、小周期未同步，顺着大方向等回踩"),
        "mix": "多空互抵，价格在区间内，等一边先破再动手",
        "bear1": ("三个周期都向下但强度不高（均线未完全排开），反弹再空，不砸"
                  if all_dn else "大周期向下、小周期未同步，反弹做空为主"),
        "bear2": "H4 / H1 / M15 同向向下，均线空排，反弹做空为主",
    }[key]
    return {"key": key, "label": TREND_TXT[key], "side": side, "score": round(total, 3),
            "bias": max(0, min(100, bias)), "parts": parts, "note": note,
            "reason": reason, "rows": rows}


# ------------------------------------------------------------------ 价位工具
def _split_levels(levels: dict, price: float) -> tuple[list[dict], list[dict]]:
    items = [x for x in ((levels or {}).get("items") or []) if _f(x.get("price"))]
    sup = sorted([x for x in items if _f(x["price"]) < price],
                 key=lambda x: -_f(x["price"]))
    res = sorted([x for x in items if _f(x["price"]) > price],
                 key=lambda x: _f(x["price"]))
    return sup, res


def _pick(seq: list[dict], *, min_price=None, max_price=None,
          prefer_strong=True) -> dict | None:
    """在候选里优先挑「强」质量、且价格约束内的最近一个。"""
    rows = [x for x in seq
            if (min_price is None or _f(x["price"]) >= min_price)
            and (max_price is None or _f(x["price"]) <= max_price)]
    if not rows:
        return None
    if prefer_strong:
        strong = [x for x in rows if x.get("quality") == "强"]
        if strong:
            return strong[0]
    return rows[0]


def _rr(entry: float, stop: float, tp: float) -> float | None:
    risk = abs(entry - stop)
    return pr(abs(tp - entry) / risk) if risk > 1e-9 else None


def _lots(balance, risk_pct, stop_usd, cap=None, contract=CONTRACT) -> float | None:
    b, p, s = _f(balance), _f(risk_pct), _f(stop_usd)
    if not b or not p or not s:
        return None
    raw = b * p / 100.0 / (s * (contract or CONTRACT))
    lots = max(0.0, math.floor(raw / 0.01) * 0.01)
    if cap:
        lots = min(lots, float(cap))
    return pr(lots)


# ------------------------------------------------------------------ 持仓决策
def _group_positions(rows: list[dict], price: float, risk_ref: float,
                     sup: list[dict], res: list[dict], trail: bool,
                     cfg: dict | None = None) -> list[dict]:
    groups: dict[str, dict] = {}
    for p in rows or []:
        side = str(p.get("side") or "?")
        vol = _f(p.get("volume"), 0.0) or 0.0
        ent = _f(p.get("price_open"))
        g = groups.setdefault(side, {"side": side, "vol": 0.0, "profit": 0.0,
                                     "entry": 0.0, "w": 0.0, "hold": None})
        g["vol"] += vol
        g["profit"] += _f(p.get("profit"), 0.0) or 0.0
        if ent is not None and vol > 0:
            g["entry"] += ent * vol
            g["w"] += vol
        h = _f(p.get("hold_h"))
        if h is not None:
            g["hold"] = max(g["hold"] or 0.0, h)

    out = []
    for side, g in groups.items():
        entry = g["entry"] / g["w"] if g["w"] else None
        if not entry:
            continue
        long = side == "buy"
        d = 1 if long else -1
        r = ((price - entry) * d / risk_ref) if risk_ref else None

        # 止损：基础 1.5×ATR；r≥1 移保本；r≥1.5 锁 0.75R（可关）
        base_stop = entry - d * risk_ref
        stop = base_stop
        trail_note = "初始止损 1.5×ATR"
        if trail and r is not None:
            if r >= 1.5:
                stop = entry + d * 0.75 * risk_ref
                trail_note = "已过 1.5R，止损锁在 +0.75R"
            elif r >= 1.0:
                stop = entry
                trail_note = "已过 1R，止损移至成本（保本）"

        # 目标：入场前方最近的两个反向关键位
        cand = [x for x in (res if long else sup)]
        if long:
            cand = [x for x in cand if _f(x["price"]) > entry + 0.4 * risk_ref]
        else:
            cand = [x for x in cand if _f(x["price"]) < entry - 0.4 * risk_ref]
        tp1 = _f(cand[0]["price"]) if cand else entry + d * 2.0 * risk_ref
        tp2 = _f(cand[1]["price"]) if len(cand) > 1 else entry + d * 3.2 * risk_ref
        tp1_name = cand[0].get("name") if cand else "2R 目标"
        tp2_name = cand[1].get("name") if len(cand) > 1 else "3R 目标"

        dist_stop = (price - stop) * d
        dist_tp1 = (tp1 - price) * d
        dist_tp2 = (tp2 - price) * d
        ctr = _contract(cfg)
        pnl_stop = (stop - entry) * d * g["vol"] * ctr
        pnl_tp1 = (tp1 - entry) * d * g["vol"] * ctr
        out.append({
            "side": side, "label": "多单" if long else "空单", "long": long,
            "vol": pr(g["vol"]), "entry": pr(entry),
            "profit": pr(g["profit"]), "r": pr(r) if r is not None else None,
            "stop": pr(stop), "base_stop": pr(base_stop),
            "trail_note": trail_note, "trailing": abs(stop - base_stop) > 1e-9,
            "tp1": pr(tp1), "tp2": pr(tp2),
            "tp1_name": tp1_name, "tp2_name": tp2_name,
            "dist_stop": pr(dist_stop), "dist_stop_pt": int(round(dist_stop / POINT)),
            "dist_tp1": pr(dist_tp1), "dist_tp1_pt": int(round(dist_tp1 / POINT)),
            "dist_tp2": pr(dist_tp2),
            "pnl_stop": pr(pnl_stop), "pnl_tp1": pr(pnl_tp1),
            "risk_ref": pr(risk_ref),
            "hold_h": round(g["hold"], 1) if g["hold"] else None,
            "struct": _pick([x for x in (sup if long else res)
                             if ((_f(x["price"]) <= entry) if long
                                 else (_f(x["price"]) >= entry))],
                            prefer_strong=True),
        })
    out.sort(key=lambda x: -x["vol"])
    return out


def hold_block(risk: dict, price: float, atr: float | None, levels: dict,
               ind: dict, trend: dict, tf_extra: dict, cfg: dict,
               journal: dict | None) -> dict:
    """持仓决策：安全持有 / 警惕减仓 / 建议平仓 + 四个价位 + 双列信号 + 动作。"""
    risk_ref = (atr or 0) * max(1.0, _f(cfg.get("stop_atr_mult"), 1.5)) if atr else 0
    rows_src = ((risk or {}).get("positions") or {}).get("list") or []
    sup, res = _split_levels(levels, price)
    rows = _group_positions(rows_src, price, risk_ref, sup, res,
                            bool(cfg.get("trail_auto", True)), cfg) if risk_ref else []

    if not rows:
        st = ((journal or {}).get("stats") or {}).get("all") or {}
        today = (journal or {}).get("today") or {}
        return {"has": False,
                "empty": {"pnl_today": today.get("pnl"),
                          "n": st.get("n"), "win_rate": st.get("win_rate"),
                          "avg_rr": st.get("avg_rr"), "streak": today.get("streak_loss"),
                          "n_open": today.get("n_open_today"),
                          "balance": ((risk or {}).get("account") or {}).get("balance"),
                          "equity": ((risk or {}).get("account") or {}).get("equity")}}

    m = rows[0]
    long = m["long"]
    d = 1 if long else -1
    closes: list[str] = []
    warns: list[str] = []
    safes: list[str] = []

    # —— 平仓级（任意 1 条）
    if m["dist_stop"] <= 0:
        closes.append(f"价格已触及止损 {m['stop']:.{px()}f}")
    if m["dist_tp1"] <= 0:
        closes.append(f"已到第一目标 {m['tp1']:.{px()}f}（{m['tp1_name']}），按计划落袋")
    side_key = "bull" if long else "bear"
    h4 = next((x for x in trend["rows"] if x["tf"] == "H4"), {})
    h1 = next((x for x in trend["rows"] if x["tf"] == "H1"), {})
    if h4.get("dir") not in (side_key, "none") and h1.get("dir") not in (side_key, "none"):
        closes.append("H4 + H1 已同向反转，原入场依据不成立")
    struct = m.get("struct") or {}
    if struct:
        sp = _f(struct.get("price"))
        if sp and ((long and price < sp - 0.15 * (atr or 0))
                   or (not long and price > sp + 0.15 * (atr or 0))):
            closes.append(f"跌破关键位 {sp:.{px()}f}（{struct.get('name')}）")
    div = ((ind or {}).get("macd") or {}).get("divergence")
    rsi = _f(((ind or {}).get("rsi") or {}).get("value"))
    ob = _f(((ind or {}).get("rsi") or {}).get("ob"), 70) or 70
    os_ = _f(((ind or {}).get("rsi") or {}).get("os"), 30) or 30
    if div == ("顶背离" if long else "底背离") and rsi is not None and (
            (long and rsi >= ob) or (not long and rsi <= os_)):
        closes.append(f"{div} + RSI {rsi:.0f}，反转共振")

    # —— 减仓级（任意 2 条）
    if m["dist_tp1"] <= 0.5 * (atr or 0) and m["dist_tp1"] > 0:
        warns.append(f"距第一目标只剩 {m['dist_tp1']:.{px()}f} 美金，先减一半")
    m15 = ((tf_extra or {}).get("M15") or {}).get("dir")
    m5 = ((tf_extra or {}).get("M5") or {}).get("dir")
    if m15 not in (side_key, "none") and m5 not in (side_key, "none"):
        warns.append("M15 + M5 已反向，短线动能衰退")
    elif m5 not in (side_key, "none"):
        warns.append("M5 已反向，注意短线回撤")
    if (long and rsi is not None and rsi >= ob) or (not long and rsi is not None and rsi <= os_):
        warns.append(f"RSI {rsi:.0f} 进入{'超买' if long else '超卖'}区")
    near_res = _pick(res if long else sup, prefer_strong=True)
    if near_res and abs(_f(near_res["price"]) - price) <= 0.6 * (atr or 0):
        warns.append(f"正测试{near_res.get('name')} {near_res['price']:.{px()}f}（{near_res.get('quality')}）")
    if (long and (m["r"] or 0) >= 1.5 and trend["side"] != "bull") or \
       (not long and (m["r"] or 0) >= 1.5 and trend["side"] != "bear"):
        warns.append("已过 1.5R 但趋势不再同向，落袋为安")

    # —— 安全信号
    if trend["side"] == side_key:
        safes.append(f"主趋势 {trend['label']}，与持仓方向一致")
    if m["dist_stop"] > 0:
        safes.append(f"未触止损，仍有 {m['dist_stop']:.{px()}f} 美金缓冲")
    if m["dist_tp1"] > 0:
        safes.append(f"距第一目标还有 {m['dist_tp1']:.{px()}f} 美金（{m['dist_tp1_pt']} pt）")
    if not div:
        safes.append("MACD 未见反向背离")
    ma_state = ((ind or {}).get("ma") or {}).get("state")
    if (long and ma_state == "多头排列") or (not long and ma_state == "空头排列"):
        safes.append(f"均线{ma_state}，结构支撑持仓")
    if (m["r"] or 0) >= 1:
        safes.append(f"已浮盈 {m['r']:.2f}R，止损已上移")
    if not safes:
        safes.append("暂无可确认的持仓优势")

    if closes:
        label, cls = "建议平仓", "close"
        reason = closes[0]
    elif len(warns) >= 2:
        label, cls = "警惕减仓", "warn"
        reason = warns[0]
    else:
        label, cls = "安全持有", "safe"
        reason = safes[0]

    acts = []
    if cls == "close":
        acts.append(("第一优先", f"立即平仓 {m['label']} {m['vol']:.2f} 手，保住 {m['profit']:+.2f} 美金"))
        acts.append(("之后", "等下一次 ⑦ 条件齐了再进，不追、不补"))
    elif cls == "warn":
        acts.append(("第一优先", f"减仓 1/2，剩余止损移至 {m['stop']:.{px()}f}"))
        acts.append(("第二优先", f"触及 {m['tp1']:.{px()}f} 再减一半，余仓看 {m['tp2']:.{px()}f}"))
    else:
        acts.append(("第一优先", f"继续持有，止损 {m['stop']:.{px()}f}（{m['trail_note']}）"))
        acts.append(("第二优先", f"触及 {m['tp1']:.{px()}f}（{m['tp1_name']}）减仓 1/2，余仓看 {m['tp2']:.{px()}f}"))
    extreme = (f"跌破 {m['stop']:.{px()}f} 立即平仓，不扛单" if long
               else f"站上 {m['stop']:.{px()}f} 立即平仓，不扛单")
    pnl_at_tp1 = m["pnl_tp1"]
    return {"has": True, "label": label, "cls": cls, "reason": reason,
            "row": m, "others": rows[1:], "safe": safes[:4], "risk": warns[:4],
            "closes": closes[:3], "actions": acts, "extreme": extreme,
            "pnl_if_tp1": pr(pnl_at_tp1), "pnl_if_stop": m["pnl_stop"]}


# ------------------------------------------------------------------ 空仓方案
def _side_plan(long: bool, price: float, atr: float, sup: list[dict], res: list[dict],
               trend: dict, ind: dict, patterns: list[dict], cfg: dict,
               balance, blocked: bool) -> dict:
    risk_ref = atr * max(1.0, _f(cfg.get("stop_atr_mult"), 1.5))
    want = "bull" if long else "bear"
    rsi = _f(((ind or {}).get("rsi") or {}).get("value"))
    ob = _f(((ind or {}).get("rsi") or {}).get("ob"), 70) or 70
    os_ = _f(((ind or {}).get("rsi") or {}).get("os"), 30) or 30
    hist = _f(((ind or {}).get("macd") or {}).get("hist"), 0) or 0
    dif = _f(((ind or {}).get("macd") or {}).get("dif"), 0) or 0

    if long:
        zone = _pick([x for x in sup if price - _f(x["price"]) <= 3.0 * atr])
        base = _f(zone["price"]) if zone else price - 0.6 * atr
        entry = base + 0.12 * atr
        stop = entry - risk_ref
        cand = [x for x in res if _f(x["price"]) >= entry + 1.0 * risk_ref]
        tp1 = _f(cand[0]["price"]) if cand else entry + 2.0 * risk_ref
        tp2 = _f(cand[1]["price"]) if len(cand) > 1 else entry + 3.2 * risk_ref
        tp1_name = cand[0].get("name") if cand else "2R 目标"
        tp2_name = cand[1].get("name") if len(cand) > 1 else "3R 目标"
        zone_name = zone.get("name") if zone else "回踩区"
        zone_q = zone.get("quality") if zone else "—"
    else:
        zone = _pick([x for x in res if _f(x["price"]) - price <= 3.0 * atr])
        base = _f(zone["price"]) if zone else price + 0.6 * atr
        entry = base - 0.12 * atr
        stop = entry + risk_ref
        cand = [x for x in sup if _f(x["price"]) <= entry - 1.0 * risk_ref]
        tp1 = _f(cand[0]["price"]) if cand else entry - 2.0 * risk_ref
        tp2 = _f(cand[1]["price"]) if len(cand) > 1 else entry - 3.2 * risk_ref
        tp1_name = cand[0].get("name") if cand else "2R 目标"
        tp2_name = cand[1].get("name") if len(cand) > 1 else "3R 目标"
        zone_name = zone.get("name") if zone else "反弹区"
        zone_q = zone.get("quality") if zone else "—"

    rr = _rr(entry, stop, tp1) or 0.0
    gap = abs(price - entry)
    min_rr = _f(cfg.get("min_rr"), 1.5) or 1.5
    pat_hit = any(p.get("dir") == want for p in (patterns or []))
    pat_opp = any(p.get("dir") != want for p in (patterns or []))

    # 5 条入场条件：ok / wait / fail
    trend_ok = trend["side"] == want
    c1 = ("ok", f"主趋势 {trend['label']}，方向匹配") if trend_ok else \
         ("wait", f"主趋势 {trend['label']}，不是{'多头' if long else '空头'}") if trend["side"] == "mix" else \
         ("fail", f"主趋势 {trend['label']}，方向相反")
    if gap <= 0.35 * atr:
        c2 = ("ok", f"价格已在{'支撑' if long else '压力'}区 {entry:.{px()}f} 附近")
    elif gap <= 1.5 * atr:
        c2 = ("wait", f"等价格{'回落' if long else '反弹'}到 {entry:.{px()}f}（差 {gap:.{px()}f}）")
    else:
        c2 = ("fail", f"离 {entry:.{px()}f} 还有 {gap:.{px()}f}，位置不合适")

    if long:
        ind_ok = (rsi is None or rsi < ob) and (hist > 0 or dif > 0)
        ind_bad = (rsi is not None and rsi >= ob) or (hist < 0 and dif < 0)
    else:
        ind_ok = (rsi is None or rsi > os_) and (hist < 0 or dif < 0)
        ind_bad = (rsi is not None and rsi <= os_) or (hist > 0 and dif > 0)
    if ind_ok:
        c3 = ("ok", f"RSI {rsi:.0f} 不极端，MACD {'红柱' if hist > 0 else '绿柱'}配合"
              if rsi is not None else "MACD 方向配合")
    elif ind_bad:
        c3 = ("fail", f"RSI {rsi:.0f} / MACD 柱 {hist:.{px()}f}，方向不配合" if rsi is not None
              else "MACD 方向不配合")
    else:
        c3 = ("wait", "指标一半配合、一半不支持，等确认")

    if pat_hit:
        c4 = ("ok", "最近一根 H1 有同向确认形态")
    elif pat_opp:
        c4 = ("fail", "最近一根 H1 是反向形态")
    else:
        c4 = ("wait", "最近一根 H1 无确认形态，等收盘")

    if rr >= min_rr:
        c5 = ("ok", f"盈亏比 {rr:.2f}:1 ≥ {min_rr:g}")
    elif rr >= 1.2:
        c5 = ("wait", f"盈亏比 {rr:.2f}:1，没到 {min_rr:g}")
    else:
        c5 = ("fail", f"盈亏比仅 {rr:.2f}:1，不值得做")

    conds = [{"name": "趋势方向匹配", "s": c1[0], "text": c1[1]},
             {"name": "关键价位触及", "s": c2[0], "text": c2[1]},
             {"name": "指标信号确认", "s": c3[0], "text": c3[1]},
             {"name": "形态确认", "s": c4[0], "text": c4[1]},
             {"name": "盈亏比达标", "s": c5[0], "text": c5[1]}]
    n_ok = sum(1 for x in conds if x["s"] == "ok")
    n_fail = sum(1 for x in conds if x["s"] == "fail")
    stars = max(1, min(5, n_ok))
    if trend["key"] == "mix":
        stars = max(1, stars - 1)
    if rr < 1.2:
        stars = min(stars, 2)

    raw = bool(n_ok >= 4 and c1[0] == "ok" and c2[0] in ("ok", "wait")
               and c5[0] == "ok" and n_fail == 0)
    qualify = raw and not blocked
    block_reason = "风险闸拦截，暂不新开" if (raw and blocked) else None
    if cfg.get("block_range") and trend["key"] == "mix":
        qualify, block_reason = False, "设置为「震荡市禁止开仓」"
    if cfg.get("block_counter") and trend["side"] not in (want, "mix"):
        qualify, block_reason = False, "设置为「逆势单禁止开仓」"
    if cfg.get("block_rr") and rr < (_f(cfg.get("min_rr"), 1.5) or 1.5):
        qualify, block_reason = False, "设置为「盈亏比不足禁止开仓」"

    lots = _lots(balance, cfg.get("risk_pct"), risk_ref, cfg.get("max_lots"), _contract(cfg))
    return {"raw": raw, "long": long, "side": "long" if long else "short",
            "label": "做多方案" if long else "做空方案",
            "entry": pr(entry), "stop": pr(stop),
            "tp1": pr(tp1), "tp2": pr(tp2),
            "tp1_name": tp1_name, "tp2_name": tp2_name,
            "rr": rr, "stars": stars, "conds": conds, "n_ok": n_ok, "n_fail": n_fail,
            "zone": zone_name, "zone_q": zone_q,
            "dist_entry": pr(gap), "dist_entry_pt": int(round(gap / _point(cfg))),
            "qualify": qualify, "block_reason": block_reason,
            "risk_usd": pr(risk_ref * (lots or 0) * _contract(cfg)) if lots else None,
            "lots": lots}


def _verdict(price: float, atr: float, long: dict, short: dict, trend: dict,
             blocked: bool, gate_text: str, cfg: dict) -> dict:
    if blocked:
        if long["raw"] or short["raw"]:
            side_txt = "做多" if long["raw"] else "做空"
            plan = long if long["raw"] else short
            return {"label": "观望等待", "cls": "wait", "side": "wait",
                    "reason": f"{side_txt}条件已齐（{plan['n_ok']}/5），但风险闸拦截："
                              f"{gate_text or '暂不新开'}"}
        return {"label": "观望等待", "cls": "wait", "side": "wait",
                "reason": gate_text or "风险闸拦截，暂不新开"}
    lq, sq = long["qualify"], short["qualify"]
    if lq and sq:
        if trend["bias"] >= 55:
            sq = False
        elif trend["bias"] <= 45:
            lq = False
        else:
            lq = sq = False
    if lq:
        return {"label": "可试做多", "cls": "long", "side": "long",
                "reason": f"主趋势 {trend['label']}，回踩 {long['entry']:.{px()}f} 有支撑，盈亏比 {long['rr']:.2f}:1"}
    if sq:
        return {"label": "可试做空", "cls": "short", "side": "short",
                "reason": f"主趋势 {trend['label']}，反弹 {short['entry']:.{px()}f} 有压力，盈亏比 {short['rr']:.2f}:1"}
    side = "long" if trend["side"] == "bull" else ("short" if trend["side"] == "bear" else "mix")
    plan = long if side == "long" else short
    if side == "mix":
        reason = "震荡整理，多空都没有共振，等一边先破"
    else:
        miss = [x for x in plan["conds"] if x["s"] != "ok"]
        if plan["block_reason"]:
            reason = plan["block_reason"]
        elif miss:
            reason = miss[0]["text"]
        else:
            reason = "条件还没齐，再等等"
    return {"label": "观望等待", "cls": "wait", "side": "wait", "reason": reason}


def entry_block(price: float, atr: float, levels: dict, ind: dict, trend: dict,
                patterns: list[dict], cfg: dict, vd: dict, balance) -> dict:
    sup, res = _split_levels(levels, price)
    blocked = bool(((vd or {}).get("gate") or {}).get("blocked"))
    gate_text = ((vd or {}).get("gate") or {}).get("text") or ""
    lo = _side_plan(True, price, atr, sup, res, trend, ind, patterns, cfg, balance, blocked)
    sh = _side_plan(False, price, atr, sup, res, trend, ind, patterns, cfg, balance, blocked)
    v = _verdict(price, atr, lo, sh, trend, blocked, gate_text, cfg)
    pick = lo if v["side"] == "long" else (sh if v["side"] == "short" else None)
    acts = []
    if pick:
        acts.append(("优先方案", f"{'回落' if pick['long'] else '反弹'}至 {pick['entry']:.{px()}f} "
                     f"{'做多' if pick['long'] else '做空'}，止损 {pick['stop']:.{px()}f}，"
                     f"目标 {pick['tp1']:.{px()}f} / {pick['tp2']:.{px()}f}"))
        act2 = sh if pick["long"] else lo
        acts.append(("备选方案", f"若{'跌破' if pick['long'] else '站上'} "
                     f"{act2['entry']:.{px()}f} 反向，止损 {act2['stop']:.{px()}f}，目标 {act2['tp1']:.{px()}f}"))
    else:
        near = lo if trend["side"] != "bear" else sh
        acts.append(("优先方案", f"等 {'回落' if near['long'] else '反弹'}到 {near['entry']:.{px()}f} 再评估"))
        acts.append(("备选方案", f"若先{'站上' if not near['long'] else '跌破'} "
                     f"{(sh if near['long'] else lo)['entry']:.{px()}f} 再转向"))
    tips = []
    if trend["key"] == "mix":
        tips.append("震荡行情，轻仓试单")
    if pick and pick.get("lots"):
        tips.append(f"止损 {pick['stop']:.{px()}f}，单笔风险按 {cfg.get('risk_pct', 1):g}% 计约 {pick['lots']} 手")
    return {"label": v["label"], "cls": v["cls"], "side": v["side"], "reason": v["reason"],
            "long": lo, "short": sh, "actions": acts, "tips": tips}


# ------------------------------------------------------------------ 指标佐证
def evidence(ind: dict, levels: dict, trend: dict, extra: dict, price: float) -> dict:
    rsi = _f(((ind or {}).get("rsi") or {}).get("value"))
    ob = _f(((ind or {}).get("rsi") or {}).get("ob"), 70) or 70
    os_ = _f(((ind or {}).get("rsi") or {}).get("os"), 30) or 30
    if rsi is None:
        rsi_txt, rsi_cls = "无数据", "dim"
    elif rsi >= ob:
        rsi_txt, rsi_cls = "超买 · 不追多", "bear"
    elif rsi <= os_:
        rsi_txt, rsi_cls = "超卖 · 不追空", "bull"
    else:
        rsi_txt, rsi_cls = "中性", "dim"

    macd = (ind or {}).get("macd") or {}
    dif, dea, hist = _f(macd.get("dif")), _f(macd.get("dea")), _f(macd.get("hist"))
    cross = "金叉" if (dif is not None and dea is not None and dif > dea) else "死叉"
    zone = "零上" if (dif or 0) > 0 else "零下"
    macd_txt = f"{cross} · {zone}"
    macd_cls = "bull" if (dif or 0) > (dea or 0) and (hist or 0) > 0 else (
        "bear" if (dif or 0) < (dea or 0) and (hist or 0) < 0 else "dim")
    if macd.get("divergence"):
        macd_txt += f" · {macd['divergence']}"

    atr = _f(((ind or {}).get("atr") or {}).get("value"))
    pct = _f(((ind or {}).get("atr") or {}).get("pct"))
    if pct is None:
        atr_lv, atr_cls = "—", "dim"
    elif pct < 0.33:
        atr_lv, atr_cls = "低波动", "dim"
    elif pct < 0.67:
        atr_lv, atr_cls = "正常", "dim"
    else:
        atr_lv, atr_cls = "高波动", "warn"

    ma = (ind or {}).get("ma") or {}
    ma_state = ma.get("state") or "—"
    ma_cls = "bull" if ma_state == "多头排列" else ("bear" if ma_state == "空头排列" else "dim")
    ma_pos = "价格在均线上方" if ma.get("above") else "价格在均线下方"

    # Pivot：只留 P / R1 / R2 / S1 / S2，标出强支撑强阻力
    piv = [x for x in ((levels or {}).get("items") or []) if str(x.get("name", "")).startswith(("Pivot", "R1", "R2", "S1", "S2"))]
    order = {"Pivot 轴心 P": 0, "R1": 1, "R2": 2, "S1": 3, "S2": 4}
    piv.sort(key=lambda x: order.get(x.get("name"), 9))
    pivots = [{"name": x.get("name"), "price": _f(x.get("price")),
               "dist": pr(price - _f(x["price"])),
               "quality": x.get("quality") or "—"} for x in piv]

    # 四周期共振
    rows = []
    for tf in ("H4", "H1", "M15", "M5"):
        t = trend["rows"] if tf in ("H4", "H1") else None
        src = next((x for x in trend["rows"] if x["tf"] == tf), None) if t else \
            (extra or {}).get(tf)
        if src is None:
            rows.append({"tf": tf, "dir": "none", "arrow": "—"})
            continue
        d = src.get("dir")
        rows.append({"tf": tf, "dir": d,
                     "arrow": "↑" if d == "bull" else ("↓" if d == "bear" else "→")})
    bulls = sum(1 for r in rows if r["dir"] == "bull")
    bears = sum(1 for r in rows if r["dir"] == "bear")
    if bulls >= 3:
        res_txt = f"{bulls}/4 周期同向向上"
    elif bears >= 3:
        res_txt = f"{bears}/4 周期同向向下"
    elif bulls or bears:
        res_txt = f"多 {bulls} / 空 {bears}，未共振"
    else:
        res_txt = "四周期无方向"

    return {"rsi": {"value": rsi, "txt": rsi_txt, "cls": rsi_cls, "ob": ob, "os": os_},
            "macd": {"dif": dif, "dea": dea, "hist": hist, "txt": macd_txt,
                     "cls": macd_cls, "divergence": macd.get("divergence")},
            "atr": {"value": atr, "lv": atr_lv, "cls": atr_cls,
                    "stop": pr(atr * 1.5) if atr else None},
            "ma": {"f": _f(ma.get("ma_fast")), "m": _f(ma.get("ma_mid")),
                   "s": _f(ma.get("ma_slow")), "state": ma_state,
                   "cls": ma_cls, "pos": ma_pos},
            "pivots": pivots, "res": rows, "res_txt": res_txt,
            "bulls": bulls, "bears": bears}


# ------------------------------------------------------------------ 入口
def build(*, cross: list[dict] | None, tf_extra: dict, levels: dict, ind: dict,
          risk: dict, price: float, cfg: dict, patterns: list[dict],
          journal: dict, vd: dict) -> dict:
    atr = _f(((ind or {}).get("atr") or {}).get("value")) or max(0.5, price * 0.002)
    trend = trend_block(cross, tf_extra, price, ind)
    hold = hold_block(risk, price, atr, levels, ind, trend, tf_extra, cfg, journal)
    balance = ((risk or {}).get("account") or {}).get("balance")
    entry = entry_block(price, atr, levels, ind, trend, patterns, cfg, vd, balance)
    ev = evidence(ind, levels, trend, tf_extra, price)
    return {"trend": trend, "hold": hold, "entry": entry, "ev": ev,
            "atr": pr(atr), "price": pr(price),
            "gate": (vd or {}).get("gate") or {}}
