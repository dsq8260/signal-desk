"""辅助页 P1：条件清单 / 关键位质量 / 时段区间 / 手数计算。

设计边界（与 assist.py 一致）：
  * 全部是纯函数：只吃已收盘 K 线 + 已算好的读数，不做 IO、不碰下单接口；
  * 任何一项算不出来就返回 "—" 或空，不抛异常、不猜；
  * 所有文字结论都是「把已有数据翻译成人话」，不是预测，也不构成入场建议。
"""
from __future__ import annotations

import pxfmt
from pxfmt import pr, px   # 价格按品种精度（黄金 2 位 / 直盘 5 位）

import datetime as dt
import math

# XAUUSD：1 手 = 100 盎司，价格每走 1 美金 = 100 美金/手
CONTRACT_USD_PER_LOT = 100.0   # 兜底（黄金）；按品种用 _contract(cfg) 覆盖

# 时段按 UTC 划（换算用服务器偏移，见 cfg["server_utc_offset"]）
SESSIONS = [
    (0, 7, "亚盘", "东京时段 · 流动性薄，突破容易假"),
    (7, 12, "欧盘", "伦敦时段 · 波动开始放大"),
    (12, 16, "欧+美重叠", "伦敦纽约重叠 · 黄金日内趋势最干净的窗口"),
    (16, 21, "美盘", "纽约时段 · 数据行情多，注意事件"),
    (21, 24, "尾盘", "流动性衰减 · 新开仓性价比低"),
]


def _f(v, default=None):
    try:
        if v is None:
            return default
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _t(s) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(str(s)[:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------ 关键位质量
def enrich_levels(levels: dict, h1: list[dict], atr: float | None,
                  ind: dict, extra_lines: list[dict] | None = None,
                  ma_len: tuple = (20, 50, 100)) -> dict:
    """给每个关键位补：ATR 距离、近 120 根触碰次数、汇合标签、质量分档。

    黄金的假突破大多发生在「孤零零的价位」上——被反复触碰、且和别的口径重合的
    价位才值得等；一次都没碰过的价位只是算出来的数字。
    """
    items = levels.get("items") or []
    if not items:
        return levels
    a = _f(atr) or 0.0
    tol = max(0.30, a * 0.25) if a else 1.0
    win = h1[-120:] if h1 else []
    ma = (ind or {}).get("ma") or {}
    ma_vals = [(f"MA{ma_len[0]}", _f(ma.get("ma_fast"))),
               (f"MA{ma_len[1]}", _f(ma.get("ma_mid"))),
               (f"MA{ma_len[2]}", _f(ma.get("ma_slow")))]

    for x in items:
        p = _f(x.get("price"))
        if p is None:
            continue
        x["atr_mult"] = pr(abs(_f(x.get("dist"), 0.0)) / a) if a else None

        # 触碰：用「整根 K 线区间覆盖 价位±tol」判定，连续若干根算一次触碰
        clusters = 0
        in_run = False
        last_touch = ""
        for b in win:
            touch = (b["l"] - tol) <= p <= (b["h"] + tol)
            if touch:
                if not in_run:
                    clusters += 1
                    in_run = True
                last_touch = b["t"]
            else:
                in_run = False
        x["touches"] = clusters
        x["last_touch"] = last_touch[:16] if last_touch else ""

        # 汇合：同价格带内还有别的口径（别的关键位 / 均线 / 前日高低 / 周开）
        conf: list[str] = []
        for y in items:
            if y is x:
                continue
            q = _f(y.get("price"))
            if q is not None and abs(q - p) <= tol:
                conf.append(str(y.get("name")))
        for name, v in ma_vals:
            if v is not None and abs(v - p) <= tol:
                conf.append(f"{name} {v:.2f}")
        for e in (extra_lines or []):
            v = _f(e.get("price"))
            if v is not None and abs(v - p) <= tol:
                conf.append(str(e.get("name")))
        # 去重 + 去掉自己这类同名标签的重复
        seen, conf2 = set(), []
        for c in conf:
            if c not in seen:
                seen.add(c)
                conf2.append(c)
        x["confluence"] = conf2[:4]

        if clusters >= 3 and conf2:
            x["quality"], x["quality_note"] = "强", f"被触碰 {clusters} 次，且有 {len(conf2)} 个口径重合"
        elif clusters >= 2 or conf2:
            x["quality"], x["quality_note"] = "中", (
                f"被触碰 {clusters} 次" if clusters >= 2 else f"仅 {len(conf2)} 个口径重合")
        else:
            x["quality"], x["quality_note"] = "弱", "近 120 根 H1 没被触碰过，孤零零的价位"
    levels["atr_tol"] = pr(tol)
    return levels


# ------------------------------------------------------------------ 时段与区间
def _session_of(h_utc: int) -> tuple[str, str]:
    for lo, hi, name, note in SESSIONS:
        if lo <= h_utc < hi:
            return name, note
    return "尾盘", SESSIONS[-1][3]


def session_info(h1: list[dict], d1: list[dict], price: float,
                 offset_h: int = 2, atr: float | None = None) -> dict:
    """时区一律用 UTC 计算；K 线时间戳是 MT5 服务器时间，按 offset_h 折算。

    只输出「区间在哪、现价在区间什么位置」，不预测突破方向。
    """
    offset_h = int(offset_h or 0)
    a = _f(atr) or 0.0
    out: dict = {"offset_h": offset_h, "lines": [], "now": {}, "note": ""}

    # 当前时段（用本机 UTC 时间 + 服务器偏移推服务器时间）
    srv = dt.datetime.utcnow() + dt.timedelta(hours=offset_h)
    name, note = _session_of(srv.hour)
    active = 0 <= srv.weekday() <= 4
    if not active:
        note = "周末休市：报价/区间只作下一交易日参考"
    out["now"] = {"name": name, "note": note,
                  "server_time": srv.strftime("%Y-%m-%d %H:%M"),
                  "active": active}

    lines: list[dict] = []

    def add(name: str, v, kind: str, note: str = "") -> None:
        v = _f(v)
        if v is None:
            return
        lines.append({"name": name, "price": pr(v), "kind": kind, "note": note,
                      "dist": pr(price - v), "abs_dist": pr(abs(price - v)),
                      "atr_mult": pr(abs(price - v) / a) if a else None})

    # 亚盘区间：最近一个交易日里 UTC 0–7 点的 H1 高低
    if h1:
        last_day = (_t(h1[-1]["t"]) or dt.datetime.utcnow()).date()
        seg = []
        for b in h1[-72:]:
            t = _t(b["t"])
            if not t:
                continue
            u = t - dt.timedelta(hours=offset_h)
            if u.date() == last_day and 0 <= u.hour < 7:
                seg.append(b)
        if len(seg) >= 3:
            hi = max(b["h"] for b in seg)
            lo = min(b["l"] for b in seg)
            add("亚盘高", hi, "resist", f"亚盘 {len(seg)} 根 H1 最高")
            add("亚盘低", lo, "support", f"亚盘 {len(seg)} 根 H1 最低")
            out["asia"] = {"high": pr(hi), "low": pr(lo),
                           "width": pr(hi - lo),
                           "width_atr": pr((hi - lo) / a) if a else None,
                           "bars": len(seg)}
            if price > hi:
                out["asia"]["pos"] = "已上破亚盘区间"
            elif price < lo:
                out["asia"]["pos"] = "已下破亚盘区间"
            else:
                out["asia"]["pos"] = "在亚盘区间内"
    if "asia" not in out:
        out["asia"] = None

    # 前日高 / 低 / 收
    if d1:
        b = d1[-1]
        add("前日高", b["h"], "resist", f"{b['t'][:10]}")
        add("前日低", b["l"], "support", f"{b['t'][:10]}")
        add("前日收", b["c"], "pivot", f"{b['t'][:10]}")
        out["prev_day"] = {"date": b["t"][:10], "high": pr(b["h"]),
                           "low": pr(b["l"]), "close": pr(b["c"])}

    # 本周开盘：本周第一根已收盘 D1 的开盘（离得太远就不进区间表，避免刷屏）
    if d1:
        last = _t(d1[-1]["t"])
        if last:
            wk = last.date() - dt.timedelta(days=last.weekday())
            for b in d1:
                t = _t(b["t"])
                if t and t.date() >= wk:
                    out["week_open"] = pr(b["o"])
                    if not a or abs(price - b["o"]) / a <= 6:
                        add("本周开盘", b["o"], "pivot", f"{t.date()}")
                    break

    lines.sort(key=lambda x: x["abs_dist"])
    out["lines"] = lines
    if not lines:
        out["note"] = "已收盘 K 线不足，算不出时段区间"
    return out


# ------------------------------------------------------------------ 条件清单
def checklist(cross: list[dict], cross_w: dict, levels: dict, ind: dict,
              patterns: list[dict], risk: dict, session: dict,
              atr: float | None) -> dict:
    """把 ②④⑤⑦ 的读数翻成 7 条「命中 / 未命中 + 一句人话」。

    每条都能被反驳，比一个 0–100 的分数更可信；命中数只用来描述「条件齐不齐」。
    """
    items: list[dict] = []

    def add(name: str, hit: bool, text: str, side: str = "", why: str = "") -> None:
        items.append({"name": name, "hit": bool(hit), "text": text,
                      "side": side, "why": why})

    by_tf = {x.get("tf"): x for x in (cross or [])}
    d1, h4, h1 = by_tf.get("D1") or {}, by_tf.get("H4") or {}, by_tf.get("H1") or {}
    up_dir = d1.get("dir") if d1.get("dir") == h4.get("dir") else None
    side = up_dir if up_dir in ("bull", "bear") else ""
    a = _f(atr) or 0.0

    # 1 上级趋势
    if side:
        word = "多" if side == "bull" else "空"
        add("上级趋势", True,
            f"D1 {d1.get('trend')} · H4 {h4.get('trend')} → 只做{word}向",
            side, f"D1 结构 {d1.get('structure')} / H4 结构 {h4.get('structure')}")
    else:
        add("上级趋势", False,
            f"D1 {d1.get('trend') or '无数据'} / H4 {h4.get('trend') or '无数据'} → 不同向，不做趋势单",
            "", "D1 与 H4 方向不一致")

    # 2 H1 触发
    h1_dir = h1.get("dir")
    if h1_dir == "none" or not h1_dir:
        add("H1 触发", False, f"H1 {h1.get('trend') or '无数据'} · 没有可用的进场触发", "",
            "H1 本身无方向")
    elif side and h1_dir == side:
        add("H1 触发", True, f"H1 {h1.get('trend')}（{h1.get('structure')}）· 与上级同向",
            side, "同向触发")
    elif not side:
        add("H1 触发", False,
            f"H1 {h1.get('trend')} · 上级没有明确方向，H1 单独的方向不构成触发",
            h1_dir, "缺上级确认")
    else:
        word_h1 = "多" if h1_dir == "bull" else "空"
        add("H1 触发", False,
            f"H1 {h1.get('trend')}（{word_h1}）· 与上级反向，属于回调/反弹段",
            h1_dir, "逆上级方向，只能等回调结束")

    # 3 位置（距最近关键位 ≤ 0.5 ATR 且方向匹配才算命中）
    allit = levels.get("items") or []
    nearest = allit[0] if allit else None
    if not allit:
        add("位置", False, "没有可用的关键价位", "", "")
    elif not side:
        add("位置", False,
            f"无方向，位置只作参考：最近 {nearest['name']} 差 "
            f"{_f(nearest.get('abs_dist')):.{px()}f} 美金（{_f(nearest.get('atr_mult'), 0):.{px()}f} ATR）",
            "", "先有方向再谈位置")
    else:
        want_kind = "support" if side == "bull" else "resist"
        good = [x for x in allit
                if x.get("kind") == want_kind and _f(x.get("abs_dist"), 9e9) <= max(0.5 * a, 1.5)]
        bad = [x for x in allit
               if x.get("kind") != want_kind and x.get("kind") != "pivot"
               and _f(x.get("abs_dist"), 9e9) <= max(0.4 * a, 1.2)]
        if good:
            x = good[0]
            add("位置", True,
                f"距 {x['name']} {_f(x.get('abs_dist')):.{px()}f} 美金（{_f(x.get('atr_mult'), 0):.{px()}f} ATR），"
                f"质量 {x.get('quality') or '—'}",
                side, f"触碰 {x.get('touches')} 次 · 汇合 {('、'.join(x.get('confluence') or []) or '无')}")
        elif bad:
            x = bad[0]
            add("位置", False,
                f"正贴着{'压力' if x.get('kind') == 'resist' else '支撑'} {x['name']} {x['price']}"
                f"（差 {_f(x.get('abs_dist')):.{px()}f} 美金）· 顺这个方向空间不足",
                "", "位置在不利一侧")
        else:
            add("位置", False,
                f"最近价位 {nearest['name']} 还差 {_f(nearest.get('abs_dist')):.{px()}f} 美金"
                f"（{_f(nearest.get('atr_mult'), 0):.{px()}f} ATR）· 中间是空档",
                "", "不在支撑/压力附近，追价性价比低")

    # 4 形态
    if not patterns:
        add("形态", False, "最近一根已收盘 H1 无确认形态", "", "没有反转/延续确认")
    else:
        same = [p for p in patterns if p.get("dir") == side] if side else []
        opp = [p for p in patterns if side and p.get("dir") != side]
        if same:
            add("形态", True, f"{same[0]['name']}（{'看涨' if side == 'bull' else '看跌'}）· 方向一致",
                side, same[0].get("why") or "")
        elif opp:
            add("形态", False, f"{opp[0]['name']}（{'看涨' if opp[0].get('dir') == 'bull' else '看跌'}）· 与方向相反",
                opp[0].get("dir") or "", opp[0].get("why") or "")
        else:
            add("形态", False, f"{patterns[0]['name']} · 方向不明确", "", patterns[0].get("why") or "")

    # 5 动量
    rsi = (ind.get("rsi") or {})
    macd = (ind.get("macd") or {})
    rv, hist = _f(rsi.get("value")), _f(macd.get("hist"))
    ob, os_ = _f(rsi.get("ob"), 70), _f(rsi.get("os"), 30)
    if rv is None:
        add("动量", False, "RSI / MACD 数据不足", "", "")
    else:
        if side == "bull":
            hit = rv >= 50 and hist is not None and hist > 0
            txt = f"RSI {rv:.1f} · MACD 柱 {hist:+.2f} · " + ("多头动量在" if hit else "动量不支持做多")
        elif side == "bear":
            hit = rv <= 50 and hist is not None and hist < 0
            txt = f"RSI {rv:.1f} · MACD 柱 {hist:+.2f} · " + ("空头动量在" if hit else "动量不支持做空")
        else:
            hit = False
            txt = f"RSI {rv:.1f} · MACD 柱 {hist:+.2f} · 无方向，动量不构成条件"
        why = ""
        if rv >= ob:
            why = f"RSI 已超买（>{ob:.0f}），追多风险高"
        elif rv <= os_:
            why = f"RSI 已超卖（<{os_:.0f}），追空风险高"
        add("动量", hit, txt, side if hit else "", why)

    # 6 波动（太低没空间、太高止损容易被扫）
    ap = _f(((ind.get("atr") or {}).get("pct")))
    av = _f(((ind.get("atr") or {}).get("value")))
    if ap is None:
        add("波动", False, "ATR 分位算不出来（历史样本不足）", "", "")
    elif ap < 0.35:
        add("波动", False, f"ATR {av:.1f}（近 200 根 {ap * 100:.0f} 分位）· 波动太小，进去就是磨",
            "", "低波动容易假突破也容易没空间")
    elif ap > 0.90:
        add("波动", False, f"ATR {av:.1f}（近 200 根 {ap * 100:.0f} 分位）· 波动过热，止损易被扫",
            "", "高波动期止损要放宽或减仓")
    else:
        add("波动", True, f"ATR {av:.1f}（近 200 根 {ap * 100:.0f} 分位）· 有空间且不过热",
            side, "1.5×ATR 止损参考")

    # 7 风险闸
    gate = (risk or {}).get("event") or {}
    q = (risk or {}).get("quote") or {}
    breached = ((risk or {}).get("manual") or {}).get("breached") or []
    blocked = bool(gate.get("no_open")) or bool(q.get("spread_alert")) or bool(breached)
    if blocked:
        rs = []
        if gate.get("no_open"):
            rs.append(f"{gate.get('name')}（{gate.get('countdown')}）")
        if q.get("spread_alert"):
            rs.append(f"点差 {q.get('spread_price')} 超阈值")
        rs += list(breached)
        add("风险闸", False, "禁止新开：" + "；".join(x for x in rs if x), "", "事件窗口 / 点差 / 当日风控")
    else:
        se = session.get("now") or {}
        add("风险闸", True,
            f"无事件窗口 · 点差正常 · 当前{se.get('name') or '—'}",
            side, se.get("note") or "")

    hits = sum(1 for x in items if x["hit"])
    total = len(items)
    if hits >= total - 1:
        grade, word = "优", "条件齐备"
    elif hits >= total - 3:
        grade, word = "中", "条件一般，等更好的位置"
    else:
        grade, word = "差", "条件不足，不动手"
    return {"items": items, "hits": hits, "total": total,
            "grade": grade, "text": f"{hits}/{total} 命中 · {word}",
            "side": side, "atr": pr(a) if a else None}


def _unit() -> str:
    """当前品种的价格单位（黄金「美元/盎司」、直盘「美元（报价货币）」）。"""
    try:
        import markets
        return str(markets.of(pxfmt.sym()).get("unit") or "价格单位")
    except Exception:  # noqa: BLE001
        return "价格单位"


# ------------------------------------------------------------------ 手数计算
def sizing(balance, risk_pct, stop_usd, step: float = 0.01,
           contract: float = CONTRACT_USD_PER_LOT) -> dict:
    """单笔风险 % → 手数。黄金 1 手 = 100 盎司，止损 1 美金 = 100 美金/手。"""
    bal, pct, stop = _f(balance), _f(risk_pct), _f(stop_usd)
    if not bal or bal <= 0:
        return {"ok": False, "note": "拿不到账户余额（数据桥没写 mt5_account.csv），先填余额"}
    if not pct or pct <= 0:
        return {"ok": False, "note": "风险 % 要大于 0"}
    if not stop or stop <= 0:
        return {"ok": False, "note": "止损距离（美金）要大于 0"}
    risk_usd = bal * pct / 100.0
    per_lot = stop * contract
    raw = risk_usd / per_lot
    lots = math.floor(raw / step) * step
    below_min = lots < step
    lots = max(step, round(lots, 2)) if not below_min else 0.0
    unit = _unit()
    return {
        "ok": True, "balance": pr(bal), "risk_pct": pr(pct),
        "risk_usd": pr(risk_usd), "stop_usd": pr(stop),
        "per_lot_usd": pr(per_lot),
        "lots": round(lots, 2), "lots_raw": round(raw, 3),
        "below_min": below_min,
        "note": (f"这一单最多亏 {risk_usd:.2f} 美金（余额的 {pct:.1f}%，单笔上限）；"
                 f"止损距离 {stop:.{px()}f}（{unit}） → 建议下单 {lots:.2f} 手，"
                 "打到止损正好亏这笔钱。当天总共最多亏多少由「日亏损线」单独控制"),

    }
