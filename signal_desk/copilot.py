"""智能辅助（Copilot）—— 实时监测 + 信息信号 + 账户建议，人工决策层。

设计边界（别改）：
  * 只把「已经发生的事实」标成偏多 / 偏空 / 中性 / 风险提示，**不预测涨跌**。
    本项目方向预测已被三次证伪（101 次网格 Deflated p=0.934 / AI 评分卡 −0.048R /
    PPO 验证段 +2.9% vs 持有 +15.1%）。信号是给人看的资料，不是交易指令。
  * 平仓 / 止损 / 止盈给的是「按 ATR 与风险预算算出的参考位」，
    是否执行完全由人决定；本模块不下单、不写任何自动执行状态。
  * 复用只读数据：ops（账户/持仓/日历/闸门/报价）、market（K 线指标）、news（关键词分级）。
    不新建数据层、不新增外部数据源；没有的数据明确列为缺口，不包装成全覆盖。
  * 与决策留痕数据层共用：brief.log_decision() / brief.logs()。
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

import brief as prebrief

try:
    import finbert_sent as fb           # 可选增强：本地 FinBERT 新闻情绪，缺失即降级
except Exception:  # noqa: BLE001
    fb = None

# 项目级数据缺口沿用简报清单，另补两条本页特有的
GAPS = [dict(x) for x in prebrief.GAPS] + [
    {"name": "新闻语义 / 基本面状态", "state": "部分",
     "why": "逐条倾向来自关键词与题材规则，不是语义阅读；只有事件级信息，没有状态与预期差"},
    {"name": "方向预测能力", "state": "缺失",
     "why": "历史回测未证明有 alpha；本页只汇总信息倾向，刻意不做涨跌预测"},
]

DISCLAIMER = ("本页只做信息汇总：逐条标偏多/偏空、给账户与持仓的参考位，"
              "不预测涨跌、不替你下单、不自动交易。参考位按 ATR 与净值 1% 风险预算计算，"
              "不含隔夜利息与真实滑点。最终进场、离场、止损、止盈都由你决定。")

DIR_LABEL = {"bull": "偏多", "bear": "偏空", "none": "中性", "caution": "风险提示"}


def _num(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _fmt(v, nd=2, suffix=""):
    return "—" if v is None else ("%.*f%s" % (nd, v, suffix))


def _clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def _sig(src, src_label, direction, title, why, weight=1, at="", extra=None):
    d = {"src": src, "src_label": src_label, "dir": direction,
         "dir_label": DIR_LABEL.get(direction, "中性"),
         "title": str(title), "why": str(why), "weight": weight, "at": at}
    if extra:
        d.update(extra)
    return d


# ------------------------------------------------------------------ 报价
def _quote(ops: dict, bars: dict) -> dict:
    for q in (ops.get("quotes") or []):
        s = str(q.get("symbol") or "").upper()
        if "GOLD" in s or "XAU" in s:
            bid, ask = _num(q.get("bid")), _num(q.get("ask"))
            if bid is not None:
                return {"symbol": str(q.get("symbol")), "bid": bid, "ask": ask,
                        "spread_pt": round((ask - bid) / 0.01, 1) if ask is not None else None,
                        "src": "实时报价"}
    live = (bars or {}).get("live") or {}
    bid, ask = _num(live.get("bid")), _num(live.get("ask"))
    if bid is not None:
        return {"symbol": (bars or {}).get("symbol") or "GOLD", "bid": bid, "ask": ask,
                "spread_pt": round((ask - bid) / 0.01, 1) if ask is not None else None,
                "src": "行情桥 tick"}
    last = (bars or {}).get("last") or {}
    return {"symbol": (bars or {}).get("symbol") or "GOLD",
            "bid": _num(last.get("c")), "ask": None, "spread_pt": None,
            "src": "K 线收盘（无实时 tick）"}


# ------------------------------------------------------------------ 信号流
def _news_signals(news: dict, sent: dict | None = None) -> list[dict]:
    """逐条新闻倾向。英文标题优先用 FinBERT 语义标签，其余沿用关键词规则。

    模型标签只是「这段文字正面/负面」，不是涨跌预测；来源写进每条信号，
    便于用户判断该信多少。
    """
    out = []
    scores = ((sent or {}).get("scores") or {})
    for x in (news.get("items") or [])[:24]:
        d = str(x.get("direction") or "none")
        d = d if d in ("bull", "bear", "none") else "none"
        impact = int(_num(x.get("impact"), 1) or 1)
        w = {3: 3, 2: 2, 1: 1}.get(impact, 1)
        label = x.get("dir_label") or DIR_LABEL[d]
        title = str(x.get("title") or "（无标题）")
        src = "新闻 · " + str(x.get("topic") or "综合")
        extra = {"impact": impact, "topic": x.get("topic"), "url": x.get("url") or ""}
        tail = " · 关键词规则"
        r = scores.get(title)
        if r and float(r.get("score") or 0) >= 0.55:
            lb = str(r.get("label") or "")
            d = {"positive": "bull", "negative": "bear", "neutral": "none"}.get(lb, d)
            label = {"positive": "模型偏多", "negative": "模型偏空",
                     "neutral": "模型中性"}.get(lb, label)
            src += " · FinBERT"
            extra["sentiment"] = {"source": "finbert", "label": lb, "score": r["score"]}
            tail = " · FinBERT %s %.2f" % (lb, float(r["score"]))
        out.append(_sig(
            "news", src, d, title,
            "%s影响 · %s · %s%s" % (x.get("impact_label") or "低", label,
                                 ("%s 分钟前" % int(_num(x.get("age_min"), 0) or 0))
                                 if x.get("age_min") is not None else (x.get("hm") or ""),
                                 tail),
            weight=w, at=str(x.get("hm") or ""), extra=extra))
    return out


def _en_signals(en: dict, sent: dict) -> list[dict]:
    """英文财经标题：FinBERT 文字情绪。

    重要：模型判的是「这段文字偏正面/负面」，不是对黄金的看涨/看跌。
    例如 "dollar strengthened" 措辞偏正面，对黄金却通常偏空，所以标签写「文字偏多」，
    并在每条理由里注明不等同方向，也不并入综合倾向。
    """
    out = []
    scores = (sent or {}).get("scores") or {}
    if not (en or {}).get("ok") or not scores:
        return out
    for x in ((en or {}).get("items") or [])[:12]:
        title = str(x.get("title") or "").strip()
        r = scores.get(title)
        if not r:
            continue
        lb = str(r.get("label") or "")
        d = {"positive": "bull", "negative": "bear", "neutral": "none"}.get(lb, "none")
        dl = {"positive": "文字偏多", "negative": "文字偏空",
              "neutral": "文字中性"}.get(lb, "文字中性")
        out.append(_sig(
            "en_news", "英文快讯 · FinBERT", d, title,
            "文字情绪 %s %.2f · 只是措辞分类，不等于黄金涨跌方向；不计入综合倾向 · WSJ Markets"
            % (lb, float(r.get("score") or 0)),
            weight=2, at=str(x.get("hm") or ""),
            extra={"dir_label": dl, "url": x.get("url") or "",
                   "sentiment": {"source": "finbert", "label": lb, "score": r.get("score")}}))
    return out


def _macro_signals(ops: dict, news: dict) -> list[dict]:
    out = []
    nx = (ops.get("calendar") or {}).get("next_high")
    if isinstance(nx, dict) and nx.get("name"):
        mins = _num(nx.get("mins"), 999)
        if mins is not None and -15 <= mins <= 60:
            out.append(_sig("macro", "宏观日历", "caution", str(nx.get("name")),
                            "T%+d 分钟，%s重要性事件前后点差与滑点常放大" %
                            (int(mins), nx.get("importance") or "高"),
                            weight=0, at=str(nx.get("time") or ""),
                            extra={"mins": int(mins)}))
    shock = news.get("shock")
    if isinstance(shock, dict) and shock.get("title"):
        out.append(_sig("macro", "突发消息", "caution", str(shock.get("title")),
                        "%s 分钟内的高影响消息，20 分钟冲击窗口内慎动" %
                        int(_num(shock.get("age_min"), 0) or 0),
                        weight=0, at=str(shock.get("hm") or "")))
    return out


def _tech_signals(bars: dict) -> list[dict]:
    b = bars or {}
    last = b.get("last") or {}
    c = _num(last.get("c"))
    ma1, ma2 = _num(last.get("ma1")), _num(last.get("ma2"))
    up, dn = _num(last.get("bo_up")), _num(last.get("bo_dn"))
    atr = _num(last.get("atr"))
    at = str(last.get("t") or "")
    out = []

    if c is not None and ma1 is not None and ma2 is not None:
        d = "bull" if (c > ma1 and ma1 > ma2) else ("bear" if (c < ma1 and ma1 < ma2) else "none")
        out.append(_sig("tech", "技术 · 均线", d, "均线排列",
                        "收盘 %s，MA20 %s，MA50 %s → %s" %
                        (_fmt(c), _fmt(ma1), _fmt(ma2),
                         "价在两线之上且多头排列" if d == "bull" else
                         ("价在两线之下且空头排列" if d == "bear" else "均线交织，方向不清")),
                        weight=2, at=at))
    else:
        out.append(_sig("tech", "技术 · 均线", "none", "均线排列", "指标数据不可用", weight=0, at=at))

    if c is not None and up is not None and dn is not None and up > dn:
        pos = (c - dn) / (up - dn) * 100
        d = "bull" if pos > 65 else ("bear" if pos < 35 else "none")
        out.append(_sig("tech", "技术 · 布林", d, "通道位置",
                        "处于布林通道 %.0f%% 分位（下轨 %s / 上轨 %s），按顺势口径标 %s" %
                        (pos, _fmt(dn), _fmt(up), DIR_LABEL[d]),
                        weight=1, at=at))
    else:
        out.append(_sig("tech", "技术 · 布林", "none", "通道位置", "指标数据不可用", weight=0, at=at))

    rows = [r for r in (b.get("bars") or []) if len(r) > 11]
    if len(rows) >= 21 and c is not None:
        c20 = _num(rows[-21][4])
        if c20:
            pct = (c - c20) / c20 * 100
            d = "bull" if pct > 0.3 else ("bear" if pct < -0.3 else "none")
            out.append(_sig("tech", "技术 · 动量", d, "近 20 根动能",
                            "收盘较 20 根前 %+.2f%%，区间高 %s / 低 %s" %
                            (pct, _fmt(max(_num(r[2]) or 0 for r in rows[-20:])),
                             _fmt(min(_num(r[3]) or 0 for r in rows[-20:]))),
                            weight=1, at=at))
    if atr is not None and c:
        atrs = [_num(r[11]) for r in rows[-200:]]
        atrs = [x for x in atrs if x is not None and x > 0]
        q = (sum(1 for x in atrs if x <= atr) / len(atrs) * 100) if atrs else None
        out.append(_sig("tech", "技术 · 波动", "none", "波动状态",
                        "ATR14 %s（占价 %.2f%%）%s；只影响止损宽度，不代表方向" %
                        (_fmt(atr), atr / c * 100,
                         ("，位于近 %d 根的 %.0f%% 分位" % (len(atrs), q)) if q is not None else ""),
                        weight=0, at=at))
    return out


def _gate_signals(ops: dict) -> list[dict]:
    out = []
    gate = ops.get("gate") or {}
    for r in (gate.get("rules") or []):
        st = r.get("state")
        if st in ("BLOCK", "WARN", "NA"):
            out.append(_sig("risk", "风控闸门", "caution", str(r.get("name") or "规则"),
                            "%s（当前 %s / 阈值 %s）%s" %
                            (st, r.get("value"), r.get("threshold"), r.get("why") or ""),
                            weight=0, extra={"state": st}))
    return out


def _bias(news: dict, tech: list[dict], nws: list[dict]) -> dict:
    nb = (news.get("bias") or {})
    n_score = _clip(_num(nb.get("score"), 0) / 10.0)
    n_up = sum(1 for x in nws if x["dir"] == "bull")
    n_dn = sum(1 for x in nws if x["dir"] == "bear")
    n_ne = len(nws) - n_up - n_dn

    tw = sum(x["weight"] for x in tech if x["weight"] > 0)
    t_raw = sum((1 if x["dir"] == "bull" else -1 if x["dir"] == "bear" else 0) * x["weight"]
                for x in tech if x["weight"] > 0)
    t_score = _clip(t_raw / tw) if tw else 0.0
    t_up = sum(1 for x in tech if x["dir"] == "bull")
    t_dn = sum(1 for x in tech if x["dir"] == "bear")

    total = round(n_score * 0.45 + t_score * 0.55, 3)
    diverge = (n_score * t_score < -0.04) and (abs(n_score) > 0.2 and abs(t_score) > 0.2)
    if diverge:
        label = "分歧"
    elif total >= 0.25:
        label = "偏多"
    elif total <= -0.25:
        label = "偏空"
    else:
        label = "中性"

    # 消息面口径直接沿用 news.digest 的原始结论，避免与侧栏「净倾向」显示不一致
    n_label = str(nb.get("label") or "中性")
    if n_label not in ("偏多", "偏空", "中性"):
        n_label = "中性"
    t_label = "偏多" if t_score >= 0.25 else ("偏空" if t_score <= -0.25 else "中性")
    summary = ("近 48h %d 条相关消息：%d 偏多 / %d 偏空 / %d 中性，消息面 %s；"
               "技术面 %d 项偏多 / %d 项偏空，%s。综合信息倾向：%s。") % (
                   len(nws), n_up, n_dn, n_ne, n_label, t_up, t_dn, t_label, label)
    if diverge:
        summary += " 注意：消息面与技术面方向相反，属于分歧状态，不宜只按一边下结论。"
    return {
        "label": label, "score": total,
        "news": {"label": n_label, "score": round(n_score, 3),
                 "n": len(nws), "n_up": n_up, "n_down": n_dn, "n_neutral": n_ne,
                 "raw_score": nb.get("score"), "raw_label": nb.get("label")},
        "tech": {"label": t_label, "score": round(t_score, 3),
                 "n": len(tech), "n_up": t_up, "n_down": t_dn},
        "n_up": n_up + t_up, "n_down": n_dn + t_dn,
        "summary": summary,
        "note": "这是已发生信息的倾向汇总，不是涨跌预测；历史回测未证明本项目有方向 alpha。",
        "diverged": bool(diverge),
    }


# ------------------------------------------------------------------ 账户 + 建议
def _account(ops: dict, q: dict) -> dict:
    a = ops.get("account") or {}
    hist = (ops.get("history") or {}).get("stats") or {}
    raw = ops.get("positions") or {}
    rows = raw.get("rows") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        rows = []
    equity = _num(a.get("equity"))
    balance = _num(a.get("balance"))
    today = _num(hist.get("today_realized"))
    day_left = (0.05 * balance + min(0.0, today)) if (balance is not None and today is not None) else None

    pos = []
    for p in rows[:12]:
        profit = _num(p.get("profit"))
        pos.append({
            "ticket": str(p.get("position_id") or p.get("ticket") or ""),
            "symbol": p.get("symbol"), "side": p.get("action") or p.get("type"),
            "lots": _num(p.get("volume")), "open": _num(p.get("price_open")),
            "last": _num(p.get("price_last")) or q["bid"],
            "profit": profit,
            "profit_pct": round(profit / equity * 100, 3) if (profit is not None and equity) else None,
            "at": p.get("create_time") or "", "comment": p.get("comment") or "",
        })
    return {
        "ok": bool(a), "login": a.get("login"), "type": a.get("type"),
        "currency": a.get("currency"), "balance": balance, "equity": equity,
        "float": _num(a.get("profit")), "margin_free": _num(a.get("margin_free")),
        "today_realized": today, "day_loss_limit_pct": -5.0, "day_loss_left": day_left,
        "n_positions": len(pos), "positions": pos,
        "total_realized": _num(hist.get("total_profit")),
        "feed": ops.get("feed") or "OFFLINE", "mode": ops.get("mode") or "",
        "quote": q,
    }


def _advice(ops: dict, bars: dict, news: dict, acct: dict, bias: dict) -> list[dict]:
    """按 ATR 与净值 1% 风险预算给参考位。全是文字建议，不产生订单。"""
    out = []
    last = (bars or {}).get("last") or {}
    atr = _num(last.get("atr"))
    equity = acct.get("equity")
    q_bid = (acct.get("quote") or {}).get("bid")
    gate = ops.get("gate") or {}
    now = dt.datetime.now().strftime("%H:%M:%S")

    # ---- 全局风险提示
    if (acct.get("day_loss_left") is not None) and acct["day_loss_left"] <= 0:
        out.append({"type": "block", "level": "URGENT", "title": "当日亏损额度已用尽",
                    "why": "已实现亏损达到 −5% 停机线；规则上今天不该再新开仓。",
                    "price": None, "ticket": "", "at": now})
    if gate.get("verdict") == "DENY" or int(_num(gate.get("n_block"), 0) or 0) > 0:
        out.append({"type": "block", "level": "CAUTION", "title": "风控闸门有拦截项",
                    "why": "%s（拦截 %s 条）；开新仓前先逐条看右侧闸门。" %
                           (gate.get("label") or "", gate.get("n_block")),
                    "price": None, "ticket": "", "at": now})
    nx = (ops.get("calendar") or {}).get("next_high")
    if isinstance(nx, dict):
        mins = _num(nx.get("mins"), 999)
        if mins is not None and -15 <= mins <= 30:
            out.append({"type": "caution", "level": "CAUTION", "title": "高影响事件临近",
                        "why": "%s T%+d 分钟，点差/滑点可能异常；已有仓位可考虑收紧保护。" %
                               (nx.get("name") or "", int(mins)),
                        "price": None, "ticket": "", "at": now})

    # ---- 逐仓参考位
    for p in acct.get("positions") or []:
        tk, side = p.get("ticket") or "", str(p.get("side") or "")
        op, cur = p.get("open"), p.get("last")
        profit, pct = p.get("profit"), p.get("profit_pct")
        base = cur if cur is not None else q_bid
        sl = tp = trail = None
        if atr and base is not None:
            sgn = 1 if side.lower().startswith("b") else -1
            sl = round(base - sgn * 2 * atr, 2)
            tp = round(base + sgn * 4 * atr, 2)
            if op is not None:
                cand = round(base - sgn * 2 * atr, 2)
                # 只朝有利方向移动保护位
                trail = round(max(op, cand), 2) if sgn > 0 else round(min(op, cand), 2)
        if sl is not None:
            out.append({"type": "sl", "level": "INFO", "ticket": tk,
                        "title": "止损参考位 %s" % _fmt(sl),
                        "why": "按当前价 ±2×ATR14（%s）计算；若已有终端内止损，以你自己的为准。" % _fmt(atr),
                        "price": sl, "at": now})
        if tp is not None:
            out.append({"type": "tp", "level": "INFO", "ticket": tk,
                        "title": "止盈参考位 %s" % _fmt(tp),
                        "why": "按当前价 ±4×ATR14；只是波动尺度换算，不是目标价预测。",
                        "price": tp, "at": now})

        if pct is not None and pct <= -1.0:
            out.append({"type": "exit", "level": "URGENT", "ticket": tk,
                        "title": "浮亏已超单笔风险预算",
                        "why": "该仓浮亏 %s（占净值 %+.2f%%），超过 1%% 单笔风险；"
                               "检查止损是否被移动过，按你自己的计划决定离场。" %
                               (_fmt(profit), pct),
                        "price": None, "at": now})
        elif profit is not None and profit > 0 and trail is not None:
            out.append({"type": "hold", "level": "INFO", "ticket": tk,
                        "title": "浮盈中 · 可考虑移动保护",
                        "why": "当前浮盈 %s；若打算继续持有，可把保护位上移到 %s 附近（不自动执行）。" %
                               (_fmt(profit), _fmt(trail)),
                        "price": trail, "at": now})
        if bias.get("label") in ("偏空", "分歧") and str(side).lower().startswith("b"):
            out.append({"type": "caution", "level": "CAUTION", "ticket": tk,
                        "title": "信息倾向与持仓方向相反",
                        "why": "综合倾向为%s，你持有多单；只作为提醒，是否减仓由你判断。" % bias["label"],
                        "price": None, "at": now})
        if bias.get("label") in ("偏多", "分歧") and str(side).lower().startswith("s"):
            out.append({"type": "caution", "level": "CAUTION", "ticket": tk,
                        "title": "信息倾向与持仓方向相反",
                        "why": "综合倾向为%s，你持有空单；只作为提醒，是否减仓由你判断。" % bias["label"],
                        "price": None, "at": now})

    if not (acct.get("positions") or []):
        out.append({"type": "hold", "level": "INFO", "title": "当前空仓",
                    "why": "没有持仓，就没有平仓/止损/止盈建议。看信号、看走势，进场由你决定。",
                    "price": None, "ticket": "", "at": now})
    return out


# ------------------------------------------------------------------ 组装
def compose(ops: dict, bars: dict, news: dict, en: dict | None = None) -> dict:
    ops, bars, news, en = ops or {}, bars or {}, news or {}, en or {}
    q = _quote(ops, bars)
    # FinBERT 只吃英文：你的中文快讯继续走关键词规则，两路来源不混算
    titles = [str(x.get("title") or "").strip() for x in (en.get("items") or [])[:12]]
    if fb is not None:
        try:
            sent = fb.analyze(titles)
        except Exception as e:  # noqa: BLE001
            sent = {"enabled": False, "source": "keyword", "scores": {}, "n_scored": 0,
                    "n_up": 0, "n_down": 0, "n_neutral": 0, "label": "—",
                    "summary": "模型调用失败，逐条信号沿用关键词规则。",
                    "note": "模型未就绪，逐条信号使用关键词规则",
                    "error": "%s: %s" % (type(e).__name__, str(e)[:160])}
    else:
        sent = {"enabled": False, "source": "keyword", "scores": {}, "n_scored": 0,
                "n_up": 0, "n_down": 0, "n_neutral": 0, "label": "—",
                "summary": "未安装模型依赖，逐条信号沿用关键词规则。",
                "note": "模型未就绪，逐条信号使用关键词规则"}
    nws = _news_signals(news, sent)
    ens = _en_signals(en, sent)
    tech = _tech_signals(bars)
    macro = _macro_signals(ops, news)
    risk = _gate_signals(ops)
    bias = _bias(news, tech, nws)
    acct = _account(ops, q)
    advice = _advice(ops, bars, news, acct, bias)
    signals = nws + ens + tech + macro + risk
    order = {"caution": 0, "bull": 1, "bear": 1, "none": 2}
    signals.sort(key=lambda x: (order.get(x["dir"], 2), -x["weight"]))
    sent_out = {k: v for k, v in sent.items() if k != "scores"}
    # worker 独立进程算分时用的英文源状态更准（可能抓失败后用了旧标题）
    wf = sent.get("feed") if isinstance(sent.get("feed"), dict) else None
    sent_out["feed"] = wf or {
        "ok": bool(en.get("ok")), "source": en.get("source") or "",
        "n": int(en.get("n") or 0), "stale": bool(en.get("stale")),
        "error": en.get("error") or ""}
    feed_stale = bool(sent_out["feed"].get("stale")) or bool(en.get("stale"))
    sent_out["feed"]["stale"] = feed_stale
    gaps = [dict(x) for x in GAPS]
    if sent.get("n_scored"):
        for g in gaps:
            if "新闻语义" in str(g.get("name") or ""):
                g["state"] = "英文标题已补"
                g["why"] = ("英文财经标题改由本地 FinBERT 标注文字情绪；中文标题仍用关键词规则。"
                            "文字情绪不等于黄金涨跌方向，两套来源不混算、不计入综合倾向。")
    gaps.append({"name": "FinBERT 新闻情绪模型",
                 "state": ("已启用 · %d 条英文" % sent["n_scored"]) if sent.get("n_scored")
                          else "未启用 / 降级",
                 "why": (sent.get("summary") or sent.get("note") or "") +
                        ("；英文源本次未取到：" + str(en.get("error") or "") if en.get("error") else "") +
                        ("；本批为上次抓到的旧标题，不是最新快讯" if feed_stale else "")})
    return {
        "ok": True,
        "ts": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "price": q["bid"], "price_src": q["src"], "spread_pt": q["spread_pt"],
        "symbol": q["symbol"], "tf": (bars.get("tf") or "H1"),
        "last_bar": (bars.get("last") or {}).get("t"),
        "feed": ops.get("feed") or "OFFLINE",
        "bias": bias,
        "signals": signals,
        "n_signals": len(signals),
        "gate": {"verdict": (ops.get("gate") or {}).get("verdict"),
                 "label": (ops.get("gate") or {}).get("label"),
                 "n_block": (ops.get("gate") or {}).get("n_block", 0),
                 "n_warn": (ops.get("gate") or {}).get("n_warn", 0)},
        "account": acct,
        "advice": advice,
        "gaps": gaps,
        "sentiment": sent_out,
        "disclaimer": DISCLAIMER,
        "news_ok": bool(news.get("ok")),
        "news_errors": news.get("errors") or [],
        "choices": prebrief.CHOICES,
    }


def log_decision(body: dict) -> dict:
    """决策留痕直接复用简报模块，账本只有一份。"""
    return prebrief.log_decision(body)


def logs(limit: int = 50) -> dict:
    return prebrief.logs(limit)


def snapshot_line(c: dict) -> dict:
    """给前端提交留痕时的最小快照，防伪造分数。"""
    return {
        "ts": c.get("ts"), "price": c.get("price"),
        "spread_pt": c.get("spread_pt"),
        "n_positions": (c.get("account") or {}).get("n_positions"),
        "today_realized": (c.get("account") or {}).get("today_realized"),
        "gate_verdict": (c.get("gate") or {}).get("verdict"),
        "n_block": (c.get("gate") or {}).get("n_block", 0),
        "n_warn": (c.get("gate") or {}).get("n_warn", 0),
        "news_shock": None, "feed": c.get("feed"),
        "bias": (c.get("bias") or {}).get("label"),
        "bias_score": (c.get("bias") or {}).get("score"),
    }
