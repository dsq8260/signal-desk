"""决策留痕与总览聚合数据层 —— 纯信息聚合、决策留痕，不给方向。

设计边界（重要，别改）：
  * 只陈述事实与「你现在的打算和硬规则的冲突」，**不输出买入/卖出建议、
    不输出总分、不产生任何订单**。方向预测在本项目已被三次证伪
    （101 次网格 Deflated p=0.934 / AI 评分卡 −0.048R / PPO 验证段 +2.9% vs 持有 +15.1%）。
  * 复用已有只读数据：ops（账户/持仓/日历/闸门/报价）、bars（MA/BOLL/ATR）、news（关键词分类）。
    不新建数据层、不新增外部数据源。
  * 明确列出「哪些数据没有」：不把覆盖不全包装成全覆盖。
  * 决策留痕是唯一科学的验证方式：下单前存一份快照 + 你选了什么，
    之后按「听了提示 / 没听提示」分组统计，验证提示有没有帮上忙。
  * 不产生任何订单，不读写任何自动执行状态；自动盯盘已整体移除。
  * 智能辅助页（copilot）复用本模块的缺口清单与决策留痕文件。

落盘：data/brief_log.jsonl（追加写，一行一条决策）
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import threading

HERE = pathlib.Path(__file__).resolve().parent
LOG = HERE / "data" / "brief_log.jsonl"
_lock = threading.Lock()

CHOICES = {
    "skip": "不下单（按提示）",
    "trade": "照常下单",
    "watch": "只观察，不动手",
    "exit": "准备离场",
    "hold": "持仓不动",
}

# 明确列出的数据缺口。state 只有「缺失 / 部分」两种，绝不写「已覆盖」。
GAPS = [
    {"name": "COT 持仓 / 恐慌指数 / 期权偏度", "state": "缺失",
     "why": "没有情绪与仓位数据源，新闻关键词倾向分不能替代"},
    {"name": "宏观状态（实际利率 / CPI 趋势 / 央行路径）", "state": "缺失",
     "why": "只有经济日历的「事件」，没有状态与预期差"},
    {"name": "新闻语义理解", "state": "部分",
     "why": "news.py 是关键词 + 影响分级规则，不是语义阅读；宁可标中性"},
    {"name": "MCP 官方新闻源", "state": "缺失",
     "why": "get_news_list 在本账户实测返回 0 条，当前用的是华尔街见闻 + 新浪"},
    {"name": "隔夜利息 swap / 真实滑点", "state": "缺失",
     "why": "行情层未建模，持仓过夜的实际成本要自己去终端看"},
]

DISCLAIMER = ("本页只聚合事实与规则冲突：不给买卖方向、不给总分、不自动下单。"
              "AI 评分卡的六维分解已被离线回测证明无 alpha（期望 −0.048R），仅可作仪表。")


def _num(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _pct(a, b):
    a, b = _num(a), _num(b)
    return None if (a is None or b is None or b == 0) else (a - b) / b * 100


def _fmt(v, nd=2, suffix=""):
    return "—" if v is None else ("%.*f%s" % (nd, v, suffix))


def _quote_of(ops: dict, bars: dict) -> dict:
    """优先桥/MCP 实时报价；没有就退回 K 线最后一根收盘。"""
    sym = str((bars or {}).get("symbol") or "GOLD")
    for q in (ops.get("quotes") or []):
        s = str(q.get("symbol") or "")
        if "GOLD" in s.upper() or "XAU" in s.upper():
            bid, ask = _num(q.get("bid")), _num(q.get("ask"))
            if bid is not None:
                return {"symbol": s, "bid": bid, "ask": ask,
                        "spread_pt": round((ask - bid) / 0.01, 1) if ask is not None else None,
                        "src": "实时报价（桥 / 市场观察）"}
    live = (bars or {}).get("live") or {}
    last = (bars or {}).get("last") or {}
    bid, ask = _num(live.get("bid")), _num(live.get("ask"))
    if bid is not None:
        return {"symbol": sym, "bid": bid, "ask": ask,
                "spread_pt": round((ask - bid) / 0.01, 1) if ask is not None else None,
                "src": "桥 tick"}
    c = _num(last.get("c"))
    return {"symbol": sym, "bid": c, "ask": None, "spread_pt": None,
            "src": "K 线收盘（无实时 tick）"}


def _bars_context(bars: dict) -> list[dict]:
    """把技术面写成陈述句，不打分、不给方向。"""
    b = bars or {}
    last = b.get("last") or {}
    c = _num(last.get("c"))
    ma1, ma2 = _num(last.get("ma1")), _num(last.get("ma2"))
    up, dn = _num(last.get("bo_up")), _num(last.get("bo_dn"))
    atr = _num(last.get("atr"))
    out = []

    if c is not None and ma1 is not None and ma2 is not None:
        d1, d2 = _pct(c, ma1), _pct(c, ma2)
        out.append({"k": "均线位置",
                    "v": "收盘 %s，相对 MA20 %s%%，相对 MA50 %s%%；MA20 %s MA50" %
                         (_fmt(c), ("+" if d1 >= 0 else "") + _fmt(d1, 2),
                          ("+" if d2 >= 0 else "") + _fmt(d2, 2),
                          "高于" if ma1 > ma2 else "低于")})
    else:
        out.append({"k": "均线位置", "v": "指标数据不可用"})

    if c is not None and up is not None and dn is not None and up > dn:
        pos = (c - dn) / (up - dn) * 100
        zone = "上轨之上" if pos > 100 else ("下轨之下" if pos < 0 else
                ("上半区" if pos > 50 else "下半区"))
        out.append({"k": "布林位置", "v": "处于通道 %s（%.0f%% 分位），带宽 %s" %
                    (zone, pos, _fmt(up - dn))})
    else:
        out.append({"k": "布林位置", "v": "指标数据不可用"})

    # ATR 分位：最近 200 根里的百分位，纯描述波动状态
    rows = [r for r in (b.get("bars") or []) if len(r) > 11]
    atrs = [_num(r[11]) for r in rows[-200:]]
    atrs = [x for x in atrs if x is not None and x > 0]
    if atr is not None and atrs:
        q = sum(1 for x in atrs if x <= atr) / len(atrs) * 100
        out.append({"k": "波动状态",
                    "v": "ATR14 %s（占价 %s%%），位于近 %d 根的 %.0f%% 分位" %
                         (_fmt(atr), _fmt(atr / c * 100 if c else None), len(atrs), q)})
    else:
        out.append({"k": "波动状态", "v": "ATR 数据不可用"})

    if len(rows) >= 21:
        c20 = _num(rows[-21][4])
        d20 = _pct(c, c20)
        out.append({"k": "近 20 根", "v": "收盘变动 %s%%；区间高 %s / 低 %s" %
                    (("+" if (d20 or 0) >= 0 else "") + _fmt(d20, 2),
                     _fmt(max(_num(r[2]) or 0 for r in rows[-20:])),
                     _fmt(min(_num(r[3]) or 0 for r in rows[-20:])))})
    return out


def compose(ops: dict, bars: dict, news: dict, symbol: str = "GOLD#") -> dict:
    """把三路只读数据合成一屏简报。不产生任何交易副作用。"""
    ops = ops or {}
    bars = bars or {}
    news = news or {}
    q = _quote_of(ops, bars)
    acct = ops.get("account") or {}
    hist = ((ops.get("history") or {}).get("stats") or {})
    cal = ops.get("calendar") or {}
    gate = ops.get("gate") or {}
    last = bars.get("last") or {}

    pos_raw = ops.get("positions") or {}
    positions = pos_raw.get("positions") if isinstance(pos_raw, dict) else pos_raw
    if not isinstance(positions, list):
        positions = []
    pos_rows = [{"symbol": p.get("symbol"), "type": p.get("type"),
                 "volume": p.get("volume"), "profit": p.get("profit")}
                for p in positions[:8]]

    balance = _num(acct.get("balance"))
    today = _num(hist.get("today_realized"))
    day_left = None
    if balance is not None and today is not None:
        # 剩余可亏额度（正数）；today 为负时吃掉一部分，<=0 表示已触停机线
        day_left = 0.05 * balance + min(0.0, today)

    nx = cal.get("next_high")
    facts = {
        "symbol": symbol,
        "price": q["bid"], "ask": q["ask"], "spread_pt": q["spread_pt"],
        "price_src": q["src"],
        "last_bar": last.get("t"), "tf": bars.get("tf") or "H1",
        "atr": _num(last.get("atr")), "ma20": _num(last.get("ma1")),
        "ma50": _num(last.get("ma2")),
        "n_positions": len(positions), "positions": pos_rows,
        "equity": _num(acct.get("equity")), "balance": balance,
        "margin_free": _num(acct.get("margin_free")),
        "today_realized": today, "day_loss_limit_pct": -5.0,
        "day_loss_left": day_left,
        "next_high": ({"name": nx.get("name"), "mins": nx.get("mins"),
                       "time": nx.get("time"), "importance": nx.get("importance")}
                      if isinstance(nx, dict) else None),
        "feed": ops.get("feed") or "OFFLINE",
    }

    nd = {
        "ok": bool(news.get("ok")),
        "n": news.get("n", 0), "high": news.get("high", 0),
        "shock": ({"title": (news.get("shock") or {}).get("title"),
                   "age_min": (news.get("shock") or {}).get("age_min")}
                  if news.get("shock") else None),
        "top": [{"title": x.get("title"), "hm": x.get("hm"),
                 "impact": x.get("impact"), "topic": x.get("topic"),
                 "dir_label": x.get("dir_label")}
                for x in (news.get("items") or [])[:4]],
        "stale": bool(news.get("stale")),
        "errors": news.get("errors") or [],
        "note": "关键词规则分级，不是语义理解；倾向分不作为方向依据",
    }

    rules = gate.get("rules") or []
    context = {
        "trend": _bars_context(bars),
        "news": nd,
        "gate": {"verdict": gate.get("verdict"), "label": gate.get("label"),
                 "n_block": gate.get("n_block", 0), "n_warn": gate.get("n_warn", 0),
                 "n_na": gate.get("n_na", 0), "enforced": gate.get("enforced", False),
                 "rules": [{"id": r.get("id"), "name": r.get("name"),
                            "state": r.get("state"), "value": r.get("value"),
                            "threshold": r.get("threshold"), "why": r.get("why")}
                           for r in rules]},
    }

    conflicts = []
    for r in rules:
        st = r.get("state")
        if st not in ("BLOCK", "WARN", "NA"):
            continue
        conflicts.append({
            "level": "BLOCK" if st == "BLOCK" else ("CAUTION" if st == "WARN" else "INFO"),
            "rule": r.get("name"), "value": r.get("value"),
            "threshold": r.get("threshold"), "why": r.get("why"),
        })
    if nd["shock"]:
        conflicts.append({"level": "CAUTION", "rule": "刚发生的高影响新闻",
                          "value": "%s（%s 分钟前）" % (nd["shock"]["title"], nd["shock"]["age_min"]),
                          "threshold": "—", "why": "20 分钟冲击窗口内点差与滑点可能异常"})
    if nx and isinstance(nx, dict) and -15 <= _num(nx.get("mins"), 999) <= 30:
        conflicts.append({"level": "CAUTION", "rule": "高重要性事件临近",
                          "value": "T%+d min" % int(_num(nx.get("mins"), 0)),
                          "threshold": "±30min", "why": str(nx.get("name") or "")})
    if facts["n_positions"]:
        conflicts.append({"level": "INFO", "rule": "已有持仓",
                          "value": "%d 笔" % facts["n_positions"],
                          "threshold": "—", "why": "先确认离场条件，再决定是否动新仓"})
    if facts["day_loss_left"] is not None and facts["day_loss_left"] <= 0:
        conflicts.append({"level": "BLOCK", "rule": "日内亏损额度",
                          "value": "已用尽", "threshold": "−5%",
                          "why": "当日已实现亏损已达停机线"})
    if not conflicts:
        conflicts.append({"level": "INFO", "rule": "无冲突项",
                          "value": "8 条闸门全 PASS", "threshold": "—",
                          "why": "这只表示没有触发硬规则，不代表行情有利"})

    return {
        "ok": True,
        "ts": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "facts": facts, "context": context,
        "conflicts": conflicts,
        "gaps": [dict(x) for x in GAPS],
        "disclaimer": DISCLAIMER,
        "choices": CHOICES,
    }


# ------------------------------------------------------------------ 决策留痕
def log_decision(body: dict) -> dict:
    """下单前存一份快照 + 你的选择。之后才能回答「这东西到底帮没帮我」。"""
    body = body or {}
    choice = str(body.get("choice") or "")
    if choice not in CHOICES:
        return {"ok": False, "error": "choice 必须是 %s 之一" % "/".join(CHOICES)}
    snap = body.get("snapshot") or {}
    if not isinstance(snap, dict):
        snap = {}
    note = re.sub(r"[\r\n\t]+", " ", str(body.get("note") or ""))[:200]
    ticket = re.sub(r"[^0-9A-Za-z_-]", "", str(body.get("ticket") or ""))[:24]
    verdict = str(snap.get("gate_verdict") or "")
    if verdict in ("DENY", "CAUTION"):
        followed = choice in ("skip", "hold", "exit", "watch")
    else:
        followed = None  # 没有否决提示，谈不上「听没听」
    rec = {
        "ts": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "choice": choice, "choice_label": CHOICES[choice],
        "note": note, "ticket": ticket or None,
        "followed": followed,
        "brief": {
            "ts": snap.get("ts"), "price": snap.get("price"),
            "spread_pt": snap.get("spread_pt"), "atr": snap.get("atr"),
            "n_positions": snap.get("n_positions"),
            "today_realized": snap.get("today_realized"),
            "next_high_mins": snap.get("next_high_mins"),
            "gate_verdict": verdict or None,
            "n_block": snap.get("n_block"), "n_warn": snap.get("n_warn"),
            "news_shock": snap.get("news_shock"), "feed": snap.get("feed"),
        },
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True, "record": rec, "note": "已留痕。攒够样本后可在智能辅助页「决策留痕」看分组统计。"}


def logs(limit: int = 50) -> dict:
    """回看 + 分组统计。收益不在这里算：需关联真实成交单号另行对账。"""
    rows = []
    if LOG.is_file():
        try:
            for line in LOG.read_text("utf-8").splitlines():
                try:
                    r = json.loads(line)
                    if isinstance(r, dict) and r.get("choice"):
                        rows.append(r)
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
    total = len(rows)
    followed = sum(1 for r in rows if r.get("followed") is True)
    ignored = sum(1 for r in rows if r.get("followed") is False)
    no_hint = sum(1 for r in rows if r.get("followed") is None)
    by_choice = {}
    for r in rows:
        k = r.get("choice")
        by_choice[k] = by_choice.get(k, 0) + 1
    linked = sum(1 for r in rows if r.get("ticket"))
    return {
        "ok": True, "n": total,
        "followed": followed, "ignored": ignored, "no_hint": no_hint,
        "linked": linked, "by_choice": by_choice,
        "recent": list(reversed(rows))[:max(1, min(limit, 200))],
        "note": ("分组口径：闸门 DENY/CAUTION 时选「不下单/持仓不动/离场/观察」记为听到提示，"
                 "选「照常下单」记为没听；闸门 ALLOW 时不参与分组。"
                 "本页不算收益——可填成交单号关联真实成交，另行对账。"),
    }
