"""AI 交易引擎：可解释评分卡 + 硬风控 + MCP 执行 + 全量审计。

这个文件只做一件事：把「AI 想怎么交易」变成「终端能执行、事后能复盘」的动作。

设计原则（每条都是为了让结果可被质疑，而不是为了让曲线好看）
------------------------------------------------------------------
1. **只做 demo**：账户 type != demo 时所有执行入口直接拒绝，不给"手滑上真钱"留口子。
2. **手数不接受输入**：由「净值 × 风险% ÷ 止损距离」反算。若反算出的小于最小手数，
   直接拒绝而不是放大风险——最小手数就已经超预算时，正确的动作是不做。
3. **决策必须能解释**：5 个维度各自打分并给出中文理由，总分不到阈值就观望。
   观望也写审计，这样"AI 为什么没出手"同样可查。
4. **单日亏损上限**：当日已实现亏损超过净值 2% 就停机，当天不再开新仓。
5. **每笔都写审计**：data/ai_audit.jsonl，一行一条，含当时的完整输入和结果。

当前的决策来源是「规则评分卡」，不是大模型黑箱：它可回测、可复现、每条理由你都能反驳。
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
CFG_FILE = DATA / "ai_config.json"
AUDIT_FILE = DATA / "ai_audit.jsonl"
STATE_FILE = DATA / "ai_state.json"

# ---------------------------------------------------------------- 默认配置
DEFAULT_CFG = {
    "enabled": False,            # 总开关：关着时任何入口都不下单
    "auto": True,                # 自动盯盘：开着时每根新 K 线自己判定一次（无需人点）
    "auto_locked": False,        # 停用锁：true 时界面任何开关都开不了自动盯盘（改本文件才能解除）
    "auto_interval": 30,         # 后台轮询间隔（秒）；实际动作仍按「新 K 线」触发
    "auto_tf": "H1",             # 自动判定用的周期
    "recheck_min": 15,           # 观望状态下隔多少分钟复判一次（盘中价格变了分数会变）
    "exec_channel": "auto",      # auto = EA 指令通道优先，MCP 兜底
    "symbol": "GOLD#",
    "allowed": ["GOLD#"],        # 白名单，改这里才能交易别的品种
    "risk_pct": 1.0,             # 单笔风险 = 净值 × 1%
    "max_lots": 0.05,            # 手数硬上限
    "max_open": 1,               # 同时持仓上限
    "daily_loss_pct": 2.0,       # 当日亏损超净值 2% 停机
    "sl_atr": 1.5,               # 止损 = ATR × 1.5
    "tp_atr": 3.0,               # 止盈 = ATR × 3.0（2:1）
    "min_score": 60,             # 开仓最低分（满分 100）
    "comment": "AIDesk",         # 成交注释，账本靠它把 AI 单分出来
    "contract": 100.0,           # GOLD 1 手 = 100 盎司
    "lot_step": 0.01,
    "lot_min": 0.01,
}

_lock = threading.Lock()
_client = None
_client_ts = 0.0
_mcp_ok: bool | None = None
_mcp_ok_ts = 0.0


# ---------------------------------------------------------------- 基础设施
def _cfg() -> dict:
    out = dict(DEFAULT_CFG)
    try:
        if CFG_FILE.exists():
            saved = json.loads(CFG_FILE.read_text("utf-8"))
            for k, v in (saved or {}).items():
                if k in out:
                    out[k] = v
    except Exception:  # noqa: BLE001
        pass
    return out


def _save_cfg(patch: dict) -> dict:
    cur = _cfg()
    for k, v in (patch or {}).items():
        if k in DEFAULT_CFG:
            cur[k] = v
    CFG_FILE.write_text(json.dumps(cur, ensure_ascii=False, indent=1), "utf-8")
    return cur


def _audit(row: dict) -> None:
    row = dict(row)
    row.setdefault("ts", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        with AUDIT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


def audit_tail(n: int = 40) -> list[dict]:
    if not AUDIT_FILE.exists():
        return []
    try:
        lines = AUDIT_FILE.read_text("utf-8").strip().splitlines()[-n:]
        return [json.loads(x) for x in lines if x.strip()]
    except Exception:  # noqa: BLE001
        return []


def mcp():
    """复用一个 MCP 连接（10 分钟重建，避免长连接被终端回收）。"""
    global _client, _client_ts
    now = time.time()
    with _lock:
        if _client is None or now - _client_ts > 600:
            from gold.mt5_mcp import Mt5Mcp
            _client = Mt5Mcp()
            _client_ts = now
    return _client


def _call(name: str, args: dict | None = None) -> dict:
    try:
        return mcp().call(name, args or {}) or {}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- 账户与仓位
def account() -> dict:
    a = (_call("get_trading_account_info") or {}).get("account") or {}
    return {
        "ok": bool(a),
        "login": a.get("login"), "type": a.get("type"),
        "currency": a.get("currency"), "balance": a.get("balance"),
        "equity": a.get("equity"), "margin_free": a.get("margin_free"),
        "profit": a.get("profit"),
    }


def open_positions(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or _cfg()
    r = _call("get_trading_open_positions")
    rows = r.get("positions") or r.get("rows") or []
    if isinstance(rows, dict):
        rows = rows.get("positions") or []
    out = []
    for p in rows or []:
        if not isinstance(p, dict):
            continue
        sym = str(p.get("symbol") or "")
        if sym.upper().replace("#", "") != cfg["symbol"].upper().replace("#", ""):
            continue
        out.append({
            "ticket": p.get("ticket") or p.get("position_ticket"),
            "symbol": sym, "type": p.get("type"),
            "volume": p.get("volume"), "price_open": p.get("price_open"),
            "sl": p.get("sl"), "tp": p.get("tp"),
            "profit": p.get("profit"), "comment": p.get("comment"),
        })
    return out


def ai_positions(cfg: dict | None = None) -> list[dict]:
    """只算 AI 自己开的仓（按 comment 认，不按品种认）。"""
    cfg = cfg or _cfg()
    tag = str(cfg["comment"]).lower()
    return [p for p in open_positions(cfg) if tag in str(p.get("comment") or "").lower()]


def today_realized() -> dict:
    """当日已平仓盈亏（含人工单，用于账户级停机保护）。"""
    now = dt.datetime.now()
    frm = now.replace(hour=0, minute=0, second=0, microsecond=0)
    r = _call("get_trading_history_positions", {
        "datetime_from": frm.strftime("%Y-%m-%dT%H:%M:%S"),
        "datetime_to": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "limit": 200,
    })
    rows = r.get("positions") or r.get("history") or []
    if isinstance(rows, dict):
        rows = rows.get("positions") or []
    total = 0.0
    n = 0
    for x in rows or []:
        if not isinstance(x, dict):
            continue
        p = x.get("profit")
        if p is None:
            continue
        try:
            total += float(p)
            n += 1
        except (TypeError, ValueError):
            continue
    return {"net": round(total, 2), "n": n}


# ---------------------------------------------------------------- 特征与评分
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _rank(vals: list[float], v: float) -> float:
    if not vals:
        return 0.5
    return sum(1 for x in vals if x < v) / len(vals)


def features(symbol: str, tf: str = "H1") -> dict:
    """从实时行情算特征。用 market.payload，和看板同一份数据。"""
    import market as mk

    d = mk.payload(symbol, tf, 400)
    if not d.get("ok"):
        return {"ok": False, "error": d.get("error") or "行情不可用"}
    bars = d["bars"]
    if len(bars) < 80:
        return {"ok": False, "error": f"K 线不足（{len(bars)} 根，至少 80）"}
    last = bars[-1]
    _, o, h, l, c, _v, _sp, ma20, ma50, bo_up, bo_dn, atr = last[:12]
    if None in (ma20, ma50, bo_up, bo_dn, atr) or not atr:
        return {"ok": False, "error": "指标未就绪（MA/BOLL/ATR 有空值）"}

    atrs = [b[11] for b in bars[-120:] if b[11]]
    closes = [b[4] for b in bars]
    mom3 = (closes[-1] - closes[-4]) if len(closes) >= 4 else 0.0
    pos_boll = ((c - bo_dn) / (bo_up - bo_dn)) if bo_up > bo_dn else 0.5
    # 服务器小时：K 线时间是 MT5 服务器时间
    try:
        srv_h = int(str(last[0])[11:13])
    except (ValueError, IndexError):
        srv_h = dt.datetime.now().hour

    return {
        "ok": True, "symbol": d.get("symbol"), "tf": tf,
        "price": c, "open": o, "high": h, "low": l,
        "ma20": ma20, "ma50": ma50, "bo_up": bo_up, "bo_dn": bo_dn,
        "atr": atr, "spread": (d.get("live") or {}).get("spread"),
        "bid": (d.get("live") or {}).get("bid") or c,
        "ask": (d.get("live") or {}).get("ask") or c,
        "live": bool((d.get("live") or {}).get("on")),
        "feed": d.get("feed"),
        "bar_time": str(last[0]),
        "server_hour": srv_h,
        "trend_atr": round((ma20 - ma50) / atr, 3),
        "dist_ma20_atr": round((c - ma20) / atr, 3),
        "pos_boll": round(pos_boll, 3),
        "mom3_atr": round(mom3 / atr, 3),
        "atr_rank": round(_rank(atrs, atr), 3),
        "atr_med": round(sorted(atrs)[len(atrs) // 2], 2) if atrs else None,
    }


def news_brief(symbol: str = "GOLD") -> dict:
    """取一份新闻摘要给决策用。新闻模块本身有缓存与容错，这里再兜一层：
    取不到就当没有新闻，绝不让它把决策流程拖挂。"""
    try:
        import news as nf
        b = nf.brief(symbol, hours=12)
        if not b.get("ok"):
            return {"ok": False, "score": 0.0, "label": "无数据", "top": [], "shock": None,
                    "error": b.get("error")}
        return b
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "score": 0.0, "label": "无数据", "top": [], "shock": None,
                "error": f"{type(e).__name__}: {e}"}


def score(f: dict, cfg: dict | None = None) -> dict:
    """5 维评分卡 → 总分 -100..100（正=做多，负=做空）+ 逐条理由。"""
    cfg = cfg or _cfg()
    parts = []

    # 1) 趋势对齐 30 分：MA20 相对 MA50 的偏离（以 ATR 归一）
    t = _clamp(f["trend_atr"] / 1.2, -1, 1) * 30
    if f["trend_atr"] > 0.3:
        parts.append(("趋势", t, f"MA20 高于 MA50 {f['trend_atr']:.2f} 个 ATR，多头排列"))
    elif f["trend_atr"] < -0.3:
        parts.append(("趋势", t, f"MA20 低于 MA50 {f['trend_atr']:.2f} 个 ATR，空头排列"))
    else:
        parts.append(("趋势", t, f"MA20/MA50 纠缠（差 {f['trend_atr']:.2f} ATR），无方向"))

    # 2) 动量 20 分：最近 3 根净位移
    m = _clamp(f["mom3_atr"] / 1.5, -1, 1) * 20
    parts.append(("动量", m, f"近 3 根净位移 {f['mom3_atr']:.2f} ATR"))

    # 3) 位置 20 分：布林带内位置（靠下轨利于做多，靠上轨扣分，避免追高）
    loc = _clamp((0.5 - f["pos_boll"]) * 2, -1, 1) * 20
    zone = "下轨附近" if f["pos_boll"] < 0.25 else ("上轨附近" if f["pos_boll"] > 0.75 else "带内中部")
    parts.append(("位置", loc, f"价格在布林带 {f['pos_boll'] * 100:.0f}% 处（{zone}）"))

    # 4) 波动率 15 分：偏好中位波动，过热过冷都扣
    vr = 1 - min(1.0, abs(f["atr_rank"] - 0.55) / 0.55)
    vol = vr * 15
    parts.append(("波动", vol, f"ATR {f['atr']:.2f}，处于近 120 根 {f['atr_rank'] * 100:.0f}% 分位"))

    # 5) 时段 15 分：伦纽重叠、伦敦盘活跃
    h = f["server_hour"]
    if 10 <= h <= 22:
        sess = 15.0
        s_txt = f"服务器 {h:02d} 时，欧美盘活跃"
    elif 7 <= h < 10 or 22 < h <= 23:
        sess = 7.0
        s_txt = f"服务器 {h:02d} 时，亚欧过渡，流动性一般"
    else:
        sess = 0.0
        s_txt = f"服务器 {h:02d} 时，亚盘清淡"
    parts.append(("时段", sess, s_txt))

    base = round(sum(p[1] for p in parts), 1)

    # 6) 新闻 ±12 分：12 小时内与品种相关的快讯净倾向（关键词规则，不是语义理解）
    nb = news_brief(f.get("symbol") or cfg["symbol"])
    n_pts = round(max(-12.0, min(12.0, float(nb.get("score") or 0) * 1.2)), 1)
    if nb.get("ok"):
        top = (nb.get("top") or [])
        lead = top[0]["title"] if top else "窗口内没有硬新闻"
        n_why = (f"{nb.get('label')} {nb.get('score'):+.1f}（{nb.get('n', 0)} 条，高影响 "
                 f"{nb.get('high', 0)} 条）· {lead[:46]}")
    else:
        n_why = f"新闻源不可用（{nb.get('error') or '—'}），这一项按 0 分算"
    parts.append(("新闻", n_pts, n_why))

    # 新闻是外部信息，不能单独决定方向：总分仍以技术面为主，新闻只做加减
    total = round(base + n_pts, 1)
    total = max(-100.0, min(100.0, total))
    side = "buy" if total >= cfg["min_score"] else ("sell" if total <= -cfg["min_score"] else "none")
    return {
        "score": total, "base": base, "news": n_pts,
        "side": side,
        "parts": [{"name": n, "score": round(sc, 1), "why": w} for n, sc, w in parts],
        "verdict": ("做多信号" if side == "buy" else "做空信号" if side == "sell"
                    else f"观望（|{total:.0f}| < 阈值 {cfg['min_score']}）"),
    }


def plan(f: dict, s: dict, acct: dict, cfg: dict | None = None) -> dict:
    """把信号变成可执行订单：止损距离 → 手数 → 价格。手数不由人填。"""
    cfg = cfg or _cfg()
    equity = float(acct.get("equity") or 0)
    if equity <= 0:
        return {"ok": False, "error": "拿不到账户净值"}
    if s["side"] == "none":
        return {"ok": False, "error": s["verdict"], "no_trade": True}

    atr = float(f["atr"])
    sl_dist = atr * float(cfg["sl_atr"])
    tp_dist = atr * float(cfg["tp_atr"])
    risk_amt = equity * float(cfg["risk_pct"]) / 100.0
    contract = float(cfg["contract"])
    step = float(cfg["lot_step"])

    raw_lots = risk_amt / (sl_dist * contract) if sl_dist > 0 else 0
    lots = math.floor(raw_lots / step) * step
    lots = round(min(lots, float(cfg["max_lots"])), 2)

    if lots < float(cfg["lot_min"]):
        return {
            "ok": False, "no_trade": True,
            "error": (f"风险预算不够开最小手数：{cfg['risk_pct']}% = {risk_amt:.0f} USD，"
                      f"而 {sl_dist:.2f} 美元止损在 0.01 手上已亏 {sl_dist * contract * 0.01:.0f} USD"),
        }

    dig = 2 if f["price"] >= 100 else 5
    if s["side"] == "buy":
        entry, sl, tp = f["ask"], f["bid"] - sl_dist, f["ask"] + tp_dist
    else:
        entry, sl, tp = f["bid"], f["ask"] + sl_dist, f["bid"] - tp_dist

    real_risk = lots * sl_dist * contract
    return {
        "ok": True, "symbol": cfg["symbol"], "side": s["side"],
        "volume": lots, "entry": round(entry, dig),
        "sl": round(sl, dig), "tp": round(tp, dig),
        "sl_dist": round(sl_dist, 2), "tp_dist": round(tp_dist, 2),
        "risk_usd": round(real_risk, 2),
        "risk_pct": round(real_risk / equity * 100, 2),
        "rr": round(tp_dist / sl_dist, 2) if sl_dist else None,
        "comment": f"{cfg['comment']}",
        "note": f"{lots:.2f} 手 × {sl_dist:.2f} 美元止损 × {contract:.0f} = 风险 {real_risk:.0f} USD",
    }


# ---------------------------------------------------------------- 决策与执行
def decide(symbol: str | None = None, tf: str = "H1") -> dict:
    """完整决策（不执行）：账户 → 持仓 → 特征 → 评分 → 计划 → 风控裁决。"""
    cfg = _cfg()
    sym = symbol or cfg["symbol"]
    out = {"ok": True, "ts": dt.datetime.now().strftime("%H:%M:%S"), "config": cfg}

    acct = account()
    out["account"] = acct
    if not acct["ok"]:
        return {**out, "ok": False, "error": "MCP 拿不到账户（MT5 是否在运行？）"}

    # 硬闸门 1：只做 demo
    if str(acct.get("type")).lower() != "demo":
        return {**out, "ok": False, "blocked": True,
                "error": f"账户类型是 {acct.get('type')}，不是 demo。这个引擎只在模拟盘执行。"}

    # 硬闸门 2：品种白名单
    if sym.upper().replace("#", "") not in [x.upper().replace("#", "") for x in cfg["allowed"]]:
        return {**out, "ok": False, "blocked": True, "error": f"{sym} 不在白名单 {cfg['allowed']}"}

    ai_pos = ai_positions(cfg)
    allday = today_realized()
    out["positions"] = ai_pos
    out["today"] = allday

    # 硬闸门 3：当日亏损超限 → 停机
    loss_limit = -abs(float(acct.get("equity") or 0) * float(cfg["daily_loss_pct"]) / 100.0)
    if allday["net"] <= loss_limit and loss_limit < 0:
        return {**out, "ok": False, "blocked": True,
                "error": f"当日已实现 {allday['net']:.2f} USD，触发 {cfg['daily_loss_pct']}% 停机线"}

    # 硬闸门 4：已有 AI 持仓就不再开
    if len(ai_pos) >= int(cfg["max_open"]):
        p = ai_pos[0]
        return {**out, "ok": True, "hold": True,
                "headline": f"已持有 {p['type']} {p['volume']} 手（#{p['ticket']}，浮盈 {p['profit']}），不重复开仓",
                "positions": ai_pos}

    f = features(sym, tf)
    out["features"] = f
    if not f.get("ok"):
        return {**out, "ok": False, "error": f.get("error")}

    s = score(f, cfg)
    out["signal"] = s
    nw = news_brief(sym)
    out["news"] = nw
    # 高影响新闻刚落地（15 分钟内）：价格由情绪定价，技术指标还没意义 → 暂停开新仓
    sh = nw.get("shock") or {}
    if sh and float(sh.get("age_min") or 999) <= 15:
        return {**out, "ok": True, "hold": True, "news_hold": True,
                "headline": (f"新闻冲击窗口：{sh.get('hm', '')} {sh.get('dir_label', '')}"
                             f"「{str(sh.get('title') or '')[:40]}」发生 "
                             f"{sh.get('age_min', 0):.0f} 分钟，等波动稳定再开仓")}
    p = plan(f, s, acct, cfg)
    out["plan"] = p
    return out


def execute(symbol: str | None = None, tf: str = "H1", confirm: str = "") -> dict:
    """执行一次：重新决策 → 下单前校验 → 真实下单 → 审计。"""
    cfg = _cfg()
    if confirm != "AI_TRADE":
        return {"ok": False, "error": "缺少确认串（confirm=AI_TRADE）"}
    if not cfg["enabled"]:
        return {"ok": False, "error": "引擎总开关是关的（enabled=false），先在界面里开启"}

    d = decide(symbol, tf)
    if d.get("blocked"):
        _audit({"action": "refuse", "reason": d.get("error")})
        return d
    if d.get("hold"):
        _audit({"action": "hold", "reason": d.get("headline")})
        return d
    if not d.get("ok"):
        return d
    pl = d.get("plan") or {}
    if not pl.get("ok"):
        _audit({"action": "no_trade", "reason": pl.get("error"),
                "signal": d.get("signal", {}).get("score")})
        return {**d, "executed": False}

    # 下单：EA 指令通道优先，MCP 交易工具兜底
    r = exec_open(pl)
    _audit({"action": "open", "plan": pl, "signal": d.get("signal"),
            "result": r, "features": {k: d["features"].get(k) for k in
                                      ("price", "atr", "trend_atr", "pos_boll", "mom3_atr", "server_hour")}})
    if not r.get("ok"):
        return {**d, "executed": False, "via": r.get("via"),
                "error": r.get("error") or "下单失败",
                "hint": _trade_hint(str(r.get("error") or ""))}
    return {**d, "executed": True, "via": r.get("via"), "order": r}


def _trade_hint(err: str) -> str:
    e = (err or "").lower()
    if "not permitted" in e:
        return ("两条路都没通：① 在 MT5 左侧「导航器」右键 → 刷新（或按 F5），"
                "让终端认出 AIDeskExec 执行器；② 或在 工具 → 选项 → MCP / AI Agent 面板"
                "勾上允许交易类工具。")
    if "10027" in e:
        return "终端 Algo Trading 按钮没开：点工具栏上的「算法交易」开关。"
    if "market closed" in e:
        return "市场休市（黄金周末/节假日）。"
    if "margin" in e:
        return "可用保证金不足。"
    return ""


# ---------------------------------------------------------------- 执行通道
# EA 指令通道：Python 写 aidesk_cmd.csv → 终端里的 AIDeskExec.mq5 读走执行
# → 写 aidesk_ack.csv 回执。不经过 MCP 的 trade_*，所以不受其权限限制。
COMMON = (Path(os.environ.get("APPDATA", "")) / "MetaQuotes" / "Terminal"
          / "Common" / "Files")
CMD_FILE = COMMON / "aidesk_cmd.csv"
ACK_FILE = COMMON / "aidesk_ack.csv"
HB_FILE = COMMON / "aidesk_exec_heartbeat.csv"


def executor() -> dict:
    """读执行器 EA 的心跳（5 秒写一次），判断这条通道能不能用。"""
    out = {"alive": False, "live": False, "magic": None, "positions": None,
           "spread": None, "ts": None, "age_s": None, "file": str(HB_FILE)}
    try:
        if not HB_FILE.exists():
            return out
        age = time.time() - HB_FILE.stat().st_mtime
        rows = HB_FILE.read_text("utf-8", errors="replace").strip().splitlines()
        if not rows:
            return out
        f = rows[-1].split(";")

        def _i(k):
            try:
                return int(f[k])
            except (IndexError, ValueError):
                return None

        out.update({"alive": age <= 30, "age_s": round(age, 1), "ts": f[0],
                    "live": _i(1) == 1, "magic": _i(2), "positions": _i(3),
                    "spread": _i(4), "account_mode": _i(5)})
    except Exception:  # noqa: BLE001
        pass
    return out


def mcp_trade_ok() -> bool:
    """MCP 的 trade_* 是否被插件放开（缓存 120 秒，别每次都探）。"""
    global _mcp_ok, _mcp_ok_ts
    now = time.time()
    if _mcp_ok is not None and now - _mcp_ok_ts < 120:
        return bool(_mcp_ok)
    r = _call("trade_check_market_order", {"symbol": _cfg()["symbol"],
                                           "type": "buy", "volume": 0.01})
    _mcp_ok = bool(r.get("ok"))
    _mcp_ok_ts = now
    return _mcp_ok


def channel() -> str:
    """实际可用的下单通道：ea / mcp / none。"""
    cfg = _cfg()
    pref = str(cfg.get("exec_channel") or "auto")
    ex = executor()
    ea_ok = bool(ex["alive"] and ex["live"])
    if pref == "ea":
        return "ea" if ea_ok else "none"
    if pref == "mcp":
        return "mcp" if mcp_trade_ok() else "none"
    if ea_ok:
        return "ea"
    return "mcp" if mcp_trade_ok() else "none"


def _ea_cmd(line: str, rid: str, timeout: float = 20.0) -> dict:
    """投一条指令给终端内执行器，等它的回执。"""
    if not COMMON.exists():
        return {"ok": False, "via": "ea", "error": "找不到 MT5 公共文件夹 Common\\Files"}
    try:
        CMD_FILE.write_text(line + "\n", "utf-8")
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "via": "ea", "error": f"写指令文件失败：{e}"}
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if ACK_FILE.exists():
                for ln in ACK_FILE.read_text("utf-8", errors="replace").splitlines():
                    f = ln.split(";")
                    if len(f) >= 7 and f[0] == rid:
                        ok = f[1] == "1"
                        return {"ok": ok, "via": "ea", "retcode": f[2], "ticket": f[3],
                                "price": f[4], "error": "" if ok else f[5], "ts": f[6]}
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.4)
    return {"ok": False, "via": "ea",
            "error": "EA 执行器没回执（终端里 AIDeskExec 在跑吗？图表右上角有没有哭脸）"}


def _rid(prefix: str = "ai") -> str:
    return prefix + dt.datetime.now().strftime("%H%M%S%f")[:12]


def exec_open(pl: dict) -> dict:
    """按计划下单：EA 通道优先，其次 MCP。"""
    ch = channel()
    if ch == "none":
        return {"ok": False, "via": "none",
                "error": "没有可用的下单通道（执行器没跑，MCP 交易权限也没开）"}
    if ch == "ea":
        rid = _rid()
        line = ";".join(["open", rid, pl["symbol"], pl["side"],
                         f"{float(pl['volume']):.2f}", f"{float(pl['sl']):.2f}",
                         f"{float(pl['tp']):.2f}", str(pl["comment"])])
        return _ea_cmd(line, rid)
    chk = _call("trade_check_market_order", {
        "symbol": pl["symbol"], "type": pl["side"],
        "volume": pl["volume"], "sl": pl["sl"], "tp": pl["tp"]})
    if not chk.get("ok"):
        err = chk.get("error") or "下单前校验未通过"
        return {"ok": False, "via": "mcp", "error": err}
    r = _call("trade_send_market_order", {
        "symbol": pl["symbol"], "type": pl["side"], "volume": pl["volume"],
        "sl": pl["sl"], "tp": pl["tp"], "comment": pl["comment"]})
    return {**r, "via": "mcp"}


def exec_close(ticket) -> dict:
    ch = channel()
    if ch == "ea":
        rid = _rid("cl")
        return _ea_cmd(";".join(["close", rid, str(ticket)]), rid)
    if ch == "mcp":
        return {**_call("trade_close_position", {"ticket": ticket}), "via": "mcp"}
    return {"ok": False, "error": "没有可用通道"}


def close_ai_positions() -> dict:
    """平掉 AI 自己的全部持仓（按 ticket 逐个平）。"""
    cfg = _cfg()
    ps = ai_positions(cfg)
    if not ps:
        return {"ok": True, "closed": 0, "note": "没有 AI 持仓"}
    done = []
    for p in ps:
        r = exec_close(p["ticket"])
        _audit({"action": "close", "ticket": p["ticket"], "result": r})
        done.append({"ticket": p["ticket"], "ok": bool(r.get("ok")), "error": r.get("error")})
    return {"ok": all(x["ok"] for x in done), "closed": sum(1 for x in done if x["ok"]), "items": done}


def stop(reason: str = "手动停机") -> dict:
    """紧急停机：关总开关 + 尝试平掉 AI 持仓。"""
    cfg = _save_cfg({"enabled": False})
    _audit({"action": "stop", "reason": reason})
    closed = close_ai_positions()
    return {"ok": True, "enabled": False, "closed": closed, "reason": reason}


def status() -> dict:
    """给前端的状态包：配置 + 账户 + 持仓 + 今日 + 最近决策 + 审计尾。"""
    cfg = _cfg()
    acct = account()
    ai_pos = ai_positions(cfg) if acct["ok"] else []
    day = today_realized() if acct["ok"] else {"net": 0, "n": 0}
    tail = audit_tail(40)
    last = next((x for x in reversed(tail)
                 if x.get("action") in ("open", "open_failed", "no_trade", "hold",
                                        "refuse", "auto_hold", "auto_refuse")), None)
    n_open = sum(1 for x in tail if x.get("action") == "open" and (x.get("result") or {}).get("ok"))
    equity = float(acct.get("equity") or 0)
    ex = executor()
    st = _state()
    # 通道判定不去探 MCP（status 每 2 秒被前端拉一次），只用缓存与心跳
    if ex["alive"] and ex["live"]:
        ch = "ea"
    elif _mcp_ok:
        ch = "mcp"
    else:
        ch = "none"
    try:
        gap = (dt.datetime.now() - dt.datetime.strptime(
            str(st.get("last_check_at") or ""), "%Y-%m-%d %H:%M:%S")).total_seconds()
    except ValueError:
        gap = None
    return {
        "ok": acct["ok"], "ts": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "enabled": bool(cfg["enabled"]),
        "config": cfg,
        "account": acct,
        "positions": ai_pos,
        "today": day,
        "budget": {
            "risk_pct": cfg["risk_pct"],
            "risk_usd": round(equity * float(cfg["risk_pct"]) / 100.0, 2),
            "daily_stop_usd": round(equity * float(cfg["daily_loss_pct"]) / 100.0, 2),
        },
        "last": last, "opens_logged": n_open,
        "audit": tail[-12:],
        "auto": {
            "on": bool(cfg["auto"]), "tf": cfg.get("auto_tf") or "H1",
            "interval": cfg.get("auto_interval") or 30,
            "last_check": st.get("last_check"), "last_bar": st.get("last_bar"),
            "last_skip": st.get("last_skip"), "last_action": st.get("last_action"),
            "age_s": round(gap, 1) if gap is not None else None,
        },
        "executor": ex,
        "channel": ch,
        # 蛛网作业台的数据源：最近一轮实际判定的五维明细与结论
        "vis": {
            "at": st.get("last_decided_at"),
            "bar": st.get("last_bar"),
            "score": st.get("last_score"),
            "side": st.get("last_side"),
            "verdict": st.get("last_verdict"),
            "parts": st.get("last_parts") or [],
            "mode": st.get("last_mode") or "score",
            "plan_ok": st.get("last_plan_ok"),
            "plan_err": st.get("last_plan_err"),
            "hold": st.get("last_hold"),
            "blocked": st.get("last_blocked"),
            "news": st.get("last_news"),
            "threshold": float(cfg["min_score"]),
        },
        "note": ("开启后每根新 K 线自动判定一次并自行下单；"
                 "不开只会给你计划，不会发单。"),
    }


# ---------------------------------------------------------------- 自动盯盘
def _state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text("utf-8")) or {}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _save_state(patch: dict) -> dict:
    st = _state()
    st.update(patch or {})
    try:
        STATE_FILE.write_text(json.dumps(st, ensure_ascii=False, indent=1), "utf-8")
    except Exception:  # noqa: BLE001
        pass
    return st


def _snap(d: dict, bar: str | None = None) -> dict:
    """把一次决策压成可视化要用的快照（五维分数 + 结论），随 state 一起落盘。

    前端「蛛网作业台」靠 last_decided_at 判断有没有新的一轮，靠 last_parts
    点亮每个维度节点。没有这份快照，动画就只能瞎编。
    """
    s = d.get("signal") or {}
    pl = d.get("plan") or {}
    return {
        "last_decided_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_bar": bar or ((d.get("features") or {}).get("bar_time")),
        "last_score": s.get("score"),
        "last_side": s.get("side"),
        "last_verdict": s.get("verdict"),
        "last_parts": [{"name": x.get("name"), "score": x.get("score"), "why": x.get("why"),
                        "ok": x.get("ok")} for x in (s.get("parts") or [])],
        "last_mode": s.get("score_mode") or "score",
        "last_news": {
            "score": s.get("news"),
            "label": ((d.get("news") or {}).get("label")),
            "title": ((((d.get("news") or {}).get("top") or [{}])[0] or {}).get("title")),
            "shock": ((d.get("news") or {}).get("shock") or {}).get("title"),
        },
        "last_plan_ok": bool(pl.get("ok")),
        "last_plan_err": pl.get("error"),
        "last_hold": d.get("headline"),
        "last_blocked": d.get("error") if d.get("blocked") else None,
    }


class _Stg:
    """把 strategy 模块延迟导入，避免 serve ↔ ai_trader 循环引用。"""

    @staticmethod
    def decide(sp):
        import strategy as stg
        return stg.decide(sp)


_stg = _Stg()


def _active_strategy() -> dict | None:
    """当前启用的口述策略（没有就返回 None，回落到评分卡）。"""
    try:
        import strategy as stg
        return stg.active()
    except Exception:  # noqa: BLE001
        return None


def auto_due() -> bool:
    """后台线程问：现在该不该跑一次自动判定。"""
    cfg = _cfg()
    if cfg.get("auto_locked") or not cfg["enabled"] or not cfg["auto"]:
        return False
    last = str(_state().get("last_check_at") or "")
    try:
        gap = (dt.datetime.now() - dt.datetime.strptime(last, "%Y-%m-%d %H:%M:%S")).total_seconds()
    except ValueError:
        gap = 1e9
    return gap >= float(cfg.get("auto_interval") or 30)


def auto_step(force: bool = False) -> dict:
    """自动盯盘一步：同一根 K 线只判定一次；达标就自己下单、不达标就记观望。"""
    cfg = _cfg()
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tf = str(cfg.get("auto_tf") or "H1")
    if cfg.get("auto_locked"):
        return {"ok": False, "skipped": "自动盯盘已停用（auto_locked）"}
    if not cfg["enabled"]:
        return {"ok": False, "skipped": "引擎总开关关闭"}
    if not cfg["auto"] and not force:
        return {"ok": False, "skipped": "自动盯盘关闭"}

    f = features(cfg["symbol"], tf)
    if not f.get("ok"):
        _save_state({"last_check_at": stamp, "last_check": stamp, "last_skip": f.get("error")})
        return {"ok": False, "skipped": f.get("error"), "at": stamp}

    bar = f["bar_time"]
    if not force and _state().get("last_bar") == bar:
        # 同一根 K 线默认只判一次；但空仓观望时允许按 recheck_min 复判——
        # 盘中价格在变，上一根 K 线的分数不代表现在。持仓时不复判（纪律优先）。
        st0 = _state()
        recheck = float(cfg.get("recheck_min") or 0)
        ok_re = False
        if recheck > 0 and st0.get("last_decided_at") and not st0.get("last_ticket"):
            try:
                g2 = (dt.datetime.now() - dt.datetime.strptime(
                    str(st0["last_decided_at"]), "%Y-%m-%d %H:%M:%S")).total_seconds()
                ok_re = g2 >= recheck * 60
            except ValueError:
                pass
        if not ok_re:
            _save_state({"last_check_at": stamp, "last_check": stamp,
                         "last_skip": f"本根 K 线（{bar}）已判定"})
            return {"ok": True, "skipped": f"本根 K 线（{bar}）已判定过", "bar": bar}

    # 有启用的口述策略时，用策略的规则替代 5 维评分卡；执行/风控/审计完全共用。
    sp = _active_strategy()
    if sp:
        d = _stg.decide(sp)
        if d.get("features"):
            bar = str((d.get("features") or {}).get("bar_time") or bar)
    else:
        d = decide(cfg["symbol"], tf)
    snap = _snap(d, bar)   # 快照进 state，供前端蛛网图回放这一轮的每个判定步骤
    if d.get("blocked"):
        _audit({"action": "auto_refuse", "reason": d.get("error"), "bar": bar})
        _save_state({"last_check_at": stamp, "last_check": stamp, "last_bar": bar,
                     "last_skip": d.get("error"), **snap})
        return {"ok": False, "skipped": d.get("error"), "bar": bar}
    if d.get("hold"):
        cl = d.get("close") or {}
        if cl.get("reason") and cl.get("ticket"):
            r = exec_close(cl["ticket"])
            _audit({"action": "close", "auto": True, "ticket": cl["ticket"],
                    "reason": cl["reason"], "result": r, "bar": bar})
            _save_state({"last_check_at": stamp, "last_check": stamp, "last_bar": bar,
                         "last_action": "close", "last_skip": cl["reason"],
                         "last_ticket": None, **snap})
            return {"ok": bool(r.get("ok")), "closed": True, "reason": cl["reason"],
                    "bar": bar, "order": r}
        _audit({"action": "auto_hold", "reason": d.get("headline"), "bar": bar})
        _save_state({"last_check_at": stamp, "last_check": stamp, "last_bar": bar,
                     "last_skip": d.get("headline"), **snap})
        return {"ok": True, "hold": True, "skipped": d.get("headline"), "bar": bar}
    if not d.get("ok"):
        _save_state({"last_check_at": stamp, "last_check": stamp, "last_bar": bar,
                     "last_skip": d.get("error"), **snap})
        return {"ok": False, "skipped": d.get("error"), "bar": bar}

    pl = d.get("plan") or {}
    sc = (d.get("signal") or {}).get("score")
    if not pl.get("ok"):
        _audit({"action": "no_trade", "auto": True, "reason": pl.get("error"),
                "signal": sc, "bar": bar})
        _save_state({"last_check_at": stamp, "last_check": stamp, "last_bar": bar,
                     "last_skip": pl.get("error"), **snap})
        return {"ok": True, "executed": False, "skipped": pl.get("error"), "bar": bar}

    r = exec_open(pl)
    _audit({"action": "open" if r.get("ok") else "open_failed", "auto": True,
            "plan": pl, "signal": sc, "result": r, "via": r.get("via"), "bar": bar})
    _save_state({"last_check_at": stamp, "last_check": stamp, "last_bar": bar,
                 "last_action": "open" if r.get("ok") else "open_failed",
                 "last_skip": "" if r.get("ok") else (r.get("error") or ""),
                 "last_ticket": r.get("ticket"), **snap})
    return {"ok": bool(r.get("ok")), "executed": bool(r.get("ok")), "bar": bar,
            "plan": pl, "order": r,
            "error": None if r.get("ok") else r.get("error"),
            "hint": None if r.get("ok") else _trade_hint(str(r.get("error") or ""))}


_LOCK_MSG = ("自动盯盘已停用（auto_locked=true），界面开关不会重新开启；"
            "如需恢复请先手动把 data/ai_config.json 的 auto_locked 改为 false")


def set_enabled(on: bool) -> dict:
    if on and _cfg().get("auto_locked"):
        return {"ok": False, "error": LOCK_MSG}
    _save_cfg({"enabled": bool(on)})
    _audit({"action": "switch", "enabled": bool(on)})
    return {"ok": True, "enabled": bool(on)}


def set_auto(on: bool) -> dict:
    if on and _cfg().get("auto_locked"):
        return {"ok": False, "error": LOCK_MSG}
    _save_cfg({"auto": bool(on)})
    _audit({"action": "auto_switch", "auto": bool(on)})
    return {"ok": True, "auto": bool(on)}


def set_threshold(v) -> dict:
    """开仓门槛：人决定「多挑剔」，AI 只负责打分。范围锁死 40~90，不让它变成 0。"""
    try:
        n = float(v)
    except (TypeError, ValueError):
        return {"ok": False, "error": "阈值必须是数字"}
    n = max(40.0, min(90.0, n))
    _save_cfg({"min_score": n})
    _audit({"action": "threshold", "min_score": n})
    return {"ok": True, "min_score": n}


if __name__ == "__main__":
    # 命令行自检：python ai_trader.py  → 打印一次完整决策（不下单）
    import pprint
    pprint.pprint(decide())
