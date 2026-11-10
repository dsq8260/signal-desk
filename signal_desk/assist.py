"""智能辅助 · 8 板块计算层（只读信息，不下单、不给入场建议）。

设计边界（重要）：
  * 所有 K 线计算 / 信号判断只使用「已收盘」K 线：每个周期丢弃最后一根正在形成的 bar。
    只有最后一根已收盘 bar 的时间戳变化时才重算，盘中不抖动、不重绘。
  * 实时报价只用于顶部全局风险栏（Bid/Ask/点差/倒计时），不参与任何信号计算。
  * 共振打分是「规则加权」，不是概率、不是预测，页面固定提示仅作观察。
  * 参数 / 阈值 / 提示音开关全部落盘 data/assist_config.json，可在配置页修改。
  * 交易录入是本地账本，用于复盘统计，不连接下单接口、不改动 MT5。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
import threading
import time
from typing import Iterable

import account_feed
import context
import market
import markets
import plan as plan_mod
import pxfmt
import verdict
from pxfmt import pr, px

HERE = pathlib.Path(__file__).resolve().parent
CONFIG = HERE / "data" / "assist_config.json"
LEDGER = HERE / "data" / "assist_trades.json"
_lock = threading.Lock()
_calc_lock = threading.Lock()      # 全量计算只串行一次，避免并发请求各算一遍
_last_snap: dict = {}             # 最近一次分析的快照，录入交易时自动带上（不靠前端传，防篡改）
_payload_cache: dict = {}         # sym -> {"at","data"}：每品种各存一份，切换品种不互相冲掉
PAYLOAD_TTL = 30.0                # 缓存 TTL；配合 SWR（过期秒回旧值+后台重算）。
                                  # EA 工厂挖掘在跑时一次全量计算要 15~17s，
                                  # 20s 的 TTL 会让后台几乎不停在算，30s 更从容。

# ------------------------------------------------------------------ 配置
# 每项：(字段, 中文名, 类型, 默认值, 分组, 说明)
FIELDS: list[tuple] = [
    ("rsi_period", "RSI 周期", "int", 14, "指标参数", "Wilder 平滑"),
    ("rsi_ob", "RSI 超买线", "num", 70, "指标参数", "默认 70"),
    ("rsi_os", "RSI 超卖线", "num", 30, "指标参数", "默认 30"),
    ("macd_fast", "MACD 快线", "int", 12, "指标参数", "EMA 周期"),
    ("macd_slow", "MACD 慢线", "int", 26, "指标参数", "EMA 周期"),
    ("macd_signal", "MACD 信号线", "int", 9, "指标参数", "DEA 周期"),
    ("atr_period", "ATR 周期", "int", 14, "指标参数", "用于止损空间与波动率"),
    ("ma_fast", "MA 快线", "int", 20, "指标参数", "均线组"),
    ("ma_mid", "MA 中线", "int", 50, "指标参数", "均线组"),
    ("ma_slow", "MA 慢线", "int", 100, "指标参数", "均线组"),
    ("swing_k", "波段识别左右根数", "int", 3, "指标参数", "分形识别参数，越大越稳"),
    ("range_lookback", "中枢回看根数", "int", 48, "指标参数", "H1 根数"),
    ("spread_alert", "点差告警阈值", "num", 1.8, "告警阈值", "单位：美金（Ask-Bid）"),
    ("near_usd", "靠近关键位距离", "num", 3.0, "告警阈值", "单位：美金，≤ 触发提醒"),
    ("max_daily_loss_usd", "单日最大亏损线", "num", 50.0, "亏损保护", "单位：美金（正数）"),
    ("max_streak_loss", "最大连续亏损次数", "int", 3, "亏损保护", "连续亏损达到即告警"),
    ("high_risk_window_min", "高风险事件窗口", "int", 30, "亏损保护", "分钟，窗口内提示禁止新开仓"),
    ("risk_pct", "单笔风险%", "num", 1.0, "仓位与止损", "余额的百分比，1 = 1%"),
    ("stop_atr_mult", "止损距离（×ATR）", "num", 1.5, "仓位与止损", "止损 = 入场 ± 该倍数 × ATR(H1)，默认 1.5"),
    ("min_rr", "最低盈亏比", "num", 1.5, "仓位与止损", "低于该盈亏比的方案不出手，默认 1.5:1"),
    ("max_lots", "最大持仓手数", "num", 1.0, "仓位与止损", "手数计算的封顶值，超过按该值算"),
    ("trail_auto", "自动移动止损", "bool", True, "仓位与止损", "过了 1R 移保本、1.5R 锁 0.75R"),
    ("require_sentiment", "必须消息同向", "bool", True, "交易过滤",
     "开=开仓必须同时满足「消息同向 + 小周期启动」双确认；关=消息只做参考、不拦截（小周期启动仍需确认）"),
    ("block_range", "震荡市禁止开仓", "bool", False, "交易过滤", "主趋势=震荡整理时不给做多/做空结论"),
    ("block_counter", "逆势单禁止开仓", "bool", True, "交易过滤", "与大周期方向相反的方案直接判为观望"),
    ("block_rr", "盈亏比不足禁止开仓", "bool", True, "交易过滤", "盈亏比低于「最低盈亏比」时判为观望"),
    ("server_utc_offset", "服务器时间 UTC 偏移", "int", 2, "时段", "MT5 服务器时间相对 UTC 的小时数，用于划分亚/欧/美盘"),
    ("sound_message", "消息提醒音", "bool", True, "提示音开关", "一般消息"),
    ("sound_price", "价格抵达提醒音", "bool", True, "提示音开关", "关键价位告警"),
    ("sound_risk", "风控告警音", "bool", True, "提示音开关", "点差 / 亏损 / 事件"),
    ("auto_refresh_sec", "自动刷新间隔", "int", 10, "页面", "秒，仅刷新风险栏；K 线部分按收盘更新"),
]
DEFAULT = {k: v for k, _, _, v, _, _ in FIELDS}
GROUPS = ["指标参数", "告警阈值", "亏损保护", "仓位与止损", "交易过滤", "时段", "提示音开关", "页面"]
# 页面只展示「交易员自己会调、也看得懂的项」。
# 「指标参数」是系统内部算指标的口径（RSI/MACD/ATR/均线周期等），仍按 DEFAULT
# 参与全部计算，只是不再出现在设置卡里——那是给代码读的，不是给人读的。
HIDDEN_GROUPS = ("指标参数",)

HIGH_EVENT = ("非农", "NFP", "payroll", "CPI", "通胀", "FOMC", "美联储",
              "Fed ", "利率决议", "Rate Decision", "PCE", "零售销售")


def config_schema() -> dict:
    """给页面渲染用的字段表：剔除纯计算口径的分组（算法内部参数不展示）。"""
    return {"fields": [{"key": k, "label": l, "type": t, "default": d,
                        "group": g, "note": n}
                       for k, l, t, d, g, n in FIELDS if g not in HIDDEN_GROUPS],
            "groups": [g for g in GROUPS if g not in HIDDEN_GROUPS]}


def current_snap() -> dict:
    """给交易录入用的最近一次收盘分析快照（只读副本）。"""
    return dict(_last_snap)


def load_config() -> dict:
    cfg = dict(DEFAULT)
    try:
        raw = json.loads(CONFIG.read_text("utf-8"))
        if isinstance(raw, dict):
            for k, _, t, d, _, _ in FIELDS:
                if k not in raw:
                    continue
                v = raw[k]
                if t == "bool":
                    cfg[k] = bool(v)
                elif t == "int":
                    try:
                        cfg[k] = max(1, min(500, int(v)))
                    except (TypeError, ValueError):
                        cfg[k] = d
                elif t == "num":
                    try:
                        cfg[k] = float(v)
                    except (TypeError, ValueError):
                        cfg[k] = d
    except (OSError, ValueError, TypeError):
        pass
    return cfg


def save_config(body: dict) -> dict:
    """只接受 schema 里的字段，校验后原子写盘；返回保存后的完整配置。"""
    cur = load_config()
    for k, _, t, d, _, _ in FIELDS:
        if k not in body:
            continue
        v = body[k]
        try:
            if t == "bool":
                cur[k] = v is True or str(v).lower() in ("1", "true", "on", "yes")
            elif t == "int":
                cur[k] = max(1, min(500, int(v)))
            elif t == "num":
                cur[k] = max(0.0, float(v))
        except (TypeError, ValueError):
            return {"ok": False, "error": f"参数 {k} 的值不合法：{v}"}
    with _lock:
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=1), "utf-8")
        tmp.replace(CONFIG)
    return {"ok": True, "config": cur}


def reset_config() -> dict:
    with _lock:
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps(DEFAULT, ensure_ascii=False, indent=1), "utf-8")
    return {"ok": True, "config": dict(DEFAULT)}


# ------------------------------------------------------------------ 交易账本
REASONS = ["顺势回踩支撑", "压力反转", "背离 + 形态", "Pivot 支撑",
           "区间上沿做空", "区间下沿做多", "其他"]


def _load_ledger() -> dict:
    try:
        d = json.loads(LEDGER.read_text("utf-8"))
        if isinstance(d, dict) and isinstance(d.get("records"), list):
            return d
    except (OSError, ValueError, TypeError):
        pass
    return {"records": [], "claimed_deals": []}


def _save_ledger(d: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(LEDGER)


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _local_utc_h() -> float:
    """本机相对 UTC 的偏移小时（含夏令时）。"""
    off = dt.datetime.now().astimezone().utcoffset()
    return (off.total_seconds() / 3600.0) if off else 8.0


def _srv_shift_h(cfg: dict | None = None) -> float:
    """本地时间 → MT5 服务器时间 要加的小时数（服务器比本地快为正）。

    桥文件里的时间（成交 time、持仓 create_time、报价 msc）全是**服务器时间**，
    直接和 datetime.now()（本机时间）相减会差出一个时区偏移。
    实测：XM 服务器 UTC+2、本机 UTC+8 → 差 -6h，导致「早上 9 点开的单」被算成
    「已持 10 小时」。所有跨源时间差都必须先过这个换算。
    """
    cfg = cfg if cfg is not None else load_config()
    try:
        srv = float(cfg.get("server_utc_offset", 2))
    except (TypeError, ValueError):
        srv = 2.0
    return srv - _local_utc_h()


def _to_srv(t: dt.datetime | None, cfg: dict | None = None) -> dt.datetime | None:
    """本地 naive 时间 → 服务器 naive 时间（用于和桥/成交时间做差）。"""
    if t is None:
        return None
    return t + dt.timedelta(hours=_srv_shift_h(cfg))


def _srv_now(cfg: dict | None = None) -> dt.datetime:
    """MT5 服务器当前时间（naive），与桥文件时间同一坐标系。"""
    return dt.datetime.now() + dt.timedelta(hours=_srv_shift_h(cfg))


def _parse_t(s: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(str(s)[:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        try:
            return dt.datetime.strptime(str(s)[:16], "%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            return None


def add_trade(body: dict, snap: dict | None = None) -> dict:
    side = str(body.get("side") or "").lower()
    if side not in ("buy", "sell"):
        return {"ok": False, "error": "方向只能是 多(buy) / 空(sell)"}
    try:
        entry = float(body["entry"])
        sl = float(body["sl"])
        tp = float(body["tp"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "入场价 / 止损价 / 止盈价 都要填数字"}
    if entry <= 0 or sl <= 0 or tp <= 0:
        return {"ok": False, "error": "价格必须大于 0"}
    if (side == "buy" and not (sl < entry < tp)) or (side == "sell" and not (tp < entry < sl)):
        return {"ok": False, "error": "多单需 止损 < 入场 < 止盈；空单需 止盈 < 入场 < 止损"}
    reason = str(body.get("reason") or "其他")
    rec = {
        "id": "T" + dt.datetime.now().strftime("%Y%m%d%H%M%S") + str(int(time.time() * 1000) % 1000),
        "open_time": str(body.get("open_time") or _now()),
        "side": side, "entry": entry, "sl": sl, "tp": tp, "reason": reason,
        "plan_rr": pr(abs(tp - entry) / abs(entry - sl)) if entry != sl else None,
        "snap": snap or {}, "max_float_loss": 0.0,
        "close_time": None, "close_price": None, "pnl": None,
        "pnl_src": None, "real_rr": None, "hold_bars": None, "close_deal": None,
    }
    with _lock:
        d = _load_ledger()
        d["records"].append(rec)
        _save_ledger(d)
    return {"ok": True, "record": rec}


def close_trade(body: dict) -> dict:
    rid = str(body.get("id") or "")
    try:
        px = float(body.get("price"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "平仓价要填数字"}
    with _lock:
        d = _load_ledger()
        rec = next((x for x in d["records"] if x.get("id") == rid), None)
        if not rec:
            return {"ok": False, "error": "找不到这笔录入"}
        if rec.get("close_time"):
            return {"ok": False, "error": "这笔已经平仓登记过了"}
        _fill_close(rec, px, body.get("close_time") or _now())
        _save_ledger(d)
    return {"ok": True, "record": rec}


def delete_trade(body: dict) -> dict:
    rid = str(body.get("id") or "")
    with _lock:
        d = _load_ledger()
        n = len(d["records"])
        d["records"] = [x for x in d["records"] if x.get("id") != rid]
        if len(d["records"]) == n:
            return {"ok": False, "error": "找不到这笔录入"}
        _save_ledger(d)
    return {"ok": True}


def _fill_close(rec: dict, px: float, ctime: str,
                pnl_account: float | None = None, deal_key: str | None = None,
                from_broker: bool = False) -> None:
    direc = 1 if rec["side"] == "buy" else -1
    price_pnl = (px - rec["entry"]) * direc
    risk = abs(rec["entry"] - rec["sl"])
    rec["close_price"] = px
    rec["close_time"] = ctime
    rec["pnl"] = pr(price_pnl)               # 价格口径（美金/单位）
    rec["pnl_account"] = pr(pnl_account) if pnl_account is not None else None
    rec["pnl_src"] = "账户成交" if pnl_account is not None else "价格差估算"
    rec["real_rr"] = pr(price_pnl / risk) if risk else None
    a, b = _parse_t(rec.get("open_time")), _parse_t(ctime)
    if from_broker:
        # open_time 是页面录入的本地时间，ctime 是服务器成交时间 → 统一到服务器坐标系
        a = _to_srv(a)
    rec["hold_bars"] = max(0, int((b - a).total_seconds() // 3600)) if (a and b) else None
    rec["close_deal"] = deal_key


def sync_closes(symbol: str = "GOLD") -> int:
    """用账户历史成交自动登记已平仓录入。

    桥文件 deals 没有 ticket，因此按「方向相反 + 时间不早于开仓 + 时间最近」匹配，
    并用 claimed_deals 去重。匹配不上的保持未平仓，页面可手动平仓。
    """
    try:
        snap = account_feed.snapshot(symbol, days=7)
    except Exception:  # noqa: BLE001
        return 0
    deals = [x for x in (snap.get("deals") or []) if str(x.get("entry")) == "out"]
    if not deals:
        return 0
    changed = 0
    with _lock:
        d = _load_ledger()
        claimed = set(d.get("claimed_deals") or [])
        for rec in d["records"]:
            if rec.get("close_time"):
                continue
            want = "sell" if rec["side"] == "buy" else "buy"
            # 录入的 open_time 是本地时间，成交 t 是服务器时间 → 比之前先换算，
            # 否则服务器慢 6 小时时，刚平掉的单会被判成「早于开仓」而匹配不上
            t0 = _to_srv(_parse_t(rec.get("open_time")))
            cand = []
            for x in deals:
                key = f"{x.get('t')}|{x.get('price')}|{x.get('profit')}"
                if key in claimed or str(x.get("side")) != want:
                    continue
                tx = _parse_t(x.get("t"))
                if t0 and tx and tx < t0:
                    continue
                cand.append((tx or dt.datetime.min, x, key))
            if not cand:
                continue
            _, best, key = max(cand, key=lambda z: z[0])
            try:
                px = float(best.get("price"))
            except (TypeError, ValueError):
                continue
            pnl = None
            try:
                pnl = float(best.get("profit"))
            except (TypeError, ValueError):
                pass
            _fill_close(rec, px, str(best.get("t") or _now()), pnl, key, from_broker=True)
            claimed.add(key)
            changed += 1
        # 同时把已平仓记录里的 deal_key 计入，避免重复认领
        for rec in d["records"]:
            if rec.get("close_deal"):
                claimed.add(rec["close_deal"])
        d["claimed_deals"] = sorted(claimed)[-500:]
        if changed:
            _save_ledger(d)
    return changed


def update_float_loss(symbol: str = "GOLD") -> None:
    """把当前持仓浮亏写回未平仓录入，用于「今日最大浮亏」。"""
    try:
        snap = account_feed.snapshot(symbol, days=1)
        pos = snap.get("positions") or []
    except Exception:  # noqa: BLE001
        return
    if not pos:
        return
    worst = min((float(x.get("profit") or 0) for x in pos), default=0.0)
    if worst >= 0:
        return
    with _lock:
        d = _load_ledger()
        hit = False
        for rec in d["records"]:
            if rec.get("close_time"):
                continue
            if worst < float(rec.get("max_float_loss") or 0):
                rec["max_float_loss"] = pr(worst)
                hit = True
        if hit:
            _save_ledger(d)


def _win_stats(rows: list[dict]) -> dict:
    done = [r for r in rows if r.get("pnl") is not None]
    wins = [r for r in done if float(r["pnl"]) > 0]
    rr = [float(r["real_rr"]) for r in done if r.get("real_rr") is not None]
    return {"n": len(done), "wins": len(wins),
            "win_rate": round(len(wins) / len(done) * 100, 1) if done else None,
            "avg_rr": pr(sum(rr) / len(rr)) if rr else None,
            "avg_pnl": pr(sum(float(r["pnl"]) for r in done) / len(done)) if done else None}


def journal() -> dict:
    d = _load_ledger()
    recs = d["records"]
    today = dt.date.today().isoformat()
    today_recs = [r for r in recs if str(r.get("open_time") or "").startswith(today)]
    closed = [r for r in recs if r.get("close_time")]
    closed.sort(key=lambda r: str(r.get("close_time")))
    # 连续亏损（按平仓时间从后往前）
    streak = 0
    for r in reversed(closed):
        if float(r.get("pnl") or 0) < 0:
            streak += 1
        else:
            break
    with_trend = [r for r in closed if (r.get("snap") or {}).get("align") == "顺势"]
    against = [r for r in closed if (r.get("snap") or {}).get("align") == "逆势"]
    by_signal = []
    for tag in ("形态", "背离", "均线", "关键价位", "超买超卖"):
        sub = [r for r in closed if tag in ((r.get("snap") or {}).get("tags") or [])]
        s = _win_stats(sub)
        s["signal"] = tag
        by_signal.append(s)
    today_pnl = sum(float(r["pnl"]) for r in today_recs if r.get("pnl") is not None)
    max_float = min([float(r.get("max_float_loss") or 0) for r in today_recs] or [0.0])
    return {
        "open": [r for r in recs if not r.get("close_time")],
        "closed": list(reversed(closed))[:30],
        "n_all": len(recs),
        "stats": {
            "with_trend": _win_stats(with_trend),
            "against_trend": _win_stats(against),
            "by_signal": by_signal,
            "all": _win_stats(closed),
        },
        "today": {"date": today, "pnl": pr(today_pnl),
                  "max_float_loss": pr(max_float),
                  "streak_loss": streak, "n_open_today": len(today_recs)},
        "reasons": REASONS,
    }


# ------------------------------------------------------------------ 指标工具
def _rows(bars: list) -> list[dict]:
    out = []
    for b in bars or []:
        try:
            out.append({"t": str(b[0]), "o": float(b[1]), "h": float(b[2]),
                        "l": float(b[3]), "c": float(b[4]),
                        "v": float(b[5] or 0), "spread": float(b[6] or 0)})
        except (IndexError, TypeError, ValueError):
            continue
    return out


TF_MIN = {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
          "H1": 60, "H4": 240, "D1": 1440, "W1": 10080}


def _closed_rows(sym: str, tf: str, n: int = 300) -> tuple[list[dict], str, dict]:
    """按时间判断最后一根是否收盘，只把「已收盘」K 线交给信号计算。

    比「无脑丢最后一根」更准：桥文件可能滞后，此时最后一根其实早就收盘了。
    只要最后一根已收盘 bar 的时间戳不变，信号就不会变 —— 盘中不抖动。
    """
    p = market.payload(sym, tf, n)
    if not p.get("ok"):
        return [], "", p
    raw = _rows(p.get("bars"))
    if not raw:
        return [], "", p
    t = _parse_t(raw[-1]["t"])
    if t is not None and t + dt.timedelta(minutes=TF_MIN.get(tf, 60)) <= dt.datetime.now():
        rows = raw                       # 最后一根已收盘
    else:
        rows = raw[:-1]                  # 最后一根正在形成，丢掉
    return rows, (rows[-1]["t"] if rows else ""), p


def _ema(vals: list[float], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(vals)
    if len(vals) < n or n < 1:
        return out
    a = 2.0 / (n + 1)
    seed = sum(vals[:n]) / n
    out[n - 1] = seed
    for i in range(n, len(vals)):
        out[i] = vals[i] * a + float(out[i - 1]) * (1 - a)
    return out


def _sma(vals: list[float], n: int) -> float | None:
    return sum(vals[-n:]) / n if len(vals) >= n and n > 0 else None


def _rsi_series(closes: list[float], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    if len(closes) < n + 1:
        return out
    dif = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(0.0, d) for d in dif]
    losses = [max(0.0, -d) for d in dif]
    ag, al = sum(gains[:n]) / n, sum(losses[:n]) / n
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n, len(dif)):
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + losses[i]) / n
        out[i + 1] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def _macd_series(closes: list[float], fast: int, slow: int, sig: int):
    ef, es = _ema(closes, fast), _ema(closes, slow)
    dif: list[float | None] = [None] * len(closes)
    for i in range(len(closes)):
        if ef[i] is not None and es[i] is not None:
            dif[i] = float(ef[i]) - float(es[i])
    start = next((i for i, v in enumerate(dif) if v is not None), None)
    dea: list[float | None] = [None] * len(closes)
    hist: list[float | None] = [None] * len(closes)
    if start is None:
        return dif, dea, hist
    tail = [float(v) for v in dif[start:] if v is not None]
    de = _ema(tail, sig)
    for j, v in enumerate(de):
        i = start + j
        dea[i] = v
        if v is not None and dif[i] is not None:
            hist[i] = 2 * (float(dif[i]) - v)
    return dif, dea, hist


def _atr_series(rows: list[dict], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(rows)
    if len(rows) < n + 1:
        return out
    trs = [0.0]
    for i in range(1, len(rows)):
        r, pc = rows[i], rows[i - 1]["c"]
        trs.append(max(r["h"] - r["l"], abs(r["h"] - pc), abs(r["l"] - pc)))
    a = sum(trs[1:n + 1]) / n
    out[n] = a
    for i in range(n + 1, len(rows)):
        a = (a * (n - 1) + trs[i]) / n
        out[i] = a
    return out


def _swings(rows: list[dict], k: int) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    hi: list[tuple[int, float]] = []
    lo: list[tuple[int, float]] = []
    for i in range(k, len(rows) - k):
        h, l = rows[i]["h"], rows[i]["l"]
        if all(rows[j]["h"] <= h for j in range(i - k, i + k + 1) if j != i):
            hi.append((i, h))
        if all(rows[j]["l"] >= l for j in range(i - k, i + k + 1) if j != i):
            lo.append((i, l))
    return hi, lo


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(round((len(sorted_vals) - 1) * q))))
    return sorted_vals[i]


# ------------------------------------------------------------------ 板块 2
def _trend(rows: list[dict], cfg: dict, tf: str) -> dict:
    c = [r["c"] for r in rows]
    if len(c) < max(cfg["ma_slow"], cfg["macd_slow"]) + 5:
        return {"tf": tf, "trend": "数据不足", "strength": 0, "structure": "—",
                "vol": "—", "dir": "none", "note": "收盘 K 线不足"}
    ma_f = _sma(c, cfg["ma_fast"])
    ma_m = _sma(c, cfg["ma_mid"])
    ma_s = _sma(c, cfg["ma_slow"])
    atrs = [a for a in _atr_series(rows, cfg["atr_period"]) if a is not None]
    atr = atrs[-1] if atrs else max(1e-6, c[-1] * 0.001)
    price = c[-1]
    # 方向：EMA 快/慢线相对位置定趋势，价格相对快线定强弱；不要求严格三线排列
    ef = _ema(c, cfg["ma_fast"])
    em = _ema(c, cfg["ma_mid"])
    e_f, e_m = ef[-1], em[-1]
    if e_f is not None and e_m is not None:
        if e_f > e_m and price > e_f:
            d, label = "bull", "上涨"
        elif e_f > e_m:
            d, label = "bull", "偏多回踩"
        elif e_f < e_m and price < e_f:
            d, label = "bear", "下跌"
        elif e_f < e_m:
            d, label = "bear", "偏空反弹"
        else:
            d, label = "none", "震荡"
    else:
        d, label = "none", "震荡"

    ema_f = _ema(c, cfg["ma_fast"])
    idx = len(c) - 1
    slope = 0.0
    if ema_f[idx] is not None and ema_f[max(0, idx - 5)] is not None:
        slope = (float(ema_f[idx]) - float(ema_f[idx - 5])) / max(atr, 1e-6)
    sep = ((ma_f - ma_m) / max(atr, 1e-6)) if (ma_f and ma_m) else 0.0
    same = 0
    for i in range(max(1, len(c) - 20), len(c)):
        if (c[i] > c[i - 1]) == (d == "bull"):
            same += 1
    consistency = same / 20.0
    # 强度 = MA 斜率 45% + 均线间距 35% + 同向 K 线占比 20%（可解释，不是概率）
    strength = round(100 * (0.45 * min(1.0, abs(slope) / 2.0)
                            + 0.35 * min(1.0, abs(sep) / 2.0)
                            + 0.20 * consistency))
    if d == "none":
        strength = min(strength, 45)
    strength = max(0, min(100, strength))

    hi, lo = _swings(rows[-120:], max(2, cfg["swing_k"]))
    if len(hi) >= 2 and len(lo) >= 2:
        hh, lh = hi[-1][1] > hi[-2][1], lo[-1][1] > lo[-2][1]
        structure = ("HH/HL 上升结构" if hh and lh else
                     "LH/LL 下降结构" if (not hh and not lh) else
                     "高低点交错 · 转换中")
    else:
        structure = "区间 / 结构未成形"

    vol = "—"
    if len(atrs) >= 30:
        base = sorted(atrs[-100:])
        cur = atrs[-1]
        rank = sum(1 for x in base if x <= cur) / len(base)
        vol = "低" if rank < 0.33 else ("中" if rank < 0.67 else "高")
    return {"tf": tf, "trend": label, "strength": strength, "structure": structure,
            "vol": vol, "dir": d, "ma": {"fast": ma_f, "mid": ma_m, "slow": ma_s},
            "atr": atr, "closed_t": rows[-1]["t"]}


def cross_timeframe(sym: str, cfg: dict) -> tuple[list[dict], dict, dict]:
    packs: dict[str, list[dict]] = {}
    out = []
    for tf in ("D1", "H4", "H1"):
        rows, ct, p = _closed_rows(sym, tf, 300)
        packs[tf] = rows
        out.append(_trend(rows, cfg, tf) if rows else
                   {"tf": tf, "trend": "无数据", "strength": 0, "structure": "—",
                    "vol": "—", "dir": "none", "closed_t": ct,
                    "note": (p.get("error") or "该周期没有已收盘 K 线")})
    h1, h4, d1 = out[2]["dir"], out[1]["dir"], out[0]["dir"]
    if h1 == "none":
        label, why, align = "中性观望", "H1 本身无明确方向，不区分顺势逆势", "中性"
    elif h1 == h4 and h1 == d1:
        label, why, align = "✅顺势", "H1 与 H4、D1 全部同向", "顺势"
    elif h4 != h1 and d1 != h1:
        label, why, align = "❌强逆", "H1 与 H4、D1 全部反向", "逆势"
    elif h4 != h1:
        label, why, align = "⚠️逆势", "H1 与 H4 反向，D1 未共振", "逆势"
    else:
        label, why, align = "✅顺势", "H1 与 H4 同向（D1 不同向，级别偏短）", "顺势"
    return out, {"label": label, "why": why, "align": align}, packs


# ------------------------------------------------------------------ 板块 3
def _near_dist(cfg: dict) -> float:
    """当前品种「靠近关键位」的距离（价格单位）：黄金 3 美金 ≈ 直盘 12 pips。"""
    return markets.near(pxfmt.sym(), cfg)


def key_levels(packs: dict, cfg: dict, price: float) -> dict:
    h1 = packs.get("H1") or []
    d1 = packs.get("D1") or []
    items: list[dict] = []
    if h1:
        win = h1[-120:]
        hi, lo = _swings(win, max(2, cfg["swing_k"]))
        for i, v in hi[-3:][::-1]:
            items.append({"name": "近期波段高点", "price": pr(v), "kind": "resist",
                          "at": win[i]["t"]})
        for i, v in lo[-3:][::-1]:
            items.append({"name": "近期波段低点", "price": pr(v), "kind": "support",
                          "at": win[i]["t"]})
        n = min(cfg["range_lookback"], len(h1))
        seg = h1[-n:]
        up = _pct(sorted(r["h"] for r in seg), 0.85)
        dn = _pct(sorted(r["l"] for r in seg), 0.15)
        items.append({"name": "震荡中枢上沿", "price": pr(up), "kind": "resist",
                      "at": f"近 {n} 根 H1 高点 85 分位"})
        items.append({"name": "震荡中枢下沿", "price": pr(dn), "kind": "support",
                      "at": f"近 {n} 根 H1 低点 15 分位"})
    pivot_note = "D1 数据不足，未计算 Pivot"
    base = d1[-1] if d1 else None
    if base:
        H, L, C = base["h"], base["l"], base["c"]
        P = (H + L + C) / 3
        rng = H - L
        for name, v, kind in (("Pivot 轴心 P", P, "pivot"),
                              ("R1", 2 * P - L, "resist"),
                              ("R2", P + rng, "resist"),
                              ("S1", 2 * P - H, "support"),
                              ("S2", P - rng, "support")):
            items.append({"name": name, "price": pr(v), "kind": kind,
                          "at": f"上一已收盘 D1 {base['t'][:10]}"})
        pivot_note = (f"标准 Floor Pivot，基于上一根已收盘 D1（{base['t'][:10]}，"
                      f"H {H:.{px()}f} / L {L:.{px()}f} / C {C:.{px()}f}）")
    for x in items:
        x["dist"] = pr(price - x["price"])
        x["abs_dist"] = pr(abs(price - x["price"]))
        x["near"] = abs(price - x["price"]) <= _near_dist(cfg)
    items.sort(key=lambda x: x["abs_dist"])
    near = [x for x in items if x["near"]]
    # 位置标签（给 K 线形态用）
    region = "一般位置"
    if h1:
        hi_vals = [x["price"] for x in items if x["kind"] == "support"]
        lo_vals = [x["price"] for x in items if x["kind"] == "resist"]
        if hi_vals and min(abs(price - v) for v in hi_vals) <= _near_dist(cfg) * 2:
            region = "支撑区域"
        elif lo_vals and min(abs(price - v) for v in lo_vals) <= _near_dist(cfg) * 2:
            region = "压力区域"
    return {"items": items, "near": near, "region": region,
            "near_usd": _near_dist(cfg), "note": pivot_note}


# ------------------------------------------------------------------ 板块 4
def indicators(packs: dict, cfg: dict, trend_row: dict) -> dict:
    rows = packs.get("H1") or []
    c = [r["c"] for r in rows]
    rsi_s = _rsi_series(c, cfg["rsi_period"])
    rsi = next((v for v in reversed(rsi_s) if v is not None), None)
    dif, dea, hist = _macd_series(c, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    last = len(c) - 1
    atrs = [a for a in _atr_series(rows, cfg["atr_period"]) if a is not None]
    atr = atrs[-1] if atrs else None
    atr_pct = None
    if len(atrs) >= 30:
        base = sorted(atrs[-200:])
        atr_pct = round(sum(1 for v in base if v <= atrs[-1]) / len(base), 3)
    ma_f = _sma(c, cfg["ma_fast"])
    ma_m = _sma(c, cfg["ma_mid"])
    ma_s = _sma(c, cfg["ma_slow"])
    env = trend_row.get("trend") or "震荡"
    trend_env = ("上涨" in env) or ("下跌" in env) or ("偏多" in env) or ("偏空" in env)
    if rsi is None:
        rsi_state, rsi_note = "数据不足", "收盘 K 线不足"
    elif rsi >= cfg["rsi_ob"]:
        rsi_state = "超买"
        rsi_note = ("趋势环境超买 = 顺势延续（不视为反转）" if trend_env
                    else "震荡环境超买 = 承压")
    elif rsi <= cfg["rsi_os"]:
        rsi_state = "超卖"
        rsi_note = ("趋势环境超卖 = 顺势延续（不视为反转）" if trend_env
                    else "震荡环境超卖 = 有支撑")
    else:
        rsi_state, rsi_note = "中性区", "未进入超买 / 超卖"

    # MACD 背离：最近两个波段高点/低点，价格与 DIF 不同步
    divergence = None
    hi, lo = _swings(rows[-120:], max(2, cfg["swing_k"]))
    off = len(rows) - len(rows[-120:])
    if len(hi) >= 2:
        (i1, p1), (i2, p2) = hi[-2], hi[-1]
        d1, d2 = dif[off + i1], dif[off + i2]
        if d1 is not None and d2 is not None and p2 > p1 and float(d2) < float(d1):
            divergence = "顶背离"
    if not divergence and len(lo) >= 2:
        (i1, p1), (i2, p2) = lo[-2], lo[-1]
        d1, d2 = dif[off + i1], dif[off + i2]
        if d1 is not None and d2 is not None and p2 < p1 and float(d2) > float(d1):
            divergence = "底背离"

    if ma_f and ma_m and ma_s:
        if c[-1] > ma_f > ma_m > ma_s:
            ma_state = "多头排列"
        elif c[-1] < ma_f < ma_m < ma_s:
            ma_state = "空头排列"
        else:
            ma_state = "缠绕震荡"
    else:
        ma_state = "数据不足"
    flags = []
    if rsi_state in ("超买", "超卖"):
        flags.append("RSI 进入超买超卖区")
    if divergence:
        flags.append(f"MACD {divergence}")
    if ma_state == "多头排列":
        flags.append("均线多头排列")
    elif ma_state == "空头排列":
        flags.append("均线空头排列")
    return {
        "rsi": {"value": round(rsi, 1) if rsi is not None else None,
                "state": rsi_state, "note": rsi_note,
                "ob": cfg["rsi_ob"], "os": cfg["rsi_os"]},
        "macd": {"dif": pr(float(dif[last])) if dif[last] is not None else None,
                 "dea": pr(float(dea[last])) if dea[last] is not None else None,
                 "hist": pr(float(hist[last])) if hist[last] is not None else None,
                 "divergence": divergence},
        "atr": {"value": pr(atr) if atr is not None else None,
                "pct": atr_pct,
                "note": "估算止损空间：常见用 1.0~1.5 × ATR"},
        "ma": {"ma_fast": pr(ma_f) if ma_f else None,
               "ma_mid": pr(ma_m) if ma_m else None,
               "ma_slow": pr(ma_s) if ma_s else None,
               "state": ma_state, "above": bool(ma_f and c[-1] > ma_f),
               "price": c[-1] if c else None},
        "flags": flags, "closed_t": rows[-1]["t"] if rows else "",
    }


# ------------------------------------------------------------------ 板块 5
def _body(r: dict) -> float:
    return abs(r["c"] - r["o"])


def candle_patterns(packs: dict, cfg: dict, levels: dict) -> list[dict]:
    rows = packs.get("H1") or []
    if len(rows) < 3:
        return []
    a, b, c = rows[-3], rows[-2], rows[-1]
    out: list[dict] = []

    def add(name: str, d: str, why: str) -> None:
        out.append({"name": name, "dir": d,
                    "text": f"H1，{levels.get('region') or '一般位置'}，{name}",
                    "why": why, "at": c["t"]})

    rng_c = max(c["h"] - c["l"], 1e-9)
    up_sh = c["h"] - max(c["o"], c["c"])
    dn_sh = min(c["o"], c["c"]) - c["l"]
    p, q = b, c
    bull_eng = (p["c"] < p["o"] and q["c"] > q["o"]
                and q["c"] >= p["o"] and q["o"] <= p["c"])
    bear_eng = (p["c"] > p["o"] and q["c"] < q["o"]
                and q["c"] <= p["o"] and q["o"] >= p["c"])
    mid_p = (p["o"] + p["c"]) / 2
    pierce = (p["c"] < p["o"] and q["c"] > q["o"]
              and q["o"] < p["l"] and mid_p < q["c"] < p["o"])
    dark_cloud = (p["c"] > p["o"] and q["c"] < q["o"]
                  and q["o"] > p["h"] and p["o"] < q["c"] < mid_p)
    if bull_eng:
        add("看涨吞没", "bull", "阴线后被阳线完全包覆，收盘确认")
    if bear_eng:
        add("看跌吞没", "bear", "阳线后被阴线完全包覆，收盘确认")
    if not bull_eng and pierce:
        add("刺透", "bull", "低开后收复前阴线实体一半以上")
    if not bear_eng and dark_cloud:
        add("乌云盖顶", "bear", "高开后跌破前阳线实体一半以下")
    # 三根星形
    big_a = _body(a) > 0.6 * (a["h"] - a["l"])
    small_b = _body(b) < 0.35 * _body(a)
    big_c = _body(c) > 0.6 * (c["h"] - c["l"])
    if (big_a and small_b and big_c and a["c"] < a["o"] and c["c"] > c["o"]
            and b["h"] < max(a["o"], a["c"]) and c["c"] > (a["o"] + a["c"]) / 2):
        add("启明星", "bull", "大阴 + 小实体 + 大阳，收复首根实体一半")
    if (big_a and small_b and big_c and a["c"] > a["o"] and c["c"] < c["o"]
            and b["l"] > min(a["o"], a["c"]) and c["c"] < (a["o"] + a["c"]) / 2):
        add("黄昏星", "bear", "大阳 + 小实体 + 大阴，跌破首根实体一半")
    # 长影线
    if (dn_sh >= 2 * max(_body(c), 1e-9) and dn_sh / rng_c >= 0.55
            and (c["c"] - c["l"]) / rng_c >= 0.6):
        add("长下影", "bull", "下影 ≥ 2 倍实体且收盘位于上 1/3")
    if (up_sh >= 2 * max(_body(c), 1e-9) and up_sh / rng_c >= 0.55
            and (c["h"] - c["c"]) / rng_c >= 0.6):
        add("长上影", "bear", "上影 ≥ 2 倍实体且收盘位于下 1/3")
    return out


# ------------------------------------------------------------------ 板块 6
def resonance(cross_w: dict, levels: dict, patterns: list[dict], ind: dict) -> dict:
    parts = []

    def add(name: str, weight: int, hit: bool, why: str) -> None:
        parts.append({"name": name, "weight": weight, "score": weight if hit else 0,
                      "hit": hit, "why": why})

    trend_hit = cross_w.get("align") == "顺势"
    add("趋势同向（H1+H4）", 30, trend_hit, cross_w.get("why") or "")
    near_hit = bool(levels.get("near"))
    add("价格抵达关键支撑 / 压力", 20, near_hit,
        (f"距 {levels['near'][0]['name']} {levels['near'][0]['abs_dist']} 美金"
         if near_hit else "当前价格未进入设定距离"))
    pat_hit = bool(patterns)
    add("K 线反转形态确认", 20, pat_hit,
        ("、".join(p["name"] for p in patterns[:3]) if pat_hit else "最近一根已收盘 H1 无确认形态"))
    rsi_hit = ind["rsi"]["state"] in ("超买", "超卖")
    add("RSI 进入超买超卖区", 15, rsi_hit,
        f"RSI {ind['rsi']['value']} · {ind['rsi']['state']}")
    div_hit = bool(ind["macd"]["divergence"])
    add("MACD 背离", 15, div_hit, ind["macd"]["divergence"] or "未检测到顶 / 底背离")
    score = sum(p["score"] for p in parts)
    grade = ("高价值观察机会" if score >= 70 else
             "普通观察" if score >= 40 else "不建议出手")
    return {"score": score, "grade": grade, "parts": parts,
            "disclaimer": "仅观察信号，不构成交易入场建议，等待自己 K 线确认"}


# ------------------------------------------------------------------ 板块 1
def risk_bar(sym: str, cfg: dict, ops: dict, packs: dict, quote: dict) -> dict:
    # 财经事件
    cal = (ops or {}).get("calendar") or {}
    cal_ok = bool(cal.get("ok") or cal.get("rows"))
    nx = cal.get("next_high")
    ev = {"name": ("暂无未来 24 小时高影响事件" if cal_ok else "财经日历本次未取到"),
          "mins": None, "importance": "—", "level": "低", "countdown": "—",
          "no_open": False, "calendar_ok": cal_ok}
    if nx:
        name = str(nx.get("name") or "未命名事件")
        mins = nx.get("mins")
        high_kw = any(k.lower() in name.lower() for k in HIGH_EVENT)
        level = "高" if high_kw else ("中" if str(nx.get("importance")) == "HIGH" else "低")
        try:
            m = int(mins)
            countdown = (f"{m // 60} 小时 {m % 60} 分" if m >= 60 else f"{m} 分钟")
            no_open = 0 <= m <= cfg["high_risk_window_min"] and level == "高"
        except (TypeError, ValueError):
            countdown, no_open, m = "—", False, None
        ev = {"name": name, "mins": m, "importance": nx.get("importance") or "—",
              "level": level, "countdown": countdown, "no_open": no_open,
              "time": nx.get("time") or "", "calendar_ok": cal_ok}

    # 报价与点差
    bid = quote.get("bid")
    ask = quote.get("ask")
    spread_price = (ask - bid) if (bid is not None and ask is not None) else None
    _pt = markets.point(sym)
    _thr = markets.spread_alert(sym, cfg)
    spread_pt = (spread_price / _pt) if spread_price is not None else None
    spread_alert = spread_price is not None and spread_price > _thr
    d1_all = _rows((market.payload(sym, "D1", 20).get("bars") or []))
    gap_usd = gap_pct = None
    if len(d1_all) >= 2:
        gap_usd = d1_all[-1]["o"] - d1_all[-2]["c"]
        gap_pct = gap_usd / d1_all[-2]["c"] * 100 if d1_all[-2]["c"] else None

    # 手工风控（来自本地账本）
    j = journal()
    today = j["today"]
    breached = []
    if today["pnl"] <= -cfg["max_daily_loss_usd"]:
        breached.append(f"今日已盈亏 {today['pnl']} 触及单日最大亏损线 -{cfg['max_daily_loss_usd']}")
    if today["streak_loss"] >= cfg["max_streak_loss"]:
        breached.append(f"连续亏损 {today['streak_loss']} 次，达到设定上限")

    # 持仓汇总
    buy_float = sell_float = 0.0
    n_untracked = 0
    total_hold_h = 0.0
    pos_list: list[dict] = []
    acct_info: dict = {"balance": None, "equity": None, "currency": "USD", "login": "", "server": ""}
    try:
        acct = account_feed.snapshot(sym, days=2)
        acc = acct.get("account") or {}
        if acc:
            acct_info = {"balance": acc.get("balance"), "equity": acc.get("equity"),
                         "margin": acc.get("margin"), "margin_free": acc.get("margin_free"),
                         "margin_level": acc.get("margin_level"),
                         "currency": acc.get("currency") or "USD",
                         "login": acc.get("login") or "", "server": acc.get("server") or "",
                         "age_s": acc.get("age_s")}
        pos = acct.get("positions") or []
        deals = acct.get("deals") or []
        for p in pos:
            try:
                v = float(p.get("profit") or 0)
            except (TypeError, ValueError):
                v = 0.0
            if str(p.get("side")) == "buy":
                buy_float += v
            else:
                sell_float += v
            # 持仓时长：同向最后一笔进场成交的时间（MT5 持仓文件里没有开仓时间）
            hold_h = None
            side = str(p.get("side") or "?")
            last_in = None
            for dl in deals:
                if dl.get("entry") == "in" and dl.get("side") == side:
                    last_in = dl.get("t")
            if last_in:
                t = _parse_t(last_in)
                if t:
                    # 成交时间是服务器时间，必须用服务器 now 相减（差 6 小时就是这么来的）
                    hold_h = round(max(0.0, (_srv_now(cfg) - t).total_seconds() / 3600), 1)
            pos_list.append({"ticket": p.get("ticket"), "side": side,
                             "volume": p.get("volume"),
                             "price_open": p.get("price_open"),
                             "price_current": p.get("price_current"),
                             "profit": pr(v), "hold_h": hold_h})
        n_untracked = max(0, len(pos) - len(j["open"]))
    except Exception:  # noqa: BLE001
        pass
    now = dt.datetime.now()
    for r in j["open"]:
        t = _parse_t(r.get("open_time"))
        if t:
            total_hold_h += max(0.0, (now - t).total_seconds() / 3600)

    alerts = []
    if ev["no_open"]:
        alerts.append({"type": "risk", "short": "事件窗口",
                       "text": f"高风险事件窗口：{ev['name']}（{ev['countdown']}）· 禁止新开仓提醒",
                       "act": "先不新开仓，等数据公布后再看；持仓单检查好止损"})
    if spread_alert:
        _spt = f"{spread_pt:.1f} 点" if px() >= 4 else f"{spread_price:.{px()}f} 美金"
        _tpt = f"{_thr / _pt:.0f} 点" if px() >= 4 else f"{_thr:g}"
        alerts.append({"type": "risk", "short": "点差过大",
                       "text": f"实时点差 {_spt} > 阈值 {_tpt}",
                       "act": "点差太宽下单成本高：先不进场，等点差回落"})
    for b in breached:
        alerts.append({"type": "risk", "short": "风控线",
                       "text": b, "act": "今天已触线：停止新开仓，管好手上的单子"})
    return {
        "event": ev,
        "account": acct_info,
        "quote": {"bid": bid, "ask": ask,
                  "ok": quote.get("ok", False),
                  "spread_price": pr(spread_price) if spread_price is not None else None,
                  "spread_pt": round(spread_pt, 1) if spread_pt is not None else None,
                  "spread_alert": spread_alert, "threshold": _thr,
                  "gap_usd": pr(gap_usd) if gap_usd is not None else None,
                  "gap_pct": round(gap_pct, 3) if gap_pct is not None else None,
                  "tick_time": quote.get("tick_time"), "age_s": quote.get("age_s")},
        "manual": {"today_pnl": today["pnl"], "max_float_loss": today["max_float_loss"],
                   "streak_loss": today["streak_loss"], "n_open_today": today["n_open_today"],
                   "max_daily_loss": cfg["max_daily_loss_usd"],
                   "max_streak": cfg["max_streak_loss"], "breached": breached},
        "positions": {"buy_float": pr(buy_float), "sell_float": pr(sell_float),
                      "total_hold_hours": round(total_hold_h, 1),
                      "n_untracked": n_untracked, "list": pos_list,
                      "note": "持仓时长仅统计已录入订单；未录入持仓不计时长"},
        "alerts": alerts,
    }


# ------------------------------------------------------------------ 聚合
def payload(sym: str = "GOLD", ops: dict | None = None, force: bool = False) -> dict:
    """服务端结果缓存 + stale-while-revalidate：过期先回旧数据，后台重算。

    为什么要 SWR：一次完整计算 1~3 秒（几万行成交记录解析），页签切换正好
    赶上缓存过期时会让页面「—」好几秒。过期后立即返回上一份结果（stale），
    同时起后台线程重算，下一轮轮询自然拿到新值——页面永远秒开。

    force=True 用于「手动刷新 / 配置保存 / 录入订单后」，仍同步重算（用户
    主动点了刷新，等 1~3 秒是合理预期），并绕过 stale 回退。
    """
    c = _payload_cache.get(sym) or {}
    fresh = bool(c.get("data")) and time.time() - c.get("at", 0) < PAYLOAD_TTL
    if fresh and not force:
        return dict(c["data"], ts=_now(), cached=True)
    if fresh and force:
        with _calc_lock:
            out = _compute_payload(sym, ops)
            _payload_cache[sym] = {"at": time.time(), "data": out}
            return dict(out, cached=False)
    # 过期 / 无缓存：有旧数据就先回旧值，后台重算
    if c.get("data") and not force:
        _refresh_async(sym, ops)
        with _calc_lock:
            return dict(c["data"], ts=_now(), cached=True, stale=True)
    with _calc_lock:
        out = _compute_payload(sym, ops)
        _payload_cache[sym] = {"at": time.time(), "data": out}
        return dict(out, cached=False)


_refreshing: set = set()
_refresh_lock = threading.Lock()


def _refresh_async(sym: str, ops: dict | None) -> None:
    """后台重算：同一时刻只允许一个，算完更新缓存；失败保持旧缓存。"""
    with _refresh_lock:
        if sym in _refreshing:
            return
        _refreshing.add(sym)

    def _job():
        t0 = time.time()
        try:
            out = _compute_payload(sym, ops)
            with _calc_lock:
                _payload_cache[sym] = {"at": time.time(), "data": out}
            dt = time.time() - t0
            if dt > 10.0:
                print(f"[辅助] 后台重算偏慢 {sym} · {dt:.1f}s"
                      "（EA 工厂挖掘在跑时会拖慢辅助计算）", flush=True)
        except Exception as e:  # noqa: BLE001  重算失败：保留旧缓存，下次再试
            print(f"[辅助] 后台重算失败 {sym}: {type(e).__name__}: {e}", flush=True)
        finally:
            with _refresh_lock:
                _refreshing.discard(sym)

    threading.Thread(target=_job, daemon=True, name="assist-swr").start()


def warmup(sym: str = "GOLD", ops: dict | None = None) -> None:
    """服务启动时后台预热一次，页面首屏直接读缓存。"""
    try:
        payload(sym, ops, force=True)
    except Exception:  # noqa: BLE001
        pass


def _compute_payload(sym: str, ops: dict | None) -> dict:
    global _last_snap
    _t0 = time.time()

    def _mark(stage):
        dt = time.time() - _t0
        if dt > 3.0:                     # 只在异常慢时打日志，避免刷屏
            print(f"[辅助] {stage} 慢 · {dt:.1f}s", flush=True)

    _mark("开始")
    pxfmt.use(sym)                     # 之后所有 pr()/f"{:.{px()}f}" 按本品种精度走
    cfg = dict(load_config(),
               _contract=markets.contract(sym), _point=markets.point(sym))
    sync_closes(sym)
    _mark("sync_closes")
    update_float_loss(sym)
    _mark("update_float_loss")
    cross, cross_w, packs = cross_timeframe(sym, cfg)
    _mark("cross_timeframe")
    # 主趋势判定要用 M15（文档权重 H4 50 / H1 30 / M15 20），共振卡还要 M5
    # 150 根足够 _trend 需要（MA100+5）；挖掘线程抢 GIL 时少读一半，快很多
    m15rows, _, _ = _closed_rows(sym, "M15", 150)
    m5rows, _, _ = _closed_rows(sym, "M5", 150)
    tf_extra = {"M15": _trend(m15rows, cfg, "M15") if m15rows else None,
                "M5": _trend(m5rows, cfg, "M5") if m5rows else None}
    _mark("tf_extra")
    h1p = market.payload(sym, "H1", 5)
    _mark("market.payload")
    live = h1p.get("live") or {}
    bid, ask = live.get("bid"), live.get("ask")
    quote_ok = bool(live.get("on")) and bid is not None
    tick_time, age_s = live.get("tick_time"), live.get("age_s")
    if not quote_ok:
        # 绝不用几小时前的旧 tick 冒充实时报价；只把「过期多久」如实带给页面
        try:
            tk = market.latest_tick(sym) or {}
            age_s = tk.get("age_s")
            if float(tk.get("age_s") or 1e9) <= 120 and tk.get("bid") is not None:
                bid, ask, quote_ok = tk["bid"], tk["ask"], True
                tick_time, age_s = tk.get("tick_time"), tk.get("age_s")
        except Exception:  # noqa: BLE001
            pass
    h1rows = packs.get("H1") or []
    last_close = h1rows[-1]["c"] if h1rows else None
    price = float(bid) if (quote_ok and bid is not None) else float(last_close or 0)
    price_src = "实时 Bid" if quote_ok else "最后已收盘 H1 收盘价（实时 tick 过期）"
    if not quote_ok:
        bid = ask = None
    quote = {"bid": bid, "ask": ask, "tick_time": tick_time,
             "age_s": age_s, "ok": quote_ok}
    levels = key_levels(packs, cfg, price)
    ind = indicators(packs, cfg, cross[-1])
    sess = context.session_info(packs.get("H1") or [], packs.get("D1") or [], price,
                                cfg.get("server_utc_offset", 2), ind["atr"]["value"])
    levels = context.enrich_levels(
        levels, packs.get("H1") or [], ind["atr"]["value"], ind,
        extra_lines=sess.get("lines"),
        ma_len=(cfg["ma_fast"], cfg["ma_mid"], cfg["ma_slow"]))
    patterns = candle_patterns(packs, cfg, levels)
    res = resonance(cross_w, levels, patterns, ind)
    risk = risk_bar(sym, cfg, ops or {}, packs, quote)
    _mark("risk_bar")
    vd = verdict.build(cross, levels, ind, risk, price)
    cond = context.checklist(cross, cross_w, levels, ind, patterns, risk, sess,
                             ind["atr"]["value"])
    size = context.sizing((risk.get("account") or {}).get("balance"), cfg["risk_pct"],
                          (ind["atr"]["value"] or 0) * 1.5,
                          contract=markets.contract(sym))
    jr = journal()
    pn = plan_mod.build(cross=cross, tf_extra=tf_extra, levels=levels, ind=ind,
                        risk=risk, price=price, cfg=cfg, patterns=patterns,
                        journal=jr, vd=vd)
    _mark("plan")
    # 当日涨跌：与上一根已收盘 D1 的收盘价比
    d1rows = packs.get("D1") or []
    prev_close = d1rows[-1]["c"] if d1rows else None
    day = {"prev_close": pr(prev_close) if prev_close else None,
           "chg": pr(price - prev_close) if prev_close else None,
           "chg_pct": pr((price - prev_close) / prev_close * 100) if prev_close else None}
    # 价格抵达告警：单独成类，受 sound_price 控制；带「接下来怎么办」的人话建议
    def _lv_action(x: dict) -> str:
        arrived = (x.get("abs_dist") or 0) <= _near_dist(cfg) * 0.05
        verb = "已到达" if arrived else "正在接近"
        if x.get("kind") == "resist":
            return (f"{verb}上方压力区 → 持多单：分批平仓止盈或把止损上移；"
                    "想做空：等回抽确认不破再进，别提前追；空仓：不追多")
        if x.get("kind") == "support":
            return (f"{verb}下方支撑区 → 持空单：分批平仓止盈或把止损下移；"
                    "想做多：等企稳信号再进，别提前抄；空仓：不追空")
        return f"{verb} Pivot 轴心 → 先看方向选择：突破跟、失守撤，不抢跑"

    for x in levels["near"]:
        risk["alerts"].append({"type": "price",
                               "short": f"{x['name']} {x['price']} · 距 {x['abs_dist']}",
                               "text": f"价格抵达关键区域：{x['name']} {x['price']}"
                                       f"（相距 {x['abs_dist']}）",
                               "act": _lv_action(x),
                               "kind": x.get("kind")})
    snap = {"align": cross_w["align"], "weight": cross_w["label"],
            "score": res["score"], "grade": res["grade"],
            "tags": list(ind["flags"]),            "closed_t": (packs.get("H1") or [{}])[-1].get("t", ""),
            "price": price}
    if patterns:
        snap["tags"].append("形态")
    if ind["macd"]["divergence"]:
        snap["tags"].append("背离")
    if any("均线" in f for f in ind["flags"]):
        snap["tags"].append("均线")
    if levels["near"]:
        snap["tags"].append("关键价位")
    if ind["rsi"]["state"] in ("超买", "超卖"):
        snap["tags"].append("超买超卖")
    snap["tags"] = sorted(set(snap["tags"]))
    _last_snap = dict(snap)
    return {
        "ok": True, "symbol": sym, "ts": _now(), "config": cfg,
        "closed": {x["tf"]: x.get("closed_t") or "" for x in cross},
        "feed": h1p.get("feed"), "live_on": bool(live.get("on")),
        "data_end": h1p.get("end"), "price": price, "price_src": price_src,
        "quote_ok": quote_ok, "tick_age_s": age_s,
        "risk": risk, "verdict": vd, "cross": cross, "cross_weight": cross_w,
        "cond": cond, "session": sess, "sizing": size, "plan": pn, "day": day,
        "levels": levels, "indicators": ind, "patterns": patterns,
        "resonance": res, "journal": jr,
        "tf_extra": {k: ({"tf": v.get("tf"), "trend": v.get("trend"),
                          "dir": v.get("dir"), "strength": v.get("strength"),
                          "structure": v.get("structure"), "vol": v.get("vol")}
                         if v else None) for k, v in tf_extra.items()},
        "reason_options": REASONS,
        "note": ("K 线相关板块只使用已收盘 K 线；实时报价只用于顶部风险栏。"
                 "收盘 bar 时间戳不变则不重绘。"),
    }
