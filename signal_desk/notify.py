# -*- coding: utf-8 -*-
"""手机推送：把告警/工单/结算结果推到手机上，不用一直开电脑。

为什么不用「手机浏览器看页面」：页面是给桌面看的单屏看板，手机上要缩放、
要点菜单、还没法主动叫你。真正需要的是「有事它找你」——一条微信/Telegram
消息，一眼看到品种、价格、该怎么做，看完就能决定。

支持渠道（选一个填一条 webhook/key 就行，都走 HTTPS，不需要公网 IP）：
  wecom      企业微信群机器人（最省事：手机微信直接收，免费、无墙）
  dingtalk   钉钉群机器人
  feishu     飞书群机器人
  telegram   Telegram Bot（chat_id + bot token）
  bark       Bark（iOS 原生推送，免费）
  serverchan Server酱（微信服务号推送）
  pushplus   PushPlus（微信推送）
  custom     自定义 webhook：POST {"title","text","kind","symbol"}

三条硬规则：
1) 推送失败绝不影响主流程（所有异常吞掉并留痕），页面该算的照算；
2) 同一件事不重复轰炸：按 dedupe 键做冷却（默认 30 分钟）+ 每日上限；
3) 夜间静默可关（默认 23:00-07:00 静默），除非是「可执行工单」。
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
CFG_FILE = DATA / "notify.json"
STATE_FILE = DATA / "notify_state.json"
LOG_FILE = DATA / "notify_log.jsonl"
LOCK = threading.Lock()
_LAST_TICK = 0.0          # 巡检守护最后一次心跳（页面可视化用）

DEFAULT = {
    "enabled": True,
    "channel": "wecom",          # 见文件头
    "webhook": "",               # 群机器人地址 / bark key / serverchan key
    "token": "",                 # telegram bot token / pushplus token
    "chat_id": "",               # telegram chat_id
    "panel_url": "",             # 可选：面板地址，消息末尾带一个入口
    "cooldown_s": 1800,          # 同一件事 30 分钟内只推一次
    "daily_max": 30,             # 每天最多推几条，防刷屏
    "quiet": "23:00-07:00",      # 夜间静默；空字符串 = 不静默
    "on_alert": True,            # 价位/风控告警
    "on_order": True,            # 三审出「可执行工单」
    "on_settle": True,           # 影子账本结算
    "on_signal": True,           # 02 页四条件达标（MT5 手动下单指导）
    "auto_scan_min": 30,         # 自动巡检三审的间隔（分钟，0=不自动跑）
}

CHANNELS = [
    ("wecom", "企业微信群机器人", "群设置 → 群机器人 → 添加 → 复制 Webhook 地址"),
    ("dingtalk", "钉钉群机器人", "群设置 → 智能群助手 → 添加机器人 → Webhook"),
    ("feishu", "飞书群机器人", "群设置 → 群机器人 → 自定义机器人 → Webhook"),
    ("telegram", "Telegram Bot", "找 @BotFather 建 bot 拿 token，chat_id 用 @getmyid_bot 查"),
    ("bark", "Bark（iOS）", "App 里复制自己的 key，形如 https://api.day.app/xxxx"),
    ("serverchan", "Server酱", "https://sct.ftqq.com 拿 SendKey"),
    ("pushplus", "PushPlus（微信）", "https://www.pushplus.plus 拿 token"),
    ("custom", "自定义 Webhook", "POST JSON：{title,text,kind,symbol}"),
]


# ------------------------------------------------------------------ 配置
def cfg() -> dict:
    c = dict(DEFAULT)
    try:
        c.update(json.loads(CFG_FILE.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        pass
    # webhook 里有 key，接口返回时打码，别把密钥显示在页面上
    return c


def public_cfg() -> dict:
    c = cfg()
    out = dict(c)
    for k in ("webhook", "token"):
        v = str(c.get(k) or "")
        out[k] = (v[:6] + "…" + v[-4:]) if len(v) > 12 else ("已设置" if v else "")
        out[k + "_set"] = bool(v)
    out["channels"] = [{"key": k, "name": n, "hint": h} for k, n, h in CHANNELS]
    return out


def public_state() -> dict:
    """巡检可视化状态：心跳/三审倒计时/今日推送计数。绝不带密钥。"""
    st = _state()
    c = cfg()
    scans = {k: float(v) for k, v in (st.get("scan") or {}).items()}
    return {
        "now": time.time(),
        "last_tick": float(_LAST_TICK or 0),
        "interval_s": 180,
        "auto_scan_min": int(c.get("auto_scan_min") or 0),
        "count": int(st.get("count") or 0),
        "daily_max": int(c.get("daily_max") or 0),
        "scan": scans,
    }


def save(patch: dict) -> dict:
    with LOCK:
        c = cfg()
        for k in DEFAULT:
            if k in patch and patch[k] is not None:
                c[k] = patch[k]
        for k in ("webhook", "token", "chat_id", "panel_url", "quiet", "channel"):
            if k in patch and isinstance(patch[k], str):
                c[k] = patch[k].strip()
        for k in ("enabled", "on_alert", "on_order", "on_settle", "on_signal"):
            if k in patch:
                c[k] = bool(patch[k])
        for k in ("cooldown_s", "daily_max", "auto_scan_min"):
            if k in patch:
                try:
                    c[k] = int(patch[k])
                except (TypeError, ValueError):
                    pass
        CFG_FILE.parent.mkdir(exist_ok=True)
        CFG_FILE.write_text(json.dumps(c, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    return {"ok": True, "config": public_cfg()}


# ------------------------------------------------------------------ 状态
def _state() -> dict:
    try:
        d = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(d, dict):
            d.setdefault("seen", {})
            d.setdefault("day", "")
            d.setdefault("count", 0)
            return d
    except Exception:  # noqa: BLE001
        pass
    return {"seen": {}, "day": "", "count": 0}


def _write_state(d: dict) -> None:
    DATA.mkdir(exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)


def reset_state() -> dict:
    with LOCK:
        _write_state({"seen": {}, "day": "", "count": 0})
    return {"ok": True}


# ------------------------------------------------------------------ 发送
def _quiet_now(quiet: str) -> bool:
    """当前是否在静默时段（支持跨零点，如 23:00-07:00）。"""
    q = str(quiet or "").strip()
    if "-" not in q:
        return False
    try:
        a, b = q.split("-", 1)
        h1, m1 = [int(x) for x in a.strip().split(":")]
        h2, m2 = [int(x) for x in b.strip().split(":")]
    except (ValueError, TypeError):
        return False
    now = dt.datetime.now()
    cur = now.hour * 60 + now.minute
    s, e = h1 * 60 + m1, h2 * 60 + m2
    return (s <= cur or cur < e) if s > e else (s <= cur < e)


def _http(url: str, payload: dict | None = None, timeout: float = 8.0) -> tuple[bool, str]:
    data = None
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(400).decode("utf-8", "ignore")
            return (200 <= r.status < 300), body
    except urllib.error.HTTPError as e:
        try:
            body = e.read(300).decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            body = ""
        return False, f"HTTP {e.code} {body}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _post(c: dict, title: str, text: str) -> tuple[bool, str]:
    ch = str(c.get("channel") or "wecom").lower()
    hook = str(c.get("webhook") or "").strip()
    token = str(c.get("token") or "").strip()
    body = f"**{title}**\n{text}" if ch in ("wecom", "dingtalk") else f"{title}\n{text}"

    if ch == "wecom":
        if not hook:
            return False, "未填企业微信 Webhook"
        return _http(hook, {"msgtype": "markdown", "markdown": {"content": body}})
    if ch == "dingtalk":
        if not hook:
            return False, "未填钉钉 Webhook"
        return _http(hook, {"msgtype": "markdown",
                            "markdown": {"title": title, "text": body}})
    if ch == "feishu":
        if not hook:
            return False, "未填飞书 Webhook"
        return _http(hook, {"msg_type": "text", "content": {"text": f"{title}\n{text}"}})
    if ch == "telegram":
        if not token:
            return False, "未填 Telegram bot token"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        return _http(url, {"chat_id": c.get("chat_id") or "", "text": f"{title}\n{text}",
                           "disable_web_page_preview": True})
    if ch == "bark":
        if not hook:
            return False, "未填 Bark key（形如 https://api.day.app/xxxx）"
        base = hook.rstrip("/")
        u = (f"{base}/{urllib.parse.quote(title)}/{urllib.parse.quote(text)}"
             f"?group=signal&level=timeSensitive&isArchive=1")
        return _http(u)
    if ch == "serverchan":
        key = hook or token
        if not key:
            return False, "未填 Server酱 SendKey"
        if not key.startswith("http"):
            key = f"https://sctapi.ftqq.com/{key}.send"
        return _http(key, {"title": title, "desp": text})
    if ch == "pushplus":
        if not token:
            return False, "未填 PushPlus token"
        return _http("https://www.pushplus.plus/send",
                     {"token": token, "title": title, "content": text, "template": "txt"})
    if ch == "custom":
        if not hook:
            return False, "未填自定义 Webhook"
        return _http(hook, {"title": title, "text": text})
    return False, f"未知渠道 {ch}"


def _log(row: dict) -> None:
    try:
        LOG_FILE.parent.mkdir(exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
        if len(lines) > 300:
            LOG_FILE.write_text("\n".join(lines[-300:]) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def logs(n: int = 30) -> list[dict]:
    if not LOG_FILE.is_file():
        return []
    try:
        rows = []
        for ln in LOG_FILE.read_text(encoding="utf-8").splitlines()[-n:]:
            try:
                rows.append(json.loads(ln))
            except Exception:  # noqa: BLE001
                continue
        return list(reversed(rows))
    except Exception:  # noqa: BLE001
        return []


def send(title: str, text: str, kind: str = "info", dedupe: str = "",
         force: bool = False, urgent: bool = False) -> dict:
    """推一条到手机。返回 {ok, skipped?, error?}，任何情况都不抛异常。"""
    c = cfg()
    if not c.get("enabled") and not force:
        return {"ok": False, "skipped": "推送已关闭"}
    if kind == "alert" and not c.get("on_alert") and not force:
        return {"ok": False, "skipped": "已关闭告警推送"}
    if kind == "order" and not c.get("on_order") and not force:
        return {"ok": False, "skipped": "已关闭工单推送"}
    if kind == "signal" and not c.get("on_signal") and not force:
        return {"ok": False, "skipped": "已关闭达标信号推送"}
    if kind == "settle" and not c.get("on_settle") and not force:
        return {"ok": False, "skipped": "已关闭结算推送"}
    if not force and not urgent and _quiet_now(str(c.get("quiet") or "")):
        return {"ok": False, "skipped": "夜间静默时段"}

    now = time.time()
    today = dt.date.today().isoformat()
    with LOCK:
        st = _state()
        if st.get("day") != today:
            st["day"], st["count"] = today, 0
        key = dedupe or (title + "|" + text[:40])
        last = float((st["seen"].get(key) or 0))
        if not force and last and now - last < float(c.get("cooldown_s") or 1800):
            _write_state(st)
            return {"ok": False, "skipped": "冷却中（同一件事刚推过）"}
        if not force and int(st.get("count") or 0) >= int(c.get("daily_max") or 30):
            _write_state(st)
            return {"ok": False, "skipped": "今日推送已达上限"}
        st["seen"][key] = now
        # 只保留最近 300 个去重键
        if len(st["seen"]) > 300:
            for k in sorted(st["seen"], key=lambda x: st["seen"][x])[:len(st["seen"]) - 300]:
                st["seen"].pop(k, None)
        st["count"] = int(st.get("count") or 0) + 1
        _write_state(st)

    ok, detail = _post(c, title, text)
    _log({"at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          "channel": c.get("channel"), "kind": kind, "title": title,
          "ok": ok, "detail": str(detail)[:200]})
    if not ok:
        if "未填" in str(detail):
            # 渠道还没配置好：保留冷却位，避免每 2 分钟空转一次；配置好后自然恢复
            pass
        else:
            # 网络/对端失败：不占冷却位，下一轮立刻重试
            with LOCK:
                st = _state()
                st["seen"].pop(key, None)
                st["count"] = max(0, int(st.get("count") or 1) - 1)
                _write_state(st)
    return {"ok": ok, "error": None if ok else detail}


def test() -> dict:
    c = cfg()
    text = ("这是一条测试消息：能收到说明推送通道已经通了。\n"
            f"渠道：{c.get('channel')} · 时间：{dt.datetime.now():%Y-%m-%d %H:%M}")
    r = send("SIGNAL DESK 推送测试", text, kind="info", force=True)
    return {**r, "channel": c.get("channel")}


# ------------------------------------------------------------------ 自动巡检
def _sym_name(sym: str) -> str:
    try:
        import markets
        return markets.name(sym)
    except Exception:  # noqa: BLE001
        return sym


def _fmt_px(sym: str, v) -> str:
    try:
        import markets
        return markets.fmt(sym, v)
    except Exception:  # noqa: BLE001
        return str(v)


def _order_text(sym: str, res: dict) -> tuple[str, str]:
    """把三审结果压成一条手机消息：方向 / 价位 / 手数 / 为什么。"""
    a1 = res.get("a1") or {}
    a2 = res.get("a2") or {}
    plan = a1.get("plan") or {}
    ez = plan.get("entry_zone") or []
    if isinstance(ez, list) and len(ez) >= 2:
        entry = (float(ez[0]) + float(ez[1])) / 2
    else:
        entry = plan.get("entry") or (res.get("facts") or {}).get("price")
    stop = plan.get("stop")
    tps = plan.get("targets") or []
    tp1 = tp2 = None
    if isinstance(tps, list) and tps:
        first = tps[0]
        tp1 = first.get("price") if isinstance(first, dict) else first
        if len(tps) > 1:
            second = tps[1]
            tp2 = second.get("price") if isinstance(second, dict) else second
    lots = ((a2.get("local_lots") or {}) if isinstance(a2.get("local_lots"), dict) else {})
    rr = None
    try:
        if entry and stop and tp1:
            rr = abs(float(tp1) - float(entry)) / abs(float(entry) - float(stop))
    except (TypeError, ValueError):
        rr = None
    name = _sym_name(sym)
    title = f"可执行工单 · {name} {'做多' if a1.get('direction') == 'buy' else '做空'}"
    lines = [
        f"入场 {_fmt_px(sym, entry)}（区间 {_fmt_px(sym, ez[0]) if isinstance(ez, list) and ez else '—'}"
        f" ~ {_fmt_px(sym, ez[1]) if isinstance(ez, list) and len(ez) > 1 else '—'}）",
        f"止损 {_fmt_px(sym, stop)} · 目标 {_fmt_px(sym, tp1)}"
        + (f" / {_fmt_px(sym, tp2)}" if tp2 else "")
        + (f" · 盈亏比 {rr:.2f}:1" if rr else ""),
        f"手数 {lots.get('lots')} 手 · 单笔风险 {lots.get('risk_usd')} 美金（净值 "
        f"{(res.get('cfg') or {}).get('risk_pct', 0.5)}%）",
        f"二审：{(a2.get('verdict') or '').upper()} · {(a2.get('reason') or '本地闸门全过')[:60]}",
        f"依据：{(a1.get('summary') or '')[:70]}",
    ]
    c = cfg()
    if c.get("panel_url"):
        lines.append(f"看板：{c['panel_url']}")
    return title, "\n".join(lines)


def _digits(sym: str) -> int:
    try:
        import markets
        for m in markets.all():
            if m.get("sym") == sym:
                return int(m.get("digits") or 2)
    except Exception:  # noqa: BLE001
        pass
    return 2


def tick(force_scan: bool = False) -> dict:
    """后台巡检一次：告警 → 三审工单 → 影子结算。返回本轮做了什么（便于排查）。"""
    out = {"alerts": 0, "orders": 0, "settle": 0, "signals": 0, "scan": [], "errors": []}
    c = cfg()
    if not c.get("enabled"):
        return out
    import markets
    syms = [x["sym"] for x in markets.enabled()]

    # 1) 价位 / 风控告警：用服务端缓存（页面没开也能算），同一段文字 30 分钟内只推一次
    try:
        import assist
        cal = {}
        for sym in syms:
            try:
                d = assist.payload(sym, {"calendar": cal})
                for a in ((d.get("risk") or {}).get("alerts") or []):
                    txt = str(a.get("text") or "")
                    if not txt:
                        continue
                    title = f"{_sym_name(sym)} · {a.get('short') or '提醒'}"
                    body = txt + (f"\n→ {a.get('act')}" if a.get("act") else "")
                    r = send(title, body, kind="alert",
                             dedupe=f"alert|{sym}|{a.get('short') or txt[:20]}")
                    if r.get("ok"):
                        out["alerts"] += 1
            except Exception as e:  # noqa: BLE001
                out["errors"].append(f"alert {sym}: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"alert: {type(e).__name__}: {e}")

    # 1.5) 达标信号：02 页四条件全绿 → 直接给 MT5 手动下单参数
    try:
        import assist
        cal = {}
        for sym in syms:
            try:
                # 上面第 1 步刚算过 payload，这里只吃缓存
                d = assist.payload(sym, {"calendar": cal})
                pe = (d.get("plan") or {}).get("entry") or {}
                side = str(pe.get("side") or "wait")
                if side not in ("long", "short"):
                    continue
                sp = pe.get(side) or {}
                if not sp.get("qualify") or str(pe.get("cls") or "wait") == "wait":
                    continue
                dg = _digits(sym)

                def _f(v, _dg=dg):
                    try:
                        return f"{float(v):.{_dg}f}"
                    except (TypeError, ValueError):
                        return "—"
                arrow = "买入 BUY" if side == "long" else "卖出 SELL"
                rr = sp.get("rr")
                rr_s = (f"{float(rr):.1f}:1" if rr is not None else "—")
                risk = sp.get("risk_usd")
                title = f"{_sym_name(sym)} · {'多头' if side == 'long' else '空头'}达标信号"
                body = (f"方向：{arrow}\n"
                        f"入场：{_f(sp.get('entry'))}（{sp.get('zone') or ''}）\n"
                        f"止损：{_f(sp.get('stop'))}\n"
                        f"目标1：{_f(sp.get('tp1'))}（{sp.get('tp1_name') or ''}）\n"
                        f"目标2：{_f(sp.get('tp2'))}（{sp.get('tp2_name') or ''}）\n"
                        f"盈亏比：{rr_s} · 建议手数：{sp.get('lots') or '—'}"
                        + (f"（风险约 ${float(risk):.0f}）" if risk else "") +
                        "\n→ MT5 按上表下单；下单后回 04 页「生成工单」可自动盯盘提醒")
                key = f"signal|{sym}|{side}|{sp.get('stop')}|{sp.get('entry')}"
                st = _state()
                seen_sig = st.setdefault("signal_seen", {})
                now_sig = time.time()
                if now_sig - float(seen_sig.get(key) or 0) < 4 * 3600:
                    continue  # 同一套方案 4 小时内不重复推
                seen_sig[key] = now_sig
                if len(seen_sig) > 100:
                    for k2 in sorted(seen_sig, key=lambda x: seen_sig[x])[:len(seen_sig) - 100]:
                        seen_sig.pop(k2, None)
                _write_state(st)
                r = send(title, body, kind="signal", dedupe=key)
                if r.get("ok"):
                    out["signals"] += 1
            except Exception as e:  # noqa: BLE001
                out["errors"].append(f"signal {sym}: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"signal: {type(e).__name__}: {e}")

    # 2) 自动巡检三审：没人在电脑前也能发现机会（间隔由 auto_scan_min 控制）
    try:
        mins = int(c.get("auto_scan_min") or 0)
        if mins > 0:
            import ai_council
            st = _state()
            scans = st.setdefault("scan", {})
            now = time.time()
            for sym in syms:
                last = float(scans.get(sym) or 0)
                if not force_scan and now - last < mins * 60:
                    continue
                scans[sym] = now
                with LOCK:
                    _write_state(st)
                try:
                    res = ai_council.run(sym, force=True)
                except Exception as e:  # noqa: BLE001
                    out["errors"].append(f"scan {sym}: {type(e).__name__}: {e}")
                    continue
                r = res.get("result") or {}
                out["scan"].append({"sym": sym, "ok": bool(res.get("ok")),
                                    "verdict": (r.get("a2") or {}).get("verdict"),
                                    "direction": (r.get("a1") or {}).get("direction")})
                a1, a2 = r.get("a1") or {}, r.get("a2") or {}
                if (a1.get("direction") in ("buy", "sell")
                        and str(a2.get("verdict") or "") == "pass"
                        and (a1.get("plan") or {}).get("stop")):
                    title, body = _order_text(sym, {**r, "cfg": ai_council._cfg()})
                    rr = send(title, body, kind="order", urgent=True,
                              dedupe=f"order|{sym}|{a1.get('direction')}|{a1.get('plan', {}).get('stop')}")
                    if rr.get("ok"):
                        out["orders"] += 1
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"scan: {type(e).__name__}: {e}")

    # 3) 影子结算：新结的单子汇总推一条（胜率/期望变化）
    try:
        import shadow
        st = _state()
        done = set(st.get("settled") or [])
        book = shadow._read().get("items") or []
        fresh = [x for x in book
                 if x.get("status") in ("won", "lost", "expired", "missed")
                 and x.get("id") not in done]
        if fresh:
            agg = shadow.state(settle_now=False)
            t = agg.get("trade") or {}
            lines = []
            for x in fresh[-5:]:
                r = x.get("net_r")
                lines.append(f"{_sym_name(x.get('sym'))} {x.get('direction')} "
                             f"{x.get('status')} · {('%.2fR' % r) if r is not None else '—'}")
            body = "\n".join(lines) + (
                f"\n累计 {t.get('closed')} 笔 · 胜率 {t.get('win_rate')}% · "
                f"平均 {t.get('avg_r')}R · 进度 {t.get('closed')}/{agg.get('target')}")
            if send(f"影子账本 · 新结算 {len(fresh)} 笔", body, kind="settle",
                    dedupe=f"settle|{fresh[-1].get('id')}").get("ok"):
                out["settle"] += 1
            st["settled"] = list(done | {x.get("id") for x in fresh})[-500:]
            with LOCK:
                _write_state(st)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"settle: {type(e).__name__}: {e}")

    return out


def run_daemon(stop: threading.Event | None = None, interval: int = 120) -> None:
    """常驻线程：每 interval 秒巡检一次（默认 120 秒，实际重活有缓存兜着）。"""
    stop = stop or threading.Event()
    time.sleep(20)                       # 让服务先起完，别抢首屏预热
    while not stop.is_set():
        global _LAST_TICK
        try:
            r = tick()
            _LAST_TICK = time.time()
            if r.get("errors"):
                print(f"[推送] 巡检异常：{r['errors'][:2]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[推送] 巡检失败：{type(e).__name__}: {e}", flush=True)
        stop.wait(interval)
