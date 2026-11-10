"""AI 三角色决策面板（M1 只读版）：AI-1 分析师 → AI-2 风控审核 → AI-3 持仓监控。

边界（别改）：
  * 本模块**绝不调用任何下单/改单/平仓接口**，只产出文字决策与理由，供人工参考；
    页面与接口都标注「仅供参考，不构成投资建议」。
  * 每次运行全量写审计 data/ai_council_audit.jsonl（含喂给模型的上下文摘要与原始输出），
    「AI 为什么这么说」永远可查。
  * LLM 只收到**结构化事实**（已收盘 K 线派生的指标/关键位/跨市场/日历/持仓），
    不发截图——数据比看图准，也不浪费 token。项目内方向预测已被三次证伪
    （见 copilot.py 头注释），所以输出定位是「可解释的观点」而非赚钱承诺。
  * 硬风控在本地用规则算（一票否决），LLM 只做复核——LLM 话术不能替代硬闸门。

数据源：assist.payload()（20 秒计算缓存）+ cross_asset.snapshot()（自带缓存）。
"""

from __future__ import annotations

import datetime as dt
import json
import os

import pxfmt
from pxfmt import pr   # 价格按品种精度格式化（模块内已有同名局部变量 px，不导入 px）
import math
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
CFG_FILE = DATA / "ai_council_config.json"
STATE_FILE = DATA / "ai_council_state.json"          # 黄金（兼容旧文件）


def _state_file(sym: str = "GOLD"):
    """每个品种各存一份三审结果：切品种时不会看到别的品种的结论。"""
    s = "".join(ch for ch in str(sym or "GOLD").replace("#", "")
                if ch.isalnum() or ch == "_").upper() or "GOLD"
    return STATE_FILE if s == "GOLD" else DATA / f"ai_council_state_{s}.json"
AUDIT_FILE = DATA / "ai_council_audit.jsonl"

DEFAULT_CFG = {
    # 决策主周期 = H1（用户交易体系：M15/M30 只做启动确认，H1 是主下单周期，
    # H4/D1 提供 1~3 周的区间框架）。页面仍可切 M15/M30/H1 做短周期试探。
    "tf": "H1",
    "model": "deepseek-chat",
    "min_interval_s": 300,      # 两次 LLM 全量分析的最小间隔（省 token）
    "llm_timeout_s": 60,
    "enabled": True,
    # 三审制硬参数（与页面上「三审制」说明一致，改这里就改了 AI 的准入线）
    "risk_pct": 0.5,            # 单笔最大账户风险：净值 0.5%
    "min_rr": 1.5,              # 最低准入盈亏比 1.5:1
    "require_sentiment": True,  # 开仓是否必须「消息同向」（双确认之一，影子期默认开）
    "event_block_h": 4.0,       # 未来 N 小时有高影响事件即否决
    "atr_period": 14,
    "adx_trend": 25,            # ADX > 25 视为趋势
    "adx_range": 20,            # ADX < 20 视为震荡
    # 包络区间（区间心电图）：上沿=MA(高)、下沿=MA(低)
    "env_period": 20,           # MA 包络的均线周期
    "env_dev_pct": 0,           # 包络带宽：0=按 ATR 自适应（推荐）；>0 才用固定百分比
    "env_width_atr": 5.0,       # 自适应带宽 = 5 × ATR(14)，上下沿各 2.5 ATR
}
_CFG_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()

# ---------------------------------------------------------------- 本地配置/状态


def _cfg() -> dict:
    cfg = dict(DEFAULT_CFG)
    try:
        cfg.update(json.loads(CFG_FILE.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        pass
    # 两个「交易准入门槛」跟 02 智能辅助设置同源：那边改一次，这里立刻跟着变，
    # 不再出现「02 面板 2.0 / 三审 1.5」两套值打架的情况。
    try:
        a = json.loads((DATA / "assist_config.json").read_text(encoding="utf-8"))
        if a.get("min_rr") is not None:
            cfg["min_rr"] = float(a["min_rr"])
        if "require_sentiment" in a:
            cfg["require_sentiment"] = bool(a["require_sentiment"])
    except Exception:  # noqa: BLE001
        pass
    return cfg


def save_cfg(patch: dict) -> dict:
    with _CFG_LOCK:
        cfg = _cfg()
        for k in DEFAULT_CFG:
            if k in patch:
                cfg[k] = patch[k]
        CFG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    return cfg


def _llm_key() -> dict:
    """读 key：优先环境变量（云端容器不用挂文件），再退回项目根 .llm_key.json。"""
    env = (os.environ.get("SD_LLM_KEY") or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if env:
        return {"key": env, "model": os.environ.get("SD_LLM_MODEL") or "deepseek-chat"}
    for p in (ROOT / ".llm_key.json", HERE / ".llm_key.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("key"):
                return d
        except Exception:  # noqa: BLE001
            continue
    return {}


def state(sym: str = "GOLD") -> dict:
    try:
        st = json.loads(_state_file(sym).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        st = {}
    return {"ok": True, "cfg": _cfg(), "llm_ready": bool(_llm_key()),
            "busy": _RUN_LOCK.locked(), "symbol": sym,
            "last": st.get("result"), "last_at": st.get("at")}


def audit_tail(n: int = 30) -> list[dict]:
    if not AUDIT_FILE.is_file():
        return []
    try:
        lines = AUDIT_FILE.read_text(encoding="utf-8").splitlines()
        rows = []
        for ln in lines[-n:]:
            try:
                rows.append(json.loads(ln))
            except Exception:  # noqa: BLE001
                continue
        return list(reversed(rows))            # 新的在前
    except Exception:  # noqa: BLE001
        return []


def _audit(row: dict) -> None:
    row["at"] = row.get("at") or dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # 审计只留最近 400 条，防止无限膨胀
    try:
        lines = AUDIT_FILE.read_text(encoding="utf-8").splitlines()
        if len(lines) > 400:
            AUDIT_FILE.write_text("\n".join(lines[-400:]) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- LLM 调用


def _llm(messages: list[dict], cfg: dict | None = None, max_tokens: int = 1200,
         json_mode: bool = False) -> str:
    """调一次对话补全。

    json_mode=False（默认）走自由文本：三角色现在都是「自然语言分析 +
    <result> 标签机器字段」的混合输出，response_format=json_object 会强迫
    整段输出变成 JSON，自然语言部分就没法写了，所以只在真正要整份 JSON 时开启。
    """
    key = _llm_key()
    if not key:
        raise RuntimeError("未配置 LLM：项目根目录缺 .llm_key.json")
    cfg = cfg or _cfg()
    body = {
        "model": key.get("model") or cfg["model"],
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        key["base"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key["key"]})
    with urllib.request.urlopen(req, timeout=cfg["llm_timeout_s"]) as r:
        out = json.loads(r.read().decode("utf-8"))
    return out["choices"][0]["message"]["content"]


def _first_json_obj(s: str) -> dict | None:
    """从字符串里抓第一个「括号配平」的 JSON 对象（容忍前后有杂字）。

    LLM 偶尔会把 JSON 压成一行、或被 max_tokens 截断在中间：这时 json.loads
    整段会失败，但只要花括号还配平就能救回来。字符串里的 {} 不会被误计。
    """
    i = s.find("{")
    if i < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(s[i:j + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:  # noqa: BLE001
                    return None
    return None


def _split_result(text: str) -> tuple[str, dict]:
    """拆「自然语言分析 + <result> 机器字段」两段输出。

    JSONDecodeError 的根因是让模型自由写大 JSON（多逗号/少引号/被截断）。
    现在机器要读的字段锁死在一个 <result> 小模板里，其余全部自然语言；
    解析失败只丢机器字段、不丢分析文本，页面照样有内容可看。

    三种情况都要兜住：正常闭合、只有 <result> 没闭合（被截断）、
    模型干脆只吐了一个 JSON。标签本身任何情况下都不许漏进正文。
    """
    t = (text or "").strip()
    m = None
    for m in re.finditer(r"<result>(.*?)</result>", t, flags=re.S | re.I):
        pass
    if m:
        analysis = (t[:m.start()] + t[m.end():]).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", m.group(1).strip(),
                     flags=re.S).strip()
        return analysis, (_first_json_obj(raw) or {})
    i = t.lower().find("<result>")
    if i >= 0:
        head = t[:i].strip()
        return head, (_first_json_obj(t[i + 8:]) or {})
    if t.startswith("{"):
        return "", (_first_json_obj(t) or {})
    return t, {}


# ---------------------------------------------------------------- 上下文压缩


def _num(x, nd=2):
    try:
        return round(float(x), nd)
    except Exception:  # noqa: BLE001
        return None


def _adx(rows: list[dict], period: int = 14) -> float | None:
    """Wilder ADX（14）：判断趋势/震荡用。数据不足返回 None。"""
    if not rows or len(rows) < period * 2 + 2:
        return None
    trs, ups, dns = [], [], []
    for i in range(1, len(rows)):
        h, l, pc = rows[i]["h"], rows[i]["l"], rows[i - 1]["c"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        up = h - rows[i - 1]["h"]
        dn = rows[i - 1]["l"] - l
        ups.append(up if (up > dn and up > 0) else 0.0)
        dns.append(dn if (dn > up and dn > 0) else 0.0)

    def _wilder(vals: list[float]) -> list[float]:
        s = sum(vals[:period])
        out = [s]
        for v in vals[period:]:
            s = s - s / period + v
            out.append(s)
        return out

    atr, pdm, mdm = _wilder(trs), _wilder(ups), _wilder(dns)
    dx = []
    for a, p, m in zip(atr, pdm, mdm):
        if a <= 0:
            continue
        pdi, mdi = 100 * p / a, 100 * m / a
        denom = pdi + mdi
        dx.append(100 * abs(pdi - mdi) / denom if denom > 0 else 0.0)
    if len(dx) < period:
        return None
    return round(sum(dx[-period:]) / period, 1)


def _sma_series(vals: list[float], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(vals)
    s = 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= n:
            s -= vals[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def _atr_at(rows: list[dict], period: int = 14) -> float | None:
    """最后一根已收盘 K 线的 ATR（简单均值口径，和包络同源，不引外部库）。"""
    if not rows or len(rows) < period + 2:
        return None
    trs = []
    for i in range(1, len(rows)):
        h, l, pc = rows[i]["h"], rows[i]["l"], rows[i - 1]["c"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period if len(trs) >= period else None


def _envelope(rows: list[dict], tf: str, period: int = 20,
              dev_pct: float = 0.0, lookback: int = 120,
              width_atr: float = 5.0) -> dict | None:
    """MA 包络区间（「区间心电图」）——MT5「Envelopes」的均线中轴 + 自适应带宽：

        中轴 mid = MA(收盘, period)
        带宽 = width_atr × ATR(14)（默认 5 个 ATR，即上下沿各 2.5 ATR）
        上沿 upper = mid + 带宽/2；下沿 lower = mid − 带宽/2
        dev_pct > 0 时改用固定百分比（mid × dev%），一般不用。

    为什么带宽按 ATR 而不是固定百分比：同一套 1.2% 在 D1 只等于 1 个 ATR（太窄，
    永远贴边），在 M15 却等于 14 个 ATR（太宽，永远在中部）——五个周期口径会打架。
    按 ATR 缩放后，每个周期的带子都等价于「自身 5 根 K 线的波动尺度」，
    上沿/下沿才是各周期真正的极值区，D1/H4/H1 的 zone 才能互相印证。
    pos 是价格在带内的分位（0=下沿，100=上沿，越界截到 0/100 并保留 zone）。
    """
    if not rows or len(rows) < period + 6:
        return None
    seg = rows[-lookback:] if len(rows) > lookback else list(rows)
    closes = [float(r["c"]) for r in seg]
    ma_s = _sma_series(closes, period)
    if ma_s[-1] is None:
        return None
    mid = float(ma_s[-1])
    if mid <= 0:
        return None
    atr = _atr_at(seg, 14) or max(1e-6, closes[-1] * 0.001)
    if dev_pct and float(dev_pct) > 0:
        dev = mid * float(dev_pct) / 100.0
    else:
        dev = max(1e-6, float(width_atr) / 2.0 * atr)
    upper, lower = mid + dev, mid - dev
    px = closes[-1]
    width = upper - lower
    raw_pos = (px - lower) / width * 100 if width else 50.0
    pos = round(max(0.0, min(100.0, raw_pos)), 1)
    zone = "upper" if pos >= 75 else ("lower" if pos <= 25 else "mid")
    slope = None
    if len(ma_s) > 6 and ma_s[-6] is not None:
        slope = pr((mid - float(ma_s[-6])) / atr)
    win = seg[-60:] if len(seg) > 60 else seg
    return {"tf": tf, "upper": _num(upper), "mid": _num(mid),
            "lower": _num(lower), "width": _num(width),
            "width_atr": _num(width / atr, 2), "pos": pos, "zone": zone,
            "out_of_band": bool(raw_pos > 100 or raw_pos < 0),
            "px": _num(px), "atr": _num(atr), "slope_atr": slope,
            "dev": _num(dev), "dev_pct": _num(dev / mid * 100, 2),
            "hi60": _num(max(r["h"] for r in win)),
            "lo60": _num(min(r["l"] for r in win)),
            "bars": len(seg), "last_t": seg[-1].get("t")}


def _kickoff(rows: list[dict], tf: str, atr: float | None = None) -> dict | None:
    """小周期启动确认（M30/M15）：最后一根已收盘 K 线是否走出同向启动。

    判定（本地规则，AI 只能引用、不能改写）：
      向上启动 = 收阳 且（实体 ≥0.25 ATR 或 收盘破前高）；或两根内净推进 + 温和实体。
      向下启动 = 镜像。实体不足 0.1 ATR 的十字星不算启动，避免被噪音骗。
    """
    if not rows or len(rows) < 5:
        return None
    r, p = rows[-1], rows[-2]
    atr = atr or _atr_at(rows, 14) or max(1e-6, r["c"] * 0.001)
    body = abs(r["c"] - r["o"])
    d = "up" if r["c"] > r["o"] else ("down" if r["c"] < r["o"] else "flat")
    brk = "up" if r["c"] > p["h"] else ("down" if r["c"] < p["l"] else "")
    mom = (r["c"] - rows[-3]["c"]) / atr
    up_ok = (d == "up" and (body / atr >= 0.25 or brk == "up"))
    dn_ok = (d == "down" and (body / atr >= 0.25 or brk == "down"))
    up_ok = up_ok or (d == "up" and r["c"] > rows[-3]["c"] and body / atr >= 0.1)
    dn_ok = dn_ok or (d == "down" and r["c"] < rows[-3]["c"] and body / atr >= 0.1)
    return {"tf": tf, "bar_t": r.get("t"), "dir": d,
            "body_atr": _num(body / atr, 2), "break": brk,
            "mom3_atr": _num(mom, 2), "up": bool(up_ok), "down": bool(dn_ok),
            "close": _num(r["c"]), "atr": _num(atr)}


def _tf_row(v: dict) -> dict:
    return {"dir": v.get("dir"), "label": v.get("trend"),
            "strength": v.get("strength"), "structure": v.get("structure"),
            "vol": v.get("vol")}


def _mtf(sym: str, snap: dict) -> dict:
    """D1/H4/H1/M30/M15 的方向与强度（包络共振法的逐周期原料）。

    D1/H1/M30 用 assist 同一套 _trend 现算（已收盘 K 线），H4 取主趋势行，
    M15 取 tf_extra——五个周期口径一致，避免「AI 自己编一个方向」。
    """
    import assist
    out: dict = {}
    acfg = assist.load_config()
    for tf in ("D1", "H1", "M30"):
        try:
            rows, _, _ = assist._closed_rows(sym, tf, 200)
            if rows:
                out[tf] = _tf_row(assist._trend(rows, acfg, tf))
        except Exception:  # noqa: BLE001
            pass
    for r in (((snap.get("plan") or {}).get("trend") or {}).get("rows") or []):
        if r.get("tf") == "H4":
            out["H4"] = {"dir": r.get("dir"), "label": r.get("label"),
                         "strength": r.get("strength"),
                         "structure": r.get("ma") or "", "vol": ""}
    tfx = (snap.get("tf_extra") or {}).get("M15")
    if tfx:
        out["M15"] = _tf_row(tfx)
    return {k: out[k] for k in ("D1", "H4", "H1", "M30", "M15") if out.get(k)}


def build_context(sym: str = "GOLD") -> dict:
    """从 assist.payload + cross_asset.snapshot 压出一份紧凑事实（纯数据，无观点）。"""
    import assist
    cfg = _cfg()
    snap = assist.payload(sym, None, False)
    ind = snap.get("indicators") or {}
    risk = snap.get("risk") or {}
    acct = risk.get("account") or {}
    q = risk.get("quote") or {}
    mtf = _mtf(sym, snap)
    adx = None
    try:
        m30, _, _ = assist._closed_rows(sym, "M30", 160)
        adx = _adx(m30, int(cfg.get("atr_period", 14)))
    except Exception:  # noqa: BLE001
        pass
    regime = ("trend" if (adx is not None and adx >= cfg["adx_trend"])
              else "range" if (adx is not None and adx <= cfg["adx_range"])
              else "unknown")

    # ---- 包络区间（每个周期一条）+ 小周期启动确认 ----
    env: dict = {}
    kick: dict = {}
    for tf, lb in (("D1", 120), ("H4", 120), ("H1", 120),
                   ("M30", 96), ("M15", 96)):
        try:
            rows, _, _ = assist._closed_rows(sym, tf, lb + 30)
            e = _envelope(rows, tf, int(cfg.get("env_period", 20)),
                          float(cfg.get("env_dev_pct", 0) or 0), lb,
                          float(cfg.get("env_width_atr", 5.0)))
            if e:
                env[tf] = e
            if tf in ("H1", "M30", "M15"):
                k = _kickoff(rows, tf, e.get("atr") if e else None)
                if k:
                    kick[tf] = k
        except Exception:  # noqa: BLE001
            continue

    # ---- 情绪/新闻（关键词规则，不是语义理解；宁可标中性，不编新闻）----
    senti: dict = {"ok": False, "label": "无数据", "score": 0.0, "top": [],
                   "shock": None}
    try:
        import news as newsfeed
        nb = newsfeed.brief(sym, hours=12)
        senti = {"ok": bool(nb.get("ok")), "label": nb.get("label") or "无数据",
                 "score": nb.get("score") or 0.0,
                 "top": nb.get("top") or [], "shock": nb.get("shock"),
                 "stale": nb.get("stale")}
    except Exception as e:  # noqa: BLE001
        senti["error"] = f"{type(e).__name__}: {e}"

    ctx = {
        "symbol": sym,
        "price": _num(snap.get("price")),
        "quote_age_s": q.get("age_s"),
        "day": snap.get("day") or {},
        "session": (snap.get("session") or {}).get("now") or {},
        "tf": cfg.get("tf"),
        "mtf": mtf,
        "env": env,
        "kickoff": kick,
        "news": senti,
        "adx": {"value": adx, "regime": regime,
                "trend_line": cfg["adx_trend"], "range_line": cfg["adx_range"]},
        "atr": _num((ind.get("atr") or {}).get("value")),
        "atr_main": ((env.get(cfg.get("tf") or "") or {}).get("atr")
                     or _num((ind.get("atr") or {}).get("value"))),
        "rules": {"risk_pct": cfg["risk_pct"], "min_rr": cfg["min_rr"],
                  "event_block_h": cfg["event_block_h"]},
        "trend": {k: v for k, v in (snap.get("plan") or {}).get("trend",
                  {}).items() if k in ("key", "label", "votes", "rows", "parts")},
        "verdict": {k: snap.get("verdict", {}).get(k)
                    for k in ("key", "action", "gate")},
        "indicators": {
            "atr": (ind.get("atr") or {}).get("value"),
            "rsi": {k: (ind.get("rsi") or {}).get(k)
                    for k in ("value", "state")},
            "macd": {k: (ind.get("macd") or {}).get(k)
                     for k in ("state", "hist")},
            "ma": {k: (ind.get("ma") or {}).get(k) for k in ("state", "note")},
        },
        "tf_extra": {tf: {kk: (v or {}).get(kk)
                          for kk in ("trend_key", "rsi", "macd_state")}
                     for tf, v in (snap.get("tf_extra") or {}).items()},
        "levels": [{"name": x.get("name"), "price": _num(x.get("price")),
                    "kind": x.get("kind"), "dist": _num(x.get("dist"), 1),
                    "touches": x.get("touches"), "quality": x.get("quality")}
                   for x in ((snap.get("levels") or {}).get("items") or [])[:9]],
        "cross": [{"name": x.get("name"), "chg_pct": _num(x.get("chg_pct"), 2),
                   "dir": x.get("dir")} for x in (snap.get("cross") or [])],
        "event": {k: (risk.get("event") or {}).get(k)
                  for k in ("name", "mins", "countdown")},
        "account": {k: _num(acct.get(k)) for k in
                    ("equity", "balance", "float_pnl", "margin_level")},
        "day_realized": risk.get("day_realized"),
        "positions": [{"ticket": p.get("ticket"), "side": p.get("side"),
                       "volume": p.get("volume"), "price_open": _num(p.get("price_open")),
                       "price_current": _num(p.get("price_current")),
                       "profit": _num(p.get("profit")),
                       "sl": _num(p.get("sl")), "tp": _num(p.get("tp")),
                       "hold_h": p.get("hold_h"),
                       "comment": p.get("comment")}
                      for p in (risk.get("positions") or {}).get("list") or []],
        "gate_reasons": (snap.get("verdict") or {}).get("gate", {}).get("reasons") or [],
        # 02 智能辅助面板的双方向方案（5 条件计数 + 盈亏比 + 是否达标），直接引用
        "plans": (snap.get("plan") or {}).get("entry") or {},
    }
    return ctx


def _hard_gates(ctx: dict, cfg: dict | None = None) -> list[dict]:
    """一审前就能判定的本地硬风控（一票否决）：规则结果不经过 LLM。"""
    cfg = cfg or _cfg()
    gates = []
    age = ctx.get("quote_age_s")
    if age is not None and age > 120:
        gates.append({"gate": "报价新鲜度", "hit": True,
                      "why": f"报价已 {age:.0f}s 未更新"})
    ev = ctx.get("event") or {}
    mins = ev.get("mins")
    block = float(cfg.get("event_block_h") or 4) * 60
    if mins is not None and 0 <= mins <= block:
        gates.append({"gate": "高影响数据窗口", "hit": True,
                      "why": f"{ev.get('name') or '高影响数据'} 约 {mins:.0f} 分钟后发布"
                             f"（{cfg.get('event_block_h')} 小时内禁止新开仓）"})
    acct = ctx.get("account") or {}
    eq, bal = acct.get("equity"), acct.get("balance")
    if eq is not None and bal is not None and bal > 0 and eq < bal * 0.98:
        gates.append({"gate": "日亏损预算", "hit": True,
                      "why": f"净值 {eq} 已低于余额 98%（回撤超 2%）"})
    for r in ctx.get("gate_reasons") or []:
        gates.append({"gate": "辅助闸门", "hit": True, "why": str(r)})
    return gates


def _plan_gates(ctx: dict, a1: dict | None, cfg: dict | None = None) -> list[dict]:
    """AI-1 出计划后才能判定的硬闸门。

    两类：
      通用风控——盈亏比、单笔风险、结构依据；
      体系闸门——区间位置（中部/方向与区间相反）、双确认（启动 K 线 + 消息面）、
                逆大周期（仅当没有区间极值加持时才算违规）。
    这些必须在本地算——LLM 自报的 rr 不可信，一票否决权也不能交给模型自己行使。
    """
    cfg = cfg or _cfg()
    out: list[dict] = []
    a1 = a1 or {}
    d = a1.get("direction")
    plan = a1.get("plan") or {}
    if d not in ("buy", "sell"):
        return out
    atr = ctx.get("atr_main") or ctx.get("atr") or 0.0
    ez = [_num(x) for x in (plan.get("entry_zone") or []) if _num(x) is not None]
    stop = _num(plan.get("stop"))
    entry = ((ez[0] + ez[1]) / 2) if len(ez) >= 2 else _num(ctx.get("price"))
    tps = []
    for t in (plan.get("targets") or []):
        p = _num(t.get("price")) if isinstance(t, dict) else _num(t)
        if p is not None:
            tps.append(p)

    if entry and stop and tps:
        risk = abs(entry - stop)
        rr = pr(abs(tps[0] - entry) / risk) if risk else 0.0
        if rr < float(cfg["min_rr"]):
            out.append({"gate": "盈亏比", "hit": True,
                        "why": f"目标1 盈亏比 {rr}:1，低于准入 {cfg['min_rr']}:1"})
    else:
        out.append({"gate": "盈亏比", "hit": True,
                    "why": "计划缺入场区/止损/目标，无法核算盈亏比"})

    eq = (ctx.get("account") or {}).get("equity")
    if entry and stop and eq:
        sl_dist = abs(entry - stop)
        min_risk = sl_dist * 100 * 0.01        # 黄金 0.01 手 = 1 盎司 → 每 1 美金 1 USD
        budget = float(eq) * float(cfg["risk_pct"]) / 100.0
        if min_risk > budget:
            out.append({"gate": "单笔风险", "hit": True,
                        "why": f"止损 {sl_dist:.{pxfmt.px()}f} 美金，最小手数风险 {min_risk:.1f} USD"
                               f" 已超 {cfg['risk_pct']}% 预算 {budget:.1f} USD"})

    env_h1 = (ctx.get("env") or {}).get("H1") or {}
    if entry and stop and atr:
        lv = [x for x in (ctx.get("levels") or []) if _num(x.get("price")) is not None]
        edges = [env_h1.get(k) for k in ("upper", "lower", "mid")]

        def _near(px: float) -> bool:
            if any(abs(_num(x.get("price")) - px) <= 0.8 * atr for x in lv):
                return True
            return any(abs(float(e) - px) <= 0.8 * atr for e in edges if e)

        if not _near(entry) or not _near(stop):
            out.append({"gate": "结构依据", "hit": True,
                        "why": "入场位或止损离关键位/包络边界超过 0.8×ATR，属随手价"})

    # ---- 体系闸门：区间位置 + 双确认 ----
    zone = env_h1.get("zone")
    pos = env_h1.get("pos")
    if zone:
        if zone == "mid":
            out.append({"gate": "区间位置", "hit": True,
                        "why": f"价格在 H1 包络中部（{pos}%），无区间极值，"
                               f"包络共振法只做上下沿，中部一律观望"})
        elif (d == "buy" and zone == "upper") or (d == "sell" and zone == "lower"):
            out.append({"gate": "区间位置", "hit": True,
                        "why": f"方向与区间相反：做{'多' if d == 'buy' else '空'}却在"
                               f"{'上沿' if zone == 'upper' else '下沿'}区（{pos}%）"})
    ks = ctx.get("kickoff") or {}
    if ks:
        key = "up" if d == "buy" else "down"
        if not any(bool((v or {}).get(key)) for v in ks.values()):
            out.append({"gate": "启动确认", "hit": True,
                        "why": "M30/M15 均未走出同向启动 K 线（双确认缺一项）"})
    nw = ctx.get("news") or {}
    if bool(cfg.get("require_sentiment", True)):
        if nw and not nw.get("ok"):
            out.append({"gate": "情绪校验", "hit": True,
                        "why": "新闻/情绪源不可用，消息面这一半无法确认"})
        elif nw:
            label = str(nw.get("label") or "")
            bias = ("bull" if ("多" in label or "利多" in label)
                    else "bear" if ("空" in label or "利空" in label) else "neutral")
            want = "bull" if d == "buy" else "bear"
            if bias != want:
                out.append({"gate": "情绪校验", "hit": True,
                            "why": f"消息面={label} 与{'做多' if d == 'buy' else '做空'}"
                                   f"方向不匹配（双确认缺消息这一半）"})

    # ---- 大周期一致：区间法允许逆势短单，但必须有区间极值加持 ----
    mtf = ctx.get("mtf") or {}
    d1 = (mtf.get("D1") or {}).get("dir")
    h4 = (mtf.get("H4") or {}).get("dir")
    if d == "buy" and d1 == "bear" and h4 == "bear" and zone != "lower":
        out.append({"gate": "趋势一致", "hit": True,
                    "why": "做多但 D1 与 H4 同时向下，且价格不在包络下沿（无区间极值加持）"})
    if d == "sell" and d1 == "bull" and h4 == "bull" and zone != "upper":
        out.append({"gate": "趋势一致", "hit": True,
                    "why": "做空但 D1 与 H4 同时向上，且价格不在包络上沿（无区间极值加持）"})
    return out


def _normal_a1(a1: dict | None) -> dict | None:
    """三审制输出是嵌套结构；补一份旧字段（entry_zone/stop/targets）给工单与页面用。"""
    if not isinstance(a1, dict):
        return a1
    plan = a1.get("plan") or {}
    if plan:
        a1.setdefault("entry_zone", plan.get("entry_zone"))
        a1.setdefault("stop", plan.get("stop"))
        tps = [t.get("price") if isinstance(t, dict) else t
               for t in (plan.get("targets") or [])]
        a1.setdefault("targets", [x for x in tps if x])
        a1.setdefault("horizon", plan.get("horizon"))
    if not a1.get("reasons") and a1.get("summary"):
        a1["reasons"] = [a1["summary"]]
    return a1


# ---------------------------------------------------------------- 本地硬事实层
# 分工原则：本地负责「算得准」，LLM 负责「想得深」。页面要展示的所有硬指标
# （区间位置、候选区、情绪匹配、启动、多周期、观望条款、持仓通道位置/健康度）
# 全部在这里算好，AI 只引用、只解释、只推演，绝不重新发明数值。


_WAIT_TEXT = {
    "①": "① 上沿出现偏多消息（或下沿出现偏空消息），多空信号混杂、方向模糊",
    "②": "② 已到候选区，但 M30/M15 还没有同向启动 K 线",
    "③": "③ 消息面没有匹配方向的驱动，或小周期没有同向启动（双确认缺项）",
    "④": "④ 大小周期严重冲突，且价格不在区间极值（无加持）",
    "⑤": "⑤ 价格在区间中部（包络 25~75），无极值不交易",
}


def _news_line(x: dict, max_len: int = 34) -> str:
    """把 news.brief 的 top 条目洗成一行人话，禁止 raw dict 漏进页面/LLM。

    {'title':..., 'impact':1, 'dir':'bear', 'topic':..., 'hm':'18:37'}
    → 「标题…」(18:37·利空)
    """
    if not isinstance(x, dict):
        return str(x)[:max_len]
    t = (x.get("title") or "").strip()
    if len(t) > max_len:
        t = t[:max_len - 1] + "…"
    d = {"bull": "利多", "bear": "利空"}.get(x.get("dir"))
    imp = x.get("impact")
    tag = x.get("hm") or ""
    if d:
        tag = (tag + "·" if tag else "") + d
    elif imp is not None and imp >= 2:
        tag = (tag + "·" if tag else "") + "高影响"
    return f"「{t}」" + (f"({tag})" if tag else "")


def _market_state(ctx: dict) -> dict:
    """本地判定行情状态（趋势 / 震荡 / 极为无序），不交给 LLM 猜。

    口径：ADX 定趋势强度，当日振幅/ATR 定无序程度。
    无序 = 波动放大但趋势读数低（来回抽），这种行情里任何体系都该降频。
    """
    adx = _num((ctx.get("adx") or {}).get("value"))
    atr = _num(ctx.get("atr_main") or ctx.get("atr")) or 0.0
    day = ctx.get("day") or {}
    hi, lo = _num(day.get("high")), _num(day.get("low"))
    rng_atr = ((hi - lo) / atr) if (hi is not None and lo is not None and atr > 0) else None
    trend_line = _num((ctx.get("adx") or {}).get("trend_line")) or 25.0
    range_line = _num((ctx.get("adx") or {}).get("range_line")) or 20.0
    if adx is None:
        return {"state": "未知", "adx": None, "range_atr": rng_atr,
                "why": "ADX 不可用，不判状态"}
    if rng_atr is not None and rng_atr >= 3.5 and adx < range_line:
        st, why = "极度无序", f"当日振幅 {rng_atr:.1f}×ATR 但 ADX 仅 {adx:.0f}，来回抽"
    elif adx >= trend_line:
        st, why = "趋势", f"ADX {adx:.0f} ≥ {trend_line:.0f}，方向性强"
    elif adx <= range_line:
        st, why = "震荡", f"ADX {adx:.0f} ≤ {range_line:.0f}，无方向性"
    else:
        st, why = "过渡", f"ADX {adx:.0f} 处于 {range_line:.0f}~{trend_line:.0f}，趋势未确立"
    return {"state": st, "adx": adx, "range_atr": rng_atr, "why": why}


def _dual_plans(ctx: dict) -> dict:
    """02 智能辅助算好的双方向方案（5 条件计数/盈亏比/是否达标），瘦身后给 AI 引用。"""
    src = ctx.get("plans") or {}
    out = {}
    for k in ("long", "short"):
        s = src.get(k) or {}
        if not s:
            continue
        out[k] = {"label": s.get("label"), "entry": s.get("entry"),
                  "stop": s.get("stop"), "rr": s.get("rr"),
                  "n_ok": s.get("n_ok"), "n_total": 5,
                  "qualify": s.get("qualify"),
                  "block_reason": s.get("block_reason"),
                  "conds": [{"name": c.get("name"), "s": c.get("s"),
                             "text": c.get("text")}
                            for c in (s.get("conds") or [])]}
    if src:
        out["verdict"] = {"label": src.get("label"), "side": src.get("side"),
                          "reason": src.get("reason")}
    return out


def _facts(ctx: dict, cfg: dict | None = None) -> dict:
    """页面要展示的硬事实（AI-1 卡片的客观部分），全部本地计算。"""
    cfg = cfg or _cfg()
    need_sent = bool(cfg.get("require_sentiment", True))
    env = ctx.get("env") or {}
    e = env.get(ctx.get("tf") or "") or env.get("H1") or {}
    zone, pos = e.get("zone"), e.get("pos")
    kick = ctx.get("kickoff") or {}
    up = any(bool((v or {}).get("up")) for v in kick.values())
    dn = any(bool((v or {}).get("down")) for v in kick.values())
    nw = ctx.get("news") or {}
    label = nw.get("label") or "无数据"
    bias = ("bull" if ("多" in label or "利多" in label)
            else "bear" if ("空" in label or "利空" in label) else "neutral")
    want = "bull" if zone == "lower" else ("bear" if zone == "upper" else None)
    matched = bool(want) and bias == want
    ko_dir = "up" if zone == "lower" else ("down" if zone == "upper" else None)
    ko_ok = bool(want) and ((ko_dir == "up" and up) or (ko_dir == "down" and dn))
    mtf = ctx.get("mtf") or {}
    dirs = [((mtf.get(t) or {}).get("dir")) for t in ("D1", "H4", "H1")]
    aligned = all(dirs) and len(set(dirs)) == 1
    # 强制观望条款（本地判，AI 解释时必须引用同一条款编号）
    rules: list[str] = []
    if zone == "mid":
        rules.append(_WAIT_TEXT["⑤"])
    if zone == "upper" and not dn:
        rules.append(_WAIT_TEXT["②"])
    if zone == "lower" and not up:
        rules.append(_WAIT_TEXT["②"])
    if want and bias != want and need_sent:
        rules.append(f"③ 消息面没有匹配方向的驱动（消息面={label}）")
    if want and not ko_ok:
        rules.append("③ 小周期没有同向启动 K 线（双确认缺项）")
    if zone == "upper" and bias == "bull":
        rules.append(_WAIT_TEXT["①"])
    if zone == "lower" and bias == "bear":
        rules.append(_WAIT_TEXT["①"])
    if not aligned and not (zone in ("upper", "lower")):
        rules.append(_WAIT_TEXT["④"])
    seen: set[str] = set()
    rules = [r for r in rules if not (r in seen or seen.add(r))]
    px = ctx.get("price")
    res_lv, sup_lv = [], []
    for x in ctx.get("levels") or []:
        p = _num(x.get("price"))
        if p is None:
            continue
        (res_lv if (px is None or p > px) else sup_lv).append(p)
    return {
        "zone": {"tf": ctx.get("tf") or "H1", "pos": pos, "zone": zone,
                 "upper": e.get("upper"), "mid": e.get("mid"),
                 "lower": e.get("lower"),
                 "why": (f"MA20 ± 2.5×ATR 自适应带宽（{e.get('width_atr')} ATR）"
                         if e else None)},
        "candidates": {"long": zone == "lower", "short": zone == "upper",
                       "why": (f"价格在 {ctx.get('tf') or 'H1'} 包络下沿 → 做多候选区"
                               if zone == "lower"
                               else f"价格在 {ctx.get('tf') or 'H1'} 包络上沿 → 做空候选区"
                               if zone == "upper"
                               else "价格在区间中部，无候选区")},
        "sentiment": {"bias": bias, "matched": matched, "label": label,
                      "required": need_sent,
                      "ok": matched or not need_sent,
                      "why": "; ".join(_news_line(x) for x in (nw.get("top") or [])[:2])
                      or ("新闻源不可用" if not nw.get("ok") else "近期无同向标题")},
        "kickoff": {"confirmed": ko_ok,
                    "tf": "、".join(k for k, v in kick.items()
                                    if (v or {}).get(ko_dir)) if ko_dir else None,
                    "why": (f"M30/M15 向{'上' if ko_dir == 'up' else '下'}启动"
                            f"{'' if ko_ok else '未'}确认"
                            if ko_dir else "中部无方向，不判启动")},
        "mtf": {t: {"dir": (mtf.get(t) or {}).get("dir"),
                    "strength": (mtf.get(t) or {}).get("strength"),
                    "why": (mtf.get(t) or {}).get("structure") or
                           (mtf.get(t) or {}).get("label") or ""}
                for t in ("D1", "H4", "H1", "M30", "M15") if mtf.get(t)},
        "resonance": {"aligned": aligned if all(dirs) else None,
                      "regime": (ctx.get("adx") or {}).get("regime"),
                      "why": ("D1/H4/H1 同向" if aligned and all(dirs)
                              else "大中小周期方向不一致" if all(dirs) else None)},
        "wait_rules": rules,
        "require_sentiment": need_sent,
        "market_state": _market_state(ctx),
        "dual_plans": _dual_plans(ctx),
        "key_levels": {"resistance": sorted(set(res_lv))[:5],
                       "support": sorted(set(sup_lv), reverse=True)[:5]},
    }


def _structured_input(ctx: dict, cfg: dict, facts: dict | None = None) -> dict:
    """喂给 LLM 的结构化行情数据（用户接口规范格式；全部本地算好的客观事实）。"""
    env = ctx.get("env") or {}
    e = env.get(ctx.get("tf") or "") or env.get("H1") or {}
    _cn = {"bull": "多", "bear": "空", "range": "震荡", "none": "震荡"}

    def _dir(d):
        return _cn.get(d, d or "?")

    mp = {}
    kick = ctx.get("kickoff") or {}
    for tf, v in (ctx.get("mtf") or {}).items():
        row = {"direction": _dir(v.get("dir")),
               "strength": v.get("strength"),
               "structure": v.get("structure") or v.get("label") or ""}
        if tf in kick:
            k = kick[tf]
            row["start_signal"] = ("向上" if k.get("up") else
                                   "向下" if k.get("down") else "无")
        mp[tf] = row
    news = ctx.get("news") or {}
    ev = ctx.get("event") or {}
    ev_mins = ev.get("mins")
    posi = ctx.get("positions") or []
    p0 = posi[0] if posi else None
    px = ctx.get("price")
    res_lv, sup_lv = [], []
    for x in ctx.get("levels") or []:
        p = _num(x.get("price"))
        if p is not None:
            (res_lv if (px is None or p > px) else sup_lv).append(p)
    ob = e.get("out_of_band")
    out = {
        "symbol": ctx.get("symbol"),
        "current_price": px,
        "main_period": ctx.get("tf") or "H1",
        "envelope": {
            "mid_line": e.get("mid"), "lower_band": e.get("lower"),
            "upper_band": e.get("upper"), "position_pct": e.get("pos"),
            "status": ("out_of_upper" if ob and (e.get("pos") or 0) >= 100
                       else "out_of_lower" if ob else
                       {"upper": "at_upper", "lower": "at_lower"}.get(e.get("zone"),
                                                                     "mid_band")),
            "band_width_atr": e.get("width_atr"),
            "all_periods": {tf: {"pos": v.get("pos"), "zone": v.get("zone"),
                                 "upper": v.get("upper"), "lower": v.get("lower")}
                            for tf, v in env.items()},
        },
        "multi_period": mp,
        "news": {
            "sentiment": news.get("label"),
            "impact_level": "高" if news.get("shock") else "低/中",
            "shock_event": bool(news.get("shock")),
            "next_24h_events": (
                f"{ev.get('name')} 约 {ev_mins:.0f} 分钟后发布"
                if ev_mins is not None and 0 <= ev_mins <= 1440 else "无高影响事件"),
            "top_headlines": [_news_line(x, 50) for x in (news.get("top") or [])[:4]],
        },
        "position": ({"direction": _dir({"buy": "bull", "sell": "bear"}.get(
                          p0.get("side"), p0.get("side"))),
                      "open_price": p0.get("price_open"),
                      "current_price": p0.get("price_current"),
                      "floating_pnl": p0.get("profit"),
                      "hold_hours": p0.get("hold_h"),
                      "volume": p0.get("volume")} if p0 else None),
        "positions": posi,
        "key_levels": {"resistance": sorted(set(res_lv))[:5],
                       "support": sorted(set(sup_lv), reverse=True)[:5]},
        "atr": ctx.get("atr_main") or ctx.get("atr"),
        "adx": {"value": (ctx.get("adx") or {}).get("value"),
                "regime": {"trend": "趋势", "range": "震荡",
                           "unknown": "未知"}.get((ctx.get("adx") or {}).get("regime"))},
        "session": ctx.get("session"),
        "account": ctx.get("account"),
        "rules": ctx.get("rules"),
    }
    if facts:
        z = (facts.get("zone") or {}).get("zone")
        out["local_verdict"] = {
            "market_state": (facts.get("market_state") or {}).get("state"),
            "zone": z,
            "candidate": ("做多候选" if z == "lower" else
                          "做空候选" if z == "upper" else "无候选（中部）"),
            "kickoff_confirmed": (facts.get("kickoff") or {}).get("confirmed"),
            "sentiment_matched": (facts.get("sentiment") or {}).get("matched"),
            "require_sentiment": facts.get("require_sentiment", True),
            "wait_rules": facts.get("wait_rules") or [],
            "plans": facts.get("dual_plans") or {},
            "market_state_detail": facts.get("market_state") or {},
            "note": "本地规则结论，只能引用，不得推翻，不得改成别的方向",
        }
    return {k: v for k, v in out.items() if v is not None}


def _plan_rr(ctx: dict, a1: dict | None) -> float | None:
    """本地重算目标1 盈亏比（LLM 自报的 rr 不采信）。"""
    plan = (a1 or {}).get("plan") or {}
    ez = [_num(x) for x in (plan.get("entry_zone") or []) if _num(x) is not None]
    stop = _num(plan.get("stop"))
    entry = ((ez[0] + ez[1]) / 2) if len(ez) >= 2 else _num(ctx.get("price"))
    tps = []
    for t in (plan.get("targets") or []):
        p = _num(t.get("price")) if isinstance(t, dict) else _num(t)
        if p is not None:
            tps.append(p)
    if entry and stop and tps:
        risk = abs(entry - stop)
        if risk:
            return pr(abs(tps[0] - entry) / risk)
    return None


def _local_checks(ctx: dict, gates: list[dict], cfg: dict,
                  a1: dict | None) -> list[dict]:
    """AI-2 的闸门勾选表：全部本地算（✓/✗ + 理由），LLM 只做风险叙述。"""
    hit = {g.get("gate"): g.get("why") for g in gates or [] if g.get("hit")}
    rr = _plan_rr(ctx, a1)

    def _add(name, ok_when_no_gate=True, why_ok=None):
        out.append({"gate": name, "pass": name not in hit,
                    "why": hit.get(name) or why_ok or "未触发"})

    out: list[dict] = []
    _add("盈亏比", why_ok=(f"目标1 盈亏比 {rr}:1 ≥ {cfg['min_rr']}:1" if rr
                           else "AI-1 未给出完整计划，跳过核算"))
    _add("单笔风险", why_ok=f"止损距离风险 ≤ 净值 {cfg['risk_pct']}% 预算")
    _add("结构依据", why_ok="入场/止损贴包络边界或关键位（0.8ATR 内）")
    _add("区间位置", why_ok="价格在候选区，方向与区间位置一致")
    _add("启动确认", why_ok="M30/M15 已出现同向启动 K 线")
    _add("情绪校验", why_ok="消息面存在匹配方向的驱动")
    _add("趋势一致", why_ok="方向与 D1/H4 一致，或有区间极值加持")
    _add("事件窗口", why_ok=f"未来 {cfg['event_block_h']} 小时无高影响事件")
    _add("报价新鲜度", why_ok="报价正常更新")
    _add("日亏损预算", why_ok="当日回撤未超预算")
    return out


def _pos_facts(ctx: dict, cfg: dict) -> list[dict]:
    """AI-3 的持仓硬事实：通道位置、健康度三项打分、触发价、本地建议，全本地算。"""
    env = ctx.get("env") or {}
    e = env.get(ctx.get("tf") or "") or env.get("H1") or {}
    atr = e.get("atr") or ctx.get("atr") or 1.0
    upper, lower = e.get("upper"), e.get("lower")
    pos, zone = e.get("pos"), e.get("zone")
    px = ctx.get("price")
    out: list[dict] = []
    for p in ctx.get("positions") or []:
        side = (p.get("side") or p.get("action") or p.get("type") or "buy").lower()
        long = side in ("buy", "bull", "long")
        cur = p.get("price_current") or px
        vol = p.get("volume") or 0
        pnl_oz = (p.get("profit") or 0.0) / (vol * 100) if vol else 0.0  # 每盎司盈亏
        # ① 通道方向（40 分）：破位才重罚；到止盈区是「该兑现」，不罚健康度
        if pos is None:
            s1 = 20.0
            s1_why = "通道位置未知"
        elif long:
            s1 = 40.0 if pos > 25 else 0.0
            s1_why = f"通道 {pos}%（{zone}）" + ("，已顶到止盈区" if pos >= 75
                                                else "，已跌回破位区" if pos <= 25 else "")
        else:
            s1 = 40.0 if pos < 75 else 0.0
            s1_why = f"通道 {pos}%（{zone}）" + ("，已砸到止盈区" if pos <= 25
                                                else "，已冲到破位区" if pos >= 75 else "")
        # ② 浮盈安全垫（30 分）：每盎司浮盈 ≥ 1×ATR 满分
        s2 = max(0.0, min(1.0, pnl_oz / max(atr, 1e-6))) * 30.0
        s2_why = f"浮盈垫 {pnl_oz / max(atr, 1e-6):.1f} ATR"
        # ③ 距破位价（30 分）：>1 ATR 满分
        if long:
            dist = (cur - lower) / max(atr, 1e-6) if (cur is not None and lower) else None
            brk = lower
        else:
            dist = (upper - cur) / max(atr, 1e-6) if (cur is not None and upper) else None
            brk = upper
        s3 = max(0.0, min(1.0, dist)) * 30.0 if dist is not None else 15.0
        s3_why = (f"距破位价 {_num(brk)} 约 {dist:.1f} ATR"
                  if dist is not None else "破位价未知")
        health = round(s1 + s2 + s3)
        # 本地默认动作 + 移动止损建议（LLM 可在其框架内细化，不能越界）
        if long:
            act = "reduce" if (pos or 0) >= 75 else ("close" if (pos or 100) <= 25
                                                     else "hold")
            tp_trig, sl_trig = upper, lower
        else:
            act = "reduce" if (pos or 0) <= 25 else ("close" if (pos or 0) >= 75
                                                     else "hold")
            tp_trig, sl_trig = lower, upper
        if pnl_oz <= -1.0 * max(atr, 1e-6):
            act = "close"
        # ④ 1R 铁律（本地硬规则，优先级高于通道规则）：
        #    浮盈到 1R → 止损必须移保本；浮亏到 -1R → 直接离场，不给扛单空间。
        entry = p.get("price_open")
        sl = _num(p.get("sl"))
        r_now = None
        if entry and sl:
            risk_oz = abs(entry - sl)
            if risk_oz > 1e-9:
                r_now = ((cur - entry) if long else (entry - cur)) / risk_oz
        trail = {"move_to_be": False, "new_stop": None, "why": ""}
        if pnl_oz >= 1.0 * max(atr, 1e-6):
            trail = {"move_to_be": True, "new_stop": _num(entry),
                     "why": f"浮盈已超 1×ATR（{pnl_oz:.1f}/oz），止损保本"}
        elif long and (pos or 0) >= 75 and upper:
            trail = {"move_to_be": True, "new_stop": _num(upper - 0.5 * atr),
                     "why": "已顶到上沿，锁定通道回吐"}
        elif not long and (pos or 100) <= 25 and lower:
            trail = {"move_to_be": True, "new_stop": _num(lower + 0.5 * atr),
                     "why": "已砸到下沿，锁定通道回吐"}
        if r_now is not None:
            if r_now >= 1.0:
                trail = {"move_to_be": True, "new_stop": _num(entry),
                         "why": f"已达 {r_now:.2f}R，止损必须移保本（1R 铁律）"}
            if r_now <= -1.0:
                act = "close"
        grade = "优" if health >= 75 else ("良" if health >= 50 else "危险")
        out.append({"ticket": p.get("ticket"), "side": side, "volume": vol,
                    "entry": _num(entry), "price": _num(cur),
                    "profit": _num(p.get("profit")), "hold_h": p.get("hold_h"),
                    "pos": pos, "zone": zone, "health": health,
                    "grade": grade, "r_now": (pr(r_now) if r_now is not None
                                              else None),
                    "health_why": [s1_why, s2_why, s3_why],
                    "action": act,
                    "close_pct": (50 if act == "reduce" else None),
                    "action_why": (f"H1 通道 {pos}%，{'止盈区' if (long and (pos or 0) >= 75) or (not long and (pos or 0) <= 25) else '破位区' if s1 <= 10 else '通道内运行'}"
                                   if pos is not None else ""),
                    "trail": trail,
                    "tp_trigger": _num(tp_trig), "sl_trigger": _num(sl_trig),
                    "break_level": _num(brk),
                    "risks": [f"破位价 {_num(brk)}（距现价 {dist:.1f} ATR）"
                              if dist is not None else "通道边界未知"]})
    return out


# ---------------------------------------------------------------- 本地裁判

def _plan_lots(ctx: dict, plan: dict | None, cfg: dict) -> dict:
    """按「净值 × 风控% ÷ 止损距离」反算手数（本地算，LLM 只允许引用）。

    XAUUSD：1 手 = 100 盎司，止损 1 美金/盎司 = 每手亏 100 美金。
    反算结果小于最小手数（0.01）时返回 0，而不是放大风险凑一手。
    """
    plan = plan or {}
    ez = [_num(x) for x in (plan.get("entry_zone") or []) if _num(x) is not None]
    stop = _num(plan.get("stop"))
    entry = ((ez[0] + ez[1]) / 2) if len(ez) >= 2 else _num(ctx.get("price"))
    equity = _num((ctx.get("account") or {}).get("equity")) or 0.0
    if entry is None or stop is None or equity <= 0:
        return {"lots": 0.0, "risk_usd": None, "per_lot": None, "dist": None}
    dist = abs(entry - stop)
    try:
        import markets
        ctr = markets.contract(pxfmt.sym())
    except Exception:  # noqa: BLE001
        ctr = 100.0
    per_lot = dist * ctr
    risk_usd = equity * cfg["risk_pct"] / 100.0
    lots = math.floor((risk_usd / per_lot) / 0.01) * 0.01 if per_lot > 0 else 0.0
    return {"lots": max(0.0, pr(lots)), "risk_usd": pr(risk_usd),
            "per_lot": pr(per_lot), "dist": pr(dist)}


def _enforce_a1(a1: dict | None, facts: dict) -> dict | None:
    """本地裁判：AI 只能提出方向，能不能成立由本地规则决定。

    LLM 说做多，但只要「区间位置 / 双确认 / 计划完整性」任何一条不满足，
    本地直接把方向改成观望并记下原方向（ai_direction + enforced），
    不做「建议性提醒」——这就是纪律执行者和分析师的区别。
    """
    if not isinstance(a1, dict):
        return a1
    d = a1.get("direction")
    if d not in ("buy", "sell"):
        return a1
    z = (facts.get("zone") or {}).get("zone")
    blockers: list[str] = []
    if z == "mid":
        blockers.append(_WAIT_TEXT["⑤"])
    elif d == "buy" and z != "lower":
        blockers.append("方向与区间相反：做多但价格不在主周期包络下沿")
    elif d == "sell" and z != "upper":
        blockers.append("方向与区间相反：做空但价格不在主周期包络上沿")
    if not (facts.get("kickoff") or {}).get("confirmed"):
        blockers.append(_WAIT_TEXT["②"])
    sn = facts.get("sentiment") or {}
    if bool(facts.get("require_sentiment", True)) and not sn.get("matched"):
        blockers.append(f"③ 消息面没有与方向匹配的驱动"
                        f"（当前 {sn.get('label') or '无数据'}）")
    mtf = facts.get("mtf") or {}
    d1 = (mtf.get("D1") or {}).get("dir")
    h4 = (mtf.get("H4") or {}).get("dir")
    if d == "buy" and d1 == "bear" and h4 == "bear" and z != "lower":
        blockers.append("逆 D1/H4 下行且无区间极值加持")
    if d == "sell" and d1 == "bull" and h4 == "bull" and z != "upper":
        blockers.append("逆 D1/H4 上行且无区间极值加持")
    # 02 面板的双方向方案：本地判定不达标的方向，AI 连提都不许提
    dp = (facts.get("dual_plans") or {}).get("long" if d == "buy" else "short") or {}
    if dp and dp.get("qualify") is False:
        blockers.append(
            f"02 面板{('做多' if d == 'buy' else '做空')}方案本地未达标"
            f"（{dp.get('n_ok')}/5 条件，盈亏比 {dp.get('rr')}）"
            + (f"：{dp.get('block_reason')}" if dp.get("block_reason") else ""))
    plan = a1.get("plan") or {}
    if not plan or not _num(plan.get("stop")) or not (plan.get("targets") or []):
        blockers.append("计划字段不完整（缺止损或目标），按无计划处理")
    if not blockers:
        return a1
    a1["ai_direction"] = d                       # 留痕：AI 原本想做哪边
    a1["ai_plan"] = a1.get("plan")               # 留痕：被拦的原始计划（影子账本要用）
    a1["direction"] = "wait"
    a1["plan"] = None
    a1["enforced"] = {"blocked": blockers,
                      "why": "本地规则拦截：AI 方向不成立，强制观望"}
    a1["wait_rules"] = facts.get("wait_rules") or blockers
    try:
        a1["confidence"] = min(int(a1.get("confidence") or 0), 39)
    except (TypeError, ValueError):
        a1["confidence"] = 39
    return a1


# ---------------------------------------------------------------- 三角色


_SYS_G = (
    "你是交易系统的【规则校验与状态解释器】，不是分析师，也没有预测权。\n"
    "【铁律】\n"
    "1. 输入 JSON 里的数值全部由本地代码精确计算：包络上下沿、pos、"
    "kickoff_confirmed、sentiment_matched、wait_rules、health、r_now——"
    "只能引用原文，绝不重算、绝不改写、绝不发明新点位；\n"
    "2. 禁止预测方向，禁止出现「接下来会涨/会跌」这类判断。"
    "你的产出只有三样：当前是什么状态、规则是否满足、按规则该做什么；\n"
    "3. 规则不满足就必须说观望，不许用「若突破则可能」绕过；\n"
    "4. 禁用模糊词：可能、大概、或许、企稳、留意、注意、择机。"
    "每个结论后面必须跟输入里的具体数字或布尔值；\n"
    "5. 不写客套话、不写免责声明、不复述任务要求。\n"
    "【体系】多周期区间包络共振法：主周期取输入 main_period（本地已按该周期算好包络），"
    "M30/M15 做启动确认，D1/H4 定框架；"
    "包络下沿=做多候选、上沿=做空候选、中部不交易；"
    "开仓双确认 = 消息同向 + 小周期同向启动（消息这一半可在设置里关闭）；\n"
    "【格式】机器可读部分必须放在 <result></result> 里，且只放 JSON。"
)

_SYS_A1 = (
    "【角色】主周期「市场状态识别器 + 信号校验器」（主周期见输入 main_period）。不给交易建议，"
    "只做两件事：把本地数据翻译成状态描述，按规则校验信号是否成立。\n"
    "【判定顺序（自上而下，命中即停）】\n"
    "1. local_verdict.zone = mid → 观望；\n"
    "2. 想做多但 zone≠lower，或想做空但 zone≠upper → 观望；\n"
    "3. local_verdict.kickoff_confirmed = false → 观望；\n"
    "4. local_verdict.require_sentiment = true 且 sentiment_matched = false → 观望；"
    "require_sentiment = false 时消息面只做参考，不构成拦截（小周期启动仍必须满足）；\n"
    "5. 以上全部通过，才允许 direction_bias=做多/做空，并按模板填 plan。\n"
    "没有「先小仓位试试」「轻仓试单」这种中间态：要么满足，要么观望。\n"
    "【双方向方案（02 面板本地算好，直接引用）】local_verdict.plans.long / short 给出"
    "入场 entry、止损 stop、盈亏比 rr、满足条件数 n_ok/5、是否达标 qualify、拦截原因 block_reason。"
    "某方向 qualify=false 时，必须明确说「该方向被系统拦截」并引用 rr 与 block_reason；"
    "不许自己判断条件是否满足，更不许把被拦的方向改写成可做。\n"
    "【plan 数值来源（不许自己算）】做多：entry_zone=[lower_band, lower_band+0.5×atr]，"
    "stop=lower_band−0.5×atr；做空：entry_zone=[upper_band−0.5×atr, upper_band]，"
    "stop=upper_band+0.5×atr；targets 依次为 mid_line、对侧通道边界、对侧边界外 0.3×atr，"
    "比例 30/40/30。atr 用输入里的 atr，边界用 envelope 里的值。\n"
    "【输出】只输出一行 <result></result>，标签里放 JSON，标签外不写任何字符：\n"
    "<result>{\"market_state\":\"趋势|震荡|极度无序（未知则原样引用）\","
    "\"direction_bias\":\"做多|做空|观望\",\"confidence\":0到100,"
    "\"analysis_summary\":\"≤50字，必须含输入里的具体数字\","
    "\"evidence\":[\"≤3条，每条≤30字，分别引用 zone/pos、kickoff、新闻标签\"],"
    "\"plan\":null,\"wait_rules\":[\"②\"],\"invalid\":\"作废的具体价格或形态\"}</result>\n"
    "做单时 plan 结构：{\"entry_zone\":[价,价],\"stop\":价,"
    "\"targets\":[{\"price\":价,\"close_pct\":30},{\"price\":价,\"close_pct\":40},"
    "{\"price\":价,\"close_pct\":30}],\"trigger_rule\":\"含价位的入场触发条件\","
    "\"horizon\":\"intraday\"}。\n"
    "观望时：plan=null、confidence≤40、wait_rules 至少一条命中编号、"
    "analysis_summary 必须写明「哪一项不满足、缺在什么数值上」。"
)

_SYS_A2 = (
    "【角色】风控审核官，一票否决。不预测行情、不改计划、不下单。\n"
    "【输入】AI-1 计划 + 本地硬闸门 + 本地反算手数。本地 gates 里 hit=true 的条目，"
    "一律 VETO，不得为它辩护。\n"
    "【硬规则（任一不满足直接 VETO）】\n"
    "1. 盈亏比 < 1.5:1；\n"
    "2. 打到止损时的亏损 > 账户净值 0.5%；\n"
    "3. 事件窗口：未来 30 分钟内或本地判定的事件封锁窗口内有高影响事件；\n"
    "4. 入场/止损距包络边界或关键位 > 0.8×ATR（无结构依据）；\n"
    "5. 方向与 D1/H4 相反且无区间极值加持。\n"
    "【禁止】不许修改入场价、止损价、目标价来「让它通过」；有瑕疵就是 VETO。\n"
    "【输出】只输出一行 <result></result>，标签里放 JSON：\n"
    "<result>{\"verdict\":\"ALLOW|WARN|VETO\","
    "\"reason\":\"≤40字：违反了哪一条硬指标；通过则写 符合所有风控要求\","
    "\"max_lot_size\":\"原样抄 local_lots.lots\","
    "\"risk_note\":\"≤40字：最需要盯的一个风险，量化到价位\"}</result>"
)

_SYS_A3 = (
    "【角色】持仓监控机器人。只对已有持仓下指令，不参与开新仓。\n"
    "【输入】每笔持仓已带本地算好的 pos/zone/health/grade/r_now/trail/触发价，"
    "只能引用，不得重算、不得推翻。\n"
    "【硬规则（本地已判定，你只负责表述）】\n"
    "1. r_now ≥ 1.0 → 止损必须移到成本价（new_stop=entry）；\n"
    "2. r_now ≤ -1.0 → 全部平仓，不许等反弹；\n"
    "3. 多单 pos≥75、空单 pos≤25（止盈区）→ 减仓 50%，剩余止损移到成本价或上沿内 0.5×ATR；\n"
    "4. 多单 pos≤25、空单 pos≥75（破位区）→ 全部平仓；\n"
    "5. 价格触及初始止损 → 止损离场，禁止扛单。\n"
    "【禁止】出现「也许/可能/再看看/建议观察」；每条指令必须带价位。\n"
    "【输出】只输出一行 <result></result>，标签里放 JSON：\n"
    "<result>{\"positions\":[{\"ticket\":\"号码\",\"grade\":\"优|良|危险\","
    "\"zone_text\":\"靠近上沿|中轴|靠近下沿\",\"action\":\"hold|reduce|close|add\","
    "\"close_pct\":50或null,\"new_stop\":价或null,"
    "\"action_why\":\"≤30字，引用包络价或成本价\"}],\"summary\":\"≤30字总评\"}</result>\n"
    "grade 直接抄输入 grade，zone_text 按 zone 映射（upper→靠近上沿、mid→中轴、lower→靠近下沿），"
    "不许自己评级。"
)


def run(sym: str = "GOLD", force: bool = False, tf: str | None = None) -> dict:
    """跑一次三角色分析。force=True 时绕过最小间隔；tf 覆盖决策主周期并持久化。"""
    with _RUN_LOCK:
        import pxfmt
        import markets
        pxfmt.use(sym)                     # 本品种价格精度（黄金 2 位 / 直盘 5 位）
        cfg = _cfg()
        cfg["_contract"] = markets.contract(sym)
        cfg["_point"] = markets.point(sym)
        st_file = _state_file(sym)
        if tf and tf in ("M15", "M30", "H1", "H4", "D1") and tf != cfg["tf"]:
            cfg["tf"] = tf
            save_cfg({"tf": tf})
        if not cfg.get("enabled"):
            return {"ok": False, "error": "AI 决策面板已在配置中停用"}
        try:
            st = json.loads(st_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            st = {}
        if not force and st.get("result") and st.get("at"):
            try:
                at = dt.datetime.strptime(st["at"], "%Y-%m-%d %H:%M:%S")
                if (dt.datetime.now() - at).total_seconds() < cfg["min_interval_s"]:
                    return {"ok": True, "cached": True,
                            "result": st["result"], "at": st["at"]}
            except Exception:  # noqa: BLE001
                pass
        ctx = build_context(sym)
        gates = _hard_gates(ctx, cfg)
        facts = _facts(ctx, cfg)                  # 页面硬事实，全部本地算
        sinput = _structured_input(ctx, cfg, facts)  # 喂给 LLM 的结构化行情数据
        result: dict = {"ctx_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "tf": cfg["tf"], "gates": gates, "errors": [],
                        "rules": ctx.get("rules"), "facts": facts}

        def _input_head() -> str:
            return (f"品种={sym}，主下单周期={cfg['tf']}（即输入的 main_period）。"
                    f"单笔最大风险={cfg['risk_pct']}% 净值，最低盈亏比={cfg['min_rr']}:1，"
                    f"ATR 周期={cfg['atr_period']}。\n"
                    "结构化行情数据 JSON（全部为本地已算好的事实，直接引用；"
                    "local_verdict 是本地规则结论，必须遵守、不得推翻）：\n")

        # AI-1 状态识别与规则校验（一审）：只翻译状态 + 引用本地结论
        try:
            raw = _llm([{"role": "system", "content": _SYS_G + "\n" + _SYS_A1},
                        {"role": "user", "content": _input_head() +
                         json.dumps(sinput, ensure_ascii=False)}], cfg, 1600)
            txt, mach = _split_result(raw)
            bias = str(mach.get("direction_bias") or "").strip()
            d = (mach.get("direction")
                 or {"做多": "buy", "buy": "buy", "bull": "buy",
                     "做空": "sell", "sell": "sell", "bear": "sell"}.get(bias, "wait"))
            ev = [str(x) for x in (mach.get("evidence") or []) if str(x).strip()]
            a1: dict = {
                "direction": d,
                "confidence": mach.get("confidence"),
                "summary": (mach.get("analysis_summary") or mach.get("summary")
                            or (txt.splitlines()[0][:60] if txt else "")
                            or "；".join((facts.get("wait_rules") or [])[:2])
                            or "本地规则未放行，本轮观望"),
                "market_state": (mach.get("market_state") or
                                 (facts.get("market_state") or {}).get("state")),
                # 正文=evidence 列表（AI 不再写长文），保证卡片有依据可读
                "analysis": txt or "\n".join(f"- {x}" for x in ev),
            }
            if not mach:
                # 机器字段没解析出来：按本地规则观望，并留痕便于以后排查输出格式
                a1["parse_fallback"] = True
            plan = mach.get("plan")
            if a1["direction"] in ("buy", "sell") and isinstance(plan, dict):
                a1["plan"] = plan
                a1["invalid"] = mach.get("invalid")
            else:
                a1["direction"] = "wait"
                a1["plan"] = None
                a1["invalid"] = mach.get("invalid")
                wr = mach.get("wait_rules") or []
                a1["wait_rules"] = [_WAIT_TEXT.get(str(c).strip(), str(c)) for c in wr
                                    if str(c).strip()] or \
                                   facts.get("wait_rules") or \
                                   ["双确认未齐备，暂不满足开仓条件"]
            # 本地裁判：AI 给的方向只要违反任一硬规则，直接改观望并留痕
            _enforce_a1(a1, facts)
            _normal_a1(a1)
            if not a1.get("reasons") and a1.get("summary"):
                a1["reasons"] = [a1["summary"]]
            result["a1"] = a1
        except Exception as e:  # noqa: BLE001
            result["errors"].append(f"AI-1: {type(e).__name__}: {e}")
            result["a1"] = None
        # 计划级硬闸门（盈亏比/单笔风险/结构依据/区间位置/双确认/大周期）——本地算
        gates += _plan_gates(ctx, result.get("a1"), cfg)
        result["gates"] = gates
        # AI-2 风控审核（二审，一票否决权在本地，LLM 只做风险穿透与执行建议）
        # AI-1 观望时没有计划可审：本地直接给 standby。让 LLM 去「否决一个空计划」
        # 会在页面上显示成「风控否决」，语义正好相反（观望不是被否，是没东西可审）。
        a1_cur = result.get("a1") or {}
        local_lots = _plan_lots(ctx, a1_cur.get("plan"), cfg)
        if a1_cur.get("direction") not in ("buy", "sell") or not a1_cur.get("plan"):
            result["a2"] = {"verdict": "standby", "checks": [],
                            "logic_review": ["AI-1 观望，本轮没有可审的交易计划"],
                            "risk_level": "low",
                            "reason": "无计划可审",
                            "max_lot_size": "0.00 手（不建仓）",
                            "exec_advice": f"等 {cfg['tf']} 到上下沿 + 消息面 + M30/M15 启动三项齐备再出计划",
                            "size_advice": "0.00 手（不建仓）"}
        else:
            checks = _local_checks(ctx, gates, cfg, a1_cur)
            try:
                raw = _llm([{"role": "system", "content": _SYS_G + "\n" + _SYS_A2},
                            {"role": "user", "content":
                             "AI-1 结论与计划（一审）：\n" +
                             json.dumps({k: a1_cur.get(k) for k in
                                         ("direction", "confidence", "summary",
                                          "plan", "invalid", "wait_rules")},
                                        ensure_ascii=False) +
                             "\n\n本地硬闸门（hit=true 必须否决，不可推翻）：\n" +
                             json.dumps(gates, ensure_ascii=False) +
                             "\n\n本地闸门勾选表（已算好，直接引用结论）：\n" +
                             json.dumps(checks, ensure_ascii=False) +
                             "\n\n本地反算手数（local_lots，只能引用不可重算）：\n" +
                             json.dumps(local_lots, ensure_ascii=False) +
                             "\n\n结构化行情数据 JSON：\n" +
                             json.dumps(sinput, ensure_ascii=False)}], cfg, 900)
                txt, mach = _split_result(raw)
                verdict = str(mach.get("verdict") or "").strip().lower()
                if gates and verdict != "veto":
                    verdict = "veto"          # 一票否决权本地行使
                if verdict not in ("allow", "warn", "veto"):
                    verdict = "veto" if gates else "warn"
                lots_txt = (f"{local_lots['lots']:.2f} 手"
                            f"（风险 {local_lots['risk_usd']:.2f} 美金）"
                            if local_lots.get("lots") else "0.00 手（不足最小手）")
                result["a2"] = {"verdict": verdict,
                                "risk_level": mach.get("risk_level") or
                                ("high" if gates else "mid"),
                                # 手数以本地反算为准，LLM 报的数字不采信
                                "max_lot_size": lots_txt,
                                "size_advice": lots_txt,
                                "reason": mach.get("reason") or
                                ("命中本地硬闸门" if gates else "符合所有风控要求"),
                                "risk_note": mach.get("risk_note"),
                                "checks": checks, "analysis": txt,
                                "local_lots": local_lots}
            except Exception as e:  # noqa: BLE001
                result["errors"].append(f"AI-2: {type(e).__name__}: {e}")
                result["a2"] = {"verdict": "veto" if gates else "warn",
                                "risk_level": "high" if gates else "mid",
                                "checks": checks, "analysis": "",
                                "exec_advice": "AI-2 调用失败，按本地硬闸门结果执行"}
        # AI-3 持仓监控（三审）：健康度/触发价本地算，LLM 出分步操作指令
        pos_facts = _pos_facts(ctx, cfg)
        if pos_facts:
            try:
                raw = _llm([{"role": "system", "content": _SYS_G + "\n" + _SYS_A3},
                            {"role": "user", "content":
                             "持仓硬事实 JSON（pos/zone/健康度/触发价已算好，直接引用）：\n" +
                             json.dumps(pos_facts, ensure_ascii=False) +
                             "\n\n结构化行情数据 JSON：\n" +
                             json.dumps({k: sinput.get(k) for k in
                                         ("current_price", "envelope", "multi_period",
                                          "news", "key_levels", "atr") if k in sinput},
                                        ensure_ascii=False)}], cfg, 1200)
                txt, mach = _split_result(raw)
                mach_pos = {str((p or {}).get("ticket")): p
                            for p in (mach.get("positions") or []) if isinstance(p, dict)}
                for f in pos_facts:
                    mm = mach_pos.get(str(f.get("ticket"))) or {}
                    act = mm.get("action")
                    if act in ("hold", "reduce", "close", "add"):
                        f["action"] = act
                    if mm.get("action_why"):
                        f["action_why"] = str(mm["action_why"])
                    if mm.get("zone_text"):
                        f["zone_text"] = str(mm["zone_text"])
                    if mm.get("close_pct") is not None and f.get("action") == "reduce":
                        try:
                            f["close_pct"] = int(float(mm["close_pct"]))
                        except (TypeError, ValueError):
                            pass
                    ns = _num(mm.get("new_stop"))
                    if ns is not None and not f["trail"].get("move_to_be"):
                        f["trail"]["new_stop"] = ns
                        f["trail"]["move_to_be"] = True
                    # grade 以本地评级为准，LLM 只能引用不能改
                result["a3"] = {"positions": pos_facts,
                                "summary": mach.get("summary") or "",
                                "analysis": txt}
            except Exception as e:  # noqa: BLE001
                result["errors"].append(f"AI-3: {type(e).__name__}: {e}")
                result["a3"] = {"positions": pos_facts,
                                "summary": "AI-3 调用失败，以下为本地通道计算",
                                "analysis": ""}
        else:
            result["a3"] = {"positions": [], "summary": "空仓，无需监控"}
        result["ok"] = not result["errors"]
        # 影子账本：只记录、不下单。记录失败不影响主流程（账本是辅助证据，不是决策依赖）
        try:
            import shadow
            result["shadow_logged"] = shadow.record(sym, cfg, result, ctx) is not None
        except Exception as e:  # noqa: BLE001
            result["shadow_logged"] = False
            result["errors"].append(f"shadow: {type(e).__name__}: {e}")
        at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st_file.write_text(json.dumps({"at": at, "result": result},
                                         ensure_ascii=False, indent=1),
                              encoding="utf-8")
        # 审计：落上下文摘要 + 三角色原始结论（可复盘「AI 为什么这么说」）
        _audit({"at": at, "symbol": sym, "tf": cfg["tf"], "gates": gates,
                "a1": result.get("a1"), "a2": result.get("a2"),
                "a3": result.get("a3"), "errors": result["errors"],
                "ctx_summary": {k: ctx.get(k) for k in
                                ("price", "quote_age_s", "trend", "event",
                                 "account")}})
        return {"ok": result["ok"], "cached": False, "result": result, "at": at}
