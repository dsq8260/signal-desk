"""口述策略台：把「你说的话」变成一套可回测、可执行、可解释的规则。

分工是刻意的，不是省事：
  · 大模型（对话层）负责「听懂 + 复述 + 追问缺什么」，把自然语言翻成结构化规格；
    这一步允许不确定、可以来回改，改错了也不花钱。
  · 运行时只跑本文件里的确定性判定函数——同一份数据、同一套规则，
    结果永远一致。所以图上每个买卖点都能对应到「哪一条条件成立」，
    也能拿历史数据回测出胜率和期望值。

规格 schema（前端可视化编辑的就是它）::

    {
      "name": "20 根突破做多", "symbol": "GOLD#", "tf": "H1",
      "direction": "long",                       # long / short / both
      "entry": {"long": [cond, ...], "short": [cond, ...]},
      "exit":  {"sl_atr": 2.0, "tp_atr": 4.0, "channel_n": 0, "max_bars": 0},
      "risk":  {"risk_pct": 1.0, "max_lots": 0.05},
      "comment": "AIDesk"
    }

条件 cond::

    {"f":"brk",  "n":20, "dir":"up"}     收盘突破前 20 根最高
    {"f":"ma",   "n":20, "op":">"}       收盘在 MA20 之上
    {"f":"cross","fast":20,"slow":50,"dir":"up"}   均线金叉
    {"f":"rsi",  "op":">", "v":65}       RSI14 大于 65
    {"f":"boll", "op":">", "v":0.8}      布林带内位置（0=下轨,1=上轨）
    {"f":"mom",  "n":3, "op":">", "v":0.5}   近 3 根净位移（ATR 单位）
    {"f":"consec","n":3,"dir":"up"}      连涨 3 根
    {"f":"hour", "from":10, "to":22}     服务器时间窗口
    {"f":"adx",  "n":14, "op":">", "v":25}   ADX 趋势强度过滤（不分多空）
    {"f":"marel","fast":20,"slow":50,"ma":"ema","op":">"}   EMA 快线在慢线上方（定方向）
    {"f":"htf",  "tf":"H4","n":50,"op":">"}   大周期顺向过滤（H4 收盘在 EMA50 上方）
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
ACTIVE = DATA / "strategy_active.json"
SAVED = DATA / "strategy_saved.json"
AUDIT = DATA / "strategy_audit.jsonl"

# 成本与合约：黄金按 100 盎司/手，外汇按 10 万基础货币/手
COST = {
    "GOLD#": {"spread": 0.33, "slip": 0.10, "contract": 100.0},
    "EURUSD": {"spread": 0.00012, "slip": 0.00004, "contract": 100000.0},
    "GBPUSD": {"spread": 0.00016, "slip": 0.00005, "contract": 100000.0},
    "USDJPY": {"spread": 0.014, "slip": 0.004, "contract": 100000.0},
    "USDCHF": {"spread": 0.00018, "slip": 0.00005, "contract": 100000.0},
}
DEFAULT_COST = COST["GOLD#"]
EQ0 = 10000.0

SYMS = {
    "黄金": "GOLD#", "金价": "GOLD#", "黄金期货": "GOLD#",
    "欧元": "EURUSD", "英镑": "GBPUSD", "日元": "USDJPY", "瑞郎": "USDCHF",
}
TF_WORDS = {"1 分钟": "M1", "5 分钟": "M5", "15 分钟": "M15", "30 分钟": "M30",
            "1 小时": "H1", "4 小时": "H4", "日线": "D1", "每天": "D1"}


# ------------------------------------------------------------------ 规格
def blank(name: str = "") -> dict:
    return {
        "name": name or "未命名策略",
        "symbol": "GOLD#", "tf": "H1", "direction": "long",
        "entry": {"long": [], "short": []},
        "exit": {"sl_atr": 2.0, "tp_atr": 4.0, "channel_n": 0, "max_bars": 0},
        "risk": {"risk_pct": 1.0, "max_lots": 0.05},
        "comment": "AIDesk",
    }


def cond_label(c: dict) -> str:
    f = str(c.get("f"))
    if f == "brk":
        return f"收盘{'突破' if c.get('dir', 'up') == 'up' else '跌破'}前 {c.get('n', 20)} 根{'最高' if c.get('dir', 'up') == 'up' else '最低'}"
    if f == "ma":
        pfx = "EMA" if str(c.get("ma", "sma")).lower() == "ema" else "MA"
        return f"收盘在 {pfx}{c.get('n', 20)} {'之上' if c.get('op', '>') == '>' else '之下'}"
    if f == "cross":
        pfx = "EMA" if str(c.get("ma", "sma")).lower() == "ema" else "MA"
        return f"{pfx}{c.get('fast', 20)} {'上穿' if c.get('dir', 'up') == 'up' else '下穿'} {pfx}{c.get('slow', 50)}"
    if f == "adx":
        op = {">": "大于", "<": "小于"}.get(c.get("op", ">"), "大于")
        return f"ADX{c.get('n', 14)} {op} {c.get('v', 25):g}（趋势够强才进场）"
    if f == "marel":
        pfx = "EMA" if str(c.get("ma", "ema")).lower() == "ema" else "MA"
        side = "上方（只做多）" if c.get("op", ">") == ">" else "下方（只做空）"
        return f"{pfx}{c.get('fast', 20)} 在 {pfx}{c.get('slow', 50)} {side}"
    if f == "htf":
        op = "上方" if c.get("op", ">") == ">" else "下方"
        return f"大周期 {c.get('tf', 'H4')}：价格在 EMA{c.get('n', 50)} {op}"
    if f == "rsi":
        op = {">": "大于", "<": "小于"}.get(c.get("op", ">"), "大于")
        return f"RSI14 {op} {c.get('v', 60)}"
    if f == "boll":
        op = {">": "高于", "<": "低于"}.get(c.get("op", ">"), "高于")
        return f"布林位置 {op} {int(float(c.get('v', 0.8)) * 100)}%"
    if f == "mom":
        return f"近 {c.get('n', 3)} 根动量 {'>' if c.get('op', '>') == '>' else '<'} {c.get('v', 0)} ATR"
    if f == "consec":
        return f"连{'涨' if c.get('dir', 'up') == 'up' else '跌'} {c.get('n', 3)} 根"
    if f == "hour":
        return f"服务器 {c.get('from', 0)}–{c.get('to', 23)} 时"
    return json.dumps(c, ensure_ascii=False)


def describe(spec: dict) -> dict:
    """把规格翻译回人话，用于「我理解成这样，对吗」的确认环节。"""
    sp = _normalize(spec)
    ex = sp["exit"]
    lines = []
    dirs = []
    if sp["direction"] in ("long", "both") and sp["entry"]["long"]:
        dirs.append("做多")
        lines.append("**做多入场**（以下条件全部成立）：" +
                     "，且".join(cond_label(c) for c in sp["entry"]["long"]))
    if sp["direction"] in ("short", "both") and sp["entry"]["short"]:
        dirs.append("做空")
        lines.append("**做空入场**（以下条件全部成立）：" +
                     "，且".join(cond_label(c) for c in sp["entry"]["short"]))
    if not lines:
        lines.append("**入场**：还没说清楚，需要至少一条可以量化的条件")

    exl = []
    if ex.get("sl_atr"):
        exl.append(f"止损 {ex['sl_atr']:g} 倍 ATR")
    if ex.get("tp_atr"):
        exl.append(f"止盈 {ex['tp_atr']:g} 倍 ATR")
    if ex.get("channel_n"):
        exl.append(f"反向突破 {ex['channel_n']} 根通道就离场")
    if ex.get("max_bars"):
        exl.append(f"最多持有 {ex['max_bars']} 根 K 线")
    if exl:
        lines.append("**出场**：" + "；".join(exl))
    lines.append(f"**仓位**：单笔风险 {sp['risk']['risk_pct']:g}% 净值，手数由止损距离反算，"
                 f"上限 {sp['risk']['max_lots']:g} 手")
    lines.append(f"**标的**：{sp['symbol']} · {sp['tf']} 周期 · {('/'.join(dirs) or '方向待定')}")
    return {"spec": sp, "lines": lines,
            "title": f"{sp['symbol']} {sp['tf']} · {sp['name']}"}


def _normalize(spec: dict) -> dict:
    """补齐缺省字段，顺手把类型收敛，避免前端传来脏数据把回测算崩。"""
    b = blank()
    sp = dict(b)
    if isinstance(spec, dict):
        for k in ("name", "symbol", "tf", "direction", "comment"):
            if spec.get(k):
                sp[k] = str(spec[k])
        if spec.get("ea"):                       # 挖掘候选的 EA 名（Mine_mXX），跨页统一标识
            sp["ea"] = re.sub(r"[^A-Za-z0-9_]", "", str(spec["ea"]))[:32]
        sp["symbol"] = sp["symbol"].upper()
        sp["tf"] = sp["tf"].upper()
        ent = spec.get("entry") or {}
        for side in ("long", "short"):
            conds = []
            for c in (ent.get(side) or []):
                if isinstance(c, dict) and c.get("f"):
                    conds.append({k: v for k, v in c.items() if v is not None})
            sp["entry"][side] = conds
        ex = spec.get("exit") or {}
        for k, dv, cast in (("sl_atr", 2.0, float), ("tp_atr", 4.0, float),
                            ("channel_n", 0, int), ("max_bars", 0, int)):
            try:
                sp["exit"][k] = cast(ex.get(k, dv))
            except (TypeError, ValueError):
                sp["exit"][k] = dv
        rk = spec.get("risk") or {}
        for k, dv in (("risk_pct", 1.0), ("max_lots", 0.05)):
            try:
                sp["risk"][k] = float(rk.get(k, dv))
            except (TypeError, ValueError):
                sp["risk"][k] = dv
        sp["risk"]["risk_pct"] = max(0.05, min(5.0, sp["risk"]["risk_pct"]))
        sp["risk"]["max_lots"] = max(0.01, min(0.5, sp["risk"]["max_lots"]))
        sp["exit"]["sl_atr"] = max(0.0, min(12.0, sp["exit"]["sl_atr"]))
        sp["exit"]["tp_atr"] = max(0.0, min(40.0, sp["exit"]["tp_atr"]))
        sp["exit"]["channel_n"] = max(0, min(200, sp["exit"]["channel_n"]))
        sp["exit"]["max_bars"] = max(0, min(2000, sp["exit"]["max_bars"]))
    if sp["direction"] not in ("long", "short", "both"):
        sp["direction"] = "long"
    if not sp["entry"]["long"] and sp["direction"] == "long":
        sp["entry"]["long"] = []
    return sp


# ------------------------------------------------------------------ 口述解析
_NUM = r"(\d+(?:\.\d+)?)"


def _clean(text: str) -> str:
    t = str(text or "")
    t = t.replace("，", "，").replace("；", "；")
    t = t.translate(str.maketrans("０１２３４５６７８９．", "0123456789."))
    return t


def parse(text: str) -> dict:
    """把口述策略翻成规格 + 复述 + 追问。

    解析是规则式的（关键词 + 数字），不是万能的；听不懂的片段会原样
    放进 unknown，让用户改口或者手工补——这比假装听懂了更安全。
    """
    raw = _clean(text)
    t = re.sub(r"\s+", "", raw)
    sp = blank()
    understood: list[str] = []
    defaults: list[str] = []
    unknown: list[str] = []
    used = [False] * len(t)

    def take(m):
        for k in range(m.start(), m.end()):
            used[k] = True

    def matched(s: str) -> bool:
        return s in t

    # 「大周期」过滤要先认出来，否则「4小时做大周期」会被当成交易周期
    htf_tf: str | None = None
    m_h = (re.search(r"(H4|4小时|日线|D1|周线|W1)[^\d，。;；]{0,6}?(?:做大?周期|大周期|大级别)", t)
           or re.search(r"大(?:周期|级别)[^\d，。;；]{0,4}?(H4|4小时|日线|D1|周线|W1)", t))
    if m_h:
        htf_tf = {"H4": "H4", "4小时": "H4", "日线": "D1", "D1": "D1",
                  "周线": "W1", "W1": "W1"}.get(m_h.group(1))
        take(m_h)

    # 标的
    for word, sym in SYMS.items():
        if word in t:
            sp["symbol"] = sym
            understood.append(f"标的：{word} → {sym}")
            break
    # 周期（如果这段话已经被「大周期过滤」认领，就不再当交易周期）
    mtf = re.search(r"(\d+)(分钟|小时)(?:线|周期|K线|图)?", t)
    m_hr = re.search(r"(?:^|[^0-9A-Za-z])([HhMm])(\d{1,2})(?![0-9A-Za-z])", t)   # H1/H4/M15 写法
    m_day = re.search(r"(?:^|[^0-9A-Za-z])[Dd]1(?![0-9A-Za-z])", t)              # D1 写法
    tf_said = False
    if mtf and not any(used[mtf.start():mtf.end()]):
        n, unit = int(mtf.group(1)), mtf.group(2)
        tf = f"M{n}" if unit == "分钟" else f"H{n}"
        if tf in ("M1", "M5", "M15", "M30", "M45", "H1", "H4", "H6", "H12"):
            sp["tf"] = tf
            understood.append(f"周期：{n} {unit} → {tf}")
            take(mtf)
        tf_said = True
    elif m_hr and not any(used[m_hr.start():m_hr.end()]):
        p, n2 = m_hr.group(1).upper(), int(m_hr.group(2))
        tf = ("M" if p == "M" else "H") + str(n2)
        if tf in ("M1", "M5", "M15", "M30", "M45", "H1", "H4", "H6", "H12"):
            sp["tf"] = tf
            understood.append(f"周期：{p}{n2} → {tf}")
            take(m_hr)
        tf_said = True
    elif m_day and not any(used[m_day.start():m_day.end()]):
        sp["tf"] = "D1"
        understood.append("周期：D1")
        take(m_day)
        tf_said = True
    elif "日线" in t:
        sp["tf"] = "D1"
        understood.append("周期：日线")
        tf_said = True
    elif "小时" in t:
        sp["tf"] = "H1"
        understood.append("周期：小时线")
        tf_said = True

    # 方向
    dir_said = False
    if re.search(r"只做多|只做多头|只买|仅做多", t):
        sp["direction"] = "long"
        understood.append("方向：只做多")
        dir_said = True
    elif re.search(r"只做空|只做空头|只卖|仅做空", t):
        sp["direction"] = "short"
        understood.append("方向：只做空")
        dir_said = True
    elif re.search(r"多空都|双向|两边都|做多做空都", t):
        sp["direction"] = "both"
        understood.append("方向：多空都做")
        dir_said = True

    # 分句：含「平仓/出场/离场/止盈」的句子算出场规则，其余算入场规则。
    # 这样「突破20根高点做多，跌破10根低点平仓」不会被误读成两条入场条件。
    _cl = [c for c in re.split(r"[，。；、,;!！?？\n]", t) if c]
    _EX = ("平仓", "出场", "离场", "退场", "就走", "止盈")
    exit_text = "".join(c for c in _cl if any(w in c for w in _EX))
    entry_text = "".join(c for c in _cl if not any(w in c for w in _EX))

    # 说「做空/做多」但没说「只」时，也要认方向
    if sp["direction"] == "long":
        if re.search(r"做空|空单|看跌", t) and not re.search(r"做多|多单|看涨", t):
            sp["direction"] = "short"
            understood.append("方向：做空")
            dir_said = True

    long_c: list[dict] = []
    short_c: list[dict] = []
    flist: list[dict] = []      # 不分多空的过滤条件（如 ADX），两边的入场都要满足

    # 突破 N 根/日 高/低点
    for m in re.finditer(_NUM + r"(?:日|根|个)?(?:内)?(?:的)?(?:最高|新高|高点)", entry_text):
        n = int(float(m.group(1)))
        if 2 <= n <= 400:
            long_c.append({"f": "brk", "n": n, "dir": "up"})
            understood.append(f"入场：收盘突破前 {n} 根最高价")
            take(m)
    for m in re.finditer(_NUM + r"(?:日|根|个)?(?:内)?(?:的)?(?:最低|新低|低点)", entry_text):
        n = int(float(m.group(1)))
        if 2 <= n <= 400:
            short_c.append({"f": "brk", "n": n, "dir": "down"})
            understood.append(f"入场：收盘跌破前 {n} 根最低价")
            take(m)
    # 唐奇安通道：认行话。经典做法是上下轨突破、多空双向
    m_don = re.search(r"唐奇安(?:通道)?\s*(\d{1,3})?", t, re.I)
    n_don = int(m_don.group(1)) if (m_don and m_don.group(1)) else 20
    if m_don:
        take(m_don)
    if m_don:
        if not any(c["f"] == "brk" for c in long_c + short_c):
            if not (dir_said or re.search(r"做多|做空|多单|空单|看涨|看跌|多头|空头", t)):
                long_c.append({"f": "brk", "n": n_don, "dir": "up"})
                short_c.append({"f": "brk", "n": n_don, "dir": "down"})
                sp["direction"] = "both"
                understood.append(f"唐奇安通道：收盘突破前 {n_don} 根最高价做多、跌破前 {n_don} 根最低价做空"
                                  f"（N 可改；这是经典的双向做法，想只做单边就说「只做多」）")
            else:
                long_side = sp["direction"] in ("long", "both")
                (long_c if long_side else short_c).append(
                    {"f": "brk", "n": n_don, "dir": "up" if long_side else "down"})
                understood.append(f"唐奇安通道突破：收盘{'突破' if long_side else '跌破'}前 {n_don} 根"
                                  f"{'最高价做多' if long_side else '最低价做空'}（N 可改）")
        elif not dir_said and not re.search(r"做多|做空|多单|空单|看涨|看跌|多头|空头", t):
            sp["direction"] = "both"
            understood.append("你提了唐奇安通道 → 经典做法是多空双向，我按双向处理")
    if not (long_c or short_c) and re.search(r"突破|新高", entry_text):
        long_c.append({"f": "brk", "n": 20, "dir": "up"})
        defaults.append("你只说了「突破」，我按最常见的 20 根最高价理解（唐奇安通道）")
    if not (long_c or short_c) and re.search(r"跌破|新低", entry_text):
        short_c.append({"f": "brk", "n": 20, "dir": "down"})
        defaults.append("你只说了「跌破」，我按 20 根最低价理解")

    # 均线
    mm = re.search(r"(?:站上|上穿|突破|收在|位于)(?:MA)?" + _NUM + r"(?:日|根)?(?:移动)?均线", t)
    if mm:
        n = int(float(mm.group(1)))
        long_c.append({"f": "ma", "n": n, "op": ">"})
        understood.append(f"入场：收盘站上 MA{n}")
        take(mm)
    mm2 = re.search(r"(?:跌破|下穿|收在)(?:MA)?" + _NUM + r"(?:日|根)?(?:移动)?均线", t)
    if mm2:
        n = int(float(mm2.group(1)))
        short_c.append({"f": "ma", "n": n, "op": "<"})
        understood.append(f"入场：收盘跌破 MA{n}")
        take(mm2)
    # 金叉/死叉：如果这句话说的是 EMA 金叉，交给下面的 EMA 分支处理，避免重复条件
    _kc = [c for c in re.split(r"[，。；、,;!！?？\n]", raw)
           if "金叉" in re.sub(r"\s+", "", c) and "EMA" not in c.upper()]
    if _kc:
        long_c.append({"f": "cross", "fast": 20, "slow": 50, "dir": "up"})
        understood.append("入场：MA20 上穿 MA50（金叉）")
    _sc = [c for c in re.split(r"[，。；、,;!！?？\n]", raw)
           if "死叉" in re.sub(r"\s+", "", c) and "EMA" not in c.upper()]
    if _sc:
        short_c.append({"f": "cross", "fast": 20, "slow": 50, "dir": "down"})
        understood.append("入场：MA20 下穿 MA50（死叉）")

    # RSI
    # RSI 的方向要看阈值含义：高位（>=60）是超买→做空侧，低位（<=40）超卖→做多侧
    mr = re.search(r"RSI(?:14)?(?:大于|高于|超过|>)" + _NUM, t, re.I)
    if mr:
        v = float(mr.group(1))
        (short_c if v >= 60 else long_c).append({"f": "rsi", "op": ">", "v": v})
        understood.append(f"入场：RSI14 > {v:g}（{'超买，做空侧' if v >= 60 else '强势，做多侧'}）")
        take(mr)
    mr2 = re.search(r"RSI(?:14)?(?:小于|低于|<)" + _NUM, t, re.I)
    if mr2:
        v = float(mr2.group(1))
        (long_c if v <= 40 else short_c).append({"f": "rsi", "op": "<", "v": v})
        understood.append(f"入场：RSI14 < {v:g}（{'超卖，做多侧' if v <= 40 else '弱势，做空侧'}）")
        take(mr2)

    # 布林
    mb = re.search(r"布林(?:带)?" + _NUM + r"%", t)
    if mb:
        p = float(mb.group(1)) / 100.0
        op = ">" if re.search(r"上轨|上方|高于", t) else "<"
        (long_c if op == ">" else short_c).append({"f": "boll", "op": op, "v": p})
        understood.append(f"入场：布林位置 {op} {p*100:.0f}%")
        take(mb)

    # 连涨/连跌
    mc = re.search(r"连涨" + _NUM + r"根", t)
    if mc:
        long_c.append({"f": "consec", "n": int(float(mc.group(1))), "dir": "up"})
        understood.append(f"入场：连涨 {mc.group(1)} 根")
        take(mc)
    mc2 = re.search(r"连跌" + _NUM + r"根", t)
    if mc2:
        short_c.append({"f": "consec", "n": int(float(mc2.group(1))), "dir": "down"})
        understood.append(f"入场：连跌 {mc2.group(1)} 根")
        take(mc2)

    # 时间段
    mh = re.search(r"(\d{1,2})点(?:到|至|~|-|－)(\d{1,2})点", t)
    if mh:
        h0, h1 = int(mh.group(1)), int(mh.group(2))
        for c in (long_c, short_c):
            c.append({"f": "hour", "from": h0, "to": h1})
        understood.append(f"只在服务器 {h0}–{h1} 时交易")
        take(mh)

    # ADX（趋势强度过滤，不分多空）——「ADX14 看动量」这种说法就落在这里
    m_adx = re.search(r"ADX\s*(\d{1,2})?", t, re.I)
    if m_adx:
        n_adx = int(m_adx.group(1) or 14)
        v_adx = 25.0
        m_adx2 = re.search(r"ADX\s*\d{0,2}\s*(?:大于|高于|超过|>)\s*(\d+(?:\.\d+)?)", t, re.I)
        if m_adx2:
            v_adx = float(m_adx2.group(1))
            take(m_adx2)
        flist.append({"f": "adx", "n": n_adx, "op": ">", "v": v_adx})
        understood.append(f"ADX{n_adx} > {v_adx:g}（趋势强度过滤）：只在趋势够强时才进场，"
                          f"25 是常见门槛，可改；盘整行情它会让你空仓")
        take(m_adx)

    # EMA：金叉/死叉是触发条件，"做方向"是趋势方向过滤——两种含义分开
    m_ecu = re.search(r"EMA\s*(\d{1,3})[\s,，/和]*(?:上穿|金叉)[\s,，/和]*(?:EMA)?\s*(\d{1,3})", raw, re.I)
    m_ecd = re.search(r"EMA\s*(\d{1,3})[\s,，/和]*(?:下穿|死叉)[\s,，/和]*(?:EMA)?\s*(\d{1,3})", raw, re.I)
    m_e2 = re.search(r"EMA\s*(\d{1,3})[\s,，/和]+(?:EMA)?\s*(\d{1,3})", raw, re.I)
    m_e1 = re.search(r"EMA\s*(\d{1,3})", raw, re.I)
    if m_ecu or m_ecd:
        if m_ecu:
            fa, sb = sorted((int(m_ecu.group(1)), int(m_ecu.group(2))))
            long_c.append({"f": "cross", "fast": fa, "slow": sb, "ma": "ema", "dir": "up"})
            understood.append(f"入场：EMA{fa} 上穿 EMA{sb}（EMA 金叉）")
        if m_ecd:
            fa, sb = sorted((int(m_ecd.group(1)), int(m_ecd.group(2))))
            short_c.append({"f": "cross", "fast": fa, "slow": sb, "ma": "ema", "dir": "down"})
            understood.append(f"入场：EMA{fa} 下穿 EMA{sb}（EMA 死叉）")
    elif m_e2:
        fa, sb = int(m_e2.group(1)), int(m_e2.group(2))
        if fa > sb:
            fa, sb = sb, fa           # 小的当快线
        long_c.append({"f": "marel", "fast": fa, "slow": sb, "ma": "ema", "op": ">"})
        short_c.append({"f": "marel", "fast": fa, "slow": sb, "ma": "ema", "op": "<"})
        understood.append(f"方向过滤：EMA{fa} 在 EMA{sb} 上方只做多、下方只做空"
                          f"（你说的「EMA 做方向」，这是均线多头/空头排列）")
    elif m_e1:
        n_e = int(m_e1.group(1))
        long_c.append({"f": "ma", "n": n_e, "op": ">", "ma": "ema"})
        short_c.append({"f": "ma", "n": n_e, "op": "<", "ma": "ema"})
        understood.append(f"方向过滤：收盘在 EMA{n_e} 上方只做多、下方只做空")

    # 大周期定方向——「H4 做大周期」：小周期进场，大周期趋势顺向才做
    if htf_tf:
        if htf_tf == sp["tf"]:
            defaults.append(f"大周期和交易周期都是 {htf_tf}，大周期过滤就没意义了，我忽略了这条")
        else:
            long_c.append({"f": "htf", "tf": htf_tf, "n": 50, "op": ">"})
            short_c.append({"f": "htf", "tf": htf_tf, "n": 50, "op": "<"})
            understood.append(f"大周期过滤：{htf_tf} 收盘在 EMA50 上方只做多、下方只做空"
                              f"（你说的「{htf_tf} 做大周期」，顺大趋势做小周期）")

    # 止损 / 止盈
    ms = re.search(r"止损(?:设在|设为|放|定在)?" + _NUM + r"(?:个|倍)?ATR", t, re.I)
    if ms:
        sp["exit"]["sl_atr"] = float(ms.group(1))
        understood.append(f"止损：{ms.group(1)} 倍 ATR")
        take(ms)
    else:
        defaults.append("你没说止损，我按 2 倍 ATR 兜底（不能让单子没有止损）")
    mt = re.search(r"止盈(?:设在|设为|放|定在)?" + _NUM + r"(?:个|倍)?ATR", exit_text or t, re.I)
    if mt:
        sp["exit"]["tp_atr"] = float(mt.group(1))
        understood.append(f"止盈：{mt.group(1)} 倍 ATR")
        take(mt)
    mp = re.search(r"盈亏比" + _NUM, t)
    if mp and not mt:
        ratio = float(mp.group(1))
        sp["exit"]["tp_atr"] = round(sp["exit"]["sl_atr"] * ratio, 2)
        understood.append(f"止盈：按盈亏比 1:{ratio:g} 反算 = {sp['exit']['tp_atr']:g} 倍 ATR")
        take(mp)
    else:
        defaults.append(f"你没说止盈，我先给 4 倍 ATR（盈亏比 1:2）；也可以改成「跌破 N 根低点就走」")

    mch = re.search(r"(?:跌破|回落|掉破)" + _NUM + r"根(?:低点|最低)", exit_text)
    if mch:
        sp["exit"]["channel_n"] = int(float(mch.group(1)))
        understood.append(f"出场：跌破 {mch.group(1)} 根最低点")
        take(mch)
    mch2 = re.search(r"(?:突破|涨破)" + _NUM + r"根(?:高点|最高)", exit_text)
    if mch2:
        sp["exit"]["channel_n"] = int(float(mch2.group(1)))
        understood.append(f"出场：突破 {mch2.group(1)} 根最高点")
        take(mch2)
    mmx = re.search(r"持有(?:最多)?" + _NUM + r"根", exit_text or t)
    if mmx:
        sp["exit"]["max_bars"] = int(float(mmx.group(1)))
        understood.append(f"出场：最多持有 {mmx.group(1)} 根")
        take(mmx)

    # 风险
    mrk = re.search(r"风险" + _NUM + r"%", t)
    if mrk:
        sp["risk"]["risk_pct"] = float(mrk.group(1))
        understood.append(f"单笔风险：{mrk.group(1)}% 净值")
        take(mrk)
    ml = re.search(r"最多" + _NUM + r"手", t)
    if ml:
        sp["risk"]["max_lots"] = float(ml.group(1))
        understood.append(f"手数上限：{ml.group(1)} 手")
        take(ml)

    def _mirror(c: dict) -> dict:
        c = dict(c)
        f = c.get("f")
        if f in ("brk", "cross", "consec"):
            c["dir"] = "down" if c.get("dir", "up") == "up" else "up"
        elif f in ("ma", "marel", "htf", "rsi", "boll", "mom"):
            c["op"] = "<" if c.get("op", ">") == ">" else ">"
        return c

    # 方向兜底：条件决定方向，冲突时以条件为准并明确告知
    if sp["direction"] == "long" and not long_c and short_c:
        sp["direction"] = "short"
        defaults.append("条件是「跌破」类（做空形态），但你没明确说方向，我按做空处理——如果是想做多请改口")
    if sp["direction"] == "short" and not short_c and long_c:
        sp["direction"] = "long"
        defaults.append("条件是「突破」类（做多形态），但你说的是做空/只做空，我按条件改成了做多——请确认")

    if sp["direction"] == "both" and long_c and not short_c:
        short_c = [_mirror(c) for c in long_c]
        defaults.append("多空都做：做空侧我按同样条件镜像（突破→跌破、站上→跌破）")
    if sp["direction"] == "both" and short_c and not long_c:
        long_c = [_mirror(c) for c in short_c]
        defaults.append("多空都做：做多侧我按同样条件镜像（跌破→突破）")

    # 只有过滤/方向条件、没有触发条件的兜底提醒
    if not any(c["f"] in ("brk", "cross", "consec", "rsi", "boll", "mom") for c in long_c + short_c):
        defaults.append("现在这些条件都是「过滤/方向」类，还缺一条「什么时候进场」的触发条件"
                        "（比如「突破20根高点」「金叉」）——光有过滤器会一直不进场或频繁进场")

    sp["entry"]["long"] = (long_c + flist) if sp["direction"] in ("long", "both") else []
    sp["entry"]["short"] = (short_c + flist) if sp["direction"] in ("short", "both") else []

    # 没匹配上的片段：按标点切句，逐句判断有没有出现可执行关键词
    if not tf_said:
        defaults.append("你没说交易周期，我按 H1（1 小时）理解；右边可以改成 M15/H4/D1 等")
    pieces = [p for p in re.split(r"[，。；、,;!！?？\n]", raw) if len(p.strip()) >= 2]
    for p in pieces:
        q = re.sub(r"\s+", "", p)
        if q and not _piece_hit(q):
            unknown.append(p.strip())

    name = _name_from(text, sp)
    sp["name"] = name
    sp = _normalize(sp)
    d = describe(sp)
    return {"ok": True, "spec": sp, "name": name,
            "understood": understood, "defaults": defaults, "unknown": unknown[:6],
            "lines": d["lines"],
            "note": "这些都是我按你说的翻译成的可执行条件；有错的地方直接改，改完再回测。"}


_PIECE_PAT = re.compile(
    r"突破|跌破|最高|最低|新高|新低|高[点位]|低[点位]|均线|MA\d|EMA|金叉|死叉|RSI|ADX|布林|"
    r"唐奇安|通道|大周期|大级别|连[涨跌]|ATR|止损|止盈|盈亏比|手|风险|分钟|小时|日线|"
    r"做多|做空|多空|双向|黄金|欧元|英镑|日元|瑞郎|点", re.I)


def _piece_hit(piece: str) -> bool:
    """这一句里有没有出现「能被引擎执行」的关键词。"""
    return bool(_PIECE_PAT.search(piece))


def _name_from(text: str, sp: dict) -> str:
    t = re.sub(r"\s+", "", text or "")
    if not t:
        return "口述策略"
    cl = [c for c in re.split(r"[，。；、,;!！?？\n]", t) if c]
    pick = next((c for c in cl if re.search(r"突破|跌破|均线|金叉|死叉|RSI|布林|连涨|连跌", c, re.I)), "")
    base = pick or t
    base = re.sub(r"^(帮我|我想|我要|给我|做一[个套]?|写一[个套]?|弄一[个套]?|用|按|策略[：:]?|价格)+", "", base)
    base = re.sub(r"(做|的|来)+策[略略]$|(做|的)策略$|做单$|做交易$", "", base)
    base = base.strip("。 ，,.")
    return (base[:16] + "…") if len(base) > 16 else (base or "口述策略")


# ------------------------------------------------------------------ 指标序列
def _sma(vals: list[float], n: int) -> list[float | None]:
    out: list[float | None] = []
    s = 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= n:
            s -= vals[i - n]
        out.append(s / n if i >= n - 1 else None)
    return out


def _rsi(closes: list[float], n: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    if len(closes) <= n:
        return out
    gain = loss = 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        gain += max(0.0, d)
        loss += max(0.0, -d)
    ag, al = gain / n, loss / n
    out[n] = 100.0 - 100.0 / (1 + (ag / al if al else 999))
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (n - 1) + max(0.0, d)) / n
        al = (al * (n - 1) + max(0.0, -d)) / n
        out[i] = 100.0 - 100.0 / (1 + (ag / al if al else 999))
    return out


def _ema(vals: list[float], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(vals)
    if n <= 0 or len(vals) < n:
        return out
    k = 2.0 / (n + 1.0)
    e = sum(vals[:n]) / n          # 用前 n 根 SMA 做种子
    out[n - 1] = e
    for i in range(n, len(vals)):
        e = vals[i] * k + e * (1 - k)
        out[i] = e
    return out


def _adx(rows: list[list], n: int = 14) -> list[float | None]:
    """Wilder ADX：只衡量趋势强度（0-100），不带方向。"""
    m = len(rows)
    out: list[float | None] = [None] * m
    if m < 2 * n + 2:
        return out
    tr_s = dp_s = dm_s = 0.0
    dxs: list[float] = []
    for i in range(1, m):
        h, l = rows[i][2], rows[i][3]
        ph, pl, pc = rows[i - 1][2], rows[i - 1][3], rows[i - 1][4]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        up, dn = h - ph, pl - l
        dp = up if up > dn and up > 0 else 0.0
        dm = dn if dn > up and dn > 0 else 0.0
        if i <= n:
            tr_s += tr
            dp_s += dp
            dm_s += dm
            if i == n:
                dpv = 100.0 * dp_s / tr_s if tr_s else 0.0
                dmv = 100.0 * dm_s / tr_s if tr_s else 0.0
                dxs.append(100.0 * abs(dpv - dmv) / (dpv + dmv) if (dpv + dmv) else 0.0)
        else:
            tr_s = tr_s - tr_s / n + tr
            dp_s = dp_s - dp_s / n + dp
            dm_s = dm_s - dm_s / n + dm
            dpv = 100.0 * dp_s / tr_s if tr_s else 0.0
            dmv = 100.0 * dm_s / tr_s if tr_s else 0.0
            dx = 100.0 * abs(dpv - dmv) / (dpv + dmv) if (dpv + dmv) else 0.0
            dxs.append(dx)
            if len(dxs) < n:
                continue
            if len(dxs) == n:
                out[i] = sum(dxs) / n
            else:
                prev = out[i - 1]
                out[i] = (prev * (n - 1) + dx) / n if prev is not None else sum(dxs[-n:]) / n
    return out


class Series:
    """一份 K 线 + 预计算指标，回测和实盘共用同一套口径。"""

    def __init__(self, rows: list[list], symbol: str = "GOLD#"):
        self.rows = rows
        self.n = len(rows)
        self.symbol = symbol
        self.close = [r[4] for r in rows]
        self.high = [r[2] for r in rows]
        self.low = [r[3] for r in rows]
        self.open = [r[1] for r in rows]
        self.atr = [r[11] for r in rows]
        self.ma20 = [r[7] for r in rows]
        self.ma50 = [r[8] for r in rows]
        self.bo_up = [r[9] for r in rows]
        self.bo_dn = [r[10] for r in rows]
        self.rsi = _rsi(self.close, 14)
        self._ema: dict[int, list] = {}
        self._adx: dict[int, list] = {}
        self._htf: dict[str, list] = {}

    def ema_series(self, n: int) -> list:
        if n not in self._ema:
            self._ema[n] = _ema(self.close, n)
        return self._ema[n]

    def adx_series(self, n: int) -> list:
        if n not in self._adx:
            self._adx[n] = _adx(self.rows, n)
        return self._adx[n]

    def htf_trend(self, tf: str, n: int = 50) -> list:
        """大周期 EMA 对齐到本周期每根 K 线。

        只用「决策时刻已收盘」的大周期 K 线：本 bar 的判定发生在它的收盘
        （= 开盘时间 + 本周期秒数），只有在此之前完成的大周期 bar 才可用。
        旧版按「大周期开盘时间早于本 bar」取 bar，会选中包含本 bar、还没
        走完的那根——用到未来数据，实盘 EA 也复现不了，已修正。
        与实盘 EA 的取法对齐：EA 在新 bar 开盘时刻取「当时已收盘的最近
        大周期 bar」的 EMA。
        """
        key = f"{tf}:{n}"
        if key in self._htf:
            return self._htf[key]
        out: list[float | None] = [None] * self.n
        rows = _load_tf_rows(self.symbol, tf)
        if rows and len(rows) > n + 2:
            e = _ema([r[4] for r in rows], n)
            hsec = _TF_SEC.get(str(tf).upper(), 14400)
            bsec = _bar_sec(self.rows)
            ht = [dt.datetime.strptime(str(r[0])[:19], _FMT) for r in rows]
            j = 0
            for i, r in enumerate(self.rows):
                t = dt.datetime.strptime(str(r[0])[:19], _FMT) + dt.timedelta(seconds=bsec)
                while j + 1 < len(rows) and ht[j + 1] + dt.timedelta(seconds=hsec) <= t:
                    j += 1
                if ht[j] + dt.timedelta(seconds=hsec) <= t and e[j] is not None:
                    out[i] = e[j]
        self._htf[key] = out
        return out

    def hour(self, i: int) -> int:
        try:
            return int(str(self.rows[i][0])[11:13])
        except (ValueError, IndexError):
            return 12

    def atr_rank(self, i: int) -> float:
        seg = [x for x in self.atr[max(0, i - 119):i + 1] if x]
        a = self.atr[i] or 0
        return sum(1 for x in seg if x < a) / len(seg) if seg else 0.5


_TF_SEC = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600,
           "H4": 14400, "D1": 86400, "W1": 604800}
_FMT = "%Y-%m-%d %H:%M:%S"


def _bar_sec(rows: list[list]) -> int:
    """相邻 bar 时间差的最常见值（周末跳过不影响众数）。"""
    from collections import Counter
    c: Counter = Counter()
    prev = None
    for r in rows[:400]:
        try:
            t = dt.datetime.strptime(str(r[0])[:19], _FMT)
        except ValueError:
            continue
        if prev is not None:
            d = int((t - prev).total_seconds())
            if d > 0:
                c[d] += 1
        prev = t
    return c.most_common(1)[0][0] if c else 3600


def _ma(s: Series, i: int, n: int) -> float | None:
    if n == 20:
        return s.ma20[i]
    if n == 50:
        return s.ma50[i]
    seg = [x for x in s.close[max(0, i - n + 1):i + 1]]
    return sum(seg) / len(seg) if len(seg) >= n else None


def _load_tf_rows(symbol: str, tf: str) -> list[list]:
    """按优先级加载另一个周期的 K 线：本地历史文件 → market 实时兜底。"""
    try:
        import backtest as bt
        rows = bt.load(symbol, tf)
        if rows:
            return rows
    except Exception:  # noqa: BLE001
        pass
    try:
        import market as mk
        d = mk.payload(symbol, tf, 3000)
        if d.get("ok"):
            return d["bars"]
    except Exception:  # noqa: BLE001
        pass
    return []


def _maval(s: Series, i: int, n: int, ema: bool) -> float | None:
    return s.ema_series(n)[i] if ema else _ma(s, i, n)


def check(s: Series, i: int, c: dict) -> bool:
    """单条条件是否成立。这里就是「策略」的全部定义，没有任何随机性。"""
    f = c.get("f")
    px, atr = s.close[i], s.atr[i] or 0
    if f == "brk":
        n = int(c.get("n", 20))
        if i < n or not atr:
            return False
        if c.get("dir", "up") == "up":
            h = max(s.high[i - n:i])
            return px > h
        lo = min(s.low[i - n:i])
        return px < lo
    if f == "ma":
        n = int(c.get("n", 20))
        m = _maval(s, i, n, str(c.get("ma", "sma")).lower() == "ema")
        if m is None:
            return False
        return px > m if c.get("op", ">") == ">" else px < m
    if f == "cross":
        fast, slow = int(c.get("fast", 20)), int(c.get("slow", 50))
        if i < 2:
            return False
        ema = str(c.get("ma", "sma")).lower() == "ema"
        f0, f1 = _maval(s, i - 1, fast, ema), _maval(s, i, fast, ema)
        s0, s1 = _maval(s, i - 1, slow, ema), _maval(s, i, slow, ema)
        if None in (f0, f1, s0, s1):
            return False
        if c.get("dir", "up") == "up":
            return f0 <= s0 and f1 > s1
        return f0 >= s0 and f1 < s1
    if f == "adx":
        n, v = int(c.get("n", 14)), float(c.get("v", 25))
        a = s.adx_series(n)[i]
        if a is None:
            return False
        return a > v if c.get("op", ">") == ">" else a < v
    if f == "marel":
        fa, sb = int(c.get("fast", 20)), int(c.get("slow", 50))
        ema = str(c.get("ma", "ema")).lower() == "ema"
        a1, b1 = _maval(s, i, fa, ema), _maval(s, i, sb, ema)
        if a1 is None or b1 is None:
            return False
        return a1 > b1 if c.get("op", ">") == ">" else a1 < b1
    if f == "htf":
        n = int(c.get("n", 50))
        e = s.htf_trend(str(c.get("tf", "H4")), n)[i]
        if e is None:
            return False
        return px > e if c.get("op", ">") == ">" else px < e
    if f == "rsi":
        v = s.rsi[i]
        if v is None:
            return False
        return v > float(c.get("v", 60)) if c.get("op", ">") == ">" else v < float(c.get("v", 40))
    if f == "boll":
        up, dn = s.bo_up[i], s.bo_dn[i]
        if up is None or dn is None or up <= dn:
            return False
        pos = (px - dn) / (up - dn)
        return pos > float(c.get("v", 0.8)) if c.get("op", ">") == ">" else pos < float(c.get("v", 0.2))
    if f == "mom":
        n = int(c.get("n", 3))
        if i < n or not atr:
            return False
        m = (px - s.close[i - n]) / atr
        return m > float(c.get("v", 0)) if c.get("op", ">") == ">" else m < float(c.get("v", 0))
    if f == "consec":
        n = int(c.get("n", 3))
        if i < n:
            return False
        ups = all(s.close[i - k] > s.open[i - k] for k in range(n))
        return ups if c.get("dir", "up") == "up" else all(
            s.close[i - k] < s.open[i - k] for k in range(n))
    if f == "hour":
        h = s.hour(i)
        h0, h1 = int(c.get("from", 0)), int(c.get("to", 23))
        return h0 <= h <= h1 if h0 <= h1 else (h >= h0 or h <= h1)
    return False


def eval_side(s: Series, i: int, conds: list[dict]) -> bool:
    return bool(conds) and all(check(s, i, c) for c in conds)


def entry_side(s: Series, i: int, spec: dict) -> str | None:
    sp = _normalize(spec)
    if sp["direction"] in ("long", "both") and eval_side(s, i, sp["entry"]["long"]):
        return "buy"
    if sp["direction"] in ("short", "both") and eval_side(s, i, sp["entry"]["short"]):
        return "sell"
    return None


def cond_status(s: Series, i: int, spec: dict) -> list[dict]:
    """逐条条件在当根 K 线成立与否——图上「为什么现在有/没有信号」就靠它。"""
    sp = _normalize(spec)
    out = []
    for side in ("long", "short"):
        for c in sp["entry"][side]:
            if side == "long" and sp["direction"] not in ("long", "both"):
                continue
            if side == "short" and sp["direction"] not in ("short", "both"):
                continue
            out.append({"side": side, "label": cond_label(c), "ok": check(s, i, c),
                        "why": _why(s, i, c)})
    return out


def _why(s: Series, i: int, c: dict) -> str:
    f = c.get("f")
    px, atr = s.close[i], s.atr[i] or 0
    if f == "brk":
        n = int(c.get("n", 20))
        if i < n:
            return "K 线不够"
        if c.get("dir", "up") == "up":
            return f"收盘 {px:.2f} vs 前 {n} 根高 {max(s.high[i - n:i]):.2f}"
        return f"收盘 {px:.2f} vs 前 {n} 根低 {min(s.low[i - n:i]):.2f}"
    if f == "ma":
        m = _maval(s, i, int(c.get("n", 20)), str(c.get("ma", "sma")).lower() == "ema")
        pfx = "EMA" if str(c.get("ma", "sma")).lower() == "ema" else "MA"
        return f"收盘 {px:.2f} vs {pfx}{c.get('n', 20)} {('%.2f' % m) if m else '—'}"
    if f == "cross":
        ema = str(c.get("ma", "sma")).lower() == "ema"
        pfx = "EMA" if ema else "MA"
        a, b = _maval(s, i, int(c.get("fast", 20)), ema), _maval(s, i, int(c.get("slow", 50)), ema)
        return f"{pfx}{c.get('fast', 20)} {('%.2f' % a) if a else '—'} / {pfx}{c.get('slow', 50)} {('%.2f' % b) if b else '—'}"
    if f == "adx":
        a = s.adx_series(int(c.get("n", 14)))[i]
        return f"ADX {('%.1f' % a) if a is not None else '—'}（门槛 {c.get('v', 25):g}）"
    if f == "marel":
        ema = str(c.get("ma", "ema")).lower() == "ema"
        pfx = "EMA" if ema else "MA"
        a, b = _maval(s, i, int(c.get("fast", 20)), ema), _maval(s, i, int(c.get("slow", 50)), ema)
        return f"{pfx}{c.get('fast', 20)} {('%.2f' % a) if a else '—'} vs {pfx}{c.get('slow', 50)} {('%.2f' % b) if b else '—'}"
    if f == "htf":
        e = s.htf_trend(str(c.get("tf", "H4")), int(c.get("n", 50)))[i]
        return f"{c.get('tf', 'H4')} EMA{c.get('n', 50)} {('%.2f' % e) if e else '—'} vs 收盘 {px:.2f}"
    if f == "rsi":
        v = s.rsi[i]
        return f"RSI14 {('%.1f' % v) if v is not None else '—'}"
    if f == "boll":
        up, dn = s.bo_up[i], s.bo_dn[i]
        if up is None or dn is None or up <= dn:
            return "布林未就绪"
        return f"位置 {(px - dn) / (up - dn) * 100:.0f}%"
    if f == "mom":
        n = int(c.get("n", 3))
        return f"{((px - s.close[i - n]) / atr):+.2f} ATR" if atr and i >= n else "—"
    if f == "consec":
        return "见 K 线实体"
    if f == "hour":
        return f"服务器 {s.hour(i):02d} 时"
    return ""


# ------------------------------------------------------------------ 回测
def load_history(symbol: str, tf: str) -> list[list]:
    import backtest as bt
    return bt.load(symbol, tf)


def run(spec: dict, rows: list[list] | None = None, date_from: str = "",
        date_to: str = "", daily_stop: bool = True, series: "Series | None" = None,
        lo: int = 0, hi: int | None = None) -> dict:
    """逐根回测：第 i 根收盘判定，第 i+1 根开盘成交；同根内先判止损（悲观）。

    series 可选：批量搜索时把同一份指标的 Series 传进来，省掉每轮重建。
    lo/hi 可选：按 bar 下标切片（walk-forward 用，比按日期切更精确）。
    """
    sp = _normalize(spec)
    rows = rows if rows is not None else load_history(sp["symbol"], sp["tf"])
    if not rows:
        return {"ok": False, "error": f"没有 {sp['symbol']} {sp['tf']} 的历史数据"}
    s = series if series is not None else Series(rows, sp["symbol"])
    c = COST.get(sp["symbol"], DEFAULT_COST)
    spread, slip, contract = c["spread"], c["slip"], c["contract"]

    eq = EQ0
    peak = eq
    mdd = 0.0
    mdd_at = ""
    trades: list[dict] = []
    skipped_risk = 0
    day_pnl: dict[str, float] = {}
    stopped: set[str] = set()
    pos = None
    n = len(rows)
    hi = n - 1 if hi is None else max(1, min(n - 1, int(hi)))
    i = max(60, 1, int(lo))
    while i < hi:
        r = rows[i]
        day = str(r[0])[:10]
        if date_to and day > date_to:
            break
        if date_from and day < date_from:
            i += 1
            continue
        atr = s.atr[i]
        if not atr:
            i += 1
            continue

        if pos is None:
            if daily_stop and (day in stopped or day_pnl.get(day, 0.0) <= -eq * 2.0 / 100.0):
                stopped.add(day)
                i += 1
                continue
            side = entry_side(s, i, sp)
            if not side:
                i += 1
                continue
            sl_k = float(sp["exit"]["sl_atr"])
            sl_dist = atr * sl_k if sl_k > 0 else 0.0
            tp_dist = atr * float(sp["exit"]["tp_atr"]) if sp["exit"]["tp_atr"] > 0 else 0.0
            # 没给止损时用 3 倍 ATR 兜底算手数，但不挂止损（规格里 sl_atr=0 表示不带止损）
            sizing_dist = sl_dist or atr * 3.0
            lots = math.floor((eq * sp["risk"]["risk_pct"] / 100.0) /
                              (sizing_dist * contract) / 0.01) * 0.01
            lots = round(min(lots, sp["risk"]["max_lots"]), 2)
            if lots < 0.01:
                skipped_risk += 1
                i += 1
                continue
            entry = rows[i + 1][1] + (spread / 2 + slip) * (1 if side == "buy" else -1)
            pos = {"side": side, "entry": entry, "lots": lots, "i": i, "t": str(r[0]),
                   "sl": (entry - sl_dist) if (side == "buy" and sl_dist) else
                         ((entry + sl_dist) if sl_dist else 0.0),
                   "tp": (entry + tp_dist) if (side == "buy" and tp_dist) else
                         ((entry - tp_dist) if tp_dist else 0.0),
                   "atr": atr, "risk0": eq * sp["risk"]["risk_pct"] / 100.0}
            i += 1
            continue

        # 持仓管理：先判硬止损/止盈，再看条件出场
        h, l, cl = r[2], r[3], r[4]
        exit_px, reason = None, ""
        if pos["side"] == "buy":
            if pos["sl"] and l <= pos["sl"]:
                exit_px, reason = pos["sl"], "SL"
            elif pos["tp"] and h >= pos["tp"]:
                exit_px, reason = pos["tp"], "TP"
            elif sp["exit"]["channel_n"]:
                k = int(sp["exit"]["channel_n"])
                if i - pos["i"] >= k and cl < min(s.low[i - k:i]):
                    exit_px, reason = cl, "CHAN"
        else:
            if pos["sl"] and h >= pos["sl"]:
                exit_px, reason = pos["sl"], "SL"
            elif pos["tp"] and l <= pos["tp"]:
                exit_px, reason = pos["tp"], "TP"
            elif sp["exit"]["channel_n"]:
                k = int(sp["exit"]["channel_n"])
                if i - pos["i"] >= k and cl > max(s.high[i - k:i]):
                    exit_px, reason = cl, "CHAN"
        if exit_px is None and sp["exit"]["max_bars"] and i - pos["i"] >= sp["exit"]["max_bars"]:
            exit_px, reason = cl, "TIME"

        if exit_px is not None:
            exit_px += -(spread / 2) * (1 if pos["side"] == "buy" else -1)
            sign = 1 if pos["side"] == "buy" else -1
            pnl = (exit_px - pos["entry"]) * sign * pos["lots"] * contract
            trades.append({
                "i": pos["i"], "t": pos["t"], "t_out": str(r[0]), "side": pos["side"],
                "entry": round(pos["entry"], 5), "exit": round(exit_px, 5),
                "sl": round(pos["sl"], 5), "tp": round(pos["tp"], 5),
                "lots": pos["lots"], "pnl": round(pnl, 2),
                "r": round(pnl / pos["risk0"], 3) if pos["risk0"] else 0,
                "reason": reason, "bars": i - pos["i"],
            })
            eq += pnl
            day_pnl[day] = day_pnl.get(day, 0.0) + pnl
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > mdd:
                mdd, mdd_at = dd, str(r[0])
            pos = None
        i += 1

    if pos is not None:   # 数据到头，按最后收盘平
        exit_px = rows[hi][4]
        sign = 1 if pos["side"] == "buy" else -1
        pnl = (exit_px - pos["entry"]) * sign * pos["lots"] * contract
        trades.append({"i": pos["i"], "t": pos["t"], "t_out": str(rows[hi][0]),
                       "side": pos["side"], "entry": round(pos["entry"], 5),
                       "exit": round(exit_px, 5), "sl": round(pos["sl"], 5),
                       "tp": round(pos["tp"], 5), "lots": pos["lots"],
                       "pnl": round(pnl, 2),
                       "r": round(pnl / pos["risk0"], 3) if pos["risk0"] else 0,
                       "reason": "EOD", "bars": hi - pos["i"]})
        eq += pnl

    st = _stats(trades, eq, mdd, mdd_at, skipped_risk)
    st["ok"] = True
    st["span"] = [str(rows[0][0])[:10], str(rows[-1][0])[:10]]
    st["bars"] = n
    st["bh"] = _bh(rows, date_from, date_to)
    st["spec"] = sp
    st["trades"] = trades
    return st


def _stats(trades: list[dict], eq: float, mdd: float, mdd_at: str, skipped: int) -> dict:
    n = len(trades)
    if not n:
        return {"n": 0, "net": 0.0, "net_pct": 0.0, "win_rate": 0.0, "pf": None,
                "expect_r": 0.0, "expect_usd": 0.0, "mdd": 0.0, "mdd_pct": 0.0,
                "worst_streak": 0, "avg_bars": 0.0, "by_year": {},
                "tp_hits": 0, "sl_hits": 0, "skipped": skipped, "buy": 0, "sell": 0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in losses)
    streak = worst = 0
    for t in trades:
        if t["pnl"] <= 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    by_year: dict[str, dict] = {}
    for t in trades:
        y = t["t"][:4]
        d = by_year.setdefault(y, {"n": 0, "w": 0, "pnl": 0.0})
        d["n"] += 1
        d["w"] += 1 if t["pnl"] > 0 else 0
        d["pnl"] += t["pnl"]
    return {
        "n": n, "buy": sum(1 for t in trades if t["side"] == "buy"),
        "sell": sum(1 for t in trades if t["side"] == "sell"),
        "win_rate": round(len(wins) / n, 4), "pf": round(gp / gl, 2) if gl else None,
        "avg_win": round(gp / len(wins), 2) if wins else 0,
        "avg_loss": round(gl / len(losses), 2) if losses else 0,
        "expect_r": round(sum(t["r"] for t in trades) / n, 3),
        "expect_usd": round(sum(t["pnl"] for t in trades) / n, 2),
        "net": round(eq - EQ0, 2), "net_pct": round((eq / EQ0 - 1) * 100, 2),
        "mdd": round(mdd, 2), "mdd_pct": round(mdd / EQ0 * 100, 2), "mdd_at": mdd_at,
        "worst_streak": worst, "avg_bars": round(sum(t["bars"] for t in trades) / n, 1),
        "tp_hits": sum(1 for t in trades if t["reason"] == "TP"),
        "sl_hits": sum(1 for t in trades if t["reason"] == "SL"),
        "chan_hits": sum(1 for t in trades if t["reason"] == "CHAN"),
        "time_hits": sum(1 for t in trades if t["reason"] == "TIME"),
        "eod": sum(1 for t in trades if t["reason"] == "EOD"),
        "skipped": skipped,
        "by_year": {y: {"n": v["n"], "wr": round(v["w"] / v["n"], 3), "pnl": round(v["pnl"], 0)}
                    for y, v in sorted(by_year.items())},
    }


def _bh(rows: list[list], d0: str = "", d1: str = "") -> dict:
    seg = [r for r in rows if (not d0 or str(r[0])[:10] >= d0) and (not d1 or str(r[0])[:10] <= d1)]
    if len(seg) < 2:
        return {"ret_pct": 0.0, "mdd_pct": 0.0}
    p0, p1 = seg[0][4], seg[-1][4]
    peak, mdd = p0, 0.0
    for r in seg:
        peak = max(peak, r[4])
        mdd = max(mdd, (peak - r[4]) / peak if peak else 0)
    return {"ret_pct": round((p1 / p0 - 1) * 100, 1), "mdd_pct": round(mdd * 100, 1),
            "p0": p0, "p1": p1}


# ------------------------------------------------------------------ 实盘判定
def _live_bars(symbol: str, tf: str) -> list[list]:
    import market as mk
    d = mk.payload(symbol, tf, 500)
    if not d.get("ok"):
        return []
    return d["bars"]


def decide(spec: dict | None = None) -> dict:
    """实盘一次判定，返回结构与 ai_trader.decide 对齐（含 parts，供蛛网图显示）。"""
    import ai_trader as ai

    sp = _normalize(spec or active() or {})
    if not sp["entry"]["long"] and not sp["entry"]["short"]:
        return {"ok": False, "blocked": True, "error": "策略没有入场条件"}
    acct = ai.account()
    if not acct.get("ok"):
        return {"ok": False, "blocked": True, "error": "拿不到账户信息（MCP 离线？）"}
    if str(acct.get("type")) != "demo":
        return {"ok": False, "blocked": True, "error": f"账户类型 {acct.get('type')}，只允许 demo"}

    rows = _live_bars(sp["symbol"], sp["tf"])
    if len(rows) < 80:
        return {"ok": False, "blocked": True,
                "error": f"{sp['symbol']} {sp['tf']} K 线不足（{len(rows)} 根）"}
    s = Series(rows, sp["symbol"])
    i = len(rows) - 1
    atr = s.atr[i] or 0
    px = s.close[i]
    status = cond_status(s, i, sp)

    f = {"ok": True, "symbol": sp["symbol"], "tf": sp["tf"], "price": px,
         "atr": atr, "bar_time": str(rows[i][0]),
         "bid": px, "ask": px, "live": False,
         "server_hour": s.hour(i)}
    parts = [{"name": c["label"][:10], "score": (100 if c["ok"] else -100),
              "why": c["why"], "ok": c["ok"]} for c in status[:5]]
    side = entry_side(s, i, sp)

    # 已有持仓：只做「条件出场」，不重复开仓
    try:
        poss = ai.ai_positions()
    except Exception:  # noqa: BLE001
        poss = []
    if poss:
        p = poss[0]
        out = None
        k = int(sp["exit"].get("channel_n") or 0)
        if k and i >= k:
            if str(p.get("type")).lower().startswith("buy") or p.get("type") == 0:
                if px < min(s.low[i - k:i]):
                    out = "跌破通道出场"
            else:
                if px > max(s.high[i - k:i]):
                    out = "突破通道出场"
        return {"ok": True, "hold": True, "features": f,
                "signal": {"score": 100 if side == "buy" else (-100 if side == "sell" else 0),
                           "side": side or "none", "parts": parts,
                           "verdict": ("持仓中 · " + out) if out else "持仓中，不重复开仓"},
                "plan": {"ok": False, "error": "已有持仓"},
                "close": {"ticket": p.get("ticket"), "reason": out} if out else None,
                "strategy": sp["name"]}

    verdict = ("条件成立 → 做多" if side == "buy" else "条件成立 → 做空" if side == "sell"
               else "条件未全部成立 → 观望")
    signal = {"score": 100 if side == "buy" else (-100 if side == "sell" else 0),
              "side": side or "none", "parts": parts, "verdict": verdict,
              "score_mode": "strategy"}
    if not side:
        return {"ok": True, "hold": False, "features": f, "signal": signal,
                "plan": {"ok": False, "no_trade": True,
                         "error": "；".join(f"{c['label']} {'✓' if c['ok'] else '✗'}"
                                           for c in status)},
                "strategy": sp["name"]}

    eq = float(acct.get("equity") or 0)
    sl_k = float(sp["exit"]["sl_atr"])
    tp_k = float(sp["exit"]["tp_atr"])
    sl_dist = atr * sl_k
    tp_dist = atr * tp_k
    contract = COST.get(sp["symbol"], DEFAULT_COST)["contract"]
    sizing = sl_dist or atr * 3.0
    lots = math.floor((eq * sp["risk"]["risk_pct"] / 100.0) / (sizing * contract) / 0.01) * 0.01
    lots = round(min(lots, sp["risk"]["max_lots"]), 2)
    if lots < 0.01:
        return {"ok": True, "hold": False, "features": f, "signal": signal,
                "plan": {"ok": False, "no_trade": True,
                         "error": f"风险预算不够最小手数（{eq * sp['risk']['risk_pct'] / 100:.0f} USD）"},
                "strategy": sp["name"]}
    dig = 2 if px >= 100 else 5
    spread = COST.get(sp["symbol"], DEFAULT_COST)["spread"]
    entry = px + (spread / 2 if side == "buy" else -spread / 2)
    sl = (entry - sl_dist) if (side == "buy" and sl_dist) else \
         ((entry + sl_dist) if sl_dist else 0.0)
    tp = (entry + tp_dist) if (side == "buy" and tp_dist) else \
         ((entry - tp_dist) if tp_dist else 0.0)
    risk_usd = lots * sizing * contract
    plan = {"ok": True, "symbol": sp["symbol"], "side": side, "volume": lots,
            "entry": round(entry, dig), "sl": round(sl, dig), "tp": round(tp, dig),
            "sl_dist": round(sl_dist, 2), "tp_dist": round(tp_dist, 2),
            "risk_usd": round(risk_usd, 2),
            "risk_pct": round(risk_usd / eq * 100, 2) if eq else 0,
            "rr": round(tp_dist / sl_dist, 2) if sl_dist else None,
            "comment": sp.get("comment") or "AIDesk",
            "note": f"{sp['name']}：{lots:.2f} 手，止损 {sl_dist:.2f}，风险 {risk_usd:.0f} USD"}
    return {"ok": True, "hold": False, "features": f, "signal": signal, "plan": plan,
            "strategy": sp["name"]}


# ------------------------------------------------------------------ 存取
def _read(p: Path, dv):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return dv


def _audit(row: dict) -> None:
    try:
        row = {"ts": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **row}
        with AUDIT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


def active() -> dict | None:
    d = _read(ACTIVE, None)
    if not d or not isinstance(d, dict) or not d.get("spec"):
        return None
    if not d.get("on"):
        return None
    return _normalize(d["spec"])


def activate(spec: dict, on: bool = True) -> dict:
    sp = _normalize(spec)
    if not sp["entry"]["long"] and not sp["entry"]["short"]:
        return {"ok": False, "error": "策略没有可执行的入场条件，不能启用"}
    ACTIVE.write_text(json.dumps({"on": bool(on), "spec": sp,
                                  "at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                                 ensure_ascii=False, indent=1), "utf-8")
    saved = _read(SAVED, [])
    if isinstance(saved, list):
        saved = [x for x in saved if x.get("name") != sp["name"]][:19]
        saved.insert(0, {"name": sp["name"], "spec": sp, "at": dt.datetime.now().strftime("%m-%d %H:%M")})
        SAVED.write_text(json.dumps(saved, ensure_ascii=False, indent=1), "utf-8")
    _audit({"action": "activate", "name": sp["name"], "symbol": sp["symbol"], "tf": sp["tf"]})
    return {"ok": True, "spec": sp}


def deactivate(reason: str = "手动停用") -> dict:
    ACTIVE.write_text(json.dumps({"on": False, "spec": {}, "at": ""}, ensure_ascii=False), "utf-8")
    _audit({"action": "deactivate", "reason": reason})
    return {"ok": True, "on": False}


def status() -> dict:
    a = _read(ACTIVE, {}) or {}
    saved = _read(SAVED, [])
    sp = a.get("spec") or None
    return {"ok": True, "on": bool(a.get("on")) and bool(sp),
            "spec": sp, "at": a.get("at") or "",
            "saved": saved if isinstance(saved, list) else [],
            "factors": [{"f": "brk", "label": "突破 N 根高/低点"},
                        {"f": "ma", "label": "站上/跌破 MA·EMA"},
                        {"f": "cross", "label": "均线金叉/死叉"},
                        {"f": "rsi", "label": "RSI14 阈值"},
                        {"f": "adx", "label": "ADX 趋势强度过滤"},
                        {"f": "marel", "label": "EMA 快慢线定方向"},
                        {"f": "htf", "label": "大周期（H4/D1）顺向过滤"},
                        {"f": "boll", "label": "布林带位置"},
                        {"f": "conj", "label": "连涨/连跌根数"},
                        {"f": "hour", "label": "交易时段"}]}


if __name__ == "__main__":
    import pprint
    import sys
    txt = " ".join(sys.argv[1:]) or "黄金1小时，突破20根高点做多，跌破10根低点平仓，止损2个ATR"
    r = parse(txt)
    pprint.pprint({k: v for k, v in r.items() if k != "spec"})
    print("\n--- 回测 ---")
    st = run(r["spec"])
    print({k: v for k, v in st.items() if k not in ("trades", "spec")})
