"""品种相关新闻 + 影响程度判定（只读，本地缓存）。

数据源
  主：华尔街见闻快讯 API（gold-channel / forex-channel / global-channel，中文实时）
  备：新浪财经滚动新闻（主源挂了才用）
  两者都不通时，退回 data/news_cache.json 里的上一份，并标 stale，绝不编新闻。

每条新闻给出三件事
  impact    高/中/低 —— 决定它值不值得打断你
  direction 利多/利空/中性 —— 对黄金（或该品种）的方向倾向
  topic     题材（美联储/通胀/就业/地缘/美元/黄金供需…）

判定是关键词规则，不是语义理解：只看「降息/加息/避险/战争/美元走弱」这类
明确措辞，不去猜「超预期」这种需要上下文的说法。宁可标中性，也不假装看懂了。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
CACHE_FILE = DATA / "news_cache.json"

TTL = 120.0          # 同一次抓取复用的秒数
TIMEOUT = 8.0
MAX_ITEMS = 60

# 题材：权重(1低 2中 3高) + 正则 + 是否属于「会突然放大波动」的事件
TOPICS = [
    (3, "美联储", r"美联储|FOMC|利率决议|鲍威尔|点阵图|加息|降息|缩表|量化宽松|QE|联邦基金|货币政策"),
    (3, "通胀", r"CPI|PCE|通胀|物价|PPI|消费者价格|通胀预期"),
    (3, "就业", r"非农|ADP|就业人口|失业率|初请|续请|裁员|职位空缺"),
    (3, "地缘", r"战争|空袭|袭击|导弹|冲突|制裁|封锁|政变|军事|打击|爆炸|恐袭|胡塞|俄乌|中东|以色列|伊朗|台海|南海"),
    (2, "增长", r"GDP|PMI|零售销售|工业产出|消费者信心|耐用品|贸易帐|经常帐|景气"),
    (2, "美元利率", r"美元指数|美债|国债收益率|收益率|美元走强|美元走弱|美元上涨|美元下跌"),
    (2, "黄金供需", r"黄金|金价|XAU|贵金属|央行购金|黄金储备|ETF|SPDR|金矿|实物金"),
    (1, "贸易能源", r"关税|贸易|OPEC|原油|库存|能源|油价|大宗"),
]

# 方向词：只认明确表态，不认「超预期」这类要看语境的词
BULL = [
    (2, r"降息|鸽派|宽松|停止加息|暂停加息"),
    (2, r"避险|risk off|资金流入|增持|ETF流入|央行购金|购金"),
    (2, r"战争|空袭|袭击|冲突升级|制裁|封锁|军事打击|恐袭|爆炸"),
    (1, r"美元走弱|美元下跌|美元回落|收益率下行|收益率下降|美债收益率下跌"),
    (1, r"通胀升温|通胀走高|衰退|违约|危机|债务上限|银行业|爆雷"),
]
BEAR = [
    (2, r"加息|鹰派|紧缩|缩减购债|收紧"),
    (2, r"停火|和谈|缓和|达成协议|解除制裁|risk on|风险偏好回升"),
    (1, r"美元走强|美元上涨|美元反弹|收益率上行|收益率上升|美债收益率上涨"),
    (1, r"减持|ETF流出|获利了结|抛售黄金|黄金抛售"),
    (1, r"通胀回落|通胀降温|通胀放缓|软着陆|经济强劲|就业强劲"),
]

_TOPIC_RE = [(w, name, re.compile(pat, re.I)) for w, name, pat in TOPICS]
_BULL_RE = [(w, re.compile(p, re.I)) for w, p in BULL]
_BEAR_RE = [(w, re.compile(p, re.I)) for w, p in BEAR]

# 品种 → 相关性关键词（不在里面、又不是高影响的，就不往该品种的新闻里塞）
SYM_TAGS = {
    "GOLD": r"黄金|金价|XAU|贵金属|美元指数|美元走强|美元走弱|美元上涨|美元下跌|美元回落|"
            r"美债|收益率|美联储|通胀|CPI|PCE|非农|ADP|就业|避险|地缘|战争|冲突|制裁|"
            r"央行购金|ETF|降息|加息|鲍威尔|FOMC",
    "SILVER": r"白银|银价|XAG|贵金属|黄金|美元|美联储|通胀|工业需求",
    "EURUSD": r"欧元|欧洲央行|ECB|拉加德|欧元区|美元|美联储|德国|法国",
    "GBPUSD": r"英镑|英国央行|BOE|贝利|英国|美元|美联储",
    "USDJPY": r"日元|日本央行|BOJ|植田|日本|美元|美联储|收益率",
    "USDCHF": r"瑞郎|瑞士央行|SNB|瑞士|美元|避险",
    "AUDUSD": r"澳元|澳洲联储|RBA|澳大利亚|中国|铁矿石|美元",
    "USDCAD": r"加元|加拿大央行|BOC|加拿大|原油|美元",
    "NZDUSD": r"纽元|新西兰联储|RBNZ|新西兰|美元",
    "USOIL": r"原油|油价|OPEC|库存|能源|中东|制裁",
    "BTCUSD": r"比特币|BTC|加密|数字资产|风险偏好|美元",
}

_lock = threading.Lock()
_cache: dict = {"at": 0.0, "items": [], "errors": [], "src": ""}


# ---------------------------------------------------------------- 抓取
def _opener():
    # 不吃系统代理：本机代理常把境外财经接口拦成 502
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
    })
    with _opener().open(req, timeout=TIMEOUT) as r:
        return r.read()


def _wscn(channel: str, limit: int = MAX_ITEMS) -> list[dict]:
    url = ("https://api-one-wscn.awtmt.com/apiv1/content/lives"
           f"?channel={channel}&limit={limit}")
    d = json.loads(_get(url).decode("utf-8", "replace"))
    out = []
    for it in (d.get("data") or {}).get("items") or []:
        txt = (it.get("content_text") or it.get("title") or "").strip()
        if not txt:
            continue
        out.append({
            "id": f"wscn-{it.get('id')}",
            "text": txt,
            "ts": int(it.get("display_time") or 0),
            "url": it.get("uri") or "",
            "src": "华尔街见闻",
        })
    return out


def _sina(limit: int = 40) -> list[dict]:
    url = ("https://feed.mix.sina.com.cn/api/roll/get"
           f"?pageid=153&lid=2516&num={limit}")
    d = json.loads(_get(url).decode("utf-8", "replace"))
    out = []
    for it in (d.get("result") or {}).get("data") or []:
        txt = (it.get("title") or "").strip()
        intro = (it.get("intro") or "").strip()
        if not txt:
            continue
        out.append({
            "id": "sina-" + str(it.get("oid") or it.get("docid") or txt[:20]),
            "text": txt + (("　" + intro) if intro and intro != txt else ""),
            "ts": int(it.get("ctime") or 0),
            "url": it.get("url") or "",
            "src": "新浪财经",
        })
    return out


def _fetch_raw(symbol: str) -> tuple[list[dict], list[str]]:
    sym = _norm_sym(symbol)
    chans = ["gold-channel", "forex-channel"] if sym == "GOLD" else \
        ["forex-channel"]
    items, errors = [], []
    seen = set()
    for ch in chans:
        try:
            for it in _wscn(ch):
                if it["id"] in seen:
                    continue
                seen.add(it["id"])
                items.append(it)
        except Exception as e:  # noqa: BLE001
            errors.append(f"华尔街见闻 {ch}：{type(e).__name__}")
    if len(items) < 8:
        try:
            for it in _sina():
                if it["id"] in seen:
                    continue
                seen.add(it["id"])
                items.append(it)
        except Exception as e:  # noqa: BLE001
            errors.append(f"新浪财经：{type(e).__name__}")
    return items, errors


# ---------------------------------------------------------------- 判定
def _norm_sym(symbol: str) -> str:
    s = (symbol or "GOLD").upper().replace("#", "").replace("/", "").strip()
    if s.startswith("XAU") or s.startswith("GOLD") or s == "GC":
        return "GOLD"
    if s.startswith("XAG") or s.startswith("SILVER"):
        return "SILVER"
    for k in ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
              "NZDUSD", "USOIL", "BTCUSD"):
        if s.startswith(k[:3]):
            return k
    return s or "GOLD"


def _topics(text: str) -> list[tuple[int, str]]:
    hit = [(w, n) for w, n, rx in _TOPIC_RE if rx.search(text)]
    return hit


def _direction(text: str) -> tuple[str, int, list[str]]:
    b = sum(w for w, rx in _BULL_RE if rx.search(text))
    s = sum(w for w, rx in _BEAR_RE if rx.search(text))
    words = [m.group(0) for _, rx in _BULL_RE + _BEAR_RE for m in [rx.search(text)] if m]
    if b == 0 and s == 0:
        return "none", 0, []
    if b > s:
        return "bull", b - s, words[:3]
    if s > b:
        return "bear", s - b, words[:3]
    return "none", 0, words[:3]


def _impact(text: str, topics: list[tuple[int, str]]) -> int:
    """影响程度：高=会立刻改变价格预期，中=改变节奏，低=背景噪音。

    只用两类硬信号给「高」：① 央行/通胀/就业的决策级信息；② 突发且有市场传导
    （提到原油/黄金/美元/封锁/制裁/全面战争）。一地伤亡数字这类没人接盘的
    地缘新闻最多算中——否则一天能刷出二十多条「高影响」，等于没有高影响。
    """
    names = {n for _, n in topics}
    if not topics:
        return 1
    high_core = re.search(
        r"利率决议|FOMC|美联储主席|鲍威尔|点阵图|加息|降息|缩表|非农|CPI|PCE|"
        r"通胀数据|ADP|美国.{0,6}失业率|初请|突发|重磅|紧急|官宣", text, re.I)
    high_geo = re.search(r"全面战争|战争升级|封锁|制裁|导弹|空袭|袭击|军事打击|地面进攻", text) and \
        re.search(r"原油|油价|黄金|金价|美元|美债|避险|霍尔木兹|红海航运|供应中断", text)
    if high_core or high_geo:
        return 3
    if names & {"美联储", "通胀", "就业", "地缘", "黄金供需"} or len(topics) >= 2:
        return 2
    return 1


def _title(text: str, limit: int = 92) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    m = re.match(r"^【(.+?)】", t)
    if m:
        return m.group(1)[:limit]
    # 没有标题格式：取第一句
    t = re.split(r"[。！？；]", t)[0]
    return (t[:limit] + "…") if len(t) > limit else t


NOISE = re.compile(r"恒生|上证|深证|创业板|科创板|涨停|跌停|个股|A股|港股|公告|"
                   r"股份|集团|财报|营收|毛利率|销售收入|中标|回购|板块|更名|人事变动", re.I)


def _noise(text: str) -> bool:
    return bool(NOISE.search(text))


def _relevance(text: str, sym: str) -> bool:
    pat = SYM_TAGS.get(sym)
    return True if not pat else bool(re.search(pat, text, re.I))


def classify(raw: list[dict], sym: str, hours: float) -> dict:
    now = time.time()
    out, skipped = [], 0
    for it in raw:
        text = it.get("text") or ""
        age_min = max(0.0, (now - (it.get("ts") or now)) / 60.0)
        if age_min > hours * 60:
            continue
        if _noise(text):
            skipped += 1
            continue
        topics = _topics(text)
        impact = _impact(text, topics)
        # 三层筛：① 命中品种关键词（黄金/美元/美联储/地缘…）→ 留；
        # ② 央行政/通胀/就业这类硬数据 → 留（就算没提黄金，也会通过美元传导）；
        # ③ 其余（公司财报、外事会见、别国股市）→ 丢。宁可少几条，不塞噪音。
        hard = {"美联储", "通胀", "就业", "黄金供需"} & {n for _, n in topics}
        if not _relevance(text, sym) and impact < 3 and not hard:
            skipped += 1
            continue
        d, strength, words = _direction(text)
        out.append({
            **it,
            "title": _title(text),
            "topic": topics[0][1] if topics else "综合",
            "impact": impact,
            "impact_label": {3: "高", 2: "中", 1: "低"}[impact],
            "direction": d,
            "dir_label": {"bull": "利多", "bear": "利空", "none": "中性"}[d],
            "dir_words": words,
            "age_min": round(age_min, 1),
            "hm": dt.datetime.fromtimestamp(it["ts"]).strftime("%H:%M") if it.get("ts") else "—",
            "date": dt.datetime.fromtimestamp(it["ts"]).strftime("%m-%d") if it.get("ts") else "",
            "strength": strength,
        })
    # 同一句话可能被两个频道各推一次：按标题前 24 字去重（保留较早的那条）
    seen_t, uniq = set(), []
    for x in out:
        k = re.sub(r"\W+", "", x["title"])[:24]
        if k and k in seen_t:
            continue
        seen_t.add(k)
        uniq.append(x)
    out = uniq
    # 清单按时间倒序（新闻要新），另给一份按影响排的「重点」供 AI 和头部使用
    out.sort(key=lambda x: -x["ts"])
    hot = sorted(out, key=lambda x: (-x["impact"], -x["strength"], -x["ts"]))
    # 倾向分：方向强度 × 影响权重 × 时间衰减（2 小时内的算全值，之后逐步衰减）
    score = 0.0
    for x in out:
        if not x["strength"]:
            continue
        decay = 1.0 if x["age_min"] <= 120 else max(0.25, 1 - (x["age_min"] - 120) / 720)
        score += (x["strength"] if x["direction"] == "bull" else -x["strength"]) * x["impact"] * decay
    score = round(max(-10.0, min(10.0, score / 3.0)), 1)
    return {"items": out, "hot": hot, "score": score, "skipped": skipped}


# ---------------------------------------------------------------- 对外
def _save_disk(raw: list[dict], sym: str) -> None:
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(
            {"at": time.time(), "symbol": sym, "items": raw[:120]}, ensure_ascii=False),
            encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _load_disk() -> list[dict]:
    try:
        d = json.loads(CACHE_FILE.read_text("utf-8"))
        return d.get("items") or []
    except Exception:  # noqa: BLE001
        return []


def _refresh(symbol: str) -> dict:
    raw, errors = _fetch_raw(symbol)
    if raw:
        _save_disk(raw, symbol)
        _cache.update({"at": time.time(), "items": raw, "errors": errors})
    return {"raw": raw, "errors": errors}


def digest(symbol: str = "GOLD", hours: float = 24, limit: int = 20,
           force: bool = False) -> dict:
    """给界面用：分类后的新闻清单 + 净倾向 + 是否有「刚发生的冲击」。"""
    sym = _norm_sym(symbol)
    if os.environ.get("SD_NEWS_OFF") == "1":
        return {"ok": False, "error": "新闻模块被 SD_NEWS_OFF 关闭", "items": [], "symbol": sym}
    with _lock:
        fresh = _cache["items"] and (time.time() - _cache["at"] < TTL)
        if force or not fresh:
            _refresh(sym)
    raw = _cache["items"]
    stale = False
    if not raw:
        raw = _load_disk()
        stale = bool(raw)
    if not raw:
        return {"ok": False, "symbol": sym, "items": [], "errors": _cache["errors"] or ["新闻源不可达"],
                "error": "新闻源不可达（本机网络或代理限制）"}
    c = classify(raw, sym, hours)
    items = c["items"][:limit]
    score = c["score"]
    label = "偏多" if score >= 1.5 else ("偏空" if score <= -1.5 else "中性")
    shock = next((x for x in c["items"]
                  if x["impact"] == 3 and x["age_min"] <= 20), None)
    return {
        "ok": True, "symbol": sym, "hours": hours,
        "updated_at": dt.datetime.fromtimestamp(_cache["at"] or time.time()).strftime("%H:%M:%S"),
        "age_s": round(time.time() - _cache["at"], 1) if _cache["at"] else None,
        "stale": stale,
        "n": len(c["items"]), "high": sum(1 for x in c["items"] if x["impact"] == 3),
        "bull": sum(1 for x in c["items"] if x["direction"] == "bull"),
        "bear": sum(1 for x in c["items"] if x["direction"] == "bear"),
        "bias": {"score": score, "label": label},
        "items": [{k: v for k, v in x.items() if k not in ("text",)} for x in items],
        "shock": ({k: shock[k] for k in ("title", "hm", "impact", "dir_label", "topic", "age_min", "url")}
                  if shock else None),
        "errors": _cache["errors"],
    }


def brief(symbol: str = "GOLD", hours: float = 12) -> dict:
    """给 AI 决策用的小包：净倾向 + 前三条 + 冲击事件。"""
    d = digest(symbol, hours=hours, limit=6)
    if not d.get("ok"):
        return {"ok": False, "score": 0.0, "label": "无数据", "top": [], "shock": None,
                "error": d.get("error")}
    return {
        "ok": True,
        "score": d["bias"]["score"], "label": d["bias"]["label"],
        "n": d["n"], "high": d["high"],
        "top": [{"title": x["title"], "impact": x["impact"], "dir": x["direction"],
                 "topic": x["topic"], "hm": x["hm"]} for x in d["items"][:3]],
        "shock": d.get("shock"),
        "stale": d.get("stale"),
    }


if __name__ == "__main__":  # 手工看一眼
    import pprint
    pprint.pprint(digest("GOLD", hours=24, limit=8))
