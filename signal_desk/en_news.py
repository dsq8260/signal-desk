"""英文财经快讯（WSJ Markets RSS）—— 给本地 FinBERT 提供可打分的英文标题。

边界：
  * 只抓公开 RSS 标题 + 时间 + 链接，不编造、不缓存超过 5 分钟。
  * 只保留与黄金/美元/利率/避险相关的标题，避免把无关公司新闻塞给模型。
  * 抓不到就返回 ok=false + 原因，页面显示「模型降级」，不伪造结果。
  * 与中文快讯（news.py）完全分开，两套来源不混算。
"""
from __future__ import annotations

import datetime as dt
import email.utils
import json
import pathlib
import re
import threading
import urllib.request
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "data" / "en_news_cache.json"
FEED = "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"
TTL = 300.0
TOPIC = re.compile(
    r"gold|silver|xau|comex|dollar|fed\b|federal reserve|rate\b|rates\b|inflation|cpi\b|"
    r"treasury|yield|safe haven|safe-haven|yen\b|franc\b|geopolit|tariff|jobs\b|"
    r"payroll|recession|central bank|oil\b|crude|bullion|metals", re.I)
_lock = threading.Lock()
_cache: dict = {"at": 0.0, "data": None}


def _fetch() -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(FEED, headers={
        "User-Agent": "Mozilla/5.0 (compatible; SignalDesk/1.0; local research)",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    })
    with opener.open(req, timeout=10) as r:
        raw = r.read(300000)
    root = ET.fromstring(raw)
    out = []
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        if not title or not TOPIC.search(title):
            continue
        pub = (it.findtext("pubDate") or "").strip()
        hm = ""
        if pub:
            try:
                d = email.utils.parsedate_to_datetime(pub)
                hm = d.astimezone().strftime("%m-%d %H:%M")
            except (TypeError, ValueError, OverflowError):
                hm = ""
        out.append({"title": title, "url": (it.findtext("link") or "").strip(),
                    "hm": hm, "topic": "英文财经"})
        if len(out) >= 12:
            break
    return {"ok": True, "source": "WSJ Markets RSS", "n": len(out), "items": out,
            "at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": "只保留黄金/美元/利率/避险相关英文标题；文字情绪不等于黄金涨跌方向"}


def _read_file_cache() -> dict | None:
    """冷启动 / 抓失败时读上次成功结果；只当「旧闻」用，页面会标 stale。"""
    try:
        d = json.loads(CACHE.read_text("utf-8"))
        return d if isinstance(d, dict) and d.get("items") else None
    except (OSError, ValueError, TypeError):
        return None


def digest(force: bool = False) -> dict:
    """带文件缓存的只读摘要；失败时保留上次成功结果并标记 stale。"""
    now = dt.datetime.now().timestamp()
    with _lock:
        if not force and _cache["data"] and now - _cache["at"] < TTL:
            return dict(_cache["data"])
        try:
            data = _fetch()
            try:
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                CACHE.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
            except OSError:
                pass
            _cache["at"], _cache["data"] = now, data
            return dict(data)
        except Exception as e:  # noqa: BLE001
            err = "%s: %s" % (type(e).__name__, str(e)[:160])
            stale = dict(_cache["data"]) if _cache["data"] else None
            if not stale:
                f = _read_file_cache()
                if f:
                    stale = dict(f)
            if stale:
                stale["stale"] = True
                stale["error"] = err
                # 抓失败后 60 秒内不再重复联网，避免把 HTTP 请求拖成 10 秒超时
                _cache["at"], _cache["data"] = now - TTL + 60, stale
                return stale
            return {"ok": False, "source": FEED, "n": 0, "items": [],
                    "error": err,
                    "note": "英文源本次没取到，FinBERT 降级为未启用"}
