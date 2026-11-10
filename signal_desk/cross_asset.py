# -*- coding: utf-8 -*-
"""跨品种联动快照：美元指数 / 美债10Y / 白银 与黄金的联动与背离。

数据源（全部免费、无 key）：
- 新浪财经 hq.sinajs.cn：美元指数(DINIW)、伦敦银(hf_XAG)、纽约金(hf_GC) 实时快照
- 美国财政部 home.treasury.gov：国债收益率日频 CSV（10Y）

只做联动观察（同向/背离/金银比/利率方向），不给买卖方向。
"""
import csv
import io
import json
import time
import urllib.request

TIMEOUT = 12

_cache: dict = {"at": 0.0, "data": None}
_ust_cache: dict = {"at": 0.0, "data": None}


def _opener():
    # 不吃系统代理：本机代理常把财经接口拦成 502
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(url: str, referer: str = "https://finance.sina.com.cn") -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "*/*",
        "Referer": referer,
    })
    with _opener().open(req, timeout=TIMEOUT) as r:
        return r.read()


def _sina_spot() -> dict:
    """新浪实时快照。返回 {dxy, silver, gold} 各含 last/chg_pct/prev/last。"""
    raw = _get("https://hq.sinajs.cn/list=DINIW,hf_XAG,hf_GC").decode("gbk", "replace")
    out: dict = {}
    for line in raw.splitlines():
        if "=" not in line:
            continue
        key = line.split("=", 1)[0].replace("var hq_str_", "").strip()
        val = line.split("=", 1)[1].strip().strip('";').split(",")
        out[key] = val
    def f(x, i):
        try:
            return float(x[i])
        except (ValueError, IndexError, TypeError):
            return None
    dxy = out.get("DINIW") or []          # [1]最新 [3]昨收
    hf_xag = out.get("hf_XAG") or []      # [0]最新 [7]昨结
    hf_gc = out.get("hf_GC") or []
    def pct(last, prev):
        if last is None or prev in (None, 0):
            return None
        return round((last - prev) / prev * 100, 2)
    return {
        "dxy": {"last": f(dxy, 1), "prev": f(dxy, 3), "chg_pct": pct(f(dxy, 1), f(dxy, 3)),
                "name": "美元指数"},
        "silver": {"last": f(hf_xag, 0), "prev": f(hf_xag, 7),
                   "chg_pct": pct(f(hf_xag, 0), f(hf_xag, 7)), "name": "伦敦银"},
        "gold": {"last": f(hf_gc, 0), "prev": f(hf_gc, 7),
                 "chg_pct": pct(f(hf_gc, 0), f(hf_gc, 7)), "name": "纽约金(代理)"},
    }


def _ust10y() -> dict:
    """美国财政部 10Y 日频。缓存 6 小时（日频数据）。"""
    if _ust_cache["data"] and time.time() - _ust_cache["at"] < 6 * 3600:
        return _ust_cache["data"]
    year = time.strftime("%Y")
    url = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
           f"daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
           f"&field_tdr_date_value={year}&page&_format=csv")
    raw = _get(url, referer="https://home.treasury.gov/").decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(raw)))
    rows.sort(key=lambda r: time.mktime(time.strptime(r["Date"], "%m/%d/%Y")), reverse=True)
    if len(rows) < 2:
        raise ValueError("treasury rows < 2")
    y10 = [r["10 Yr"] for r in rows]
    last, prev = float(y10[0]), float(y10[1])
    chg_bp = round((last - prev) * 100, 1)
    d = {"last": last, "chg_bp": chg_bp, "date": rows[0]["Date"],
         "prev_date": rows[1]["Date"]}
    _ust_cache.update({"at": time.time(), "data": d})
    return d


def _flags(gold_chg, dxy_chg, ratio, ust) -> list[dict]:
    """规则化联动提示：只描述事实，不给方向。"""
    out = []
    if gold_chg is not None and dxy_chg is not None:
        if gold_chg > 0.15 and dxy_chg > 0.15:
            out.append({"level": "WARN", "text":
                f"黄金 {gold_chg:+.2f}% 与美元 {dxy_chg:+.2f}% 同涨——负相关常态被打破，"
                "可能是避险与美元双强，注意黄金涨势的成色"})
        elif gold_chg < -0.15 and dxy_chg < -0.15:
            out.append({"level": "WARN", "text":
                f"黄金 {gold_chg:+.2f}% 与美元 {dxy_chg:+.2f}% 同跌——可能与收益率上行共振，"
                "留意实际利率方向"})
        elif abs(gold_chg) > 0.15:
            out.append({"level": "OK", "text":
                f"黄金 {gold_chg:+.2f}% 与美元 {dxy_chg:+.2f}% 反向，负相关关系正常"})
    if ratio is not None:
        if ratio >= 85:
            out.append({"level": "WARN", "text":
                f"金银比 {ratio:.1f}（≥85）：白银相对黄金被明显抛售，历史上多出现在避险单边段"})
        elif ratio <= 75:
            out.append({"level": "WARN", "text":
                f"金银比 {ratio:.1f}（≤75）：白银相对黄金明显走强，工业/投机属性占上风"})
        else:
            out.append({"level": "OK", "text": f"金银比 {ratio:.1f}，处于常规区间 75–85"})
    if ust and ust.get("chg_bp") is not None:
        bp = ust["chg_bp"]
        if abs(bp) >= 5:
            out.append({"level": "WARN" if bp > 0 else "OK", "text":
                f"美债10Y 日变动 {bp:+.1f}bp（{ust.get('date','')}）："
                + ("收益率上行，持金机会成本抬升" if bp > 0 else "收益率回落，对黄金偏友好")})
    return out


def snapshot(force: bool = False) -> dict:
    """对外入口。带 120s 缓存；单源失败不炸整个接口。"""
    if not force and _cache["data"] and time.time() - _cache["at"] < 120:
        return _cache["data"]
    errs, spot, ust = [], None, None
    try:
        spot = _sina_spot()
    except Exception as e:
        errs.append("新浪行情: " + str(e)[:80])
    try:
        ust = _ust10y()
    except Exception as e:
        errs.append("美债CSV: " + str(e)[:80])
    ratio = None
    if spot and spot.get("gold", {}).get("last") and spot.get("silver", {}).get("last"):
        try:
            ratio = round(spot["gold"]["last"] / spot["silver"]["last"], 1)
        except (TypeError, ZeroDivisionError):
            ratio = None
    flags = _flags(
        spot.get("gold", {}).get("chg_pct") if spot else None,
        spot.get("dxy", {}).get("chg_pct") if spot else None,
        ratio, ust) if (spot or ust) else []
    data = {
        "ok": bool(spot or ust), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spot": spot, "ust": ust, "ratio": ratio, "flags": flags,
        "errors": errs,
        "src": "新浪财经(实时) + 美国财政部(日频) · 只读快照",
    }
    if spot:
        _cache.update({"at": time.time(), "data": data})
    return data


if __name__ == "__main__":
    print(json.dumps(snapshot(True), ensure_ascii=False, indent=1)[:1200])
