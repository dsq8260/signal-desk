"""行情数据层 —— 给 03 页「行情看板」供数（纯只读，不连 MT5、不下单）。

数据来源（按优先级）：
  1. MT5 文件桥 —— %APPDATA%/MetaQuotes/Terminal/Common/Files/gold_bars_<SYM>_<TF>.csv
     GoldBridge EA 在终端里实时写出，文件 mtime 随 tick 更新，可当实时源。
     另有 gold_tick_<SYM>.csv（time_msc,bid,ask），用来把最后一根 K 线接到最新价。
  2. 本地快照 —— data/market/<SYM>_H1.csv（离线兜底，只有 H1，H4/D1 由 H1 聚合）。

约定：
  * 只读 CSV，不写任何文件。
  * 全部用标准库，不引入 pandas/numpy，冷启动更快。
  * 单品种全量指标结果按 (symbol, tf, mtime) 缓存，桥文件一变才重算。
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import pathlib
import threading
import time
from bisect import bisect_left

ROOT = pathlib.Path(__file__).resolve().parent.parent
MKT = ROOT / "data" / "market"

# 桥支持的周期（EA 导出的文件名后缀）
TFS = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1")
# 只有 H1 能靠聚合兜底
AGG_FROM_H1 = ("H4", "D1")
TF_SEC = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
          "H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800}
PARAMS = {"ma1": 20, "ma2": 50, "atr": 14, "boll": 20, "kb": 2.0}
MAX_N = 3000        # 单次返回上限（一页）
MIN_N = 40
WARM = 260          # 指标预热：MA50/ATR14 需要的前置根数（别把开头的 None 喂给前端）

_lock = threading.Lock()
# key = (symbol, tf) -> {"key": (mtime, size), "rows": [...]}
_cache: dict = {}
# key = "syms" -> {"ts": float, "rows": [...]}
_sym_cache: dict = {}


def bridge_dir() -> pathlib.Path:
    """桥文件目录：云端用 SD_BRIDGE_DIR（数据中继落地目录），本机用 MT5 公共目录。"""
    env = os.environ.get("SD_BRIDGE_DIR")
    if env:
        return pathlib.Path(env)
    return (pathlib.Path(os.environ.get("APPDATA", pathlib.Path.home()))
            / "MetaQuotes" / "Terminal" / "Common" / "Files")


def safe_name(sym: str) -> str:
    return "".join(ch for ch in str(sym) if ch.isalnum() or ch == "_") or "SYM"


def norm(sym: str) -> str:
    """GOLD# / GOLD 都指向桥里的 GOLD。"""
    return safe_name(str(sym or "").replace("#", "").rstrip(".")).upper()


def _bridge_bars(sym: str, tf: str) -> pathlib.Path | None:
    p = bridge_dir() / f"gold_bars_{norm(sym)}_{tf}.csv"
    return p if p.is_file() else None


def _snapshot_h1(sym: str) -> pathlib.Path | None:
    """本地快照只认 index.json 登记过的文件。"""
    p = MKT / "index.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    for name, meta in (d.get("symbols") or {}).items():
        if norm(name) == norm(sym):
            f = MKT / str(meta.get("file") or "")
            return f if f.is_file() else None
    return None


def _read_csv(path: pathlib.Path) -> list[list]:
    """返回 [[时间字符串, o, h, l, c, v, spread], ...]，跳过残缺行。"""
    rows: list[list] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rd = csv.reader(fh)
            head = next(rd, None)
            if not head:
                return rows
            for r in rd:
                if len(r) < 6:
                    continue
                try:
                    o, h, l, c = float(r[1]), float(r[2]), float(r[3]), float(r[4])
                    v = float(r[5])
                except (TypeError, ValueError):
                    continue
                sp = 0.0
                if len(r) > 6:
                    try:
                        sp = float(r[6])
                    except (TypeError, ValueError):
                        sp = 0.0
                rows.append([r[0].strip(), o, h, l, c, v, sp])
    except Exception:  # noqa: BLE001
        return []
    return rows


def _read_bars_file(path: pathlib.Path) -> list[list]:
    """读主文件 + 分页（gold_bars_X_H1_p1.csv ...），按时间去重排序。

    桥的 time 列是 unix 秒；本地快照是 'YYYY-MM-DD HH:MM:SS' 字符串。
    统一输出 'YYYY-MM-DD HH:MM:SS'。
    """
    frames = [path]
    base = path.with_suffix("")
    frames += sorted(base.parent.glob(f"{base.name}_p*.csv"))
    seen = set()
    out: list[list] = []
    for p in frames:
        for r in _read_csv(p):
            t = r[0]
            try:
                ts = int(float(t))
                r[0] = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            except (TypeError, ValueError):
                pass
            key = r[0]
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    out.sort(key=lambda x: x[0])
    return out


def _agg(rows: list[list], tf: str) -> list[list]:
    """H1 -> H4 / D1（只在没有桥文件时兜底）。"""
    if tf == "H1" or not rows:
        return rows
    out: list[list] = []
    cur_key = None
    cur = None
    for r in rows:
        try:
            d = dt.datetime.strptime(r[0][:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if tf == "H4":
            k = (d.year, d.month, d.day, d.hour // 4)
        else:
            k = (d.year, d.month, d.day)
        if k != cur_key:
            if cur is not None:
                out.append(cur)
            cur_key = k
            cur = [r[0], r[1], r[2], r[3], r[4], r[5], r[6]]
        else:
            cur[2] = max(cur[2], r[2])
            cur[3] = min(cur[3], r[3])
            cur[4] = r[4]
            cur[5] += r[5]
            cur[6] = r[6]
    if cur is not None:
        out.append(cur)
    return out


def _sma(vals: list[float], n: int) -> list[float | None]:
    out: list[float | None] = [None] * len(vals)
    if n <= 0 or len(vals) < n:
        return out
    s = 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= n:
            s -= vals[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def _stdev(vals: list[float], n: int, means: list[float | None]) -> list[float | None]:
    out: list[float | None] = [None] * len(vals)
    for i in range(len(vals)):
        m = means[i]
        if m is None:
            continue
        seg = vals[i - n + 1:i + 1]
        var = sum((x - m) ** 2 for x in seg) / n
        out[i] = var ** 0.5
    return out


def _with_indicators(rows: list[list]) -> list[list]:
    """给每根算出 ma1 / ma2 / boll 上下轨 / atr，追加到行尾。"""
    n = len(rows)
    close = [r[4] for r in rows]
    high = [r[2] for r in rows]
    low = [r[3] for r in rows]

    ma1 = _sma(close, PARAMS["ma1"])
    ma2 = _sma(close, PARAMS["ma2"])
    bmid = _sma(close, PARAMS["boll"])
    bsd = _stdev(close, PARAMS["boll"], bmid)

    tr: list[float] = []
    for i in range(n):
        if i == 0:
            tr.append(high[i] - low[i])
        else:
            pc = close[i - 1]
            tr.append(max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc)))
    atr = _sma(tr, PARAMS["atr"])

    k = PARAMS["kb"]
    out = []
    for i, r in enumerate(rows):
        mid, sd = bmid[i], bsd[i]
        up = None if mid is None or sd is None else mid + k * sd
        dn = None if mid is None or sd is None else mid - k * sd
        out.append([r[0], r[1], r[2], r[3], r[4], r[5], r[6],
                    ma1[i], ma2[i], up, dn, atr[i]])
    return out


def _rd(x, nd=5):
    if x is None:
        return None
    return round(float(x), nd)


# ---------------------------------------------------------------- tick ----
_off_cache: dict = {"mtime": None, "h": 2.0}


def _srv_off_h() -> float:
    """读 assist_config 里的服务器 UTC 偏移（不反向 import assist，避免循环依赖）。"""
    p = MKT.parent / "assist_config.json"
    try:
        st = p.stat()
        if _off_cache["mtime"] != st.st_mtime:
            d = json.loads(p.read_text(encoding="utf-8"))
            _off_cache.update(mtime=st.st_mtime,
                              h=float(d.get("server_utc_offset", 2)))
    except Exception:  # noqa: BLE001
        pass
    return _off_cache["h"]


def latest_tick(sym: str = "GOLD") -> dict | None:
    """读 gold_tick_<SYM>.csv 尾部：{'time_msc','bid','ask','age_s'}。"""
    p = bridge_dir() / f"gold_tick_{norm(sym)}.csv"
    if not p.is_file():
        return None
    try:
        st = p.stat()
        size = st.st_size
        with p.open("rb") as f:
            if size > 8192:
                f.seek(size - 8192)
            raw = f.read().decode("ascii", errors="ignore")
    except Exception:  # noqa: BLE001
        return None
    for line in reversed(raw.splitlines()):
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        try:
            msc = int(parts[0])
            bid = float(parts[1])
            ask = float(parts[2])
        except ValueError:
            continue
        # 桥写的 msc 是「服务器墙钟当成 UTC 纪元」，fromtimestamp 会再叠加本机偏移
        # （服务器 07:24 → 显示 15:24）。减掉服务器 UTC 偏移才是本机时间。
        local_t = (dt.datetime.fromtimestamp(msc / 1000)
                   - dt.timedelta(hours=_srv_off_h()))
        return {"time_msc": msc, "bid": bid, "ask": ask,
                "age_s": round(max(0.0, time.time() - st.st_mtime), 1),
                "tick_time": local_t.strftime("%Y-%m-%d %H:%M:%S")}
    return None


def _digits(price: float) -> int:
    """按价格量级猜小数位：GOLD 41xx -> 2 位，EURUSD 1.08 -> 5 位。"""
    if price >= 100:
        return 2
    if price >= 10:
        return 3
    return 5


def _tail_rows(path: pathlib.Path, k: int = 3) -> list[list]:
    """只读文件尾部，取最后 k 根（顶栏报价用，不解析全文件）。"""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - 16384))
            raw = f.read().decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return []
    out: list[list] = []
    for line in raw.splitlines()[-k - 1:]:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            out.append([parts[0], float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]), float(parts[5])])
        except ValueError:
            continue
    return out


def quotes() -> list[dict]:
    """顶栏实时报价条：每品种最新价 + 当日涨跌。

    价格优先级：新鲜 tick（<=120s）> D1 末根 close > H1 末根 close。
    只有 GOLD 有 tick 文件，其余品种靠 bars 文件 mtime 判活。
    """
    out = []
    for s in symbols():
        sym = s["symbol"]
        tk = latest_tick(sym)
        fresh = bool(tk and tk["age_s"] <= 120)
        p = _bridge_bars(sym, "D1")
        d1 = _tail_rows(p, 2) if p else []
        if not d1:
            d1 = _tail_rows(_bridge_bars(sym, "H1") or pathlib.Path("x"), 2)
        prev_close = d1[-2][4] if len(d1) >= 2 else (d1[-1][4] if d1 else None)
        price = (tk["bid"] if fresh else None) or (d1[-1][4] if d1 else s.get("last"))
        if price is None:
            continue
        dig = _digits(price)
        chg = (price - prev_close) if prev_close else None
        pct = (chg / prev_close * 100) if (prev_close and chg is not None) else None
        bars_age = s.get("bars_age_s")
        out.append({
            "symbol": sym,
            "price": _rd(price, dig),
            "bid": _rd(tk["bid"], dig) if fresh else None,
            "ask": _rd(tk["ask"], dig) if fresh else None,
            "chg": _rd(chg, dig) if chg is not None else None,
            "chg_pct": _rd(pct, 3) if pct is not None else None,
            "tick_time": tk["tick_time"] if fresh else None,
            "tick_age_s": tk["age_s"] if fresh else None,
            "bars_age_s": bars_age,
            "live": fresh or bool(bars_age is not None and bars_age <= 180),
            "source": "tick" if fresh else "bar",
            "digits": dig,
        })
    return out


# ------------------------------------------------------------ symbols ----
def _sym_meta(sym: str) -> dict:
    """品种元信息：只 stat + 读尾部，不解析全量 K 线（顶栏 5 品种要快）。"""
    p = _bridge_bars(sym, "H1")
    if p is None:
        h1 = _snapshot_h1(sym)
        if h1 is None:
            return {"symbol": norm(sym), "available": False, "source": "none"}
        rows = _tail_rows(h1, 1)
        return {"symbol": norm(sym), "file": h1.name, "rows": None,
                "start": None, "end": rows[-1][0] if rows else None,
                "available": bool(rows), "source": "snapshot"}

    try:
        st = p.stat()
    except OSError:
        return {"symbol": norm(sym), "available": False, "source": "none"}
    d1 = _bridge_bars(sym, "D1")
    tail = _tail_rows(d1, 1) if d1 else []
    tk = latest_tick(sym)
    fresh = bool(tk and tk["age_s"] <= 120)
    return {"symbol": norm(sym), "file": p.name, "rows": None,
            "start": None, "end": tail[-1][0] if tail else None,
            "available": True, "source": "bridge",
            "mtime": st.st_mtime,
            "bars_age_s": round(max(0.0, time.time() - st.st_mtime), 1),
            "last": (tk["bid"] if fresh else (tail[-1][4] if tail else None)),
            "tick_age_s": tk["age_s"] if tk else None}


def symbols(force: bool = False) -> list[dict]:
    """可交易品种清单：桥优先，缺失时补本地快照。结果缓存 5s。"""
    now = time.time()
    hit = _sym_cache.get("syms")
    if hit and not force and now - hit["ts"] < 5.0:
        return hit["rows"]

    names: list[str] = []
    d = bridge_dir()
    if d.is_dir():
        for p in d.glob("gold_bars_*_H1.csv"):
            nm = p.name[len("gold_bars_"):-len("_H1.csv")]
            if nm:
                names.append(nm)
    if not names and MKT.is_dir():
        idx = MKT / "index.json"
        try:
            data = json.loads(idx.read_text(encoding="utf-8"))
            names = [n.replace("#", "") for n in (data.get("symbols") or {})]
        except Exception:  # noqa: BLE001
            names = []

    rows = [_sym_meta(n) for n in sorted(set(names))]
    _sym_cache["syms"] = {"ts": now, "rows": rows}
    return rows


# ------------------------------------------------------------- payload ----
def _load(sym: str, tf: str) -> tuple[list[list], str]:
    """取原始 OHLCV（7 列，未算指标）与来源标识。

    缓存的是裸 K 线；指标在 payload 里按「这次要返回的窗口 + 预热」现算。
    这样前端往左翻历史时，不必为 3 万根全量算指标，也不会被尾部窗口截断。
    """
    p = _bridge_bars(sym, tf)
    src = "bridge"
    if p is None:
        # 桥没有这个周期：用 H1 快照聚合兜底
        h1 = _snapshot_h1(sym)
        if h1 is None:
            return [], "none"
        p, src = h1, "snapshot"
        tf_eff = tf
    else:
        tf_eff = tf

    try:
        st = p.stat()
    except OSError:
        return [], "none"
    ck = (norm(sym), tf_eff)
    key = (src, str(p), st.st_mtime_ns, st.st_size)
    with _lock:
        hit = _cache.get(ck)
        if hit and hit["key"] == key:
            return hit["rows"], src

    rows = _read_bars_file(p)
    if src == "snapshot" and tf_eff in AGG_FROM_H1:
        rows = _agg(rows, tf_eff)
    with _lock:
        _cache[ck] = {"key": key, "rows": rows}
    return rows, src


def payload(symbol: str, tf: str = "H1", n: int = 700, before: str = "") -> dict:
    """返回给前端的行情包。失败时 ok=False 并给中文原因。

    before：只取这个时间点（不含）之前的数据，用于「往左翻历史」分页。
            空 = 最新一页（会接 tick 做实时修正）。
    """
    tf = str(tf or "H1").upper()
    if tf not in TFS:
        return {"ok": False, "error": f"周期只支持 {'/'.join(TFS)}"}
    n = max(MIN_N, min(MAX_N, int(n or 700)))

    rows, src = _load(symbol, tf)
    if not rows:
        return {"ok": False, "error": f"没有 {norm(symbol)} 的 {tf} 行情（桥未导出该周期，本地也无快照）",
                "symbols": symbols()}

    # ---- 分页 + 按窗口算指标 ----
    before = str(before or "").strip()[:19]
    earlier = 0
    if before:
        idx = bisect_left([r[0] for r in rows], before)
        earlier = idx
        rows = rows[:idx]
        if not rows:
            return {"ok": True, "symbol": norm(symbol), "tf": tf, "count": 0,
                    "bars": [], "start": "", "end": "", "params": dict(PARAMS),
                    "feed": "bridge" if src == "bridge" else "snapshot",
                    "live": {"on": False}, "more": False, "total": idx}
    total = len(rows) if not before else earlier
    win = rows[-(n + WARM):]
    rows = _with_indicators([list(r) for r in win])[-n:]
    more = len(win) > n or (before != "" and earlier > len(win))

    # ---- 实时修正：把最后一根接到最新 tick（只在最新一页） ----
    tk = latest_tick(symbol) if (src == "bridge" and not before) else None
    if tk and tk["age_s"] > 120:
        tk = None        # 旧 tick 宁可不用，别把几天前的价格接到今天的 K 线上
    live = {"on": False}
    if tk:
        bid, ask = tk["bid"], tk["ask"]
        dig = _digits(bid)
        step = 10 ** -dig
        last = rows[-1]
        bar_start = None
        try:
            bar_start = dt.datetime.strptime(last[0][:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
        tick_dt = dt.datetime.fromtimestamp(tk["time_msc"] / 1000)
        inside = bar_start is not None and tick_dt >= bar_start
        if inside:
            last[4] = bid
            last[2] = max(last[2], bid)
            last[3] = min(last[3], bid)
        live = {
            "on": tk["age_s"] <= 120,
            "bid": _rd(bid, dig), "ask": _rd(ask, dig),
            "spread": round((ask - bid) / step, 1),
            "tick_time": tk["tick_time"], "age_s": tk["age_s"],
            "quote_in_bar": inside,
            "bars_age_s": None,
        }
        try:
            live["bars_age_s"] = round(max(0.0, time.time() - _bridge_bars(symbol, tf).stat().st_mtime), 1)
        except Exception:  # noqa: BLE001
            pass

    last = rows[-1]
    if live.get("bid") is not None and live.get("spread") is not None:
        last[6] = live["spread"]     # 桥 bars 无 spread 列，用当前点差补末根
    prev = rows[-2] if len(rows) > 1 else last
    chg = last[4] - prev[4]
    hi = max(r[2] for r in rows)
    lo = min(r[3] for r in rows)
    spreads = sorted(r[6] for r in rows if r[6] > 0)

    dig = _digits(last[4])
    return {
        "ok": True,
        "symbol": norm(symbol),
        "tf": tf,
        "count": len(rows),
        "start": rows[0][0],
        "end": last[0],
        "more": bool(more),
        "total": int(total),
        "params": dict(PARAMS),
        "feed": "bridge" if src == "bridge" else "snapshot",
        "live": live,
        "bars": [[r[0], _rd(r[1], dig), _rd(r[2], dig), _rd(r[3], dig), _rd(r[4], dig),
                  _rd(r[5], 2), _rd(r[6], 1), _rd(r[7], dig), _rd(r[8], dig),
                  _rd(r[9], dig), _rd(r[10], dig), _rd(r[11], dig)]
                 for r in rows],
        "last": {
            "t": last[0], "o": _rd(last[1], dig), "h": _rd(last[2], dig), "l": _rd(last[3], dig),
            "c": _rd(last[4], dig), "v": _rd(last[5], 2), "spread": _rd(last[6], 1),
            "chg": _rd(chg, dig), "chg_pct": _rd(chg / prev[4] * 100 if prev[4] else 0, 3),
            "ma1": _rd(last[7], dig), "ma2": _rd(last[8], dig),
            "bo_up": _rd(last[9], dig), "bo_dn": _rd(last[10], dig), "atr": _rd(last[11], dig),
        },
        "win": {"high": _rd(hi, dig), "low": _rd(lo, dig),
                "spread_med": _rd(spreads[len(spreads) // 2], 1) if spreads else None},
        "source": (f"桥 gold_bars_{norm(symbol)}_{tf}.csv" if src == "bridge"
                   else f"本地快照 {(p.name if (p := _snapshot_h1(symbol)) else '—')}"),
        "symbols": symbols(),
    }
