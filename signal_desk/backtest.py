"""离线回测：用实盘同一套评分卡，在桥导出的历史 K 线上跑一遍。

这个脚本要回答一个很朴素的问题：**这套规则到底有没有正期望？**

为了不骗自己，定了几条硬规矩：
1. 只用「当时已知」的数据：第 i 根收盘后决策，第 i+1 根开盘成交，绝不用未来函数。
2. 成本必须计：点差 + 双边滑点。不计成本的回测结果一律虚高。
3. 同一根 K 线内止损和止盈都触及时，按「先触止损」处理（悲观原则）。
4. 不改实盘配置，纯只读；数据源与看板同一份（Common\\Files 的 bars CSV）。

用法：
    python signal_desk/backtest.py                       # GOLD H1，默认参数
    python signal_desk/backtest.py --sweep                # 参数扫描
    python signal_desk/backtest.py --sym GOLD# --tf H1 --thr 55
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ai_trader as ai   # noqa: E402
import market as mk      # noqa: E402

DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

# 回测假设（与实盘配置解耦，写死在这里方便对比）
BASE = {
    "risk_pct": 1.0,
    "max_lots": 0.05,
    "lot_min": 0.01,
    "lot_step": 0.01,
    "contract": 100.0,     # GOLD 1 手 = 100 盎司
    "equity0": 10000.0,
    "spread": 0.33,        # GOLD 实测 bid/ask 差约 0.33 美元
    "slip": 0.10,          # 单边滑点，美元/盎司
    "daily_stop_pct": 2.0,
}


# ---------------------------------------------------------------- 数据
def load(sym: str, tf: str) -> list[list]:
    """读全量历史 + 指标。market._load 只算尾部 960 根，回测要全量。"""
    p = mk._bridge_bars(sym, tf)
    if p is None:
        raise SystemExit(f"桥没有 {sym} {tf} 的 bars（检查 Common\\Files 里有没有 gold_bars_*.csv）")
    rows = mk._read_bars_file(p)
    if not rows:
        raise SystemExit(f"{p} 读出来是空的")
    return mk._with_indicators(rows)


def _rank(seg: list[float], v: float) -> float:
    if not seg:
        return 0.5
    return sum(1 for x in seg if x < v) / len(seg)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def score_v(f: dict, variant: str = "base", thr_hint: float = 60) -> tuple[float, dict]:
    """参数化版评分卡：用来对比"改因子"和"调参数"哪个真有用。

    base   = 与实盘 ai_trader.score 逐行一致（用于验证回测与实盘同源）
    posflip= 位置因子改成顺势（贴上轨加分、贴下轨扣分）
    nopos  = 去掉位置因子，20 分权重给趋势（趋势 50 / 动量 20 / 波动 15 / 时段 15）
    """
    t = _clamp(f["trend_atr"] / 1.2, -1, 1)
    m = _clamp(f["mom3_atr"] / 1.5, -1, 1)
    loc = _clamp((0.5 - f["pos_boll"]) * 2, -1, 1)
    vr = 1 - min(1.0, abs(f["atr_rank"] - 0.55) / 0.55)
    h = f["server_hour"]
    sess = 15.0 if 10 <= h <= 22 else (7.0 if (7 <= h < 10 or 22 < h <= 23) else 0.0)

    if variant == "posflip":
        parts = {"趋势": t * 30, "动量": m * 20, "位置": -loc * 20, "波动": vr * 15, "时段": sess}
    elif variant == "nopos":
        parts = {"趋势": t * 50, "动量": m * 20, "波动": vr * 15, "时段": sess}
    else:
        parts = {"趋势": t * 30, "动量": m * 20, "位置": loc * 20, "波动": vr * 15, "时段": sess}
    return round(sum(parts.values()), 1), {k: round(v, 1) for k, v in parts.items()}


def prep(rows: list[list], variant: str = "base") -> list[dict]:
    """把每根 bar 的评分开销提前算掉：扫描参数时只重跑交易循环。

    特征口径与 ai_trader.features 逐字段对齐，否则回测和实盘就是两套东西。
    """
    cfg = dict(ai.DEFAULT_CFG)
    atrs = [r[11] for r in rows]
    out: list[dict] = []
    n = len(rows)
    for i in range(50, n - 1):
        r = rows[i]
        atr, ma20, ma50, up, dn, c = r[11], r[7], r[8], r[9], r[10], r[4]
        if not atr or ma20 is None or ma50 is None or up is None or dn is None or up <= dn:
            continue
        lo = max(0, i - 119)
        seg = [x for x in atrs[lo:i + 1] if x]
        pos = (c - dn) / (up - dn)
        mom3 = (c - rows[i - 3][4]) if i >= 3 else 0.0
        try:
            hour = int(str(r[0])[11:13])
        except (ValueError, IndexError):
            hour = 12
        f = {"atr": atr, "atr_rank": _rank(seg, atr),
             "trend_atr": (ma20 - ma50) / atr, "pos_boll": pos,
             "mom3_atr": mom3 / atr, "server_hour": hour, "price": c}
        total, parts = score_v(f, variant)
        out.append({"i": i, "t": r[0], "score": total,
                    "parts": parts,
                    "atr": atr, "close": c, "hour": hour,
                    "trend": f["trend_atr"], "pos": pos})
    return out


# ---------------------------------------------------------------- 单次回测
def run(pre: list[dict], rows: list[list], thr: float = 60.0, sl_k: float = 1.5,
        tp_k: float = 3.0, cost: dict | None = None, daily_stop: bool = True,
        date_from: str = "", date_to: str = "") -> dict:
    a = dict(BASE)
    if cost:
        a.update(cost)
    eq = float(a["equity0"])
    peak, mdd, mdd_at = eq, 0.0, None
    trades: list[dict] = []
    skipped_risk = 0
    day_pnl: dict[str, float] = {}
    stopped_days: set[str] = set()

    k = 0
    npre = len(pre)
    nrow = len(rows)
    while k < npre:
        s = pre[k]
        i = s["i"]
        day = s["t"][:10]
        if date_from and day < date_from:
            k += 1
            continue
        if date_to and day > date_to:
            break
        if abs(s["score"]) < thr:
            k += 1
            continue
        # 当日亏损停机（与实盘同一条规则）
        if daily_stop and day in stopped_days:
            k += 1
            continue
        if daily_stop and day_pnl.get(day, 0.0) <= -eq * a["daily_stop_pct"] / 100.0:
            stopped_days.add(day)
            k += 1
            continue

        atr = s["atr"]
        sl_dist = atr * sl_k
        tp_dist = atr * tp_k
        if sl_dist <= 0:
            k += 1
            continue
        lots = math.floor((eq * a["risk_pct"] / 100.0) / (sl_dist * a["contract"]) / a["lot_step"]) * a["lot_step"]
        lots = round(min(lots, a["max_lots"]), 2)
        if lots < a["lot_min"]:
            skipped_risk += 1
            k += 1
            continue

        side = "buy" if s["score"] > 0 else "sell"
        # 下一根开盘成交
        entry_open = rows[i + 1][1]
        entry = entry_open + (a["spread"] / 2 + a["slip"]) * (1 if side == "buy" else -1)
        sl = entry - sl_dist if side == "buy" else entry + sl_dist
        tp = entry + tp_dist if side == "buy" else entry - tp_dist

        exit_px = None
        reason = ""
        j = i + 1
        while j < nrow:
            h, l = rows[j][2], rows[j][3]
            if side == "buy":
                if l <= sl:                       # 悲观：同根先判止损
                    exit_px, reason = sl, "SL"
                    break
                if h >= tp:
                    exit_px, reason = tp, "TP"
                    break
            else:
                if h >= sl:
                    exit_px, reason = sl, "SL"
                    break
                if l <= tp:
                    exit_px, reason = tp, "TP"
                    break
            j += 1
        if exit_px is None:                       # 数据到头，按最后收盘平
            j = nrow - 1
            exit_px = rows[j][4]
            reason = "EOD"
        exit_px += -1 * (a["spread"] / 2) * (1 if side == "buy" else -1)

        sign = 1 if side == "buy" else -1
        pnl = (exit_px - entry) * sign * lots * a["contract"]
        risk_at_entry = eq * a["risk_pct"] / 100.0
        trades.append({
            "i": i, "t": s["t"], "side": side, "score": s["score"],
            "entry": round(entry, 2), "exit": round(exit_px, 2),
            "sl": round(sl, 2), "tp": round(tp, 2), "lots": lots,
            "pnl": round(pnl, 2), "r": round(pnl / risk_at_entry, 3) if risk_at_entry else 0,
            "reason": reason, "bars": j - i, "hour": s["hour"],
        })
        eq += pnl
        day_pnl[day] = day_pnl.get(day, 0.0) + pnl
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > mdd:
            mdd, mdd_at = dd, s["t"]
        k = j + 1 if j > i else k + 1
    return {"trades": trades, "equity": eq, "mdd": mdd, "mdd_at": mdd_at,
            "skipped_risk": skipped_risk}


# ---------------------------------------------------------------- 统计
def stats(res: dict, pre: list[dict], thr: float, sl_k: float, tp_k: float) -> dict:
    tr = res["trades"]
    n = len(tr)
    if not n:
        return {"thr": thr, "sl": sl_k, "tp": tp_k, "n": 0}
    wins = [t for t in tr if t["pnl"] > 0]
    losses = [t for t in tr if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in losses)
    eq0 = BASE["equity0"]
    # 最大连亏
    streak = worst = 0
    for t in tr:
        if t["pnl"] <= 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    avg_bars = sum(t["bars"] for t in tr) / n
    # 按年
    by_year: dict[str, dict] = {}
    for t in tr:
        y = t["t"][:4]
        d = by_year.setdefault(y, {"n": 0, "w": 0, "pnl": 0.0})
        d["n"] += 1
        d["w"] += 1 if t["pnl"] > 0 else 0
        d["pnl"] += t["pnl"]
    return {
        "thr": thr, "sl": sl_k, "tp": tp_k,
        "n": n, "buy": sum(1 for t in tr if t["side"] == "buy"),
        "sell": sum(1 for t in tr if t["side"] == "sell"),
        "win_rate": round(len(wins) / n, 4),
        "pf": round(gp / gl, 2) if gl else None,
        "avg_win": round(gp / len(wins), 2) if wins else 0,
        "avg_loss": round(gl / len(losses), 2) if losses else 0,
        "expect_r": round(sum(t["r"] for t in tr) / n, 3),
        "expect_usd": round(sum(t["pnl"] for t in tr) / n, 2),
        "net": round(res["equity"] - eq0, 2),
        "net_pct": round((res["equity"] / eq0 - 1) * 100, 2),
        "mdd": round(res["mdd"], 2),
        "mdd_pct": round(res["mdd"] / eq0 * 100, 2),
        "worst_streak": worst,
        "tp_hits": sum(1 for t in tr if t["reason"] == "TP"),
        "sl_hits": sum(1 for t in tr if t["reason"] == "SL"),
        "eod": sum(1 for t in tr if t["reason"] == "EOD"),
        "avg_bars": round(avg_bars, 1),
        "skipped_risk": res["skipped_risk"],
        "by_year": {y: {"n": v["n"], "wr": round(v["w"] / v["n"], 3), "pnl": round(v["pnl"], 0)}
                    for y, v in sorted(by_year.items())},
    }


# ---------------------------------------------------------------- 因子检验
def buy_hold(rows: list[list]) -> dict:
    """基准：同期无脑买入持有多头。策略跑不赢它，就没有存在的理由。"""
    p0, p1 = rows[0][4], rows[-1][4]
    peak, mdd = p0, 0.0
    for r in rows:
        c = r[4]
        if c > peak:
            peak = c
        dd = (peak - c) / peak if peak else 0
        if dd > mdd:
            mdd = dd
    return {"p0": p0, "p1": p1, "ret_pct": round((p1 / p0 - 1) * 100, 1),
            "mdd_pct": round(mdd * 100, 1)}


def ic_analysis(pre: list[dict], rows: list[list], horizon: int = 24, nbin: int = 5) -> dict:
    """因子分档检验：每个维度按数值分 5 档，看每档未来 horizon 根的平均收益（ATR 单位）。

    这是判断「因子到底有没有预测力」最直接的办法。五档收益没有单调性，
    就说明这个维度是噪声，调权重只是在拟合噪声。
    """
    keys = ["score", "趋势", "动量", "位置", "波动", "时段"]
    samples = []
    for s in pre:
        j = s["i"] + horizon
        if j >= len(rows):
            break
        fut = (rows[j][4] - s["close"]) / s["atr"] if s["atr"] else 0.0
        rec = {"fut": fut, "score": s["score"]}
        for k in ("趋势", "动量", "位置", "波动", "时段"):
            rec[k] = s["parts"].get(k)
        samples.append(rec)

    # 去掉市场漂移：减去同期均值，只剩「相对平均的超额收益」，这才是因子的选股能力
    base = sum(r["fut"] for r in samples) / len(samples) if samples else 0.0
    for r in samples:
        r["fut"] -= base
    print(f"（已扣除同期平均收益 {base:+.3f} ATR，下表是超额部分）")

    out: dict = {"_base": round(base, 3)}
    for k in keys:
        vals = [(r[k], r["fut"]) for r in samples if r[k] is not None]
        if len(vals) < 100:
            continue
        vals.sort(key=lambda x: x[0])
        step = len(vals) // nbin
        bins = []
        for b in range(nbin):
            seg = vals[b * step:(b + 1) * step] if b < nbin - 1 else vals[b * step:]
            if not seg:
                continue
            bins.append({"range": (round(seg[0][0], 1), round(seg[-1][0], 1)),
                         "n": len(seg), "avg": round(sum(x[1] for x in seg) / len(seg), 3)})
        out[k] = {"bins": bins, "spread": round(bins[-1]["avg"] - bins[0]["avg"], 3),
                  "n": len(vals)}
    return out


def print_ic(res: dict) -> None:
    for k, v in res.items():
        if not isinstance(v, dict):
            continue
        print(f"\n{k}  n={v['n']}  最高档-最低档 = {v['spread']:+.3f} ATR")
        for b in v["bins"]:
            n = max(0, min(16, int(abs(b["avg"]) * 160)))
            bar = ("+" if b["avg"] >= 0 else "-") + "█" * n
            print(f"  [{b['range'][0]:>6} ~ {b['range'][1]:>6}] n={b['n']:>5}  {bar:<18} {b['avg']:+.3f}")


def factor_scan(rows: list[list], horizon: int = 24, nbin: int = 5,
                neutral: bool = False, win: int = 200) -> dict:
    """单因子扫描：一批候选因子各自的「分档超额收益」跨度。

    跨度越大、越单调，说明这个因子越能区分未来的涨跌。
    neutral=True 时减去「邻近 ±win 根的滚动均值」，用来剔除
    "这个信号恰好都出现在牛市里" 这种时间上的巧合。
    """
    atrs = [r[11] for r in rows]
    samples = []
    for i in range(60, len(rows) - horizon):
        r = rows[i]
        atr = r[11]
        if not atr or r[7] is None or r[8] is None:
            continue
        c = r[4]
        hi = max(x[2] for x in rows[i - 20:i])
        lo = min(x[3] for x in rows[i - 20:i])
        seg = [x for x in atrs[max(0, i - 119):i + 1] if x]
        vals = {
            "动量3根": (c - rows[i - 3][4]) / atr,
            "动量6根": (c - rows[i - 6][4]) / atr,
            "动量12根": (c - rows[i - 12][4]) / atr,
            "动量24根": (c - rows[i - 24][4]) / atr,
            "距MA20": (c - r[7]) / atr,
            "距MA50": (c - r[8]) / atr,
            "MA20-MA50": (r[7] - r[8]) / atr,
            "布林位置": (c - r[10]) / (r[9] - r[10]) if r[9] > r[10] else 0.5,
            "20根突破": 1.0 if c >= hi else (-1.0 if c <= lo else 0.0),
            "ATR分位": _rank(seg, atr),
            "当根实体": (c - r[1]) / atr,
            "上影比例": (r[2] - max(c, r[1])) / atr,
            "连涨根数": next((k for k in range(1, 9) if i - k < 0 or (rows[i - k][4] - rows[i - k][1]) * (c - r[1]) <= 0), 8),
        }
        fut = (rows[i + horizon][4] - c) / atr
        samples.append((vals, fut))

    futs = [s[1] for s in samples]
    n = len(futs)
    if neutral and n:
        # 前缀和：O(1) 取任意窗口均值，避免 O(n·win)
        ps = [0.0]
        for x in futs:
            ps.append(ps[-1] + x)
        adj = []
        for k in range(n):
            lo_i, hi_i = max(0, k - win), min(n, k + win + 1)
            adj.append(futs[k] - (ps[hi_i] - ps[lo_i]) / (hi_i - lo_i))
        futs = adj
        base = 0.0
        tag = f"已扣邻近 ±{win} 根滚动均值"
    else:
        base = sum(futs) / n if n else 0.0
        futs = [x - base for x in futs]
        tag = f"已扣同期均值 {base:+.3f}"

    out: dict = {"_base": round(base, 3), "_n": n, "_tag": tag}
    for name in samples[0][0].keys():
        pairs = sorted(((v[name], f) for (v, _), f in zip(samples, futs) if v.get(name) is not None))
        if len(pairs) < 200:
            continue
        step = len(pairs) // nbin
        bins = []
        for b in range(nbin):
            chunk = pairs[b * step:(b + 1) * step] if b < nbin - 1 else pairs[b * step:]
            if chunk:
                bins.append(round(sum(x[1] for x in chunk) / len(chunk), 3))
        out[name] = {"bins": bins, "spread": round(bins[-1] - bins[0], 3),
                     "n": len(pairs)}
    return out


def print_scan(res: dict, horizon: int) -> None:
    print(f"\n=== 单因子扫描（未来 {horizon} 根，超额 ATR，{res.get('_tag', '')}）===")
    items = [(k, v) for k, v in res.items() if isinstance(v, dict)]
    items.sort(key=lambda x: -abs(x[1]["spread"]))
    print(f"{'因子':<12} {'跨度':>7}  分档超额（低 → 高）")
    for k, v in items:
        bars = " ".join(f"{b:+.2f}" for b in v["bins"])
        flag = "  ← 有信号" if abs(v["spread"]) >= 0.20 else ""
        print(f"{k:<12} {v['spread']:+7.3f}  {bars}{flag}")
    print("\n跨度 < 0.15 ATR 的基本是噪声；单调（一路递增/递减）才算真信号。")


# ---------------------------------------------------------------- 突破策略
def run_donchian(rows: list[list], n_entry: int = 20, n_exit: int = 10,
                 sl_k: float = 2.0, cost: dict | None = None,
                 date_from: str = "", date_to: str = "", allow_short: bool = True) -> dict:
    """唐奇安通道突破：收盘破前 N 根高点做多，破前 M 根低点平仓；反向做空。

    与评分卡的区别：不预测价格该在哪，只跟随「已经发生」的突破，
    止损靠 ATR 兜底。这是趋势品种上最朴素也最经得起检验的一类打法。
    """
    a = dict(BASE)
    if cost:
        a.update(cost)
    eq = float(a["equity0"])
    trades: list[dict] = []
    pos = None
    nrow = len(rows)
    i = max(55, n_entry + 2)
    while i < nrow - 1:
        r = rows[i]
        c, atr = r[4], r[11]
        day = str(r[0])[:10]
        if date_to and day > date_to:
            break
        if date_from and day < date_from:
            i += 1
            continue
        if not atr:
            i += 1
            continue

        if pos is None:
            hi = max(x[2] for x in rows[i - n_entry:i])
            lo = min(x[3] for x in rows[i - n_entry:i])
            side = "buy" if c >= hi else ("sell" if (allow_short and c <= lo) else None)
            if not side:
                i += 1
                continue
            sl_dist = atr * sl_k
            lots = math.floor((eq * a["risk_pct"] / 100.0) / (sl_dist * a["contract"]) / a["lot_step"]) * a["lot_step"]
            lots = round(min(lots, a["max_lots"]), 2)
            if lots < a["lot_min"]:
                i += 1
                continue
            entry = rows[i + 1][1] + (a["spread"] / 2 + a["slip"]) * (1 if side == "buy" else -1)
            pos = {"side": side, "entry": entry, "lots": lots, "i": i,
                   "t": str(r[0]), "atr": atr,
                   "sl": entry - sl_dist if side == "buy" else entry + sl_dist,
                   "peak": entry, "trough": entry}
            i += 1
            continue

        # 持仓中：先判止损，再判通道反向
        h, l = r[2], r[3]
        exit_px, reason = None, ""
        if pos["side"] == "buy":
            pos["peak"] = max(pos["peak"], c)
            if l <= pos["sl"]:
                exit_px, reason = pos["sl"], "SL"
            else:
                lo2 = min(x[3] for x in rows[i - n_exit:i])
                if c <= lo2:
                    exit_px, reason = c, "CHAN"
        else:
            pos["trough"] = min(pos["trough"], c)
            if h >= pos["sl"]:
                exit_px, reason = pos["sl"], "SL"
            else:
                hi2 = max(x[2] for x in rows[i - n_exit:i])
                if c >= hi2:
                    exit_px, reason = c, "CHAN"

        if exit_px is not None:
            exit_px += -1 * (a["spread"] / 2) * (1 if pos["side"] == "buy" else -1)
            sign = 1 if pos["side"] == "buy" else -1
            pnl = (exit_px - pos["entry"]) * sign * pos["lots"] * a["contract"]
            risk0 = eq * a["risk_pct"] / 100.0
            trades.append({"i": pos["i"], "t": pos["t"], "side": pos["side"], "score": 0,
                           "entry": round(pos["entry"], 2), "exit": round(exit_px, 2),
                           "sl": round(pos["sl"], 2), "tp": 0, "lots": pos["lots"],
                           "pnl": round(pnl, 2), "r": round(pnl / risk0, 3) if risk0 else 0,
                           "reason": reason, "bars": i - pos["i"], "hour": 0})
            eq += pnl
            pos = None
        i += 1

    if pos is not None:   # 数据到头，按最后收盘平
        exit_px = rows[-1][4]
        sign = 1 if pos["side"] == "buy" else -1
        pnl = (exit_px - pos["entry"]) * sign * pos["lots"] * a["contract"]
        risk0 = eq * a["risk_pct"] / 100.0
        trades.append({"i": pos["i"], "t": pos["t"], "side": pos["side"], "score": 0,
                       "entry": round(pos["entry"], 2), "exit": round(exit_px, 2),
                       "sl": round(pos["sl"], 2), "tp": 0, "lots": pos["lots"],
                       "pnl": round(pnl, 2), "r": round(pnl / risk0, 3) if risk0 else 0,
                       "reason": "EOD", "bars": nrow - 1 - pos["i"], "hour": 0})
        eq += pnl
    mdd, peak = 0.0, float(a["equity0"])
    curve = float(a["equity0"])
    for t in trades:
        curve += t["pnl"]
        peak = max(peak, curve)
        mdd = max(mdd, peak - curve)
    return {"trades": trades, "equity": eq, "mdd": mdd, "mdd_at": "", "skipped_risk": 0}


def show(st: dict) -> str:
    if not st["n"]:
        return f"thr={st['thr']} sl={st['sl']} tp={st['tp']} → 没有成交（门槛过高或数据不足）"
    yn = " ".join(f"{y}:{v['n']}笔/{v['pnl']:+.0f}" for y, v in st["by_year"].items())
    return (
        f"thr={st['thr']:<3} sl={st['sl']:<3} tp={st['tp']:<3} | "
        f"{st['n']:>3}笔(多{st['buy']}/空{st['sell']}) 胜率{st['win_rate']*100:5.1f}% "
        f"盈亏比{st['pf'] if st['pf'] else '—':>5} 期望{st['expect_r']:+.3f}R "
        f"净{st['net']:+8.0f}({st['net_pct']:+6.1f}%) 回撤{st['mdd_pct']:4.1f}% "
        f"连亏{st['worst_streak']:>2} TP{st['tp_hits']}/SL{st['sl_hits']} | {yn}"
    )


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="GOLD#")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--thr", type=float, default=60)
    ap.add_argument("--sl", type=float, default=1.5)
    ap.add_argument("--tp", type=float, default=3.0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ic", action="store_true")
    ap.add_argument("--compare", action="store_true", help="因子变体 × 训练/验证集对比")
    ap.add_argument("--factors", action="store_true", help="单因子扫描")
    ap.add_argument("--neutral", action="store_true", help="因子检验做滚动中性化")
    ap.add_argument("--donchian", action="store_true", help="唐奇安突破策略回测")
    ap.add_argument("--variant", default="base", choices=["base", "posflip", "nopos"])
    ap.add_argument("--dfrom", default="")
    ap.add_argument("--dto", default="")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rows = load(args.sym, args.tf)
    pre = prep(rows, args.variant)
    span = f"{rows[0][0][:10]} ~ {rows[-1][0][:10]}"
    print(f"{args.sym} {args.tf} · {len(rows)} 根 · {span} · 可判定 {len(pre)} 根")
    bh = buy_hold(rows)
    print(f"买入持有基准：{bh['p0']:.2f} → {bh['p1']:.2f}  {bh['ret_pct']:+.1f}%  最大回撤 {bh['mdd_pct']:.1f}%")

    if args.compare:
        splits = [("训练 2021-2024", "2021-01-01", "2024-12-31", 1),
                  ("验证 2025-2026", "2025-01-01", "2026-12-31", 2)]
        for v in ("base", "posflip", "nopos"):
            p2 = prep(rows, v)
            print(f"\n### 变体 {v}")
            for name, d0, d1, _ in splits:
                # 该区间买入持有基准
                seg = [r for r in rows if d0 <= str(r[0])[:10] <= d1]
                b = ""
                if len(seg) > 10:
                    b = f"   [同期买入持有 {(seg[-1][4]/seg[0][4]-1)*100:+.0f}%]"
                for thr in (50, 55, 60):
                    r = run(p2, rows, thr, args.sl, args.tp, date_from=d0, date_to=d1)
                    st = stats(r, p2, thr, args.sl, args.tp)
                    print(f"  {name} {show(st)}{b if thr == 50 else ''}")
        return

    if args.donchian:
        for n_e, n_x in ((20, 10), (55, 20), (20, 20), (55, 10)):
            for side_tag, short in (("双向", True), ("只做多", False)):
                print(f"\n--- 突破 {n_e} / 退出 {n_x} / {side_tag} ---")
                for name, d0, d1 in (("训练 2021-2024", "2021-01-01", "2024-12-31"),
                                     ("验证 2025-2026", "2025-01-01", "2026-12-31"),
                                     ("全样本", "", "")):
                    r = run_donchian(rows, n_e, n_x, 2.0, date_from=d0, date_to=d1, allow_short=short)
                    st = stats(r, pre, 0, 2.0, 0)
                    print(f"  {name:14} {show(st)}")
        return

    if args.ic:
        print(f"\n=== 因子分档检验（未来 {args.horizon} 根，单位 ATR）===")
        res = ic_analysis(pre, rows, args.horizon)
        print_ic(res)
        if args.json:
            Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
            print(f"\n已写 {args.json}")
        return

    if args.factors:
        res = factor_scan(rows, args.horizon, neutral=args.neutral)
        print_scan(res, args.horizon)
        if args.json:
            Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
            print(f"\n已写 {args.json}")
        return

    if args.sweep:
        print("\n=== 参数扫描（thr × 止损倍数 × 止盈倍数）===")
        out = []
        for thr in (40, 45, 50, 55, 60, 65, 70, 75):
            for sl_k in (1.0, 1.5, 2.0, 2.5):
                for tp_k in (2.0, 3.0, 4.0):
                    r = run(pre, rows, thr, sl_k, tp_k)
                    st = stats(r, pre, thr, sl_k, tp_k)
                    out.append(st)
                    if st["n"] >= 10:
                        print(show(st))
        ok = [x for x in out if x["n"] >= 20 and x.get("pf")]
        ok.sort(key=lambda x: -x["expect_r"])
        print("\n=== 期望值前 10（至少 20 笔）===")
        for st in ok[:10]:
            print(show(st))
        if args.json:
            Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
            print(f"\n已写 {args.json}")
        return

    r = run(pre, rows, args.thr, args.sl, args.tp, date_from=args.dfrom, date_to=args.dto)
    st = stats(r, pre, args.thr, args.sl, args.tp)
    print("\n=== 基准结果 ===")
    print(show(st))
    print(f"\n被跳过（风险预算不够 0.01 手）：{st['skipped_risk']} 次")
    print(f"最大回撤发生：{r['mdd_at']}")
    print("\n按年：")
    for y, v in st["by_year"].items():
        print(f"  {y}  {v['n']:>3} 笔  胜率 {v['wr']*100:5.1f}%  盈亏 {v['pnl']:+8.0f} USD")
    if args.json:
        payload = {"meta": {"sym": args.sym, "tf": args.tf, "rows": len(rows), "span": span},
                   "stats": st, "trades": r["trades"][-200:]}
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=1), "utf-8")
        print(f"\n已写 {args.json}")


if __name__ == "__main__":
    main()
