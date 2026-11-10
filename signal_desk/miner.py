"""策略挖掘器：批量搜索 → 样本外验证 → walk-forward → 稳健性体检 → 晋级 demo 实盘。

为什么不是「跑一遍参数优化就完事」：
1. 单次最优参数几乎一定是过拟合——参数越多、搜得越狠，越容易从噪声里挑出
   「历史上很漂亮」的巧合。所以这里把「发现」和「验证」彻底分开：
   搜索只看开发段（前 75%），最后 25% 当封存样本（holdout），只评一次、不参与挑选。
2. walk-forward：把开发段切成多个滚动窗口，每个窗口内重新选参数，只统计
   「下一个窗口」的样本外结果并拼接。它回答的是：这套规则隔一段时间还找不找得到
   有效参数（WFE = 样本外期望 / 样本内期望，>0.4 才算把边际带出了训练区）。
3. 三项体检：参数邻域（不是孤立尖峰）、Monte Carlo 自助重采样（边际是不是靠
   几笔运气）、窗口一致性（不是某一波行情撑起来的）。
4. 全过才是 A 级；A/B 才允许晋级 demo 实盘跟踪。实盘验收标准（2026-11-05 与
   用户确定，见 验收标准.md）：跟踪满 28 天且 ≥8 笔、周均 ≥2 笔、净利为正、
   实盘最大回撤 ≤15%、实盘期望R 不低于回测基线的一半，才标「实盘就绪」。
   回测再漂亮也不跳过这一步。

所有回测都走 strategy.run —— 与实盘判定同一套代码、同一份成本（点差+滑点），
避免「回测一套、实盘另一套」。
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import random
import threading
import time
from pathlib import Path

import strategy as st

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
LATEST = DATA / "mining_latest.json"
PROMOTED = DATA / "mining_promoted.json"
POOL = DATA / "ea_pool.json"          # 跨轮累积的「优秀 EA 池」，只存达标候选
AUTO_CFG = DATA / "mining_auto.json"  # 不间断挖掘开关与进度

# 「不行的直接不要」：只有同时满足这三条的候选才配进池子 / 出现在页面上。
# C、D 级以及样本外不正、WFE 不达标的，一轮结束就地丢弃，既不展示也不入库。
KEEP_GRADES = ("A", "B")
POOL_MAX = 60
AUTO_GAP = 20                        # 两轮之间的间隔秒数（给 CPU / 磁盘留口气）

# ---- 验证参数（改动会改变结论，写死在模块里，界面上只展示不可调） ----
DEV_SHARE = 0.75        # 前 75% 开发，后 25% 封存
IS_BARS = 3000          # walk-forward 训练窗 ≈ 6 个月 H1
OOS_BARS = 900          # 验证窗 ≈ 7 周
NWIN = 6                # 滚动窗口数（研究报告建议 ≥6 个样本外段）
MIN_IS_TRADES = 25      # 开发段最少成交笔数
TOP_K = 10              # 进入深度验证的候选数
MC_ITERS = 800          # 自助重采样次数
PROMOTE_MIN_DAYS = 28   # demo 跟踪观察期：4 周（验收标准）
PROMOTE_MIN_TRADES = 8  # 观察期总笔数下限（周均 ≥2 笔 × 4 周）
PROMOTE_MIN_WEEKLY = 2  # 周均成交笔数下限（策略活性）
PROMOTE_MAX_DD = 0.15   # 实盘跟踪最大回撤上限 15%（红线）
PROMOTE_DEV_FLOOR = 0.5 # 实盘期望R ≥ 回测基线 × 0.5（衰减过半=回测不可信）

FAMILIES = {
    "trend_breakout": {"label": "趋势突破", "desc": "收盘破前 N 根高/低点跟进，ADX/EMA 定方向"},
    "ma_pullback": {"label": "趋势回调", "desc": "EMA 趋势内 RSI 回落到阈值接多（反向接空）"},
    "range_revert": {"label": "区间反转", "desc": "布林两端 + RSI 极值反手，短止损、限期离场"},
    "momentum": {"label": "动量延续", "desc": "N 根动量超阈值跟进，可选 H4 顺向过滤"},
    "trend_breakdown": {"label": "空头趋势", "desc": "只做空：EMA 空排 + 跌破 N 根低点 + ADX 过滤"},
    "range_fade": {"label": "震荡反打", "desc": "只在 ADX 低（无趋势）时，布林两端反打、回中轨离场"},
}

_JOB = {"running": False, "stage": "", "done": 0, "total": 0, "pct": 0,
        "t0": 0.0, "elapsed": 0.0, "log": [], "error": "", "cancel": False,
        "params": {}, "phase": ""}
_LOCK = threading.Lock()
_SER = {}


# ---------------------------------------------------------------- 数据与缓存
def _series(symbol: str, tf: str):
    key = (symbol.upper(), tf.upper())
    if key not in _SER:
        import backtest as bt
        rows = bt.load(key[0], key[1])
        _SER[key] = (rows, st.Series(rows, key[0]))
    return _SER[key]


def _sub(rows: list[list], d0: str = "", d1: str = "") -> list[list]:
    if not d0 and not d1:
        return rows
    return [r for r in rows if (not d0 or str(r[0])[:10] >= d0) and (not d1 or str(r[0])[:10] <= d1)]


# ---------------------------------------------------------------- 候选生成
def _spec(fam: str, name: str, direction: str, long: list, short: list,
          sl: float, tp: float, channel: int = 0, max_bars: int = 0,
          symbol: str = "GOLD#", tf: str = "H1") -> dict:
    sp = st.blank(name)
    sp["symbol"], sp["tf"], sp["direction"] = symbol, tf, direction
    # 只做空也必须带上 short 条件：早先只在 both 时保留 short，导致空头族永远空条件
    sp["entry"] = {"long": long if direction in ("long", "both") else [],
                   "short": short if direction in ("short", "both") else []}
    sp["exit"] = {"sl_atr": sl, "tp_atr": tp, "channel_n": channel, "max_bars": max_bars}
    sp["risk"] = {"risk_pct": 1.0, "max_lots": 0.05}
    sp["comment"] = "AIDesk"
    sp["_family"] = fam
    return sp


def gen_trend_breakout(rng: random.Random, symbol: str, tf: str) -> dict:
    n = rng.choice([10, 15, 20, 30, 40, 55])
    adx = rng.choice([0, 0, 20, 25, 30])
    use_marel = rng.random() < 0.55
    both = rng.random() < 0.55
    sl = rng.choice([1.5, 2.0, 2.5, 3.0])
    chan = rng.random() < 0.62
    tp = 0.0 if chan else rng.choice([3.0, 4.0, 6.0])
    chn = rng.choice([10, 20, 30]) if chan else 0
    long = [{"f": "brk", "n": n, "dir": "up"}]
    short = [{"f": "brk", "n": n, "dir": "down"}]
    if adx:
        long.append({"f": "adx", "n": 14, "op": ">", "v": adx})
        short.append({"f": "adx", "n": 14, "op": ">", "v": adx})
    if use_marel:
        long.append({"f": "marel", "fast": 20, "slow": 50, "ma": "ema", "op": ">"})
        short.append({"f": "marel", "fast": 20, "slow": 50, "ma": "ema", "op": "<"})
    bits = [f"突破{n}"]
    if adx:
        bits.append(f"ADX>{adx}")
    if use_marel:
        bits.append("EMA20/50")
    bits.append("通道离场" if chan else f"TP{tp:g}")
    return _spec("trend_breakout", " · ".join(bits), "both" if both else "long",
                 long, short, sl, tp, chn, 0, symbol, tf)


def gen_ma_pullback(rng: random.Random, symbol: str, tf: str) -> dict:
    fast = rng.choice([10, 20, 30])
    rsi_th = rng.choice([35, 40, 45, 50])
    sl = rng.choice([1.5, 2.0, 2.5])
    tp = rng.choice([2.0, 3.0, 4.0, 6.0])
    both = rng.random() < 0.4
    mb = rng.choice([0, 48, 96])
    long = [{"f": "marel", "fast": fast, "slow": 50, "ma": "ema", "op": ">"},
            {"f": "rsi", "n": 14, "op": "<", "v": rsi_th}]
    short = [{"f": "marel", "fast": fast, "slow": 50, "ma": "ema", "op": "<"},
             {"f": "rsi", "n": 14, "op": ">", "v": 100 - rsi_th}]
    return _spec("ma_pullback", f"EMA{fast}>50 · RSI<{rsi_th} · TP{tp:g}",
                 "both" if both else "long", long, short, sl, tp, 0, mb, symbol, tf)


def gen_range_revert(rng: random.Random, symbol: str, tf: str) -> dict:
    bth = rng.choice([0.10, 0.15, 0.20, 0.25])
    rth = rng.choice([25, 30, 35])
    sl = rng.choice([1.5, 2.0, 2.5])
    tp = rng.choice([1.5, 2.0, 3.0])
    mb = rng.choice([24, 48, 96])
    both = rng.random() < 0.72
    long = [{"f": "boll", "op": "<", "v": bth}, {"f": "rsi", "n": 14, "op": "<", "v": rth}]
    short = [{"f": "boll", "op": ">", "v": round(1 - bth, 2)},
             {"f": "rsi", "n": 14, "op": ">", "v": 100 - rth}]
    return _spec("range_revert", f"布林<{int(bth*100)}% · RSI<{rth} · {mb}根限时",
                 "both" if both else "long", long, short, sl, tp, 0, mb, symbol, tf)


def gen_momentum(rng: random.Random, symbol: str, tf: str) -> dict:
    n = rng.choice([3, 6, 12])
    v = rng.choice([0.3, 0.5, 0.8, 1.2, 1.6])      # 1.2/1.6 只在高波加速段触发
    htf = rng.random() < 0.5
    adx = rng.choice([0, 0, 22, 25, 30])
    sl = rng.choice([1.5, 2.0, 2.5])
    tp = rng.choice([3.0, 4.0, 6.0, 8.0])
    both = rng.random() < 0.6
    long = [{"f": "mom", "n": n, "op": ">", "v": v}]
    short = [{"f": "mom", "n": n, "op": "<", "v": -v}]
    if adx:
        long.append({"f": "adx", "n": 14, "op": ">", "v": adx})
        short.append({"f": "adx", "n": 14, "op": ">", "v": adx})
    if htf:
        long.append({"f": "htf", "tf": "H4", "n": 50, "op": ">"})
        short.append({"f": "htf", "tf": "H4", "n": 50, "op": "<"})
    bits = [f"动量{n}根>{v:g}ATR"]
    if adx:
        bits.append(f"ADX>{adx}")
    if htf:
        bits.append("H4顺向")
    bits.append(f"TP{tp:g}")
    return _spec("momentum", " · ".join(bits), "both" if both else "long",
                 long, short, sl, tp, 0, 0, symbol, tf)


def gen_trend_breakdown(rng: random.Random, symbol: str, tf: str) -> dict:
    """纯空头趋势族：方向固定 short，不带多头侧。

    存在的意义是补上「趋势下行」这一格——原四族里空头只是 both 的附带一侧，
    在长期上涨的黄金样本里多头那半把整体指标撑住了，空头单独看是亏的，
    所以 both 版本过不了体检，池子里就永远只剩多头。单独成族才能评出空头到底行不行。
    """
    n = rng.choice([10, 15, 20, 30, 40])
    adx = rng.choice([0, 22, 25, 30])
    ema = rng.random() < 0.7
    sl = rng.choice([1.5, 2.0, 2.5, 3.0])
    chan = rng.random() < 0.6
    tp = 0.0 if chan else rng.choice([3.0, 4.0, 6.0])
    chn = rng.choice([10, 20, 30]) if chan else 0
    short = [{"f": "brk", "n": n, "dir": "down"}]
    if ema:
        short.append({"f": "marel", "fast": 20, "slow": 50, "ma": "ema", "op": "<"})
    if adx:
        short.append({"f": "adx", "n": 14, "op": ">", "v": adx})
    bits = [f"跌破{n}", "EMA20<50"] if ema else [f"跌破{n}"]
    if adx:
        bits.append(f"ADX>{adx}")
    bits.append("通道离场" if chan else f"TP{tp:g}")
    return _spec("trend_breakdown", " · ".join(bits), "short",
                 [], short, sl, tp, chn, 0, symbol, tf)


def gen_range_fade(rng: random.Random, symbol: str, tf: str) -> dict:
    """震荡反打族：先要求 ADX 低（没趋势）再谈反转。

    原 range_revert 没有趋势过滤，单边行情里反复在布林上轨摸顶、下轨摸底，
    被趋势按着打。这一族把「ADX<阈值」写进入场条件，只在真正的震荡里开仓。
    止盈放到 2~4 ATR、限时 48~120 根：TP 太小（1 ATR）时扣掉点差滑点后
    样本外几乎不剩利润，也凑不满 OOS≥25 笔的体检门槛，等于白测。
    """
    adx_max = rng.choice([18, 20, 22, 25])
    bth = rng.choice([0.10, 0.15, 0.20])
    rth = rng.choice([25, 30, 35, 40])
    sl = rng.choice([2.0, 2.5, 3.0])
    tp = rng.choice([2.0, 2.5, 3.0, 4.0])
    mb = rng.choice([48, 72, 96, 120])
    both = rng.random() < 0.85
    filt = {"f": "adx", "n": 14, "op": "<", "v": adx_max}
    long = [filt, {"f": "boll", "op": "<", "v": bth},
            {"f": "rsi", "n": 14, "op": "<", "v": rth}]
    short = [dict(filt), {"f": "boll", "op": ">", "v": round(1 - bth, 2)},
             {"f": "rsi", "n": 14, "op": ">", "v": 100 - rth}]
    return _spec("range_fade", f"ADX<{adx_max} · 布林<{int(bth*100)}% · RSI<{rth} · {mb}根限时",
                 "both" if both else "long", long, short, sl, tp, 0, mb, symbol, tf)


_GEN = {"trend_breakout": gen_trend_breakout, "ma_pullback": gen_ma_pullback,
        "range_revert": gen_range_revert, "momentum": gen_momentum,
        "trend_breakdown": gen_trend_breakdown, "range_fade": gen_range_fade}


def _sig(sp: dict) -> str:
    keep = {"direction": sp["direction"], "entry": sp["entry"], "exit": sp["exit"]}
    return json.dumps(keep, sort_keys=True, ensure_ascii=False)


def sample(n: int, fams: list[str], seed: int, symbol: str, tf: str) -> list[dict]:
    """随机采样候选。同一份规则只留一个（去重），避免把搜索预算浪费在重复上。"""
    rng = random.Random(seed)
    fams = [f for f in fams if f in _GEN] or list(_GEN)
    seen, out = set(), []
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        fam = fams[len(out) % len(fams)]
        sp = _GEN[fam](rng, symbol, tf)
        k = _sig(sp)
        if k in seen:
            continue
        seen.add(k)
        out.append(sp)
    return out


# ---------------------------------------------------------------- 参数邻域
def _mut(sp: dict, fn) -> dict:
    c = copy.deepcopy(sp)
    fn(c)
    return c


def _shift(spec: dict, fam: str, knob: str, direction: int) -> dict:
    """沿一个参数轴挪一格（+1/-1），多空两侧同步改，保持规则对称。"""
    def fn(c: dict):
        ex = c["exit"]
        if knob == "sl":
            ex["sl_atr"] = max(0.8, min(6.0, ex["sl_atr"] + 0.5 * direction))
        elif knob == "tp":
            ex["tp_atr"] = max(0.0, min(12.0, ex["tp_atr"] + 1.0 * direction))
        elif knob == "chan":
            ex["channel_n"] = max(0, min(60, ex["channel_n"] + 10 * direction))
        elif knob == "bars":
            ex["max_bars"] = max(0, min(240, ex["max_bars"] + 24 * direction))
        else:
            for side in ("long", "short"):
                for cond in c["entry"][side]:
                    f = cond.get("f")
                    if knob == "brk" and f == "brk":
                        cond["n"] = max(5, min(80, int(cond["n"]) + 5 * direction))
                    elif knob == "rsi" and f == "rsi":
                        cond["v"] = max(10, min(90, float(cond["v"]) + 5 * direction * (1 if side == "long" else -1)))
                    elif knob == "boll" and f == "boll":
                        cond["v"] = max(0.02, min(0.98, float(cond["v"]) + 0.05 * direction * (1 if side == "long" else -1)))
                    elif knob == "mom_n" and f == "mom":
                        cond["n"] = max(2, min(24, int(cond["n"]) + (2 if direction > 0 else -2)))
                    elif knob == "mom_v" and f == "mom":
                        cond["v"] = cond["v"] + 0.2 * direction * (1 if side == "long" else 1)
                    elif knob == "adx" and f == "adx":
                        base = float(cond.get("v") or 0)
                        cond["v"] = max(10.0, min(45.0, base + 2.0 * direction))
                    elif knob == "fast" and f == "marel":
                        cond["fast"] = max(5, min(60, int(cond["fast"]) + 5 * direction))
    return _mut(spec, fn)


def knobs(sp: dict) -> list[str]:
    fam = sp.get("_family")
    if fam == "trend_breakout":
        return ["brk", "sl", "tp"] if sp["exit"]["tp_atr"] else ["brk", "sl", "chan"]
    if fam == "trend_breakdown":
        return ["brk", "sl", "tp"] if sp["exit"]["tp_atr"] else ["brk", "sl", "chan"]
    if fam == "ma_pullback":
        return ["fast", "sl", "tp"]
    if fam == "range_revert":
        return ["boll", "sl", "tp"]
    if fam == "range_fade":
        return ["adx", "boll", "sl"]
    if fam == "momentum":
        return ["mom_n", "sl", "tp"]
    return ["sl", "tp"]


def variants(sp: dict) -> list[tuple[str, dict]]:
    out = [("原参数", sp)]
    for k in knobs(sp):
        for d in (-1, 1):
            v = _shift(sp, sp.get("_family", ""), k, d)
            if _sig(v) != _sig(sp):
                out.append((f"{k}{'+' if d > 0 else '-'}", v))
    return out


# ---------------------------------------------------------------- 单次评估
def ev(sp: dict, symbol: str, tf: str, lo: int = 0, hi: int | None = None) -> dict:
    rows, ser = _series(symbol, tf)
    r = st.run(sp, rows=rows, series=ser, lo=lo, hi=hi)
    if not r.get("ok"):
        return {"n": 0, "ok": False, "error": r.get("error", "")}
    keep = ("n", "win_rate", "pf", "expect_r", "expect_usd", "net", "net_pct",
            "mdd", "mdd_pct", "worst_streak", "buy", "sell", "avg_bars")
    out = {k: r.get(k) for k in keep}
    out["ok"] = True
    out["trades"] = r.get("trades") or []
    return out


def _agg(trades: list[dict]) -> dict:
    """把多段样本外交易拼起来算一份统计（walk-forward 的「真实业绩」）。"""
    n = len(trades)
    if not n:
        return {"n": 0, "expect_r": 0.0, "pf": None, "net": 0.0, "win_rate": 0.0,
                "mdd_pct": 0.0, "mdd": 0.0}
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    eq = 10000.0
    peak = eq
    mdd = 0.0
    for t in trades:
        eq += t["pnl"]
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return {"n": n, "expect_r": round(sum(t.get("r", 0) for t in trades) / n, 3),
            "pf": round(gp / gl, 2) if gl else None,
            "net": round(eq - 10000.0, 2), "net_pct": round((eq / 10000.0 - 1) * 100, 2),
            "win_rate": round(sum(1 for t in trades if t["pnl"] > 0) / n, 4),
            "mdd": round(mdd, 2), "mdd_pct": round(mdd / 10000.0 * 100, 2)}


def _curve(trades: list[dict], points: int = 120) -> list[list]:
    """净值曲线（抽样成 points 个点，前端画小图用）。"""
    if not trades:
        return []
    eq, out = 10000.0, []
    for t in trades:
        eq += t["pnl"]
        out.append([t.get("t", ""), round(eq, 2)])
    if len(out) > points:
        step = len(out) / points
        out = [out[min(len(out) - 1, int(i * step))] for i in range(points)]
    return out


# ---------------------------------------------------------------- walk-forward
def _splits(n: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    """返回 (封存起点, [(is_lo, is_hi, oos_lo, oos_hi), ...])。

    开发段 = 前 DEV_SHARE；封存段只用于最终一次性验证。
    窗口在开发段内滚动：训练 IS_BARS 根，验证 OOS_BARS 根，步长自动拉满 NWIN 个窗。
    """
    dev = max(int(n * DEV_SHARE), IS_BARS + OOS_BARS + 100)
    dev = min(dev, n - 100)
    usable = dev - IS_BARS - OOS_BARS
    step = max(OOS_BARS, usable // max(1, NWIN - 1)) if NWIN > 1 else usable
    wins = []
    a = 0
    while a + IS_BARS + OOS_BARS <= dev and len(wins) < NWIN:
        wins.append((a, a + IS_BARS, a + IS_BARS, a + IS_BARS + OOS_BARS))
        a += step
    return dev, wins


def wfa(sp: dict, symbol: str, tf: str, dev: int, wins: list[tuple[int, int, int, int]]) -> dict:
    """每个窗口内重新选参数 → 只统计下一窗样本外 → 拼接。"""
    vs = variants(sp)
    win_rows, stitched, is_rs = [], [], []
    for (a, b, c, d) in wins:
        best, best_r = None, None
        for label, v in vs:
            r = ev(v, symbol, tf, lo=a, hi=b)
            if r["n"] < 4:
                continue
            if best_r is None or (r["expect_r"] or -9) > best_r:
                best, best_r, best_label = v, r["expect_r"], label
        if best is None:
            win_rows.append({"d0": _d(symbol, tf, a), "d1": _d(symbol, tf, b), "pick": "无",
                             "is_r": None, "oos_n": 0, "oos_net": 0.0, "oos_r": None})
            continue
        o = ev(best, symbol, tf, lo=c, hi=d)
        is_rs.append(best_r)
        stitched.extend(o["trades"])
        win_rows.append({"d0": _d(symbol, tf, a), "d1": _d(symbol, tf, c), "pick": best_label,
                         "is_r": round(best_r, 3), "oos_n": o["n"],
                         "oos_net": round(o["net"], 2), "oos_r": o["expect_r"]})
    agg = _agg(stitched)
    pos_win = sum(1 for w in win_rows if (w["oos_net"] or 0) > 0 and (w["oos_n"] or 0) > 0)
    tested = sum(1 for w in win_rows if (w["oos_n"] or 0) > 0)
    wfe = None
    if is_rs and agg["expect_r"] is not None and sum(is_rs):
        is_mean = sum(is_rs) / len(is_rs)
        wfe = round(agg["expect_r"] / is_mean, 2) if is_mean > 0 else None
    return {"windows": win_rows, "stitched": agg, "wfe": wfe,
            "pos_win": pos_win, "tested": tested,
            "worst_win": min([w["oos_net"] for w in win_rows if w["oos_n"]] or [0.0]),
            "trades": stitched}


def _d(symbol: str, tf: str, i: int) -> str:
    rows, _ = _series(symbol, tf)
    i = max(0, min(len(rows) - 1, i))
    return str(rows[i][0])[:10]


# ---------------------------------------------------------------- 体检
def mc_test(trades: list[dict], iters: int = MC_ITERS, seed: int = 7) -> dict:
    """自助重采样：从已有交易里有放回地抽 n 笔，看均值 R 的分布。

    p_neg = 均值 R ≤ 0 的占比。它回答「这点边际会不会只是几笔运气」。
    顺序打乱对总盈亏没影响，所以这里重采样而不是洗牌。
    """
    rs = [float(t.get("r", 0)) for t in trades]
    n = len(rs)
    if n < 8:
        return {"iters": 0, "n": n, "p_neg": None, "mean_r": None,
                "p5": None, "p50": None, "p95": None}
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += rs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return {"iters": iters, "n": n,
            "mean_r": round(sum(rs) / n, 3),
            "p_neg": round(sum(1 for x in means if x <= 0) / len(means), 3),
            "p5": round(means[int(0.05 * len(means))], 3),
            "p50": round(means[int(0.50 * len(means))], 3),
            "p95": round(means[int(0.95 * len(means)) - 1], 3)}


def neighbors(sp: dict, symbol: str, tf: str, dev: int) -> dict:
    """参数邻域：把每个参数挪 ±1 格，看是不是周围一圈也能赚钱。

    孤立尖峰 = 过拟合的典型特征；一片高原才是可实盘的东西。
    """
    items = []
    for label, v in variants(sp)[1:]:
        r = ev(v, symbol, tf, lo=0, hi=dev)
        if r["n"] < 10:
            continue
        items.append({"k": label, "n": r["n"], "r": r["expect_r"],
                      "net": round(r["net"], 1), "pf": r["pf"]})
    if not items:
        return {"n": 0, "pos": 0, "pos_share": None, "median_r": None, "items": []}
    rs = sorted(x["r"] for x in items)
    med = rs[len(rs) // 2] if len(rs) % 2 else (rs[len(rs) // 2 - 1] + rs[len(rs) // 2]) / 2
    pos = sum(1 for x in items if (x["r"] or 0) > 0)
    return {"n": len(items), "pos": pos, "pos_share": round(pos / len(items), 2),
            "median_r": round(med, 3), "items": items}


# ---------------------------------------------------------------- 评级
def grade(c: dict) -> tuple[str, dict, float]:
    is_, oos, w, nb, mc = c["is"], c["holdout"], c["wfa"], c["nb"], c["mc"]
    g = {}
    g["样本外为正"] = bool(oos["n"] >= 12 and (oos["net"] or 0) > 0 and (oos["pf"] or 0) >= 1.15)
    g["样本量够"] = bool(is_["n"] >= MIN_IS_TRADES)
    g["WFE≥0.4"] = bool(w["wfe"] is not None and w["wfe"] >= 0.4)
    g["窗口一致"] = bool(w["tested"] >= 4 and w["pos_win"] >= max(3, math.ceil(w["tested"] * 0.6)))
    # 邻域不能「只有这一点赚钱」：中位邻居至少要拿到候选本身一半的期望
    nb_ok = bool(nb["pos_share"] is not None and nb["pos_share"] >= 0.6
                 and (nb["median_r"] or -9) >= 0.5 * max(0.0, (is_["expect_r"] or 0)))
    g["邻域稳健"] = nb_ok
    g["MC p≤0.15"] = bool(mc["p_neg"] is not None and mc["p_neg"] <= 0.15)
    g["回撤可控"] = bool((oos["mdd_pct"] or 99) <= 15 and (w["stitched"]["mdd_pct"] or 99) <= 18)
    ok = sum(1 for v in g.values() if v)
    gr = "A" if ok == len(g) else "B" if ok == len(g) - 1 else "C" if ok >= 5 else "D"
    score = ((is_["expect_r"] or 0) * min(1.0, is_["n"] / 40)
             * (0.5 + 0.5 * min(1.0, (is_["pf"] or 0) / 1.4))
             * (1 - min(0.9, (is_["mdd_pct"] or 0) / 25)))
    return gr, g, round(score, 3)


# ---------------------------------------------------------------- 派生质量指标
# 说明：下面这些不参与 grade() 的 7 项体检，避免悄悄改掉历史评级。
# 它们是叠加在 A/B 之上的「优秀线」分档：同样叫 A 级，质量差得很远。
QUALITY = {
    "pf": 2.0,        # 盈利因子下限（优秀线，沿用验收标准里的 PF>2.0）
    "annual": 12.0,   # 封存段年化收益 % 下限
    "wfe": 1.0,       # 走样效率下限
    "oos_n": 40,      # 样本外笔数下限：防小样本撞运气
    "mdd": 12.0,      # 封存段最大回撤上限 %
    "ok_pf": 1.5,     # 「合格」下限：能吃但撑不起主力仓位
    "ok_annual": 8.0,
    "ok_wfe": 0.8,
    "ok_n": 25,
    "ok_mdd": 15.0,
}
YEAR_MIN_DAYS = 90     # 首尾年份不满 90 天的不算「年度」样本，避免半年当一年


def _curve_years(curve: list) -> tuple[float, float, list[dict]]:
    """从封存段净值曲线反推 (年数, 总收益率%, 各自然年收益)。

    curve 形如 [[时间, 净值], ...]，时间就是成交时间，足够还原年度表现。
    """
    pts = []
    for row in curve or []:
        try:
            pts.append((dt.datetime.fromisoformat(str(row[0])), float(row[1])))
        except (TypeError, ValueError, IndexError):
            continue
    if len(pts) < 2:
        return 0.0, 0.0, []
    years = max((pts[-1][0] - pts[0][0]).days / 365.25, 0.0)
    total = (pts[-1][1] / pts[0][1] - 1) * 100 if pts[0][1] else 0.0
    buckets: dict[int, list] = {}
    for t, eq in pts:
        b = buckets.setdefault(t.year, [eq, eq, t, t])
        b[1] = eq
        b[3] = t
    out = []
    for y, (first, last, t0, t1) in sorted(buckets.items()):
        if (t1 - t0).days < YEAR_MIN_DAYS:
            continue
        out.append({"y": y, "pct": round((last / first - 1) * 100, 1)})
    return round(years, 3), round(total, 2), out


def derive(c: dict) -> dict:
    """给候选补上可排序、可比的质量字段（幂等，可反复调用）。"""
    o = c.get("holdout") or {}
    w = c.get("wfa") or {}
    curve = c.get("hold_curve") or []
    years, total, yearly = _curve_years(curve)
    if years <= 0:
        years = 1.41          # 拿不到曲线时退回当前封存段长度，宁可保守不用 0 除
        total = float(o.get("net_pct") or 0)
    ann = total / years if years else 0.0
    dd = o.get("mdd_pct")
    rr = (ann / dd) if dd else None
    pf = o.get("pf") or 0.0
    wfe = w.get("wfe")
    n = o.get("n") or 0

    if curve:
        o = dict(o)
        o["span"] = [curve[0][0], curve[-1][0]]
        c["holdout"] = o
    c["annual_pct"] = round(ann, 2)
    c["rr"] = round(rr, 2) if rr is not None else None
    c["years"] = years
    c["yearly"] = yearly
    c["years_pos"] = sum(1 for x in yearly if x["pct"] > 0)
    c["years_n"] = len(yearly)

    if (pf >= QUALITY["pf"] and ann >= QUALITY["annual"]
            and (wfe or 0) >= QUALITY["wfe"] and n >= QUALITY["oos_n"]
            and (dd or 99) <= QUALITY["mdd"]):
        c["quality"] = "elite"          # 过优秀线：真能当主力候选看
    elif (pf >= QUALITY["ok_pf"] and ann >= QUALITY["ok_annual"]
          and (wfe or 0) >= QUALITY["ok_wfe"] and n >= QUALITY["ok_n"]
          and (dd or 99) <= QUALITY["ok_mdd"]):
        c["quality"] = "ok"             # 合格：可做组合搭配，单只撑不起仓位
    else:
        c["quality"] = "watch"          # 及格但某条明显短：只观察

    # 行情适配摘要：这笔策略实际在哪类行情里赚过钱（只描述，不做淘汰）
    reg = c.get("regime") or {}
    rows = [(k, v) for k, v in reg.items()
            if isinstance(v, dict) and (v.get("n") or 0) > 0]
    tot = sum(int(v.get("n") or 0) for _k, v in rows)
    if tot:
        main_k, main_v = max(rows, key=lambda kv: int(kv[1].get("n") or 0))
        good = [k for k, v in rows if (v.get("expect_r") or 0) > 0]
        c["regime_fit"] = {
            "main": main_k,
            "main_label": REG_LABEL.get(main_k, main_k),
            "share": round(int(main_v.get("n") or 0) / tot, 3),
            "covered": len(good),
            "total": len(REG_KINDS),
            "single": round(int(main_v.get("n") or 0) / tot, 3) >= 0.75,
            "missing": [REG_LABEL.get(k, k) for k in REG_KINDS
                        if not reg.get(k) or not (reg.get(k) or {}).get("n")],
        }
    else:
        c["regime_fit"] = None
    return c


# ---------------------------------------------------------------- 行情状态分桶
# 目的：回答「这个 EA 是不是只在趋势行情里赚钱」——把封存段每笔成交按入场时的
# 行情状态分桶统计，震荡期没做过 / 做了就亏的候选一眼可见，防止同一家族集体偏科。
_REG_CACHE: dict = {}

REG_KINDS = ("trend_up", "trend_dn", "range", "high_vol")
REG_LABEL = {"trend_up": "趋势上行", "trend_dn": "趋势下行", "range": "震荡",
             "high_vol": "高波"}


def _regime_labels(symbol: str, tf: str) -> tuple[list[str], list[bool]]:
    """给每根 bar 打行情状态标签（一份 K 线只算一次，缓存复用）。

    互斥三态（ADX(14)≥22 判趋势方向用 MA20/MA50 相对位置，与实盘 assist 口径同源）：
      trend_up  ADX≥22 且 MA20>MA50
      trend_dn  ADX≥22 且 MA20<MA50
      range     其余（ADX<22 → 震荡）
    另记每根「高波」标记：ATR 高于过去 1000 根的 75 分位（滚动窗口，
    不用全历史分位——黄金近年波动整体抬升，全历史分位会把整个封存段判成高波）。
    返回 (labels, hivol)。
    """
    key = (symbol.upper(), tf.upper())
    if key in _REG_CACHE:
        return _REG_CACHE[key]
    import bisect
    rows, ser = _series(symbol, tf)
    ma20, ma50, atr = ser.ma20, ser.ma50, ser.atr
    adx = ser.adx_series(14)
    n = len(rows)
    labels: list[str] = ["range"] * n
    for i in range(n):
        a, b = ma20[i], ma50[i]
        if a and b and adx[i] is not None and adx[i] >= 22:
            labels[i] = "trend_up" if a > b else "trend_dn"
    hivol: list[bool] = [False] * n
    start, W = 60, 1000
    win: list[float] = sorted(v for v in atr[max(0, start - W):start] if v)
    for i in range(start, n):
        v = atr[i]
        if not v:
            continue
        if len(win) >= W:
            win.pop(0)                      # 滑出窗口的最旧值（FIFO）
        bisect.insort(win, v)
        q75 = win[max(0, int(len(win) * 0.75) - 1)]
        hivol[i] = v >= q75
    _REG_CACHE[key] = (labels, hivol)
    return labels, hivol


def _regime_agg(trades: list[dict]) -> dict:
    n = len(trades)
    if not n:
        return {"n": 0}
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    return {"n": n,
            "win_rate": round(sum(1 for t in trades if t["pnl"] > 0) / n, 3),
            "expect_r": round(sum(t.get("r", 0) for t in trades) / n, 3),
            "pf": round(gp / gl, 2) if gl else None}


def _regime_split(trades: list[dict], symbol: str, tf: str) -> dict:
    """把封存段成交按入场 bar 的行情状态分桶统计。"""
    labels, hivol = _regime_labels(symbol, tf)
    buckets: dict[str, list] = {k: [] for k in REG_KINDS}
    for t in trades or []:
        i = t.get("i")
        if not isinstance(i, int) or i < 0 or i >= len(labels):
            continue
        buckets["high_vol" if hivol[i] else labels[i]].append(t)
    out = {k: _regime_agg(v) for k, v in buckets.items()}
    # 偏科判定：震荡期没做过（untested）/ 做了就亏（loss）/ 验证过且不亏（ok）
    rng = out["range"]
    out["bias"] = ("untested" if rng.get("n", 0) == 0
                   else "loss" if (rng.get("expect_r") or 0) < 0
                   else "ok")
    return out


def regime_backfill() -> int:
    """给池里还没有 regime 字段的旧候选补算（重启后台跑一次，不阻塞请求）。

    重新跑一遍封存段回测只为拿成交序列：不改动原有指标，只补 regime。
    """
    n_done = 0
    items = _pool_read()
    changed = False
    for x in items:
        if x.get("regime") or not x.get("spec"):
            continue
        try:
            sp = x["spec"]
            rows, _ = _series(sp.get("symbol", "GOLD#"), sp.get("tf", "H1"))
            dev, _w = _splits(len(rows))
            hold = ev(sp, sp.get("symbol", "GOLD#"), sp.get("tf", "H1"), lo=dev)
            x["regime"] = _regime_split(hold.get("trades") or [],
                                        sp.get("symbol", "GOLD#"), sp.get("tf", "H1"))
            changed = True
            n_done += 1
        except (Exception, SystemExit):  # noqa: BLE001  桥缺数据会抛 SystemExit
            continue
    if changed:
        POOL.write_text(json.dumps(items, ensure_ascii=False, indent=1), "utf-8")
    return n_done


# ---------------------------------------------------------------- 任务
def _log(msg: str) -> None:
    with _LOCK:
        _JOB["log"].append(f"{dt.datetime.now():%H:%M:%S} {msg}")
        _JOB["log"] = _JOB["log"][-60:]


def _set(stage: str = "", done: int | None = None, total: int | None = None,
         phase: str = "") -> None:
    with _LOCK:
        if stage:
            _JOB["stage"] = stage
        if phase:
            _JOB["phase"] = phase
        if done is not None:
            _JOB["done"] = done
        if total is not None:
            _JOB["total"] = max(1, total)
        _JOB["pct"] = round(100 * _JOB["done"] / max(1, _JOB["total"]), 1)
        _JOB["elapsed"] = round(time.time() - _JOB["t0"], 1)


def status() -> dict:
    with _LOCK:
        j = dict(_JOB)
    latest = _read(LATEST, {}) or {}
    return {"ok": True, "job": j,
            "meta": latest.get("meta") or {},
            "at": latest.get("at") or "",
            "count": len(latest.get("candidates") or []),
            "promoted": promoted(),
            "families": [{"k": k, **v} for k, v in FAMILIES.items()],
            "auto": auto_status(),
            "pool_n": len(_pool_read()),
            "rules": {"dev_share": DEV_SHARE, "nwin": NWIN, "is_bars": IS_BARS,
                      "oos_bars": OOS_BARS, "min_trades": MIN_IS_TRADES,
                      "mc_iters": MC_ITERS, "promote_days": PROMOTE_MIN_DAYS,
                      "promote_trades": PROMOTE_MIN_TRADES,
                      "promote_weekly": PROMOTE_MIN_WEEKLY,
                      "promote_max_dd": int(PROMOTE_MAX_DD * 100)}}


def results() -> dict:
    """对外只给「配留下来的」：本轮合格候选 + 跨轮优秀 EA 池。

    C/D 级、样本外不正、WFE <0.4 的候选在这一步被挡掉，页面上看不见、
    也不能拿来生成 EA。原始全量仍留在 mining_latest.json 供内部统计。
    """
    d = _read(LATEST, {}) or {}
    d.pop("_full", None)
    raw = d.get("candidates") or []
    kept = [c for c in raw if _qualified(c)]
    items = _pool_read()
    seen = {x.get("id") for x in items}
    merged = list(items)
    for c in kept:                       # 本轮刚跑完、还没来得及入池的也别漏
        pid = _pid(c["spec"])
        if pid not in seen:
            merged.append(_pool_entry(c, d.get("at") or "", (d.get("meta") or {})))
            seen.add(pid)
    merged = [derive(x) for x in merged]
    merged.sort(key=lambda x: ({"A": 0, "B": 1}.get(x.get("grade"), 2),
                               -{"elite": 2, "ok": 1}.get(x.get("quality"), 0),
                               -(x.get("score") or 0)))
    d["candidates"] = merged
    d["discarded"] = len(raw) - len(kept)
    d["pool_n"] = len(items)
    d["auto"] = auto_status()
    return d


def _read(p: Path, dv):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return dv


def start(n: int = 160, fams: list[str] | None = None, seed: int | None = None,
          symbol: str = "GOLD#", tf: str = "H1") -> dict:
    if _JOB["running"]:
        return {"ok": False, "error": "已经有一轮在跑，等它结束或先停止"}
    n = max(20, min(600, int(n or 160)))
    fams = [f for f in (fams or list(FAMILIES)) if f in FAMILIES]
    seed = int(seed if seed is not None else time.time() % 100000)
    with _LOCK:
        _JOB.update({"running": True, "cancel": False, "stage": "准备数据", "done": 0,
                     "total": n, "pct": 0, "t0": time.time(), "elapsed": 0.0,
                     "log": [], "error": "", "phase": "",
                     "params": {"n": n, "families": fams, "seed": seed,
                                "symbol": symbol, "tf": tf}})
    threading.Thread(target=_worker, args=(n, fams, seed, symbol, tf),
                     name="miner", daemon=True).start()
    return {"ok": True, "seed": seed}


def stop() -> dict:
    with _LOCK:
        _JOB["cancel"] = True
    return {"ok": True}


def _pick_top(scored: list[dict], fams: list[str], k: int = TOP_K) -> list[dict]:
    """深验名额按族轮转分配，而不是全局按分数取前 K。

    开发段分数天然偏向顺势族：趋势突破在单边上涨样本里几乎必然跑赢空头/震荡族，
    全局取前 10 的结果就是 —— 160 个候选里 6 个族都测了，深验却只留趋势族，
    最后进池的清一色多头突破。这不是别的策略不行，是名额分配把它们挡在门外。
    改成每族先各取最优 1 个，再轮第二圈，名额没用完才按全局分数补。
    """
    by: dict[str, list[dict]] = {}
    for c in scored:
        by.setdefault(str((c.get("spec") or {}).get("_family") or "?"), []).append(c)
    for v in by.values():
        v.sort(key=lambda x: -x["_score"])
    order = [f for f in fams if f in by] + [f for f in by if f not in fams]
    picked: list[dict] = []
    i = 0
    while len(picked) < k:
        added = False
        for f in order:
            v = by.get(f) or []
            if i < len(v):
                picked.append(v[i])
                added = True
                if len(picked) >= k:
                    break
        if not added:
            break
        i += 1
    return picked


def _worker(n: int, fams: list[str], seed: int, symbol: str, tf: str) -> None:
    t0 = time.time()
    try:
        rows, ser = _series(symbol, tf)
        dev, wins = _splits(len(rows))
        _log(f"数据 {len(rows)} 根（{_d(symbol, tf, 0)} ~ {_d(symbol, tf, len(rows)-1)}）"
             f"，开发段 {dev} 根、封存 {len(rows)-dev} 根")
        _log(f"候选 {n} 个 · 家族 {('、'.join(FAMILIES[f]['label'] for f in fams))} · seed {seed}")
        cands = sample(n, fams, seed, symbol, tf)
        _set("① 搜索：开发段批量回测", 0, len(cands))
        scored = []
        for i, sp in enumerate(cands):
            if _JOB["cancel"]:
                raise RuntimeError("已停止")
            r = ev(sp, symbol, tf, lo=0, hi=dev)
            if r["n"] >= MIN_IS_TRADES and (r["net"] or 0) > 0 and (r["pf"] or 0) >= 1.05:
                c = {"spec": sp, "is": _strip(r)}
                sc = ((r["expect_r"] or 0) * min(1.0, r["n"] / 40)
                      * (0.5 + 0.5 * min(1.0, (r["pf"] or 0) / 1.4))
                      * (1 - min(0.9, (r["mdd_pct"] or 0) / 25)))
                c["_score"] = sc
                scored.append(c)
            if i % 5 == 0:
                _set(done=i + 1)
                if i and i % 25 == 0:
                    _log(f"已测 {i}/{len(cands)}，通过初筛 {len(scored)}")
        _set("① 搜索：开发段批量回测", len(cands))
        scored.sort(key=lambda x: -x["_score"])
        top = _pick_top(scored, fams)
        _log(f"初筛通过 {len(scored)} 个，送深验 {len(top)} 个"
             f"（按族轮转：{'、'.join(sorted({str((c.get('spec') or {}).get('_family') or '?') for c in top}))}）")
        if not top:
            raise RuntimeError("没有候选通过初筛（样本内笔数≥25 且净利为正）。"
                               "可加候选数、换家族，或放宽周期。")

        _set("② 封存段 + walk-forward + 体检", 0, len(top) * 3)
        out = []
        for idx, c in enumerate(top):
            if _JOB["cancel"]:
                raise RuntimeError("已停止")
            sp = c["spec"]
            _set(phase=f"候选 {idx+1}/{len(top)}：{sp['name']}")
            hold = ev(sp, symbol, tf, lo=dev)
            wf = wfa(sp, symbol, tf, dev, wins)
            nb = neighbors(sp, symbol, tf, dev)
            mc = mc_test(wf["trades"])
            c2 = {"id": f"m{idx+1:02d}", "name": sp["name"], "family": sp.get("_family", ""),
                  "spec": sp, "is": c["is"], "holdout": _strip(hold),
                  "desc": st.describe(sp)["lines"],
                  "wfa": {k: v for k, v in wf.items() if k != "trades"},
                  "mc": mc, "nb": nb,
                  "curve": _curve(wf["trades"] or hold["trades"]),
                  "hold_curve": _curve(hold["trades"]),
                  "regime": _regime_split(hold.get("trades") or [], symbol, tf)}
            g, gates, score = grade(c2)
            c2.update({"grade": g, "gates": gates, "score": score})
            out.append(c2)
            _set(done=(idx + 1) * 3)
            _log(f"{sp['name'][:26]} → {g} 级（{sum(1 for x in gates.values() if x)}/{len(gates)} 项通过）")
        out.sort(key=lambda x: ({"A": 0, "B": 1, "C": 2, "D": 3}[x["grade"]], -x["score"]))
        out = [derive(c) for c in out]
        meta = {"symbol": symbol, "tf": tf, "n": n, "fams": fams, "seed": seed,
                "rows": len(rows), "dev": dev, "span": [ _d(symbol, tf, 0), _d(symbol, tf, len(rows)-1)],
                "tested": len(cands), "passed": len(scored), "elapsed": round(time.time() - t0, 1),
                "cost": {"spread": st.COST.get(symbol, st.DEFAULT_COST)["spread"],
                         "slip": st.COST.get(symbol, st.DEFAULT_COST)["slip"]}}
        # 族审计：每族测了多少 / 过初筛多少 / 进深验多少 / 最终入池多少。
        # 用户看到池里全是趋势突破时，需要知道「别的族是测了没过，还是压根没测」。
        stat = {f: {"tested": 0, "passed": 0, "deep": 0, "kept": 0} for f in fams}
        for sp0 in cands:
            k = str(sp0.get("_family") or "")
            if k in stat:
                stat[k]["tested"] += 1
        for c0 in scored:
            k = str((c0.get("spec") or {}).get("_family") or "")
            if k in stat:
                stat[k]["passed"] += 1
        for c0 in top:
            k = str((c0.get("spec") or {}).get("_family") or "")
            if k in stat:
                stat[k]["deep"] += 1
        for c0 in out:
            k = str(c0.get("family") or "")
            if k in stat and c0.get("grade") in ("A", "B"):
                stat[k]["kept"] += 1
        meta["fams_stat"] = stat
        meta["round"] = _AUTO.get("round") or 1
        LATEST.write_text(json.dumps({"at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                      "meta": meta, "candidates": out},
                                     ensure_ascii=False, indent=1), "utf-8")
        ing = _pool_ingest(out, meta)
        _set("完成", len(top) * 3, len(top) * 3)
        _log(f"完成：{len(out)} 个候选中留下 {ing['kept']} 个，淘汰 {ing['discarded']} 个，"
             f"优秀 EA 池现有 {ing['pool_n']} 个，用时 {time.time()-t0:.0f}s")
    except Exception as e:  # noqa: BLE001
        with _LOCK:
            _JOB["error"] = f"{type(e).__name__}: {e}"
        _log(f"中断：{_JOB['error']}")
    finally:
        with _LOCK:
            _JOB["running"] = False
            _JOB["elapsed"] = round(time.time() - _JOB["t0"], 1)


def _strip(r: dict) -> dict:
    return {k: r.get(k) for k in ("n", "win_rate", "pf", "expect_r", "net", "net_pct",
                                  "mdd", "mdd_pct", "worst_streak", "buy", "sell")}


# ---------------------------------------------------------------- 晋级 / 实盘跟踪
def promote(cid: str) -> dict:
    cands = results().get("candidates") or []   # 只看达标候选 / 优秀 EA 池
    c = next((x for x in cands if x.get("id") == cid), None)
    if not c:
        return {"ok": False, "error": f"没有候选 {cid}"}
    if c.get("grade") not in ("A", "B"):
        return {"ok": False, "error": f"{c.get('grade')} 级不允许晋级：先满足 6/7 项体检再上 demo"}
    sp = copy.deepcopy(c["spec"])
    ea = f"Mine_{cid}"                 # 统一标识：排行/详情/跟踪/03 回测页/生成的 .mq5 同名
    sp["name"] = "挖-" + sp["name"]
    sp["ea"] = ea
    r = st.activate(sp, True)
    if not r.get("ok"):
        return r
    base_equity = 0.0                  # 晋级时账户净值：实盘回撤百分比的分母
    try:
        import account_feed as _af
        _s = _af.snapshot(sp["symbol"], days=1)
        base_equity = float(((_s.get("account") or {}).get("equity")) or 0)
    except Exception:  # noqa: BLE001
        pass
    lst = _read(PROMOTED, []) or []
    if not isinstance(lst, list):
        lst = []
    rec = {"id": cid, "ea": ea, "at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "name": sp["name"], "symbol": sp["symbol"], "tf": sp["tf"],
           "grade": c.get("grade"), "spec": sp,
           "base_equity": round(base_equity, 2),
           "baseline": {"is": c.get("is"), "holdout": c.get("holdout"),
                        "wfa": {k: v for k, v in (c.get("wfa") or {}).items() if k != "windows"}},
           "status": "paper", "ticks": 0}
    lst = [x for x in lst if x.get("id") != cid]
    lst.insert(0, rec)
    PROMOTED.write_text(json.dumps(lst[:20], ensure_ascii=False, indent=1), "utf-8")
    return {"ok": True, "note": f"已晋级到 demo 实盘跟踪：{sp['name']}。"
                                f"自动盯盘开着就会按这套规则判定；验收标准：跑满 {PROMOTE_MIN_DAYS} 天 / "
                                f"{PROMOTE_MIN_TRADES} 笔、周均 ≥{PROMOTE_MIN_WEEKLY} 笔、净利为正、"
                                f"回撤 ≤{int(PROMOTE_MAX_DD*100)}%、期望R 不低于基线一半，"
                                f"全部达标才谈上实盘（详见 验收标准.md）。"}


def unpromote(cid: str) -> dict:
    lst = [x for x in (_read(PROMOTED, []) or []) if x.get("id") != cid]
    PROMOTED.write_text(json.dumps(lst, ensure_ascii=False, indent=1), "utf-8")
    act = st.status()
    if act.get("on"):
        st.deactivate("挖矿策略撤下")
    return {"ok": True, "note": "已撤下并停用当前启用策略"}


def promoted() -> list[dict]:
    """晋级列表 + demo 实盘跟踪进度（天数 / 成交 / 与回测基线的偏离）。"""
    lst = _read(PROMOTED, []) or []
    if not isinstance(lst, list):
        return []
    try:
        import account_feed as af
    except Exception:  # noqa: BLE001
        af = None
    out = []
    for x in lst:
        rec = dict(x)
        try:
            at = dt.datetime.strptime(x.get("at", ""), "%Y-%m-%d %H:%M:%S")
        except Exception:  # noqa: BLE001
            at = dt.datetime.now()
        days = round((dt.datetime.now() - at).total_seconds() / 86400, 1)
        deals, net, cum, peak, maxdd = 0, 0.0, 0.0, 0.0, 0.0
        if af is not None:
            try:
                snap = af.snapshot(x.get("symbol") or "GOLD#", days=max(1, int(days) + 1))
                for d in snap.get("deals") or []:
                    if d.get("entry") != "out":
                        continue
                    if "AIDesk" not in str(d.get("comment") or ""):
                        continue
                    deals += 1
                    p = float(d.get("profit") or 0)
                    net += p
                    cum += p                      # 累计净利曲线 → 交易级最大回撤
                    peak = max(peak, cum)
                    maxdd = max(maxdd, peak - cum)
            except Exception:  # noqa: BLE001
                pass
        base_eq = float(x.get("base_equity") or 0)
        dd_pct = round(maxdd / base_eq * 100, 1) if base_eq > 0 and maxdd > 0 else 0.0
        weekly = round(deals * 7.0 / days, 1) if days > 0 else 0.0
        base_r = float(((x.get("baseline") or {}).get("holdout") or {}).get("expect_r") or 0)
        per_risk = base_eq * 0.01               # 单笔风险 ≈ 净值 1%（risk_pct=1%）
        expect_live = round(net / (deals * per_risk), 3) if deals >= 3 and per_risk > 0 else None
        ok_weekly = days < 7 or weekly >= PROMOTE_MIN_WEEKLY      # 首周不判周均
        ok_dd = base_eq <= 0 or dd_pct <= PROMOTE_MAX_DD * 100   # 无基准时先不拦
        ok_dev = expect_live is None or base_r <= 0 or expect_live >= PROMOTE_DEV_FLOOR * base_r
        ready = (days >= PROMOTE_MIN_DAYS and deals >= PROMOTE_MIN_TRADES
                 and ok_weekly and net > 0 and ok_dd and ok_dev)
        base = ((x.get("baseline") or {}).get("holdout") or {})
        rec.update({"days": days, "deals": deals, "live_net": round(net, 2),
                    "weekly": weekly, "live_dd": dd_pct,
                    "expect_live": expect_live,
                    "ready": ready,
                    "need_days": max(0, round(PROMOTE_MIN_DAYS - days, 1)),
                    "need_deals": max(0, PROMOTE_MIN_TRADES - deals),
                    "base_expect_r": base.get("expect_r")})
        out.append(rec)
    return out


# ---------------------------------------------------------------- 优秀 EA 池 / 不间断挖掘
# 目标：EA 工厂只干一件事——不停地挖，达标的沉淀进池子，不达标的当场丢掉。
_AUTO = {"on": False, "n": 120, "fams": None, "gap": AUTO_GAP, "round": 0,
         "started_at": "", "last_at": "", "kept_total": 0, "discarded_total": 0,
         "error": "", "thread": None}
_AUTO_LOCK = threading.Lock()


def _qualified(c: dict) -> bool:
    """入围线：等级 + 样本外为正 + WFE 达标。任一不满足就地淘汰。"""
    if (c or {}).get("grade") not in KEEP_GRADES:
        return False
    oos, w = c.get("holdout") or {}, c.get("wfa") or {}
    return bool((oos.get("expect_r") or -9) > 0 and (oos.get("net") or -9) > 0
                and w.get("wfe") is not None and w["wfe"] >= 0.4)


def _pid(spec: dict) -> str:
    """稳定 id：同一套规则跨轮只留一条，也避免每轮 m00/m01 重名。"""
    return "p" + hashlib.md5(_sig(spec).encode("utf-8")).hexdigest()[:8]


def _pool_read() -> list[dict]:
    v = _read(POOL, []) or []
    items = v if isinstance(v, list) else []
    return [derive(x) for x in items]


def _pool_entry(c: dict, at: str, meta: dict) -> dict:
    pid = _pid(c["spec"])
    e = dict(c)                      # 保留详情/曲线/生成 EA 需要的全部字段
    e.update({"id": pid, "origin": c.get("id"), "ea": "Mine_" + pid,
              "found_at": at, "round": (meta or {}).get("round") or 1,
              "seed": (meta or {}).get("seed"), "seen": 1, "last_at": at})
    return e


def _pool_trim(items: list[dict], cap: int = POOL_MAX) -> list[dict]:
    """按族轮转截断，保证池子里不是清一色一个族。

    原先直接按 grade+score 取前 60：趋势突破族分数高，池子一满就把后进的新族
    全挤掉，越跑越单一。改成每族轮流取，族内按分数排序——同样的容量，至少让
    每个族都有发声的机会。
    """
    by: dict[str, list[dict]] = {}
    for x in items:
        k = str(((x.get("spec") or {}).get("_family")) or x.get("family") or "?")
        by.setdefault(k, []).append(x)
    for v in by.values():
        v.sort(key=lambda x: ({"A": 0, "B": 1}.get(x.get("grade"), 2),
                              -{"elite": 2, "ok": 1}.get(x.get("quality"), 0),
                              -(x.get("score") or 0)))
    order = list(by)
    out: list[dict] = []
    i = 0
    while len(out) < cap:
        added = False
        for f in order:
            v = by[f]
            if i < len(v):
                out.append(v[i])
                added = True
                if len(out) >= cap:
                    break
        if not added:
            break
        i += 1
    return out


def _pool_ingest(out: list[dict], meta: dict) -> dict:
    kept = [c for c in (out or []) if _qualified(c)]
    disc = len(out or []) - len(kept)
    at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    items = _pool_read()
    idx = {x.get("id"): i for i, x in enumerate(items)}
    new = 0
    for c in kept:
        e = _pool_entry(c, at, meta or {})
        i = idx.get(e["id"])
        if i is None:
            items.append(e)
            idx[e["id"]] = len(items) - 1
            new += 1
        else:
            old = items[i]
            old["seen"] = int(old.get("seen") or 1) + 1
            old["last_at"] = at
            if (e.get("score") or 0) > (old.get("score") or 0):
                seen = old["seen"]
                items[i] = dict(e)
                items[i]["seen"] = seen
    items.sort(key=lambda x: ({"A": 0, "B": 1}.get(x.get("grade"), 2),
                              -(x.get("score") or 0)))
    items = _pool_trim(items, POOL_MAX)
    POOL.write_text(json.dumps(items, ensure_ascii=False, indent=1), "utf-8")
    with _AUTO_LOCK:
        _AUTO["kept_total"] += len(kept)
        _AUTO["discarded_total"] += disc
        _AUTO["last_at"] = at
    return {"kept": len(kept), "discarded": disc, "new": new, "pool_n": len(items)}


def pool_bootstrap() -> dict:
    """首次启动时，把上一轮已经跑出来、且达标的候选收进池子（不重跑回测）。"""
    if POOL.exists():
        return {"ok": True, "imported": 0}
    d = _read(LATEST, {}) or {}
    r = _pool_ingest(d.get("candidates") or [], d.get("meta") or {})
    return {"ok": True, "imported": r["kept"]}


def pool() -> dict:
    items = _pool_read()
    return {"ok": True, "items": items, "n": len(items), "max": POOL_MAX,
            "auto": auto_status(),
            "note": "只存通过七项体检且样本外为正 / WFE≥0.4 的候选；未达标的一轮结束即丢弃。"}


def auto_status() -> dict:
    with _AUTO_LOCK:
        a = {k: v for k, v in _AUTO.items() if k != "thread"}
        th = _AUTO.get("thread")
    a["running"] = bool(_JOB.get("running"))
    a["thread_alive"] = bool(th and th.is_alive())
    return a


def _write_auto_cfg() -> None:
    with _AUTO_LOCK:
        cfg = {k: v for k, v in _AUTO.items() if k != "thread"}
    AUTO_CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), "utf-8")


def auto_set(on: bool, n: int = 120, fams=None, gap: int = AUTO_GAP) -> dict:
    """开/关不间断挖掘。开关状态落盘，服务重启后 auto_resume() 自动接着跑。"""
    fams = [f for f in (fams or []) if f in FAMILIES] or None
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _AUTO_LOCK:
        _AUTO.update({"on": bool(on), "n": max(20, min(600, int(n or 120))),
                      "fams": fams, "gap": max(5, min(600, int(gap or AUTO_GAP))),
                      "error": ""})
        if on and not _AUTO.get("started_at"):
            _AUTO["started_at"] = now
        th = _AUTO.get("thread")
        need = bool(on) and not (th and th.is_alive())
    if need:
        t = threading.Thread(target=_auto_loop, name="auto-miner", daemon=True)
        with _AUTO_LOCK:
            _AUTO["thread"] = t
        t.start()
    _write_auto_cfg()
    state = auto_status()
    return {"ok": True, "auto": state,
            "note": ("已开启：每轮结束自动换 seed 再挖，达标入池、不达标丢弃。"
                     if on else "已关闭：当前这一轮跑完即停，池子保留。")}


def _auto_loop() -> None:
    while True:
        with _AUTO_LOCK:
            if not _AUTO["on"]:
                return
            n, fams, gap = _AUTO["n"], _AUTO["fams"], _AUTO["gap"]
            _AUTO["round"] = int(_AUTO.get("round") or 0) + 1
            rnd = _AUTO["round"]
        if _JOB.get("running"):       # 用户手点的一轮还没完，先别抢
            time.sleep(2)
            continue
        try:
            r = start(n, fams, seed=int(time.time()) + rnd * 7919)
            if not r.get("ok"):
                time.sleep(5)
                continue
            _log(f"不间断挖掘 · 第 {rnd} 轮（seed {r['seed']}）")
            while _JOB.get("running"):
                time.sleep(2)
            _write_auto_cfg()
            _log(f"第 {rnd} 轮结束，休息 {gap}s 后继续")
        except Exception as e:  # noqa: BLE001
            with _AUTO_LOCK:
                _AUTO["error"] = f"{type(e).__name__}: {e}"
            _log(f"自动轮异常：{e}")
        for _ in range(max(1, int(gap) // 2)):
            with _AUTO_LOCK:
                if not _AUTO["on"]:
                    return
            time.sleep(2)


def auto_resume() -> dict:
    """服务启动时恢复上次的不间断挖掘开关（不会在命令行挖掘时自动启动）。"""
    cfg = _read(AUTO_CFG, {}) or {}
    if not isinstance(cfg, dict):
        return {"ok": True, "resumed": False}
    with _AUTO_LOCK:
        for k in ("round", "kept_total", "discarded_total", "last_at", "started_at"):
            if k in cfg:
                _AUTO[k] = cfg[k]
    if not cfg.get("on"):
        return {"ok": True, "resumed": False}
    return {"ok": True, "resumed": True,
            **auto_set(True, cfg.get("n") or 120, cfg.get("fams"), cfg.get("gap") or AUTO_GAP)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--fams", default=",".join(FAMILIES))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sym", default="GOLD#")
    ap.add_argument("--tf", default="H1")
    a = ap.parse_args()
    print(start(a.n, a.fams.split(","), a.seed, a.sym, a.tf))
    while True:
        s = status()["job"]
        print(f"\r{s['stage']} {s['pct']}% {s['phase'][:30]:30}", end="", flush=True)
        if not s["running"]:
            break
        time.sleep(1)
    print()
    for line in status()["job"]["log"][-12:]:
        print(line)
    for c in (results().get("candidates") or [])[:8]:
        print(f"{c['grade']} {c['score']:>6} {c['name'][:38]:38} "
              f"IS {c['is']['n']:>3}笔 {c['is']['expect_r']:+.3f}R | "
              f"OOS {c['holdout']['n']:>3}笔 {c['holdout']['expect_r']:+.3f}R | "
              f"WFE {c['wfa']['wfe']} 窗口 {c['wfa']['pos_win']}/{c['wfa']['tested']} "
              f"nb {c['nb']['pos_share']} mc {c['mc']['p_neg']}")
