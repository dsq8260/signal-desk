# -*- coding: utf-8 -*-
"""历史回放评分：把「02 智能辅助达标方案」在过去行情里重放一遍。

要回答的问题与 backtest.py 相同：**这套规则到底有没有正期望？**
但重放对象是当前 02 页主链路：cross → key_levels → indicators → patterns
→ verdict → plan.build（5 条件双方向方案），不是旧版 ai_trader 评分卡。

与影子账本完全同源的约定（shadow.py）：
1. 只用「当时已知」数据：assist._closed_rows 被打上时间截断补丁，任何时点
   只能看到该时点之前已收盘的 K 线（market.payload(before=...) 同窗口口径）。
2. 挂单语义：信号 = 回踩/反弹限价单。24h（M5×288 / H1×24）没碰到入场价
   → 记 missed，不计盈亏；入场后 48h 仍未到止损/目标 → 按收盘价结账。
3. 同一根 K 线同时触及止损与目标 → 按止损计（悲观原则）。
4. 成本必须计：点差 + 滑点按 markets.py 品种默认值，cost_r = 成本/单笔风险。
5. 单品种同一时刻只持有一笔（挂单中也算占用）；不使用 LLM 委员（无法离线
   重放），所以本脚本测的是「本地规则内核 + 用户配置的拦截开关」。
6. 账户侧风控（余额/持仓冲突）在回放里不存在：fake risk 无告警、无持仓，
   因此 qualify 与 raw 等价（除非配置里的 block_range/block_counter/block_rr 命中）。

用法：
    python replay_score.py                     # 三品种，近 6 个月
    python replay_score.py --months 12 --syms GOLD
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import market  # noqa: E402
import markets  # noqa: E402
import pxfmt  # noqa: E402
import assist  # noqa: E402
import context  # noqa: E402
import verdict  # noqa: E402
import plan as plan_mod  # noqa: E402

DATA = HERE / "data"
OUT_JSON = DATA / "replay_score.json"

BALANCE = 10000.0
EXPIRE_H = 24          # 挂单有效期（小时）
MAX_HOLD_H = 48        # 入场后最长持仓（小时）


# ---------------------------------------------------------------- 数据准备
def preload(sym: str) -> dict[str, list[list]]:
    tfs = {}
    for tf in ("D1", "H4", "H1", "M30", "M15", "M5"):
        rows, _src = market._load(sym, tf)
        tfs[tf] = rows or []
    return tfs


def _t_of(r: list) -> str:
    return r[0][:19]


# ------------------------------------------------------------ 截断补丁
_ORIG_CLOSED_ROWS = assist._closed_rows


def _install_cut(cut: str) -> None:
    """让 assist._closed_rows 只看到 cut 之前已收盘的 K 线。

    cut = 决策 bar 的收盘时刻。market.payload(before=cut) 返回开盘时间
    严格早于 cut 的 bars（指标在同一窗口内现算，与实盘 WARM 口径一致），
    其中最后一根就是刚收盘的决策 bar。
    """

    def _closed_rows(sym: str, tf: str, n: int = 300):
        p = market.payload(sym, tf, n, before=cut)
        if not p.get("ok"):
            return [], "", p
        raw = assist._rows(p.get("bars"))
        return raw, (raw[-1]["t"] if raw else ""), p

    assist._closed_rows = _closed_rows


def _restore() -> None:
    assist._closed_rows = _ORIG_CLOSED_ROWS


# ------------------------------------------------------------ 条件消融
ABLATE_KEY = {"trend": "趋势方向匹配", "levels": "关键价位触及",
              "indicator": "指标信号确认", "pattern": "形态确认"}


def _requalify(side_plan: dict, ablate: tuple) -> bool:
    """在「去掉若干条件」的口径下重判该方案是否达标（不触碰 plan.py）。

    与原版 raw 的严格度对齐：剩余条件中最多 1 个非 ok（只能是 wait）、
    不允许任何 fail；盈亏比达标(c5)永远保留（门槛由 min_rr 控制）；
    趋势匹配(c1)与关键位触及(c2)在被消融时豁免自己的原判。
    """
    conds = {c["name"]: c["s"] for c in side_plan.get("conds") or []}
    keys = [ABLATE_KEY.get(a) for a in ablate]
    if not keys or any(k not in conds for k in keys):
        return bool(side_plan.get("qualify") and side_plan.get("raw"))
    rem = {k: v for k, v in conds.items() if k not in keys}
    n_ok = sum(1 for v in rem.values() if v == "ok")
    n_fail = sum(1 for v in rem.values() if v == "fail")
    if n_fail or n_ok < len(rem) - 1:
        return False
    if "trend" not in ablate and rem.get("趋势方向匹配") != "ok":
        return False
    if "levels" not in ablate and rem.get("关键价位触及") not in ("ok", "wait"):
        return False
    return rem.get("盈亏比达标") == "ok"


def _pick_side(entry: dict, ablate: tuple | None) -> list[dict]:
    """按基线或消融口径挑出达标方案（可能 0/1/2 个）。"""
    cands = []
    for k in ("long", "short"):
        s = entry.get(k)
        if not s:
            continue
        if ablate:
            ok = _requalify(s, ablate)
        else:
            ok = bool(s.get("qualify") and s.get("raw"))
        if ok:
            cands.append(s)
    cands.sort(key=lambda c: (c.get("n_ok") or 0, c.get("rr") or 0), reverse=True)
    return cands


def evaluate(sym: str, cfg: dict, cut: str, ablate: tuple | None = None) -> dict | None:
    """在 cut 时刻跑一遍 02 页计算链，返回达标方案或 None。"""
    pxfmt.use(sym)
    cross, cross_w, packs = assist.cross_timeframe(sym, cfg)
    h1rows = packs.get("H1") or []
    if not h1rows:
        return None
    m15rows, _, _ = assist._closed_rows(sym, "M15", 150)
    m5rows, _, _ = assist._closed_rows(sym, "M5", 150)
    tf_extra = {"M15": assist._trend(m15rows, cfg, "M15") if m15rows else None,
                "M5": assist._trend(m5rows, cfg, "M5") if m5rows else None}
    price = float(h1rows[-1]["c"])
    # 伪造「新鲜报价」：实盘中页面开着 tick 就是新鲜的，报价闸不是被测变量。
    # 不伪造的话 verdict._gate 会把所有 raw 方案判成 blocked，回放永远 0 信号。
    quote = {"bid": price, "ask": price, "tick_time": cut, "age_s": 0,
             "ok": True, "spread_alert": False}
    fake_risk = {
        "alerts": [], "event": {}, "quote": quote,
        "account": {"balance": BALANCE, "equity": BALANCE},
        "positions": {"list": []},
        "manual": {"breached": [], "today_pnl": 0, "max_float_loss": 0,
                   "streak_loss": 0, "n_open_today": 0},
    }
    levels = assist.key_levels(packs, cfg, price)
    ind = assist.indicators(packs, cfg, cross[-1])
    sess = context.session_info(h1rows, packs.get("D1") or [], price,
                                cfg.get("server_utc_offset", 2),
                                ind["atr"]["value"])
    levels = context.enrich_levels(
        levels, h1rows, ind["atr"]["value"], ind,
        extra_lines=sess.get("lines"),
        ma_len=(cfg["ma_fast"], cfg["ma_mid"], cfg["ma_slow"]))
    patterns = assist.candle_patterns(packs, cfg, levels)
    vd = verdict.build(cross, levels, ind, fake_risk, price)
    jr = {"items": [], "today": {}, "stats": {}}
    pn = plan_mod.build(cross=cross, tf_extra=tf_extra, levels=levels, ind=ind,
                        risk=fake_risk, price=price, cfg=cfg, patterns=patterns,
                        journal=jr, vd=vd)
    entry = pn.get("entry") or {}
    cands = _pick_side(entry, ablate)
    if not cands:
        return None
    # 两边同时达标：按 (条件数, 盈亏比) 择优——与前端 briefPlan 的择优精神一致
    pick = cands[0]
    hour = int(cut[11:13])
    return {
        "sym": sym, "at": cut, "side": pick["side"],
        "direction": "buy" if pick["long"] else "sell",
        "entry": pick["entry"], "stop": pick["stop"], "tp1": pick["tp1"],
        "rr": pick["rr"], "stars": pick["stars"], "n_ok": pick["n_ok"],
        "conds": [(c["name"], c["s"]) for c in pick["conds"]],
        "trend_label": cross_w.get("label"), "align": cross_w.get("align"),
        "trend_side": (pn.get("trend") or {}).get("side"),
        "zone": pick.get("zone"), "hour": hour,
        "both_raw": len(cands) > 1,
        "risk": abs(float(pick["entry"]) - float(pick["stop"])),
    }


# ------------------------------------------------------------ 结算（shadow 同款）
def settle(sig: dict, tfs: dict) -> dict:
    sym = sig["sym"]
    use_m5 = sig["at"] >= (tfs["M5"][0][0][:19] if tfs["M5"] else "9")
    tf_rows = tfs["M5"] if use_m5 else tfs["H1"]
    if not tf_rows:
        return {"status": "no_data"}
    times = [r[0][:19] for r in tf_rows]
    start = bisect.bisect_left(times, sig["at"])  # 决策收盘之后的 bar 才算未来
    pend = tf_rows[start:]
    if not pend:
        return {"status": "no_data"}
    expire = 288 if use_m5 else EXPIRE_H
    max_hold = 576 if use_m5 else MAX_HOLD_H
    buy = sig["direction"] == "buy"
    entry, stop, tp1 = float(sig["entry"]), float(sig["stop"]), float(sig["tp1"])
    risk = sig["risk"]
    if risk <= 1e-9:
        return {"status": "no_data"}

    hit = None
    for j, r in enumerate(pend[:expire]):
        if (r[3] <= entry) if buy else (r[2] >= entry):
            hit = j
            break
    if hit is None:
        last = pend[min(len(pend), expire) - 1]
        return {"status": "missed", "settle_tf": "M5" if use_m5 else "H1",
                "entry_bar": None, "exit": round(last[4], markets.digits(sym)),
                "closed_at": last[0][:19], "r": None, "hold_bars": None}

    work = pend[hit:]
    for k, r in enumerate(work[:max_hold]):
        hi, lo, c = r[2], r[3], r[4]
        if (lo <= stop) if buy else (hi >= stop):
            raw_r = -1.0
            status, exit_px, bt = "lost", stop, r[0][:19]
            return _pack(sig, status, raw_r, exit_px, bt, k, risk, use_m5)
        if (hi >= tp1) if buy else (lo <= tp1):
            raw_r = abs(tp1 - entry) / risk
            status, exit_px, bt = "won", tp1, r[0][:19]
            return _pack(sig, status, raw_r, exit_px, bt, k, risk, use_m5)
    last = work[min(len(work), max_hold) - 1]
    raw_r = ((last[4] - entry) / risk) if buy else ((entry - last[4]) / risk)
    return _pack(sig, "expired", raw_r, last[4], last[0][:19],
                 min(len(work), max_hold) - 1, risk, use_m5)


def _pack(sig: dict, status: str, raw_r: float, exit_px: float,
          closed_at: str, hold: int, risk: float, use_m5: bool) -> dict:
    sym = sig["sym"]
    cost = markets.spread_default(sym) + markets.slip(sym)
    return {"status": status, "settle_tf": "M5" if use_m5 else "H1",
            "entry_bar": sig["at"], "closed_at": closed_at,
            "exit": round(exit_px, markets.digits(sym)),
            "r": round(raw_r, 3), "cost_r": round(cost / risk, 3),
            "net_r": round(raw_r - cost / risk, 3), "hold_bars": hold}


# ------------------------------------------------------------ 统计
def bucket(trades: list[dict]) -> dict:
    closed = [t for t in trades if t["status"] in ("won", "lost", "expired")]
    won = [t for t in closed if t["status"] == "won"]
    lost = [t for t in closed if t["status"] == "lost"]
    rs = [t["net_r"] for t in closed]
    gw = sum(r for r in rs if r > 0)
    gl = -sum(r for r in rs if r < 0)
    cum = 0.0
    peak = 0.0
    dd = 0.0
    for t in sorted(closed, key=lambda x: x["closed_at"] or ""):
        cum += t["net_r"]
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    return {
        "signals": len(trades), "closed": len(closed),
        "won": len(won), "lost": len(lost),
        "expired": sum(1 for t in closed if t["status"] == "expired"),
        "missed": sum(1 for t in trades if t["status"] == "missed"),
        "win_rate": round(len(won) / len(closed) * 100, 1) if closed else None,
        "avg_r": round(sum(rs) / len(rs), 3) if rs else None,
        "sum_r": round(sum(rs), 2),
        "pf": round(gw / gl, 2) if gl > 0 else None,
        "max_dd_r": round(dd, 2),
    }


def by_key(trades: list[dict], key) -> dict:
    groups: dict = {}
    for t in trades:
        groups.setdefault(key(t), []).append(t)
    return {k: bucket(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


# ------------------------------------------------------------ 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--syms", default="GOLD,EURUSD,GBPUSD")
    ap.add_argument("--min-rr", type=float, default=None,
                    help="覆盖 assist 配置的 min_rr（盈亏比门槛）")
    ap.add_argument("--block-counter", action="store_true",
                    help="开启「逆势单禁止开仓」")
    ap.add_argument("--block-range", action="store_true",
                    help="开启「震荡市禁止开仓」")
    ap.add_argument("--tag", default="", help="输出文件名后缀，用于对比不同配置")
    ap.add_argument("--ablate", default=None,
                    help="条件消融，逗号分隔可组合：trend,levels,indicator,pattern "
                         "（trend 被消融时同时豁免禁逆势）")
    args = ap.parse_args()

    ablate: tuple | None = None
    if args.ablate:
        parts = tuple(a.strip() for a in args.ablate.split(",") if a.strip())
        bad = [p for p in parts if p not in ABLATE_KEY]
        if bad:
            ap.error(f"--ablate 无效条件 {bad}，可选 {sorted(ABLATE_KEY)}")
        ablate = parts

    t0 = (dt.datetime.now() - dt.timedelta(days=args.months * 30)) \
        .strftime("%Y-%m-%d %H:%M:%S")
    syms = [s.strip().upper() for s in args.syms.split(",") if s.strip()]
    overrides = {}
    if args.min_rr is not None:
        overrides["min_rr"] = args.min_rr
    if args.block_counter:
        overrides["block_counter"] = True
    if args.block_range:
        overrides["block_range"] = True
    print(f"回放窗口 {t0} 起 · 品种 {syms} · 月数 {args.months} · "
          f"配置覆盖 {overrides or '无（用实盘配置）'} · 消融 {ablate or '无'}",
          flush=True)

    all_trades: list[dict] = []
    eval_count = 0
    err_count = 0
    t_start = time.time()
    for sym in syms:
        cfg = dict(assist.load_config(),
                   _contract=markets.contract(sym), _point=markets.point(sym))
        cfg.update(overrides)
        if ablate and "trend" in ablate:
            # 趋势匹配被消融时，禁逆势开关同属「趋势条件族」，一并豁免
            cfg["block_counter"] = False
        tfs = preload(sym)
        h1 = tfs["H1"]
        start_idx = max(1, bisect.bisect_left([_t_of(r) for r in h1], t0))
        print(f"[{sym}] H1 {len(h1)} 根 · 决策点从第 {start_idx} 根起", flush=True)
        resume_at = None          # 有单在身（含挂单期）时，跳过这段决策点
        for i in range(start_idx, len(h1)):
            cut = (dt.datetime.strptime(_t_of(h1[i]), "%Y-%m-%d %H:%M:%S")
                   + dt.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
            if cut > dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"):
                break
            if resume_at and cut <= resume_at:
                continue
            _install_cut(cut)
            try:
                sig = evaluate(sym, cfg, cut, ablate=ablate)
            except Exception as e:  # noqa: BLE001
                err_count += 1
                if err_count <= 5:
                    print(f"[{sym}] {cut} 计算异常: {type(e).__name__}: {e}", flush=True)
                continue
            finally:
                _restore()
            eval_count += 1
            if not sig:
                continue
            res = settle(sig, tfs)
            sig.update(res)
            all_trades.append(sig)
            # 结算是同步走完的：把决策点推进到这笔单结束（含 24h 挂单期）之后
            resume_at = res.get("closed_at")
            if len(all_trades) % 20 == 0:
                print(f"  已记 {len(all_trades)} 笔 · {cut}", flush=True)
        print(f"[{sym}] 完成 · 用时 {time.time() - t_start:.0f}s", flush=True)

    report = {
        "ok": True, "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "window_from": t0, "months": args.months, "symbols": syms,
        "config_overrides": overrides,
        "ablate": list(ablate) if ablate else None,
        "eval_points": eval_count, "errors": err_count,
        "config_note": "使用 assist_config.json 实际配置；回放无账户侧风控与 LLM 委员",
        "all": bucket(all_trades),
        "by_sym": by_key(all_trades, lambda t: t["sym"]),
        "by_dir": by_key(all_trades, lambda t: t["direction"]),
        "by_align": by_key(all_trades, lambda t: t.get("align") or "?"),
        "by_settle": by_key(all_trades, lambda t: t.get("settle_tf") or "?"),
        "by_session": by_key(all_trades, lambda t: (
            "亚盘" if t["hour"] < 7 else "欧盘" if t["hour"] < 14 else "美盘")),
        "by_rr": by_key(all_trades, lambda t: (
            "<1.5" if t["rr"] < 1.5 else "1.5-2.5" if t["rr"] < 2.5 else ">=2.5")),
        "trades": sorted(all_trades, key=lambda x: x["at"]),
    }
    out_path = DATA / (f"replay_score_{args.tag}.json" if args.tag
                       else "replay_score.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    b = report["all"]
    print("\n===== 回放总览 =====")
    print(f"信号 {b['signals']} · 已结 {b['closed']} · 胜率 {b['win_rate']}% · "
          f"均R {b['avg_r']} · 累计R {b['sum_r']} · PF {b['pf']} · 最大回撤 {b['max_dd_r']}R")
    for name, key in (("分品种", "by_sym"), ("分方向", "by_dir"),
                      ("分顺逆", "by_align"), ("分时段", "by_session"),
                      ("分盈亏比", "by_rr")):
        print(f"-- {name}")
        for k, v in report[key].items():
            print(f"   {k}: {v}")
    print(f"结果已写 {out_path} · 总用时 {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
