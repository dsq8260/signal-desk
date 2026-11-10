"""多品种影子副账本（Multi-Pair Shadow Ledger）

用途：让 EURUSD / GBPUSD 在**后台默默记影子样本**，加快样本积累，
      但绝不污染黄金主账本（shadow_book.json），也绝不进入主系统的工单面板。

为什么不用三审（ai_council）来记外汇：
  三审的上下文是黄金口径的（`_num(x, 2)` 截断），外汇价格会变成 1.12、ATR 变成 0.0，
  用它记外汇样本等于记垃圾。副账本因此改走 **02 智能辅助同源规则链**
  （assist.payload → plan.build 的 5 条件 + min_rr=2.5 + 禁逆势 + verdict 闸门），
  这条链按品种精度计算，是经过三品种实跑验证的正确路径，且不消耗 LLM。

隔离方式（物理级，全部在本进程内生效，不改任何既有模块）：
  * shadow.BOOK  -> data/shadow_book_multi.json   （主账本文件不被写入）
  * markets.json 保持 FX disabled —— 主服务、watchlist、推送、预热都不碰外汇
  * 不写 ai_council 的审计/状态文件，04 页看不到任何外汇记录

记录口径与主账本完全一致（复用 shadow.record / shadow.settle）：
  真实点差优先（拿不到用品种默认值）、点差+滑点计入 cost_r、M5 前向悲观结算。

用法：
  python shadow_multi.py --once     # 跑一轮（结算 + 各品种按 02 页规则入账）
  python shadow_multi.py --daemon   # 常驻，默认每 30 分钟一轮（与 M30 收盘对齐）
  python shadow_multi.py --state    # 打印副账本汇总 JSON
  python shadow_multi.py --reset --yes

环境变量：SD_MULTI_SYMS=EURUSD,GBPUSD  SD_MULTI_MIN=30  SD_MULTI_TARGET=100
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import socket
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE / "data"

import shadow       # noqa: E402
import markets      # noqa: E402
import assist       # noqa: E402

MAIN_BOOK = shadow.BOOK                       # 记下来只做校验，绝不写它
MULTI_BOOK = DATA / "shadow_book_multi.json"
LOG_FILE = DATA / "shadow_multi.log"

SYMS = [s.strip().upper() for s in
        (os.environ.get("SD_MULTI_SYMS") or "EURUSD,GBPUSD").split(",") if s.strip()]
INTERVAL_MIN = float(os.environ.get("SD_MULTI_MIN") or 30)
TARGET = int(os.environ.get("SD_MULTI_TARGET") or 100)
LOCK_PORT = int(os.environ.get("SD_MULTI_LOCK_PORT") or 8799)
CYCLE = "multi-pair-v1"


def log(msg: str) -> None:
    line = f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        DATA.mkdir(exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------ 隔离绑定
_CUR = {"sym": None}          # 当前正在记账的品种（供点差补丁使用）


def bind() -> None:
    """把本进程的账本路径改到副账本文件；主账本文件永远只读。

    同时给 shadow._spread 打一个「按当前品种取精度」的补丁：
    record 内部调用 `_spread(ctx)` 不带品种参数（默认黄金 2 位），
    外汇点差 0.00012 会被舍成 0.0，导致条目里点差字段失真。
    """
    shadow.BOOK = MULTI_BOOK
    shadow.CFG["target"] = TARGET
    assert shadow.BOOK != MAIN_BOOK, "隔离失败：主账本路径未被改写"

    if not getattr(shadow, "_multi_spread_patched", False):
        _orig = shadow._spread

        def _spread_with_sym(ctx: dict, sym: str | None = None) -> float:
            return _orig(ctx, sym or _CUR.get("sym") or "GOLD")

        shadow._spread = _spread_with_sym
        shadow._multi_spread_patched = True


def init_book() -> None:
    if MULTI_BOOK.exists():
        return
    d = {
        "items": [],
        "cycle": CYCLE,
        "cycle_note": "多品种副账本：仅影子记录，不入主账本、不进工单面板；"
                      "规则与 02 智能辅助同源（5 条件 + min_rr=2.5 + 禁逆势）",
        "cycle_started": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbols": list(SYMS),
        "engine": "assist.payload → plan.build（按品种精度，不经三审 LLM）",
        "main_book": str(MAIN_BOOK.name),
    }
    shadow._write(d)
    log(f"副账本已创建 · {MULTI_BOOK.name} · 品种 {SYMS}")


# ------------------------------------------------------------ 信号适配
def _pick_side(entry: dict) -> tuple[dict | None, dict | None]:
    """返回 (达标方案, 被拦方案)。择优口径与前端 briefPlan 一致：条件数→盈亏比。"""
    good, blocked = [], []
    for k in ("long", "short"):
        s = entry.get(k) or {}
        if s.get("entry") is None or s.get("stop") is None or s.get("tp1") is None:
            continue
        if s.get("qualify") and s.get("raw"):
            good.append(s)
        elif s.get("raw"):
            blocked.append(s)
    key = lambda c: ((c.get("n_ok") or 0), (c.get("rr") or 0))  # noqa: E731
    good.sort(key=key, reverse=True)
    blocked.sort(key=key, reverse=True)
    return (good[0] if good else None), (blocked[0] if blocked else None)


def _plan_of(s: dict) -> dict:
    return {"entry_zone": [s.get("entry"), s.get("entry")],
            "stop": s.get("stop"),
            "targets": [{"price": s.get("tp1")}]}


def _ctx_of(payload: dict, sym: str) -> dict:
    """构造记录上下文。

    注意：shadow.record 内部 `_spread(ctx)` 不带品种参数（默认黄金），所以
    必须在这里显式给出本品种点差，否则外汇会被记成黄金的 0.3，成本放大几千倍。
    无实时 tick 时用品种默认点差（EURUSD 1.2 点 / GBPUSD 1.5 点）。
    """
    q = payload.get("quote") or {}
    sp = q.get("spread")
    if not sp or float(sp) <= 0:
        sp = markets.spread_default(sym)
    return {"price": payload.get("price"), "spread": float(sp),
            "quote": q, "sym": sym}


def evaluate(sym: str) -> dict:
    """跑一次 02 页同源计算链，并把结果适配成 shadow.record 需要的三审结构。"""
    d = assist.payload(sym)
    entry = (d.get("plan") or {}).get("entry") or {}
    good, blocked = _pick_side(entry)
    cfg = dict(assist.load_config(),
               _contract=markets.contract(sym), _point=markets.point(sym))
    sizing = d.get("sizing") or {}
    lots = {"lots": sizing.get("lots"), "risk_usd": sizing.get("risk_usd")}
    ctx = _ctx_of(d, sym)
    lv = d.get("levels") if isinstance(d.get("levels"), dict) else {}
    zone_name = (lv.get("zone") or {}).get("zone") if isinstance(lv.get("zone"), dict) else None
    facts = {"zone": {"zone": zone_name},
             "market_state": (d.get("verdict") or {}).get("state")}
    res = {"a1": {"direction": None, "plan": None},
           "a2": {"verdict": "standby", "local_lots": lots},
           "facts": facts}
    if good:
        s = good[0]
        res["a1"] = {"direction": "buy" if s.get("long") else "sell",
                     "plan": _plan_of(s), "confidence": s.get("stars"),
                     "summary": f"{s.get('n_ok')}/5 条件达标，盈亏比 {s.get('rr')}",
                     "market_state": facts.get("market_state")}
        res["a2"]["verdict"] = "pass"
    elif blocked:
        s = blocked[0]
        res["a1"] = {"direction": None,
                     "ai_direction": "buy" if s.get("long") else "sell",
                     "ai_plan": _plan_of(s),
                     "summary": f"被拦：{s.get('block_reason')}",
                     "market_state": facts.get("market_state")}
        res["a2"]["verdict"] = "blocked"
    item = None
    if good or blocked:
        _CUR["sym"] = sym
        try:
            item = shadow.record(sym, cfg, res, ctx)     # 与主账本同一个记录函数
        finally:
            _CUR["sym"] = None
    return {"ok": bool(d.get("ok")), "price": d.get("price"),
            "min_rr": (d.get("config") or {}).get("min_rr"),
            "side": (res["a1"]["direction"] or res["a1"].get("ai_direction")),
            "verdict": res["a2"]["verdict"],
            "logged": item is not None,
            "rr": (good[0] if good else blocked[0]).get("rr") if (good or blocked) else None}


# ------------------------------------------------------------ 主流程
def run_once() -> dict:
    shadow.settle()                       # 先把上一轮的挂单/持仓推进
    out = {"at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "symbols": {}, "skipped": []}
    enabled = {(m.get("sym") if isinstance(m, dict) else m) for m in markets.enabled()}
    for sym in SYMS:
        if sym in enabled:
            # 用户若在 UI 里重新启用该品种：主账本会自己记，副账本让位，避免双记
            out["skipped"].append({"sym": sym, "why": "markets.enabled 已启用，主账本负责"})
            log(f"{sym} 已在主系统启用，副账本跳过（避免双记）")
            continue
        try:
            r = evaluate(sym)
        except Exception as e:  # noqa: BLE001
            out["symbols"][sym] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            log(f"{sym} 计算异常: {type(e).__name__}: {e}")
            continue
        out["symbols"][sym] = r
        log(f"{sym} · 价 {r.get('price')} · 门槛 rr {r.get('min_rr')} · "
            f"方向 {r.get('side') or '观望'} · {r.get('verdict')} · 入账={r.get('logged')}")
    shadow.settle()                       # 刚入账的单立刻尝试结算（挂单期没到就留着）
    return out


def summary() -> dict:
    shadow.settle()
    d = shadow._read()
    items = d.get("items") or []
    trade = [x for x in items if x.get("kind") == "trade"]
    blocked = [x for x in items if x.get("kind") == "blocked"]
    st = shadow._bucket(trade)
    st["target"] = TARGET
    st["progress"] = round(min(100.0, len(trade) / TARGET * 100), 1) if TARGET else 0.0
    by_sym = {}
    for sym in sorted({(x.get("sym") or "?") for x in items}):
        rows = [x for x in items if (x.get("sym") or "?") == sym and x.get("kind") == "trade"]
        by_sym[sym] = shadow._bucket(rows)
    spread_real = sum(1 for x in trade if (x.get("spread") or 0) > 0)
    return {
        "ok": True, "book": str(MULTI_BOOK.name), "cycle": d.get("cycle"),
        "cycle_started": d.get("cycle_started"), "symbols": d.get("symbols"),
        "target": TARGET, "items": len(items), "trade": st,
        "blocked_n": len(blocked), "by_sym": by_sym,
        "spread_recorded": spread_real,
        "main_book_untouched": str(MAIN_BOOK),
        "updated": d.get("updated"),
    }


def reset(confirm: bool) -> dict:
    if not confirm:
        return {"ok": False, "error": "需要 --yes 确认清空副账本"}
    shadow.reset()
    d = shadow._read()
    d.update({"cycle": CYCLE, "cycle_started": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "symbols": list(SYMS), "cycle_note": "副账本重置（仅影子记录）"})
    shadow._write(d)
    log("副账本已清空并重置周期")
    return {"ok": True, "items": 0}


def daemon() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", LOCK_PORT))
        s.listen(1)
    except OSError:
        log(f"端口 {LOCK_PORT} 已被占用：已有副账本实例在跑，本实例退出")
        return
    bind()
    init_book()
    log(f"副账本守护启动 · 品种 {SYMS} · 每 {INTERVAL_MIN:.0f} 分钟一轮 · "
        f"目标 {TARGET} 笔 · 主账本 {MAIN_BOOK.name} 只读不写")
    while True:
        t0 = time.time()
        try:
            run_once()
        except Exception as e:  # noqa: BLE001
            log(f"本轮异常: {type(e).__name__}: {e}")
            log(traceback.format_exc(limit=3))
        time.sleep(max(30.0, INTERVAL_MIN * 60 - (time.time() - t0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="跑一轮后退出")
    ap.add_argument("--daemon", action="store_true", help="常驻守护")
    ap.add_argument("--state", action="store_true", help="打印副账本汇总")
    ap.add_argument("--reset", action="store_true", help="清空副账本")
    ap.add_argument("--yes", action="store_true", help="配合 --reset 的确认")
    args = ap.parse_args()

    bind()
    init_book()

    if args.reset:
        print(json.dumps(reset(args.yes), ensure_ascii=False, indent=1))
        return
    if args.state:
        print(json.dumps(summary(), ensure_ascii=False, indent=1))
        return
    if args.once:
        out = run_once()
        print(json.dumps({"run": out, "summary": summary()}, ensure_ascii=False, indent=1))
        return
    try:
        daemon()
    except KeyboardInterrupt:
        log("收到中断，副账本退出（主账本与主服务不受影响）")


if __name__ == "__main__":
    main()
