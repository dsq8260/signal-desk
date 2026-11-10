"""Tick → M1 bar 收集器（本地方案，绕开 MT5 历史不可用）。

用法：
    .venv\\Scripts\\python -m gold.ticker --minutes 60      # 采 60 分钟后退出
    .venv\\Scripts\\python -m gold.ticker                  # 一直采，Ctrl+C 停

它每 1 秒读一次 MT5 的 GOLD# tick，累积成 1 分钟 bar 写进 data/paper.db 的 bars 表（tf='M1'）。
M5/M15/M30/H1/H4 由 M1 聚合（见 gold/timeframes.py），无需再采。

MT5 的 Python API 没有 tick 回调，只能轮询；1 秒一次对本地终端负担极小。
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sqlite3
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gold.paper import connect  # noqa: E402
from gold.timeframes import PAPER_DB  # noqa: E402

SYMBOL_CANDIDATES = ("GOLD#", "XAUUSD", "GOLD")


def _minute_key(ts: dt.datetime) -> int:
    return int(ts.replace(second=0, microsecond=0).timestamp())


def collect(minutes: float = 0, interval: float = 1.0,
            db: str = PAPER_DB, symbol: str | None = None,
            verbose: bool = True) -> dict:
    """采集 tick 并累积 M1 bar。返回统计。"""
    try:
        import MetaTrader5 as mt5
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"未安装 MetaTrader5: {e}"}

    if not mt5.initialize(path=r"C:\Program Files\XM Global MT5\terminal64.exe"):
        return {"ok": False, "error": f"连不上终端：{mt5.last_error()}"}

    sym = symbol or next((s for s in SYMBOL_CANDIDATES if mt5.symbol_select(s, True)),
                         None)
    if not sym:
        mt5.shutdown()
        return {"ok": False, "error": "找不到黄金品种（GOLD#/XAUUSD）"}

    con = connect(db)
    bars: dict[int, dict] = {}
    ticks = 0
    written = 0
    deadline = time.time() + minutes * 60 if minutes > 0 else None

    try:
        while True:
            if deadline and time.time() >= deadline:
                break
            t = mt5.symbol_info_tick(sym)
            if t is None or t.bid <= 0:
                time.sleep(interval)
                continue
            ticks += 1
            ts = dt.datetime.fromtimestamp(t.time)
            key = _minute_key(ts)
            price = float(t.bid)
            cur = bars.get(key)
            if cur is None:
                bars[key] = {"open": price, "high": price, "low": price, "close": price}
            else:
                cur["high"] = max(cur["high"], price)
                cur["low"] = min(cur["low"], price)
                cur["close"] = price

            # 上一分钟收口了就落盘（只写已完成的 bar）
            # 注意：必须用 tick 自带时间判断，不能用本地 now——
            # 券商服务器时区与本地常差数小时，混用会导致永远不落盘。
            for k in sorted(bars):
                if k >= key:
                    continue
                b = bars.pop(k)
                con.execute(
                    "INSERT OR REPLACE INTO bars(tf,ts,open,high,low,close) "
                    "VALUES('M1',?,?,?,?,?)",
                    (k, b["open"], b["high"], b["low"], b["close"]))
                written += 1
            con.commit()

            if verbose and ticks % 60 == 0:
                print(f"  tick {ticks} · 已写入 M1 {written} 根 · 当前价 {price:.2f}",
                      flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        con.commit()
        con.close()
        mt5.shutdown()

    return {"ok": True, "ticks": ticks, "bars_written": written,
            "symbol": sym, "minutes": minutes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=0, help="采集时长（分钟），0=不限")
    ap.add_argument("--interval", type=float, default=1.0, help="轮询间隔秒")
    ap.add_argument("--db", default=PAPER_DB)
    ap.add_argument("--symbol", default=None)
    args = ap.parse_args()

    print(f"开始采集（间隔 {args.interval}s，时长 "
          f"{'不限' if not args.minutes else f'{args.minutes} 分钟'}）…")
    r = collect(args.minutes, args.interval, args.db, args.symbol)
    if not r.get("ok"):
        print(f"失败：{r['error']}")
        return 1
    print(f"结束：{r['symbol']} 收到 {r['ticks']} 个 tick，"
          f"写入 {r['bars_written']} 根 M1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
