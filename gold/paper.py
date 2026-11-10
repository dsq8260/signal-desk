"""多策略并行模拟盘（forward paper trading）。

为什么需要它：MT5 的「策略测试器」一次只能跑一个 EA，10 个策略要手动跑 10 遍；
而这里可以让 N 个策略共用一条实时行情，各自记自己的成交与净值，你只看一张表。

设计要点
--------
1. **只读**：不调用任何下单接口，成交是本地记账。
2. **无未来函数**：信号在第 t 根**已收盘**的 bar 上计算，成交记在 t 的收盘价上；
   下一根 bar 收盘时才能再决策一次（与回测口径一致）。
3. **共用行情**：所有策略读同一张 bars 表，各自维护 position / cash / equity。
4. **落盘**：SQLite（`data/paper.db`），程序退出不丢，第二天接着跑。
5. **可核对**：每笔成交记录 signal_bar_time、fill_price、fee、pnl，能追到具体 bar。
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from dataclasses import dataclass

import pandas as pd

from gold.symbols import spread_for  # noqa: E402

DEFAULT_DB = "data/paper.db"
INIT_CASH = 10_000.0
LOT_UNITS = 1.0          # 0.01 手 = 1 盎司（XM GOLD# 合约 100 盎司）

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    tf TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL,
    PRIMARY KEY (tf, ts)
);
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL, kind TEXT NOT NULL, params TEXT NOT NULL,
    tf TEXT NOT NULL DEFAULT 'D1', units REAL NOT NULL DEFAULT 1.0,
    cash0 REAL NOT NULL DEFAULT 10000.0, created TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL, ts INTEGER NOT NULL,
    side TEXT NOT NULL, price REAL NOT NULL, units REAL NOT NULL,
    fee REAL NOT NULL, pnl REAL NOT NULL, note TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
    strategy_id INTEGER NOT NULL, ts INTEGER NOT NULL,
    position INTEGER NOT NULL, units REAL NOT NULL,
    cash REAL NOT NULL, equity REAL NOT NULL, price REAL NOT NULL,
    PRIMARY KEY (strategy_id, ts)
);
"""


def connect(db: str | pathlib.Path = DEFAULT_DB) -> sqlite3.Connection:
    p = pathlib.Path(db)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.executescript(SCHEMA)
    return con


# ---------------------------------------------------------------- 登记策略
def add_strategy(con: sqlite3.Connection, name: str, kind: str,
                 params: dict | None = None, tf: str = "D1",
                 units: float = LOT_UNITS, cash0: float = INIT_CASH,
                 created: str | None = None) -> int:
    cur = con.execute(
        "INSERT INTO strategies(name,kind,params,tf,units,cash0,created) "
        "VALUES(?,?,?,?,?,?,?)",
        (name, kind, json.dumps(params or {}, ensure_ascii=False), tf,
         float(units), float(cash0), created or pd.Timestamp.now().isoformat(timespec="seconds")))
    con.commit()
    return int(cur.lastrowid)


def list_strategies(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT id,name,kind,params,tf,units,cash0,created FROM strategies "
        "ORDER BY id", con)


def remove_strategy(con: sqlite3.Connection, sid: int) -> None:
    for t in ("trades", "snapshots", "strategies"):
        col = "id" if t == "strategies" else "strategy_id"
        con.execute(f"DELETE FROM {t} WHERE {col}=?", (sid,))
    con.commit()


# ---------------------------------------------------------------- 行情
def upsert_bars(con: sqlite3.Connection, df: pd.DataFrame, tf: str = "D1") -> int:
    """把新 bar 写进库（重复时间戳覆盖）。返回总根数。"""
    if df is None or len(df) == 0:
        return 0
    rows = [(tf, int(pd.Timestamp(ts).timestamp()), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]))
            for ts, r in df.iterrows()]
    con.executemany("INSERT OR REPLACE INTO bars(tf,ts,open,high,low,close) "
                    "VALUES(?,?,?,?,?,?)", rows)
    con.commit()
    return int(con.execute("SELECT COUNT(*) FROM bars WHERE tf=?", (tf,)).fetchone()[0])


def load_bars(con: sqlite3.Connection, tf: str = "D1") -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT ts,open,high,low,close FROM bars WHERE tf=? ORDER BY ts", con, params=(tf,))
    if len(df) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df.index = pd.to_datetime(df.pop("ts"), unit="s")
    return df[["open", "high", "low", "close"]].astype(float)


# ---------------------------------------------------------------- 引擎
@dataclass
class StepResult:
    strategy_id: int = 0
    name: str = ""
    action: str = ""          # buy / sell / hold
    price: float = 0.0
    pnl: float = 0.0
    equity: float = 0.0


def _target_position(bars: pd.DataFrame, kind: str, params: dict,
                     offset: int = 1) -> float:
    """取第 -1-offset 根 bar 的仓位信号（含过滤层，与回放同一口径）。"""
    if len(bars) < 3 + offset:
        return 0.0
    series = signal_series(bars, kind, params)
    if series is None or len(series) == 0:
        return 0.0
    idx = -1 - int(offset)
    if len(series) < abs(idx):
        return 0.0
    return float(series.iloc[idx])


def signal_series(bars: pd.DataFrame, kind: str, params: dict) -> pd.Series:
    """一次算出整段信号（0/1/-1），供回放与推进共用。

    支持三个"元参数"（从 params 里取出，不传给策略函数）：
      ``_filter`` 状态过滤名（无过滤/低波动/强趋势）
      ``_sl`` / ``_tp`` 止损止盈百分比

    为什么需要：自主设计出来的 EA 常带「+低波动」「+TP5%」这类附加条件，
    如果引擎忽略它们，前向验证跑的就是另一个策略，结论无效。

    历史说明：早期还有一个 ``_zhi``（日柱地支）元参数，随干支维度一起移除。
    旧账本里残留的 ``_zhi`` 会当作未知元参数丢弃（这些策略也已从账本清掉）。
    """
    from gold.combo import ENTRIES, FILTERS
    from gold.entries_ext import EXT_ENTRIES, apply_stops
    from gold.factors import build_features, regime_mask
    from gold.symbols import spread_for
    from gold.strategies import STRATEGIES

    params = dict(params or {})
    params.pop("_zhi", None)
    flt_name = params.pop("_filter", None)
    sl_p = params.pop("_sl", None)
    tp_p = params.pop("_tp", None)
    params.pop("_allow_short", None)
    # 丢掉所有未知的元参数（以 _ 开头），避免传给策略函数报 TypeError
    params = {k: v for k, v in params.items() if not str(k).startswith("_")}

    if kind in STRATEGIES:
        fn = STRATEGIES[kind]["fn"]
        kw = {**STRATEGIES[kind]["params"], **params}
    elif kind in ENTRIES:
        fn, default = ENTRIES[kind]
        kw = {**default, **params}
    elif kind in EXT_ENTRIES:
        # 自主设计新增的进场（TSMOM / 时变夏普 / 鳄鱼线）
        fn = EXT_ENTRIES[kind][0]
        kw = {**EXT_ENTRIES[kind][1], **params}
    else:
        return pd.Series(0.0, index=bars.index)
    if len(bars) < 3:
        return pd.Series(0.0, index=bars.index)

    out = pd.Series(fn(bars, **kw), index=bars.index).fillna(0.0)

    if flt_name and flt_name in FILTERS and FILTERS[flt_name] is not None:
        name, q, mode = FILTERS[flt_name]
        feat = build_features(bars)
        out = out * regime_mask(feat, name, q, mode).astype(float)

    if sl_p is not None or tp_p is not None:
        out = apply_stops(bars, out, sl_p, tp_p)

    return out.fillna(0.0)


def reset_strategy(con: sqlite3.Connection, sid: int) -> None:
    """清空某个策略的账本（成交 + 净值），保留登记信息。"""
    con.execute("DELETE FROM trades WHERE strategy_id=?", (sid,))
    con.execute("DELETE FROM snapshots WHERE strategy_id=?", (sid,))
    con.commit()


def replay_window(con: sqlite3.Connection, df,
                  start=None, end=None, spread_usd: float | None = None,
                  tf: str = "D1", reset: bool = True,
                  strategy_ids: list[int] | None = None,
                  slippage_bps: float = 0.0, leverage: float = 0.0,
                  stopout_pct: float = 0.5) -> pd.DataFrame:
    """回放一段历史。``spread_usd=None`` 时按每个策略登记的品种自动取点差。"""
    """在**指定历史时段**上回放所有已登记策略，并写入账本。

    df 可以是：
      - 一个 DataFrame：所有策略共用这份 K 线（按 tf 参数）；
      - 一个 {周期: DataFrame} 字典：每个策略用**自己登记的周期**回放。

    执行口径：第 t 根 bar 收盘执行 t-1 的信号（shift(1)），不使用未来数据。

    撮合参数（默认全关，历史结果不变）：
    - ``slippage_bps``：每笔滑点（价格的万分之几，买入加价、卖出减价）；
    - ``leverage``：杠杆（如 100）；>0 才计算保证金占用；
    - ``stopout_pct``：保证金率低于该值强制平仓（XM 约 0.5，即 50%）。
    """
    empty = pd.DataFrame(columns=["策略", "周期", "开始", "结束", "收益",
                                  "最大回撤", "成交", "期末净值", "bar数"])
    if df is None:
        return empty
    is_map = isinstance(df, dict)
    if not is_map and len(df) == 0:
        return empty

    strat = list_strategies(con)
    if strategy_ids:
        strat = strat[strat["id"].isin([int(x) for x in strategy_ids])]
    if len(strat) == 0:
        return empty

    fee = float(spread_usd) if spread_usd is not None else float(
        spread_for(str(strat.iloc[0].get("symbol") or "GOLD"))
        if len(strat) else 0.33)
    rows = []
    for row in strat.to_dict("records"):
        sid = int(row["id"])
        params = json.loads(row["params"] or "{}")
        units = float(row["units"])
        cash0 = float(row["cash0"])
        stf = str(row.get("tf") or tf)

        src = df.get(stf) if is_map else df
        if src is None or len(src) == 0:
            continue

        win = src
        if start is not None:
            win = win[win.index >= pd.Timestamp(start)]
        if end is not None:
            win = win[win.index <= pd.Timestamp(end)]
        if len(win) < 5:
            continue

        if reset:
            reset_strategy(con, sid)

        sig = signal_series(win, row["kind"], params)
        pos = sig.shift(1).fillna(0.0)      # t 的信号 t+1 生效

        cash, held = cash0, 0.0
        entry_px, margin = 0.0, 0.0
        n_trades = 0
        eq_curve = []
        for ts, price in win["close"].items():
            price = float(price)
            want = float(pos.loc[ts]) > 0.5
            holding = held > 0

            # ---- 撮合：滑点（买入加价、卖出减价）----
            slip = price * float(slippage_bps) / 10000.0
            buy_px, sell_px = price + slip, price - slip

            def _equity(px: float) -> float:
                """权益：旧口径=现金+持仓市值；杠杆口径=现金+保证金+浮盈。"""
                if leverage > 0:
                    return cash + margin + held * (px - entry_px)
                return cash + held * px

            # ---- 强平：保证金率低于阈值时强制平仓 ----
            stopped = False
            if leverage > 0 and held > 0:
                m_used = held * price / float(leverage)
                m_level = (_equity(price) / m_used) if m_used > 0 else float("inf")
                if m_level < float(stopout_pct):
                    cash += margin + held * (sell_px - entry_px) - held * fee
                    n_trades += 1
                    con.execute(
                        "INSERT INTO trades(strategy_id,ts,side,price,units,fee,pnl,note) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (sid, int(ts.timestamp()), "SELL", sell_px, units,
                         units * fee, 0.0,
                         f"强平@{ts.date()} 保证金率{m_level:.2f}"))
                    held, margin, entry_px = 0.0, 0.0, 0.0
                    holding = False
                    stopped = True

            if want and not holding and not stopped:
                if leverage > 0:
                    need_m = units * buy_px / float(leverage)
                    need = need_m + units * fee
                else:
                    need_m = 0.0
                    need = units * buy_px + units * fee
                if cash >= need:
                    cash -= need
                    held = units
                    entry_px = buy_px
                    margin = need_m
                    n_trades += 1
                    con.execute(
                        "INSERT INTO trades(strategy_id,ts,side,price,units,fee,pnl,note) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (sid, int(ts.timestamp()), "BUY", buy_px, units, units * fee,
                         0.0, f"replay@{ts.date()}"))
            elif not want and holding:
                if leverage > 0:
                    cash += margin + held * (sell_px - entry_px) - held * fee
                else:
                    cash += held * sell_px - held * fee
                n_trades += 1
                con.execute(
                    "INSERT INTO trades(strategy_id,ts,side,price,units,fee,pnl,note) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (sid, int(ts.timestamp()), "SELL", sell_px, units,
                     units * fee, 0.0, f"replay@{ts.date()}"))
                held, margin, entry_px = 0.0, 0.0, 0.0

            equity = _equity(price)
            eq_curve.append(equity)
            con.execute(
                "INSERT OR REPLACE INTO snapshots"
                "(strategy_id,ts,position,units,cash,equity,price) VALUES(?,?,?,?,?,?,?)",
                (sid, int(ts.timestamp()), 1 if held > 0 else 0, held, cash,
                 equity, price))

        eq = pd.Series(eq_curve, index=win.index[: len(eq_curve)])
        peak = eq.cummax()
        rows.append({
            "策略": row["name"],
            "周期": stf,
            "开始": str(win.index[0].date()),
            "结束": str(win.index[-1].date()),
            "收益": float(eq.iloc[-1] / cash0 - 1.0) if cash0 else float("nan"),
            "最大回撤": float((eq / peak - 1.0).min()) if len(eq) else float("nan"),
            "成交": n_trades,
            "期末净值": float(eq.iloc[-1]) if len(eq) else float("nan"),
            "bar数": int(len(win)),
        })

    con.commit()
    return pd.DataFrame(rows)


def step(con: sqlite3.Connection, spread_usd: float | None = None,
         tf: str = "D1") -> list[StepResult]:
    """对所有已登记策略推进一根 bar（就地记账，幂等：同一 bar 不重复成交）。"""
    bars = load_bars(con, tf)
    if len(bars) < 3:
        return []
    ts = int(bars.index[-1].timestamp())
    price = float(bars["close"].iloc[-1])
    fee_per_unit = float(spread_usd) if spread_usd is not None else float(
        spread_for(str(bars.get("symbol") if hasattr(bars, "get") else "")
                   or "GOLD"))
    out: list[StepResult] = []

    for row in list_strategies(con).to_dict("records"):
        sid = int(row["id"])
        params = json.loads(row["params"] or "{}")
        units = float(row["units"])

        snap = con.execute(
            "SELECT position,units,cash,equity,ts FROM snapshots "
            "WHERE strategy_id=? ORDER BY ts DESC LIMIT 1", (sid,)).fetchone()
        pos_units, cash = (snap[1], snap[2]) if snap else (0.0, float(row["cash0"]))
        last_ts = int(snap[4]) if snap else 0

        if ts <= last_ts:            # 这根 bar 已经处理过
            out.append(StepResult(sid, row["name"], "hold", price, 0.0,
                                  float(cash + pos_units * price)))
            continue

        target = _target_position(bars, row["kind"], params, offset=1)
        want = target > 0.5
        holding = pos_units > 0
        action, pnl = "hold", 0.0

        if want and not holding:
            cost = units * price + units * fee_per_unit
            if cash >= cost:
                cash -= cost
                pos_units = units
                action = "buy"
                con.execute(
                    "INSERT INTO trades(strategy_id,ts,side,price,units,fee,pnl,note) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (sid, ts, "BUY", price, units, units * fee_per_unit, 0.0,
                     f"signal@{pd.to_datetime(ts, unit='s').date()}"))
        elif not want and holding:
            cash += pos_units * price - pos_units * fee_per_unit
            pnl = 0.0
            pos_units = 0.0
            action = "sell"
            con.execute(
                "INSERT INTO trades(strategy_id,ts,side,price,units,fee,pnl,note) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (sid, ts, "SELL", price, units, units * fee_per_unit, 0.0,
                     f"signal@{pd.to_datetime(ts, unit='s').date()}"))

        equity = cash + pos_units * price
        con.execute(
            "INSERT OR REPLACE INTO snapshots(strategy_id,ts,position,units,cash,equity,price) "
            "VALUES(?,?,?,?,?,?,?)",
            (sid, ts, 1 if pos_units > 0 else 0, pos_units, cash, equity, price))
        out.append(StepResult(sid, row["name"], action, price, pnl, equity))

    con.commit()
    return out


# ---------------------------------------------------------------- 每日看板
def dashboard(con: sqlite3.Connection) -> pd.DataFrame:
    """每个策略一行：持仓、成交次数、净值、当日盈亏、最大回撤。"""
    rows = []
    for row in list_strategies(con).to_dict("records"):
        sid = int(row["id"])
        snaps = pd.read_sql_query(
            "SELECT ts,equity,position,price FROM snapshots WHERE strategy_id=? "
            "ORDER BY ts", con, params=(sid,))
        trades = pd.read_sql_query(
            "SELECT ts,side,price,units,fee FROM trades WHERE strategy_id=? "
            "ORDER BY ts", con, params=(sid,))
        if len(snaps) == 0:
            rows.append({"策略": row["name"], "状态": "等待第一根 bar",
                         "持仓": "—", "成交": 0, "净值": float(row["cash0"]),
                         "当日盈亏": 0.0, "累计收益": 0.0, "最大回撤": 0.0,
                         "最近更新": "—"})
            continue

        eq = snaps["equity"].astype(float)
        cash0 = float(row["cash0"])
        day_pnl = float(eq.iloc[-1] - eq.iloc[-2]) if len(eq) > 1 else 0.0
        peak = eq.cummax()
        mdd = float((eq / peak - 1.0).min()) if len(eq) else 0.0
        rows.append({
            "策略": row["name"],
            "状态": "持仓中" if int(snaps["position"].iloc[-1]) else "空仓",
            "持仓": f"{row['units']:g} 单位 @ {snaps['price'].iloc[-1]:.2f}"
                    if int(snaps["position"].iloc[-1]) else "—",
            "成交": int(len(trades)),
            "净值": float(eq.iloc[-1]),
            "当日盈亏": day_pnl,
            "累计收益": float(eq.iloc[-1] / cash0 - 1.0),
            "最大回撤": mdd,
            "最近更新": pd.to_datetime(int(snaps["ts"].iloc[-1]), unit="s")
                        .strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows)


def seed_defaults(con: sqlite3.Connection) -> int:
    """首次使用时登记排行榜前几名（幂等：已存在的名字跳过）。"""
    existing = set(list_strategies(con)["name"]) if len(list_strategies(con)) else set()
    want = [
        ("SMA金叉 20/60", "双均线", {"fast": 20, "slow": 60}),
        ("SMA金叉 20/90", "双均线", {"fast": 20, "slow": 90}),
        ("唐奇安 20/10", "唐奇安通道突破", {"entry": 20, "exit_": 10}),
    ]
    n = 0
    for name, kind, params in want:
        if name in existing:
            continue
        add_strategy(con, name, kind, params)
        n += 1
    return n
