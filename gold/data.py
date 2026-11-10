"""黄金行情数据：MT5 优先，COMEX 日线兜底。

为什么要有兜底：
MT5 的 `copy_rates_*` 需要终端先下载过该品种历史（打开过图表），
否则返回 (-1, 'Terminal: Call failed')。在历史就绪之前，用 AkShare 的
COMEX 黄金期货（GC）日线顶上——它与现货金基差通常只有几美元，
做日线级策略研究足够；界面会明确标注当前用的是哪个源。
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import time
from dataclasses import dataclass, field

import pandas as pd

MT5_TERMINAL = r"C:\Program Files\XM Global MT5\terminal64.exe"
MT5_GOLD_SYMBOLS = ("GOLD#", "XAUUSD", "GOLD", "XAUUSDm", "GOLD24-7#")

# 内部周期名 → MT5 常量名
from gold import bridge as mt5_file_bridge  # noqa: E402
from gold.timeframes import (  # noqa: E402
    ALL_TFS,
    INTRADAY_TFS,
    TF_NAMES,
    intraday_bars,
    resample_bars,
)


@dataclass
class Bars:
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: str = ""
    note: str = ""
    symbol: str = ""
    period: str = "D1"
    fetched_at: str = ""

    @property
    def ok(self) -> bool:
        return len(self.df) > 0


# MT5 Python IPC 熔断状态（进程级 + 落盘）：
# 终端 build 高于 MetaTrader5 包版本时，initialize 会卡满 60 秒才失败（timeout 参数
# 对 IPC 握手无效），所以失败一次就写标记文件，之后所有调用立即跳过，
# 直到手动 reset_ipc() 或删掉标记文件。
_IPC_STATE: dict = {"failed": False, "reason": ""}
_IPC_MARKER = pathlib.Path("data/.mt5_ipc_broken.json")


def _load_ipc_marker() -> None:
    try:
        if _IPC_MARKER.exists():
            data = json.loads(_IPC_MARKER.read_text(encoding="utf-8"))
            _IPC_STATE["failed"] = True
            _IPC_STATE["reason"] = str(data.get("reason", "MT5 IPC 不可用"))
    except Exception:  # noqa: BLE001
        pass


def _save_ipc_marker(reason: str) -> None:
    try:
        _IPC_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _IPC_MARKER.write_text(
            json.dumps({"ts": dt.datetime.now().isoformat(timespec="seconds"),
                        "reason": reason}, ensure_ascii=False),
            encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def ipc_status() -> dict:
    """给界面显示用：MT5 Python 通道是否可用、失败原因。"""
    return {"failed": bool(_IPC_STATE["failed"]),
            "reason": str(_IPC_STATE["reason"])}


def reset_ipc() -> None:
    """手动重置熔断（升级了包或换了终端之后）。"""
    _IPC_STATE["failed"] = False
    _IPC_STATE["reason"] = ""
    try:
        if _IPC_MARKER.exists():
            _IPC_MARKER.unlink()
    except Exception:  # noqa: BLE001
        pass


_load_ipc_marker()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名、索引与类型。"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in d.columns]
    if "close" not in keep:
        return pd.DataFrame()
    d = d[keep]
    if not isinstance(d.index, pd.DatetimeIndex):
        try:
            d.index = pd.to_datetime(d.index)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    d = d[~d.index.duplicated(keep="last")].sort_index()
    for c in ("open", "high", "low"):
        if c not in d.columns:
            d[c] = d["close"]
    if "volume" not in d.columns:
        d["volume"] = 0.0
    return d[["open", "high", "low", "close", "volume"]].astype(float).dropna(subset=["close"])


def from_mt5(period: str = "D1", count: int = 60000,
             symbol: str | None = None,
             terminal: str = MT5_TERMINAL) -> Bars:
    """从 MT5 读 K 线。连不上 / 没历史都返回空 Bars，不抛异常。"""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        import MetaTrader5 as mt5
    except Exception:  # noqa: BLE001
        return Bars(source="mt5", note="未安装 MetaTrader5 包", period=period,
                    fetched_at=now)

    # 熔断：终端 build 高于 Python 包版本时，每次 initialize 都会卡 60 秒才失败，
    # 一旦失败就在本进程内不再重试，避免把界面拖死。
    if _IPC_STATE["failed"]:
        return Bars(source="mt5", note=_IPC_STATE["reason"], period=period,
                    fetched_at=now)

    # timeout 参数对 IPC 握手无效（内部固定 60 秒），所以只试一次；
    # 失败后写标记，本机后续所有调用立即跳过。
    ok = mt5.initialize(path=terminal)
    if not ok:
        _IPC_STATE["failed"] = True
        _IPC_STATE["reason"] = (
            f"MT5 Python IPC 不可用：{mt5.last_error()} "
            f"（Python 包 {getattr(mt5, '__version__', '?')} 低于终端 build 时会这样，"
            f"等官方发新版包或改用数据桥）")
        _save_ipc_marker(_IPC_STATE["reason"])
        return Bars(source="mt5", note=_IPC_STATE["reason"],
                    period=period, fetched_at=now)
    try:
        tf = getattr(mt5, TF_NAMES.get(period, "TIMEFRAME_D1"), mt5.TIMEFRAME_D1)
        cands = [symbol] if symbol else list(MT5_GOLD_SYMBOLS)
        for sym in cands:
            if not sym or not mt5.symbol_select(sym, True):
                continue
            rates = mt5.copy_rates_from_pos(sym, tf, 0, count)
            if rates is None or len(rates) == 0:
                continue
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.set_index("time")
            out = _clean(df)
            if len(out):
                return Bars(df=out, source="mt5", symbol=sym, period=period,
                            note=f"MT5 {sym} {period}", fetched_at=now)
        return Bars(source="mt5", note=f"终端无历史数据 {mt5.last_error()}",
                    period=period, fetched_at=now)
    finally:
        mt5.shutdown()


def from_akshare(period: str = "D1") -> Bars:
    """兜底：COMEX 黄金期货（GC）日线。只有日线。"""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if period.upper() not in ("D1", "D", "1D"):
        return Bars(source="akshare", note="兜底源只有日线", period=period,
                    fetched_at=now)
    raw = None
    last_err = ""
    for attempt in range(3):
        try:
            import akshare as ak

            raw = ak.futures_foreign_hist(symbol="GC")
            if raw is not None and len(raw):
                break
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:80]}"
            time.sleep(1.5 * (attempt + 1))       # 1.5s / 3s / 4.5s
    if raw is None or len(raw) == 0:
        return Bars(source="akshare",
                    note=f"取数失败（重试 3 次）：{last_err or '空结果'}",
                    period=period, fetched_at=now)
    if raw is None or len(raw) == 0:
        return Bars(source="akshare", note="返回空", period=period, fetched_at=now)
    df = raw.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    out = _clean(df)
    return Bars(df=out, source="akshare", symbol="GC", period="D1",
                note="COMEX 黄金期货 GC 日线（降级源）", fetched_at=now)


def load_gold_bars(period: str = "D1", count: int = 60000,
                   prefer: str = "mt5", symbol: str | None = None,
                   terminal: str = MT5_TERMINAL) -> Bars:
    """按 prefer 顺序取数：mt5 → akshare；也可 prefer='akshare' 直连兜底。"""
    # 0) 数据桥（GoldBridge.mq5 写到公共文件夹的 CSV）——真 MT5 数据，最优先
    #    注意：新版 EA 写的是 gold_bars_<SYM>_<TF>.csv，旧版写 gold_bars_<TF>.csv。
    #    必须先读带品种名的文件，否则会读到旧 EA 留下的、早已停止更新的文件。
    br = None
    # 指定了品种就只读它；没指定才按 GOLD → 旧版单品种文件 的顺序回退
    _cands = (symbol,) if symbol else ("GOLD", None)
    for _sym in _cands:
        try:
            br = mt5_file_bridge.bars_any(period, max_bars=count, sym=_sym)
        except Exception:  # noqa: BLE001
            br = None
        if br is not None and len(br):
            break
    if br is not None and len(br):
        return Bars(df=br, source="bridge",
                    symbol=(symbol or "GOLD"), period=period,
                    note=f"MT5 数据桥 GoldBridge · {len(br)} 根",
                    fetched_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 分钟级：本地 M1 聚合（MT5 历史不可用时的唯一来源）
    if period in INTRADAY_TFS:
        local = intraday_bars(period)
        if len(local):
            return Bars(df=local, source="local", symbol="GOLD#", period=period,
                        note=f"本地 M1 聚合 · {len(local)} 根 · 由 gold.ticker 采集",
                        fetched_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return Bars(source="local", period=period,
                    note="分钟级暂无数据：先跑 python -m gold.ticker 采集 M1",
                    fetched_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 周线：先取日线，再本地聚合（没有任何源直接提供周线）
    if period == "W1":
        daily = load_gold_bars(period="D1", count=count, prefer=prefer,
                               symbol=symbol, terminal=terminal)
        if daily.ok:
            daily.df = resample_bars(daily.df, "W1")
            daily.period = "W1"
            daily.note = f"{daily.note} → 周线聚合（{len(daily.df)} 根）"
            return daily
        return Bars(source="none", period="W1", note=f"周线无源：{daily.note}",
                    fetched_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    order = ["mt5", "akshare"] if prefer == "mt5" else ["akshare", "mt5"]
    notes = []
    for src in order:
        bars = (from_mt5(period=period, count=count, symbol=symbol, terminal=terminal)
                if src == "mt5" else from_akshare(period=period))
        if bars.ok:
            if notes:
                bars.note = f"{bars.note}（{'；'.join(notes)}）"
            return bars
        notes.append(f"{src}: {bars.note}")
    return Bars(source="none", note=" | ".join(notes), period=period,
                fetched_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
