"""读取 MT5 数据桥（GoldBridge.mq5）写到公共文件夹的 CSV。

为什么要这条通道
----------------
本机 MT5 Python API 的 `copy_rates_*` 一直返回 `(-1, 'Terminal: Call failed')`，
拿不到历史 K 线；但 EA 里的 `CopyRates()` 走终端内部通道是正常的。
于是让 EA 把数据写成文件，Python 读文件 —— 绕开那个 bug。

文件位置
--------
MT5 公共文件夹：`%APPDATA%\\MetaQuotes\\Terminal\\Common\\Files\\`
  - gold_tick.csv         追加写入，每行 `time_msc,bid,ask`
  - gold_bars_<TF>.csv     整份重写，`time,open,high,low,close,tick_volume`

注意：这些文件由 EA 写，属于外部输入；解析时对坏行/缺列一律跳过，不抛异常。
"""

from __future__ import annotations

import os
import pathlib

import pandas as pd

TFS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]


def safe_name(sym: str) -> str:
    """与 EA 的 SafeName() 保持一致：只保留字母数字和下划线。"""
    return "".join(ch for ch in str(sym) if ch.isalnum() or ch == "_") or "SYM"


def common_dir() -> pathlib.Path:
    """MT5 公共文件夹（所有终端共享）。

    云服务器上没有 MT5，也没有 Windows 的 APPDATA——用 SD_BRIDGE_DIR 指向
    「数据中继」落地目录（本机 push_bridge.py 把桥文件推上去），其余逻辑不变。
    """
    env = os.environ.get("SD_BRIDGE_DIR")
    if env:
        return pathlib.Path(env)
    return (pathlib.Path(os.environ.get("APPDATA", pathlib.Path.home()))
            / "MetaQuotes" / "Terminal" / "Common" / "Files")


def _tick_path(d: pathlib.Path, sym: str | None = None) -> pathlib.Path:
    if sym:
        return d / f"gold_tick_{safe_name(sym)}.csv"
    return d / "gold_tick.csv"


def _bars_path(d: pathlib.Path, tf: str, sym: str | None = None) -> pathlib.Path:
    """sym=None 读旧版单品种文件；给了 sym 读 gold_bars_<SYM>_<TF>.csv。"""
    if sym:
        return d / f"gold_bars_{safe_name(sym)}_{tf}.csv"
    return d / f"gold_bars_{tf}.csv"


def symbols(d: pathlib.Path | None = None) -> list[str]:
    """扫描公共文件夹，列出桥导出的所有品种（按 H1 文件推断）。"""
    d = pathlib.Path(d) if d else common_dir()
    if not d.exists():
        return []
    out = []
    for p in d.glob("gold_bars_*_H1.csv"):
        name = p.name[len("gold_bars_"):-len("_H1.csv")]
        if name:
            out.append(name)
    return sorted(out)


def available(d: pathlib.Path | None = None) -> bool:
    """桥是否在工作（任一文件存在即算）。"""
    d = pathlib.Path(d) if d else common_dir()
    if not d.exists():
        return False
    return _tick_path(d).exists() or any(_bars_path(d, tf).exists() for tf in TFS)


def latest_tick(d: pathlib.Path | None = None,
                sym: str | None = None) -> dict | None:
    """返回最后一笔 tick：{'time_msc','bid','ask'}；没有则 None。

    sym=None 读旧版 gold_tick.csv；给了 sym 读 gold_tick_<SYM>.csv。
    """
    d = pathlib.Path(d) if d else common_dir()
    p = _tick_path(d, sym)
    if not p.exists() and sym:
        p = _tick_path(d)                   # 回退到旧版单品种 tick 文件
    if not p.exists():
        return None
    try:
        # 只读文件尾部 8KB，避免 tick 文件变大后拖慢
        size = p.stat().st_size
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
            return {"time_msc": int(parts[0]), "bid": float(parts[1]),
                    "ask": float(parts[2])}
        except ValueError:
            continue
    return None


def bars(tf: str, d: pathlib.Path | None = None,
         max_bars: int | None = None,
         sym: str | None = None) -> pd.DataFrame:
    """读某个周期（可选某品种）的 K 线；没有文件或解析失败返回空 DataFrame。

    sym=None 读旧版单品种文件 gold_bars_<TF>.csv；
    给了 sym 读多品种文件 gold_bars_<SYM>_<TF>.csv。
    """
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    d = pathlib.Path(d) if d else common_dir()
    p = _bars_path(d, tf, sym)
    if not p.exists() and sym:
        p = _bars_path(d, tf)               # 回退到旧版单品种文件
    if not p.exists():
        return empty
    try:
        df = pd.read_csv(p)
        # 合并分页文件（gold_bars_<SYM>_<TF>_p1.csv ...），去掉重复时间戳
        base = p.with_suffix("")
        pages = sorted(base.parent.glob(f"{base.name}_p*.csv"))
        if pages:
            frames = [df]
            for pp in pages:
                try:
                    frames.append(pd.read_csv(pp))
                except Exception:  # noqa: BLE001
                    continue
            df = pd.concat(frames, ignore_index=True)
            df = df.drop_duplicates(subset=["time"]).sort_values("time")
    except Exception:  # noqa: BLE001
        return empty
    if len(df) == 0 or "time" not in df.columns:
        return empty

    need = {"open", "high", "low", "close"}
    if not need.issubset(df.columns):
        return empty

    df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    if "tick_volume" in df.columns:
        df["volume"] = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0

    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out = out[~out.index.duplicated(keep="last")]
    if max_bars:
        out = out.tail(int(max_bars))
    return out


def panel(tf: str, syms: list[str] | None = None,
          d: pathlib.Path | None = None,
          max_bars: int | None = None) -> dict[str, pd.DataFrame]:
    """读多品种面板：{symbol: bars}。syms 缺省时自动扫描。"""
    d = pathlib.Path(d) if d else common_dir()
    targets = syms or symbols(d)
    out = {}
    for s in targets:
        df = bars(tf, d, max_bars, sym=s)
        if len(df):
            out[s] = df
    return out


def bars_any(tf: str, d: pathlib.Path | None = None,
             max_bars: int | None = None,
             sym: str | None = None) -> pd.DataFrame:
    """同 bars()，但对不认识的周期名返回空（不抛异常）。"""
    if tf not in TFS:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return bars(tf, d, max_bars, sym=sym)


def status(d: pathlib.Path | None = None) -> dict:
    """桥的健康状态，给界面显示用。"""
    d = pathlib.Path(d) if d else common_dir()
    rows = {tf: _bars_path(d, tf).exists() for tf in TFS}
    tick = latest_tick(d)
    return {
        "dir": str(d),
        "dir_exists": d.exists(),
        "tick_file": _tick_path(d).exists(),
        "tick": tick,
        "bars": rows,
        "symbols": symbols(d),
        "any": any(rows.values()) or tick is not None,
    }
