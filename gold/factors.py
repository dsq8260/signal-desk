"""因子层：市场状态特征 + 状态过滤。

思路：不追求"更准的进场信号"，而是回答"这个策略在什么状态下该关掉"。
常见有效状态：波动率分位、趋势强度、ATR 占比、距高点回撤。
用法：`gate = regime_mask(feat, "vol_pct", 0.7)` → `pos * gate`。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pct_rank(s: pd.Series, lookback: int) -> pd.Series:
    """滚动分位（0-1）：当前值在过去 lookback 期里的百分位。

    性能：原实现用 ``rolling.apply(lambda w: ...)``，每根 bar 调一次 Python 函数，
    33k 根要 0.43 秒；这里用滑窗视图**整段向量化**，降到几十毫秒。
    分块处理控制内存（每块约 400 万个数）。
    """
    lb = max(20, int(lookback))
    minp = max(20, lb // 4)
    arr = np.asarray(s, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < lb:
        return pd.Series(out, index=s.index)

    win = np.lib.stride_tricks.sliding_window_view(arr, lb)   # (n-lb+1, lb)
    rows = win.shape[0]
    chunk = max(1, 4_000_000 // lb)
    for start in range(0, rows, chunk):
        stop = min(rows, start + chunk)
        block = win[start:stop]
        last = block[:, -1]
        cnt = (block <= last[:, None]).sum(axis=1)
        valid = block.shape[1] - np.isnan(block).sum(axis=1)   # 有效样本数
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(valid >= minp, cnt / np.maximum(valid, 1), np.nan)
        out[start + lb - 1: stop + lb - 1] = frac

    return pd.Series(out, index=s.index)


def realized_vol(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["close"].astype(float).pct_change().rolling(
        int(n), min_periods=int(n) // 2).std(ddof=1) * np.sqrt(252)


def atr_pct(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR 占价格比例：跨品种可比的波动度量。

    直接复用 `strategies.atr`（Wilder 平滑），与 MT5 的 iATR 同口径；
    以前这里另写了一份「TR 简单均值」，会和导出 EA 里的过滤对不上。
    """
    from gold.strategies import atr as _atr

    return _atr(df, int(n)) / df["close"].astype(float)


def trend_strength(df: pd.DataFrame, n: int = 60) -> pd.Series:
    """趋势强度 = |N 期收益| / N 期平均 ATR 占比，越大越单边。"""
    ret = df["close"].astype(float).pct_change(int(n)).abs()
    vol = atr_pct(df, 14).rolling(int(n), min_periods=5).mean()
    return ret / vol.replace(0.0, np.nan)


def drawdown_state(df: pd.DataFrame, n: int = 250) -> pd.Series:
    """当前价距 N 期滚动高点的回撤（<=0）。"""
    close = df["close"].astype(float)
    peak = close.rolling(int(n), min_periods=20).max()
    return close / peak - 1.0


def build_features(df: pd.DataFrame, lookback: int = 250) -> pd.DataFrame:
    """汇总所有状态特征（含分位版本，便于做阈值过滤）。"""
    feat = pd.DataFrame(index=df.index)
    feat["vol"] = realized_vol(df, 20)
    feat["atr_pct"] = atr_pct(df, 14)
    feat["trend"] = trend_strength(df, 60)
    feat["dd"] = drawdown_state(df, 250)
    feat["vol_pct"] = _pct_rank(feat["vol"], lookback)
    feat["atr_pct_pct"] = _pct_rank(feat["atr_pct"], lookback)
    feat["trend_pct"] = _pct_rank(feat["trend"], lookback)
    return feat


def htf_trend(df: pd.DataFrame, mult: int = 4, span: int = 20) -> pd.Series:
    """大周期方向（默认 H4 = 4×H1）：H4 收盘相对 span 根前是否上行。

    返回布尔序列（与 df 同索引，已前向填充：H4 只在收盘后改变）。
    """
    import pandas as pd

    close = df["close"].astype(float)
    rule = f"{int(mult)}h"
    try:
        h4 = close.resample(rule).last().dropna()
    except Exception:  # noqa: BLE001
        # 无 DatetimeIndex 时退化为按 mult 根聚合
        h4 = close.groupby(close.index // int(mult)).last()
        h4.index = close.index[:: int(mult)][: len(h4)]
    up = h4 > h4.shift(int(span))
    return up.reindex(df.index, method="ffill").fillna(False)


def session_ok(df: pd.DataFrame, start_hour: int = 7,
               end_hour: int = 17) -> pd.Series:
    """时段过滤：只在这些小时（UTC）交易。默认 7-17 = 伦敦+纽约重叠。"""
    try:
        hours = df.index.hour
    except Exception:  # noqa: BLE001
        return pd.Series(True, index=df.index)
    return pd.Series((hours >= int(start_hour)) & (hours <= int(end_hour)),
                     index=df.index)


def regime_mask(feat: pd.DataFrame, name: str, q: float,
                mode: str = "below") -> pd.Series:
    """按分位阈值生成开关：below=只在分位低于 q 时交易；above=反之。

    `name` 用 `*_pct` 列（已是 0-1 分位）；NaN 期间默认关（不交易）。
    """
    if name not in feat.columns:
        raise KeyError(f"缺少特征列：{name}")
    col = feat[name].astype(float)
    mask = (col <= float(q)) if mode == "below" else (col >= float(q))
    return mask.fillna(False)


def gate_positions(pos: pd.Series, mask: pd.Series) -> pd.Series:
    """把信号乘上状态开关（不满足状态的期间强制空仓）。"""
    p = pd.Series(pos, dtype=float).reindex(mask.index).fillna(0.0)
    return (p * mask.astype(float)).clip(0.0, 1.0)


def factor_ic(feat: pd.DataFrame, fwd: pd.Series, name: str,
              method: str = "spearman") -> dict:
    """单因子与未来一期收益的相关性（IC），并给有效样本数。"""
    if name not in feat.columns:
        raise KeyError(f"缺少特征列：{name}")
    x = pd.Series(feat[name], dtype=float)
    y = pd.Series(fwd, dtype=float)
    df = pd.concat([x, y], axis=1).dropna()
    n = len(df)
    if n < 30:
        return {"ic": float("nan"), "n": n, "t": float("nan")}
    ic = float(df.iloc[:, 0].corr(df.iloc[:, 1], method=method))
    t = ic * np.sqrt(max(1, n - 2)) / max(1e-9, np.sqrt(max(1e-9, 1 - ic ** 2)))
    return {"ic": ic, "n": n, "t": float(t)}


FEATURE_DOCS = {
    "vol": "20 日已实现波动（年化）",
    "atr_pct": "ATR 占价格比例",
    "trend": "趋势强度（N 期收益 / 平均 ATR 占比）",
    "dd": "距 250 期滚动高点的回撤",
    "vol_pct": "波动率的滚动分位（0-1）",
    "atr_pct_pct": "ATR 占比的滚动分位",
    "trend_pct": "趋势强度的滚动分位",
}


# ---------------------------------------------------------------- 特征缓存
_FEAT_CACHE: dict = {}


def _df_key(df) -> tuple:
    """用"长度 + 首尾时间 + 收盘和"做轻量指纹，避免哈希整个 DataFrame。"""
    try:
        return (len(df), str(df.index[0]), str(df.index[-1]),
                round(float(df["close"].iloc[-1]), 6),
                round(float(df["close"].sum()), 3))
    except Exception:  # noqa: BLE001
        return (id(df),)


def build_features_cached(df):
    """带缓存的 build_features。

    组合实验室会为每个成分算一次特征；同一份数据重复算要 0.9 秒，
    缓存后只算一次。
    """
    key = _df_key(df)
    hit = _FEAT_CACHE.get(key)
    if hit is not None:
        return hit
    out = build_features(df)
    if len(_FEAT_CACHE) > 8:          # 只留最近几份，防内存膨胀
        _FEAT_CACHE.clear()
    _FEAT_CACHE[key] = out
    return out
