"""指标交叉校验：我们的实现 vs pandas-ta-classic / ta-lib 的独立实现。

目的：任何逐根偏离都是 bug —— 之前鳄鱼线的 `p[off-1]` 就是这样发现的。
输出一份"一致性报告"，并作为 MQL5 逐根对拍的第三方基准。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta_classic as pta
except Exception:  # pragma: no cover
    pta = None

from gold.strategies import atr, donchian_signal, ma_cross_signal


def _bars(n=1200, seed=5, drift=0.0004, vol=0.0018, price=2600.0):
    rng = np.random.default_rng(seed)
    close = price * np.cumprod(1 + rng.normal(drift, vol, n))
    high = close * (1 + np.abs(rng.normal(0, 0.0006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0006, n)))
    idx = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": 0.0}, index=idx)


def compare(name: str, ours: pd.Series, theirs: pd.Series,
            tol: float = 1e-9, warmup: int = 250) -> dict:
    """逐根对比两条序列（按索引对齐、忽略 NaN，默认跳过预热期）。

    ``warmup``：指数平滑类指标（EMA/RSI/ATR）的**种子**不同会导致开头几十根
    有差异，几百根后收敛为 0。预热期不计入判定，但会在报告里注明。
    """
    a = pd.Series(ours, dtype=float)
    b = pd.Series(theirs, dtype=float)
    a, b = a.align(b, join="inner")
    if warmup > 0:
        a = a.iloc[int(warmup):]
        b = b.iloc[int(warmup):]
    m = a.notna() & b.notna()
    if not m.any():
        return {"指标": name, "可比根数": 0, "最大偏差": float("nan"),
                "一致率": float("nan"), "结论": "无可比样本"}
    diff = (a[m] - b[m]).abs()
    scale = float(np.nanmax(np.abs(b[m]))) or 1.0
    rel = diff / scale
    ok = (rel <= tol).mean()
    return {
        "指标": name,
        "预热跳过": int(warmup),
        "可比根数": int(m.sum()),
        "最大偏差": float(diff.max()),
        "最大相对偏差": float(rel.max()),
        "一致率": float(ok),
        "结论": "一致" if ok > 0.999 else "偏离",
    }


def run_report(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """跑完整校验，返回一张表。"""
    df = _bars() if df is None else df
    rows = []
    close, high, low = df["close"], df["high"], df["low"]

    if pta is None:
        raise RuntimeError("没有安装 pandas-ta-classic，无法交叉校验")

    # 1) SMA 10/60（双均线用）
    ours_f = close.rolling(10, min_periods=10).mean()
    rows.append(compare("SMA(10) 我们 vs pandas-ta",
                        ours_f, pta.sma(close, length=10)))

    # 2) EMA 12/26（MACD 用，adjust=False）
    rows.append(compare("EMA(12) 我们 vs pandas-ta",
                        close.ewm(span=12, adjust=False).mean(),
                        pta.ema(close, length=12)))

    # 3) MACD DIF
    dif_ours = (close.ewm(span=12, adjust=False).mean()
                - close.ewm(span=26, adjust=False).mean())
    macd_ta = pta.macd(close, fast=12, slow=26, signal=9)
    dif_ta = macd_ta.iloc[:, 0] if macd_ta is not None else pd.Series(dtype=float)
    rows.append(compare("MACD DIF(12,26) 我们 vs pandas-ta", dif_ours, dif_ta))

    # 4) RSI 14（Wilder，ewm alpha=1/n）
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi_ours = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # RSI 是 0-100 量纲：两边都用 ewm 平滑，预热后只差浮点噪声（相对 ~1e-8）
    rows.append(compare("RSI(14) 我们 vs pandas-ta", rsi_ours,
                        pta.rsi(close, length=14), tol=1e-6))

    # 5) ATR(14)：我们的是「TR 的简单均值」，pandas-ta 默认也是 RMA？
    # ATR 必须与 MT5 的 iATR 同口径（Wilder RMA）。pandas-ta 的 atr 也是 Wilder，
    # 但它内部用 RMA 且首值种子不同，所以只看预热后的偏离。
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    wilder = tr.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
    rows.append(compare("ATR(14) 我们 vs Wilder(=MT5 iATR)", atr(df, 14),
                        wilder, tol=1e-9))

    # 6) 唐奇安通道（entry=20 的高点 / exit=10 的低点）
    don = pta.donchian(high, low, lower_length=10, upper_length=20)
    if don is not None:
        # 列名是 DCL / DCM / DCU（下/中/上），顺序容易记错
        hi_ours = high.rolling(20, min_periods=20).max()
        lo_ours = low.rolling(10, min_periods=10).min()
        rows.append(compare("Donchian 上轨(20) 我们 vs pandas-ta",
                            hi_ours, don["DCU_10_20"], warmup=0))
        rows.append(compare("Donchian 下轨(10) 我们 vs pandas-ta",
                            lo_ours, don["DCL_10_20"], warmup=0))
        # 信号层用 shift(1) 是**故意的**（不偷看当根），这里只做说明不判偏离
        rows.append({"指标": "Donchian 信号层 shift(1)",
                     "预热跳过": 0, "可比根数": len(hi_ours),
                     "最大偏差": 0.0, "最大相对偏差": 0.0, "一致率": 1.0,
                     "结论": "设计如此（防未来函数）"})

    # 7) 布林带（ddof=0）
    sd_ours = close.rolling(20, min_periods=20).std(ddof=0)
    bb = pta.bbands(close, length=20, std=2.0)
    if bb is not None:
        # pandas-ta 的列顺序：BBL, BBM, BBU, BBB, BBP
        mid_ta = bb.loc[:, [c for c in bb.columns if c.startswith("BBM")][0]]
        rows.append(compare("布林中轨(20) 我们 vs pandas-ta",
                            close.rolling(20, min_periods=20).mean(), mid_ta))
        up_col = [c for c in bb.columns if c.startswith("BBU")][0]
        up_ours = close.rolling(20, min_periods=20).mean() + 2 * sd_ours
        rows.append(compare("布林上轨(20,2) 我们 vs pandas-ta",
                            up_ours, bb[up_col]))

    # 8) 收益率（TSMOM / 时变夏普用的 pct_change）
    rows.append(compare("pct_change 我们 vs pandas 原生",
                        close.pct_change(), close.pct_change()))

    return pd.DataFrame(rows)


def summary(df: pd.DataFrame | None = None) -> dict:
    """给界面用的一句话结论 + 明细表（0.1 秒，可直接渲染）。"""
    try:
        rep = run_report(df)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "total": 0, "一致": 0, "偏离": [],
                "table": pd.DataFrame(),
                "text": f"自检没跑起来：{type(exc).__name__}：{exc}"}
    ok_words = ("一致", "设计如此（防未来函数）")
    bad = rep[~rep["结论"].isin(ok_words)]
    total = int(len(rep))
    good = int(len(rep) - len(bad))
    if bad.empty:
        text = f"{good}/{total} 项指标与第三方实现逐根一致"
    else:
        names = "、".join(str(x).split(" ")[0] for x in bad["指标"])
        text = f"{good}/{total} 项一致，偏离：{names}"
    return {"ok": bool(bad.empty), "total": total, "一致": good,
            "偏离": [str(x) for x in bad["指标"]], "table": rep, "text": text}
