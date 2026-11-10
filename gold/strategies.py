"""黄金 EA 策略库（纯函数：吃 OHLC、吐仓位序列）。

约定
----
- 输入 `df` 至少含 open/high/low/close，索引为 DatetimeIndex。
- 输出 `pd.Series`，取值 {1.0, 0.0}：1 = 持多，0 = 空仓。
  （现货黄金普通账户不做空，与 MT5 单一多空持仓语义一致；如需做空请扩展。）
- **信号只用 t 及之前的数据；仓位在回测里统一 shift(1)，即 t 收盘出信号、t+1 开盘执行。**

参考来源（公式为公开方法论，代码为本项目自实现）：
- RSRS：光大证券《基于阻力支撑相对强度（RSRS）的市场择时》
- LLT：广发证券《低延迟趋势线与交易择时》
- 唐奇安通道突破 / ATR 跟踪：经典海龟法则
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 基础指标
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """真实波幅均值（ATR），**Wilder 平滑**口径。

    为什么必须是 Wilder 而不是简单均值
    ----------------------------------
    MT5 的内置 `iATR()` 就是 Wilder RMA（alpha = 1/n 的指数平滑，首值用 TR 起始）。
    如果 Python 侧用「TR 的简单均值」，两边的 ATR 不同 → 由 ATR 派生的
    「强趋势过滤」「atr_pct」在 EA 里会和回测对不上。
    2026-11 用 pandas-ta 交叉校验时发现并统一（见 gold/indicator_check.py）。
    """
    h, l, c = df["high"], df["low"], df["close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / int(n), adjust=False, min_periods=1).mean()


def rolling_ols_slope_r2(y: pd.Series, x: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    """滚动一元回归的斜率与 R²（RSRS 的核心计算）。"""
    n = int(n)
    my = y.rolling(n, min_periods=n).mean()
    mx = x.rolling(n, min_periods=n).mean()
    cov = (y * x).rolling(n, min_periods=n).mean() - my * mx
    varx = (x * x).rolling(n, min_periods=n).mean() - mx * mx
    vary = (y * y).rolling(n, min_periods=n).mean() - my * my
    beta = cov / varx.replace(0.0, np.nan)
    corr = cov / np.sqrt(varx * vary).replace(0.0, np.nan)
    return beta, corr ** 2


# ---------------------------------------------------------------- 策略实现
def rsrs_signal(df: pd.DataFrame, n: int = 18, m: int = 250,
                upper: float = 0.7, lower: float = -0.7) -> pd.Series:
    """RSRS 标准分择时：右偏标准分 > upper 持多，< lower 空仓。"""
    beta, r2 = rolling_ols_slope_r2(df["high"], df["low"], n)
    raw = beta * r2
    mu = raw.rolling(int(m), min_periods=max(20, int(m) // 5)).mean()
    sd = raw.rolling(int(m), min_periods=max(20, int(m) // 5)).std(ddof=1)
    z = (raw - mu) / sd.replace(0.0, np.nan)
    pos = pd.Series(0.0, index=df.index)
    pos[z.shift(1) > upper] = 1.0          # 用上一根的分数，避免同日信息
    z_prev = z.shift(1)
    # 持仓保持：跌破 lower 才平，中间维持原仓位
    state, out = 0.0, []
    for v in z_prev.to_numpy():
        if v == v:
            if v > upper:
                state = 1.0
            elif v < lower:
                state = 0.0
        out.append(state)
    return pd.Series(out, index=df.index)


def llt_signal(df: pd.DataFrame, n: int = 60, alpha: float | None = None) -> pd.Series:
    """低延迟趋势线：收盘在 LLT 之上持多，之下空仓。"""
    p = df["close"].astype(float).to_numpy()
    a = float(alpha) if alpha else 2.0 / (int(n) + 1.0)
    llt = np.full(len(p), np.nan)
    if len(p) > 2:
        llt[0] = p[0]
        llt[1] = p[1]
        for t in range(2, len(p)):
            llt[t] = ((a - a * a / 4.0) * p[t]
                      + (a * a / 2.0) * p[t - 1]
                      - (a - 3.0 * a * a / 4.0) * p[t - 2]
                      + 2.0 * (1.0 - a) * llt[t - 1]
                      - (1.0 - a) ** 2 * llt[t - 2])
    line = pd.Series(llt, index=df.index)
    pos = (df["close"].shift(1) > line.shift(1)).astype(float)
    return pos.fillna(0.0)


def donchian_signal(df: pd.DataFrame, entry: int = 20, exit_: int = 10) -> pd.Series:
    """唐奇安通道突破：突破 N 日高点入场，跌破 M 日低点出场。"""
    entry, exit_ = int(entry), int(exit_)
    hi = df["high"].rolling(entry, min_periods=entry).max().shift(1)
    lo = df["low"].rolling(exit_, min_periods=exit_).min().shift(1)
    close = df["close"]
    state, out = 0.0, []
    for i in range(len(df)):
        c = close.iloc[i]
        if state == 0.0 and c > hi.iloc[i]:      # 突破高点 -> 开多
            state = 1.0
        elif state == 1.0 and c < lo.iloc[i]:    # 跌破低点 -> 平仓
            state = 0.0
        out.append(state)
    return pd.Series(out, index=df.index)


def ma_cross_signal(df: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.Series:
    """双均线：快线在慢线上方持多。"""
    f = df["close"].rolling(int(fast), min_periods=int(fast)).mean()
    s = df["close"].rolling(int(slow), min_periods=int(slow)).mean()
    return ((f > s).astype(float)).fillna(0.0)


def atr_trailing_signal(df: pd.DataFrame, n: int = 14, mult: float = 3.0,
                        entry_ma: int = 20) -> pd.Series:
    """ATR 跟踪止损（海龟式）：收盘站上均线开多，跌破 最高价 - k*ATR 平仓。"""
    close = df["close"]
    ma = close.rolling(int(entry_ma), min_periods=int(entry_ma)).mean()
    a = atr(df, n)
    state, peak, out = 0.0, 0.0, []
    for i in range(len(df)):
        c = float(close.iloc[i])
        av = float(a.iloc[i]) if a.iloc[i] == a.iloc[i] else 0.0
        if state == 0.0:
            if c > float(ma.iloc[i]) if ma.iloc[i] == ma.iloc[i] else False:
                state, peak = 1.0, c
        else:
            peak = max(peak, c)
            if av > 0 and c < peak - mult * av:
                state = 0.0
        out.append(state)
    return pd.Series(out, index=df.index)


def bollinger_reversion_signal(df: pd.DataFrame, n: int = 20, k: float = 2.0,
                               exit_mid: bool = True,
                               both: bool = False) -> pd.Series:
    """布林带回归：跌破下轨开多，回到中轨（或上轨）平仓。

    both=True 时改为双向：跌破下轨做多、突破上轨做空，回到中轨平仓。
    """
    close = df["close"]
    mid = close.rolling(int(n), min_periods=int(n)).mean()
    sd = close.rolling(int(n), min_periods=int(n)).std(ddof=0)
    lower = mid - float(k) * sd
    upper = mid + float(k) * sd
    state, out = 0.0, []
    for i in range(len(df)):
        c = float(close.iloc[i])
        lo, up, md = lower.iloc[i], upper.iloc[i], mid.iloc[i]
        if state == 0.0:
            if lo == lo and c < float(lo):
                state = 1.0
            elif both and up == up and c > float(up):
                state = -1.0
        elif state > 0:
            target = md if exit_mid else upper.iloc[i]
            if target == target and c > float(target):
                state = 0.0
        else:                                   # 空头：回到下轨平仓
            if md == md and c < float(md):
                state = 0.0
        out.append(state)
    return pd.Series(out, index=df.index)


def macd_momentum_signal(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                         signal: int = 9) -> pd.Series:
    """MACD 动量：DIF 上穿 DEA（金叉）持多，死叉空仓。"""
    close = df["close"]
    ema_f = close.ewm(span=int(fast), adjust=False).mean()
    ema_s = close.ewm(span=int(slow), adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=int(signal), adjust=False).mean()
    state, out = 0.0, []
    for i in range(len(df)):
        d, e = dif.iloc[i], dea.iloc[i]
        if d == d and e == e:
            if d > e:
                state = 1.0
            else:
                state = 0.0
        out.append(state)
    return pd.Series(out, index=df.index)


def rsi_reversion_signal(df: pd.DataFrame, n: int = 14, buy: float = 30.0,
                         sell: float = 55.0, both: bool = False) -> pd.Series:
    """RSI 反转：RSI 跌破 buy 开多，回到 sell 之上平仓。

    both=True 时改为双向：超卖做多、超买（100-buy，默认 70）做空。
    """
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / int(n), adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / int(n), adjust=False).mean()
    # 关键：loss=0（单边上涨）时 RSI 应为 100，不能变成 NaN —— 否则永远等不到"超买"。
    rs = gain / loss.where(loss > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(loss > 0, 100.0).where(gain.notna())
    overbought = 100.0 - float(buy)          # 默认 70
    cover = 100.0 - float(sell)              # 默认 45：空头回补线
    state, out = 0.0, []
    for i in range(len(df)):
        v = rsi.iloc[i]
        if v == v:
            if state == 0.0:
                if v < float(buy):
                    state = 1.0
                elif both and v > overbought:
                    state = -1.0
            elif state > 0:
                if v > float(sell):
                    state = 0.0
            else:
                if v < cover:
                    state = 0.0
        out.append(state)
    return pd.Series(out, index=df.index)



# ---------------------------------------------------------------- 注册表
STRATEGIES = {
    "RSRS择时": {
        "fn": rsrs_signal,
        "params": {"n": 18, "m": 250, "upper": 0.7, "lower": -0.7},
        "desc": "阻力支撑相对强度：高点对低点的滚动回归斜率×R²，标准分择时",
    },
    "低延迟趋势线LLT": {
        "fn": llt_signal,
        "params": {"n": 60},
        "desc": "二阶递推低延迟均线，收盘在线上持多",
    },
    "唐奇安通道突破": {
        "fn": donchian_signal,
        "params": {"entry": 20, "exit_": 10},
        "desc": "突破 N 日高点开多，跌破 M 日低点平仓（海龟法则）",
    },
    "双均线": {
        "fn": ma_cross_signal,
        "params": {"fast": 20, "slow": 60},
        "desc": "快慢均线金叉持多，死叉空仓",
    },
    "ATR跟踪止损": {
        "fn": atr_trailing_signal,
        "params": {"n": 14, "mult": 3.0, "entry_ma": 20},
        "desc": "收盘站上均线开多，跌破 最高价-k×ATR 平仓（海龟式跟踪止损）",
    },
    "布林带回归": {
        "fn": bollinger_reversion_signal,
        "params": {"n": 20, "k": 2.0},
        "desc": "跌破布林下轨开多，回到中轨平仓（均值回归）",
    },
    "MACD动量": {
        "fn": macd_momentum_signal,
        "params": {"fast": 12, "slow": 26, "signal": 9},
        "desc": "DIF 在 DEA 上方持多，死叉空仓",
    },
    "RSI反转": {
        "fn": rsi_reversion_signal,
        "params": {"n": 14, "buy": 30.0, "sell": 55.0},
        "desc": "RSI 跌破 30 开多，回到 55 以上平仓",
    },
}


# 双向模式下"反向即做空"的策略（趋势类）：多头信号消失 = 持空
TREND_LIKE = {"RSRS择时", "低延迟趋势线LLT", "唐奇安通道突破", "双均线",
              "MACD动量", "ATR跟踪止损"}
# 均值回归类自己在函数里处理 both（超买做空），不在这里取反
MEAN_REVERT = {"布林带回归", "RSI反转"}


def apply_strategy(df: pd.DataFrame, name: str,
                   params: dict | None = None,
                   both: bool = False) -> pd.Series:
    """按名字取策略并执行。

    both=False（默认）：返回 {1,0} 仓位（只做多），与历史口径一致。
    both=True：返回 {-1,0,1}；趋势类策略"信号消失即做空"，
    均值回归类策略"超买做空/超卖做多"。
    """
    if name not in STRATEGIES:
        raise KeyError(f"未知策略：{name}")
    spec = STRATEGIES[name]
    kw = dict(spec["params"])
    if params:
        kw.update({k: v for k, v in params.items() if k in kw})

    if both and name in MEAN_REVERT:
        kw["both"] = True                      # 函数内部处理多空
    pos = pd.Series(spec["fn"](df, **kw), index=df.index).fillna(0.0)

    if not both:
        return pos.clip(0.0, 1.0)
    if name in TREND_LIKE:
        # 1 -> 1（多），0 -> -1（空）
        return pos.apply(lambda x: 1.0 if x > 0.5 else -1.0)
    return pos.clip(-1.0, 1.0)
