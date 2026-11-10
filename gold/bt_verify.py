"""用 backtrader 独立复算，交叉验证我们自己的回测引擎。

为什么需要
----------
我们自己写的 `gold/backtest.py` 只有一套实现，算错了也看不出来。
这里把**同一份仓位序列**喂给 backtrader（第三方引擎），比较：
收益 / 最大回撤 / 成交笔数。两者接近 → 我们的账算得对；
差得远 → 我们的引擎有问题（这是之前 MT5 数字对不上时最需要的能力）。

口径对齐
--------
- 我们：pos[t] = 信号[t-1]，收益 = pos × 收盘-to-收盘收益率 − 换手×费率
- backtrader：`cheat_on_close` 让市价单在本根收盘成交，喂进去的就是 pos 序列，
  于是成交时点与我们一致；费率按百分比（=点差/价格）设到 commission。

撮合时点（实测校准）
--------------------
我们把 pos 喂给 backtrader 时必须**前移一根**（extra_lag=-1），
因为两边对"哪一根收盘成交"的定义差半拍：
- 我们：信号在 t 收盘形成 → 用 t 到 t+1 的收盘收益（等于 t 收盘进场）
- backtrader `set_coo`：t 收盘下的单在 t 收盘成交，但信号线读到的是 t 的旧值
实测 lag=-1 时四个策略的成交笔数**完全一致**，收益差 0.3~1.9 个百分点，
残差来自 0.5% 现金缓冲与佣金舍入。

已知差异
--------
- 我们按"全仓比例"复利；backtrader 按"股数×价格"记账，留 0.5% 现金防拒单
- 我们按换手计费；backtrader 按成交金额计费
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:                                    # 允许在未安装 backtrader 时降级
    import backtrader as bt
    HAS_BT = True
except Exception:                       # noqa: BLE001
    bt = None
    HAS_BT = False

from gold.backtest import spread_fee_bps
from gold.strategies import apply_strategy


class _SignalData(bt.feeds.PandasData if HAS_BT else object):
    """带一条 signal 线的数据源。"""
    if HAS_BT:
        lines = ("signal",)
        params = (("signal", "signal"),)


class _FollowSignal(bt.Strategy if HAS_BT else object):
    """完全跟随 signal：>0.5 满仓买入，否则清仓。"""

    def __init__(self):
        self.equity = []
        self.buys = 0
        self.sells = 0

    def next(self):
        want = float(self.data.signal[0]) > 0.5
        if want and not self.position:
            # 留 0.5% 余量：否则加上佣金后现金不足，买单被拒，下一根又重试
            size = (0.995 * self.broker.getcash()
                    / max(1e-9, float(self.data.close[0])))
            if size > 0:
                self.buy(size=size)
                self.buys += 1
        elif not want and self.position:
            self.close()
            self.sells += 1
        self.equity.append(float(self.broker.getvalue()))


def our_engine(df: pd.DataFrame, signal: pd.Series,
               spread_usd: float = 0.33, initial: float = 10000.0) -> dict:
    """我们自己引擎的口径：pos 前移一根，按收盘收益率复利。"""
    pos = pd.Series(signal, dtype=float).reindex(df.index).fillna(0.0).shift(1).fillna(0.0)
    ret = df["close"].astype(float).pct_change(fill_method=None)
    fee = spread_fee_bps(float(df["close"].mean()), spread_usd) / 1e4
    turn = pos.diff().abs().fillna(pos.abs())
    net = (pos * ret - turn * fee).fillna(0.0)
    eq = (1.0 + net).cumprod() * initial
    mdd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else float("nan")
    changes = int((pos.diff().fillna(0) != 0).sum())
    return {
        "累计收益": float(eq.iloc[-1] / initial - 1.0) if len(eq) else float("nan"),
        "最大回撤": mdd,
        "期末净值": float(eq.iloc[-1]) if len(eq) else float("nan"),
        "成交变动": changes,
        "引擎": "本系统",
    }


def bt_engine(df: pd.DataFrame, signal: pd.Series,
              spread_usd: float = 0.33, initial: float = 10000.0,
              extra_lag: int = -1) -> dict:
    """把同一份仓位序列喂给 backtrader 复算。"""
    if not HAS_BT:
        return {"累计收益": float("nan"), "最大回撤": float("nan"),
                "期末净值": float("nan"), "成交变动": 0,
                "引擎": "backtrader（未安装）"}

    pos = pd.Series(signal, dtype=float).reindex(df.index).fillna(0.0).shift(1).fillna(0.0)
    if extra_lag:
        pos = pos.shift(int(extra_lag)).fillna(0.0)
    feed_df = pd.DataFrame({
        "open": df["open"].astype(float), "high": df["high"].astype(float),
        "low": df["low"].astype(float), "close": df["close"].astype(float),
        "volume": 0.0, "signal": pos.to_numpy(),
    }, index=df.index)

    cerebro = bt.Cerebro()
    cerebro.broker.set_coo(True)   # 收盘价成交（与我们的口径对齐）
    cerebro.adddata(_SignalData(dataname=feed_df))
    cerebro.addstrategy(_FollowSignal)
    cerebro.broker.setcash(initial)
    fee_rate = spread_fee_bps(float(df["close"].mean()), spread_usd) / 1e4
    cerebro.broker.setcommission(commission=fee_rate)
    strat = cerebro.run()[0]

    eq = pd.Series(strat.equity, dtype=float)
    mdd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else float("nan")
    return {
        "累计收益": float(eq.iloc[-1] / initial - 1.0) if len(eq) else float("nan"),
        "最大回撤": mdd,
        "期末净值": float(eq.iloc[-1]) if len(eq) else float("nan"),
        "成交变动": int(strat.buys + strat.sells),
        "引擎": "backtrader",
    }


def compare(df: pd.DataFrame, strategy: str, params: dict | None = None,
            spread_usd: float = 0.33) -> dict:
    """同一策略、同一份信号，两个引擎各算一遍。"""
    sig = apply_strategy(df, strategy, params or {})
    a = our_engine(df, sig, spread_usd)
    b = bt_engine(df, sig, spread_usd)
    return {
        "策略": strategy,
        "参数": params or {},
        "本系统": a,
        "backtrader": b,
        "笔数差": a["成交变动"] - b["成交变动"],
        "收益差": (a["累计收益"] or 0) - (b["累计收益"] or 0),
        "回撤差": (a["最大回撤"] or 0) - (b["最大回撤"] or 0),
    }
