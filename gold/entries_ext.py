"""新增可移植进场信号（扩基因空间用）。

全部是**单序列**信号，只用价格自身，不需要外部数据：
- 时序动量 TSMOM（Moskowitz/Ooi/Pedersen 2012）
- 时变夏普（光大证券研报思路）
- 鳄鱼线 Alligator（Bill Williams）

为什么放在单独模块
------------------
`combo.ENTRIES` 被 `combo.build_candidates()` 用来枚举 195 个组合，
直接往里加会改变那个数字、破坏既有口径与测试。新信号只进 forge 的基因空间。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _close(df: pd.DataFrame) -> pd.Series:
    return df["close"].astype(float)


def tsmom(df: pd.DataFrame, n: int = 60, thr: float = 0.0) -> pd.Series:
    """时序动量：过去 n 期收益 > 阈值则持多。

    最简的趋势定义，没有平滑、没有交叉，纯粹看累计涨幅。
    """
    c = _close(df)
    mom = c / c.shift(int(n)) - 1.0
    out = (mom > float(thr)).astype(float)
    out[mom.isna()] = 0.0
    return out


def time_varying_sharpe(df: pd.DataFrame, n: int = 60,
                        thr: float = 0.0) -> pd.Series:
    """时变夏普：滚动窗口内 收益均值 / 收益标准差，比值 > 阈值持多。

    比 TSMOM 多一层波动率归一化，意图是"单位风险赚得多才做"。
    """
    c = _close(df)
    r = c.pct_change(fill_method=None)
    mu = r.rolling(int(n), min_periods=max(5, int(n) // 3)).mean()
    sd = r.rolling(int(n), min_periods=max(5, int(n) // 3)).std(ddof=1)
    ratio = (mu / sd.replace(0.0, np.nan))
    out = (ratio > float(thr)).astype(float)
    out[ratio.isna()] = 0.0
    return out


def alligator(df: pd.DataFrame, jaw: int = 13, teeth: int = 8, lips: int = 5,
              shift: int = 3) -> pd.Series:
    """鳄鱼线：三条平滑均线（下颚/牙齿/嘴唇）多头排列时持多。

    原版用 SMMA 且各自向前平移 8/5/3 根；这里用 SMA + 统一平移，
    保留"三线发散方向定趋势"的核心含义，避免过度依赖未收盘的当前 bar。
    """
    c = _close(df)
    j = c.rolling(int(jaw), min_periods=int(jaw)).mean().shift(int(shift))
    t = c.rolling(int(teeth), min_periods=int(teeth)).mean().shift(int(shift))
    l = c.rolling(int(lips), min_periods=int(lips)).mean().shift(int(shift))
    out = ((l > t) & (t > j) & (c > l)).astype(float)
    out[out.isna()] = 0.0
    return out


# 名称 -> (函数, 默认参数, 参数网格, 最少数据长度)
EXT_ENTRIES = {
    "时序动量TSMOM": (
        tsmom, {"n": 60, "thr": 0.0},
        [{"n": 20, "thr": -0.02}, {"n": 60, "thr": 0.0},
         {"n": 120, "thr": 0.0}, {"n": 250, "thr": 0.0}],
        260,
    ),
    "时变夏普": (
        time_varying_sharpe, {"n": 60, "thr": 0.0},
        [{"n": 20, "thr": 0.0}, {"n": 60, "thr": 0.0},
         {"n": 120, "thr": 0.0}, {"n": 60, "thr": 0.5}],
        130,
    ),
    "鳄鱼线": (
        alligator, {"jaw": 13, "teeth": 8, "lips": 5, "shift": 3},
        [{"jaw": 13, "teeth": 8, "lips": 5, "shift": 3},
         {"jaw": 21, "teeth": 13, "lips": 8, "shift": 5}],
        30,
    ),
}


def apply_stops(df: pd.DataFrame, pos: pd.Series,
                sl: float | None = None,
                tp: float | None = None) -> pd.Series:
    """按 K 线逐根模拟止损/止盈，命中后立即平仓（下一根起可再进）。

    sl/tp 用**小数**表示（0.02 = 2%）。None 表示不启用。
    只用当根 bar 的 high/low 判断是否触发，不看未来。

    实现说明：路径依赖，无法完全向量化；但用 numpy 数组 + 原生 float 循环，
    比逐行 ``Series.loc`` 快一个数量级（20000 根 × 数百候选时差别明显）。
    """
    if sl is None and tp is None:
        return pos

    idx = df.index
    p_arr = pd.Series(pos, dtype=float).reindex(idx).fillna(0.0).to_numpy()
    hi = df["high"].astype(float).to_numpy()
    lo = df["low"].astype(float).to_numpy()
    cl = df["close"].astype(float).to_numpy()

    out = p_arr.copy()
    entry = 0.0
    blocked = False

    for i in range(len(p_arr)):
        want = p_arr[i]
        if blocked:
            # 触发过止损/止盈的那一根不允许重进
            if want == out[i - 1] if i else want == 0.0:
                out[i] = 0.0
                blocked = False
                entry = 0.0
                continue
            blocked = False
        if entry == 0.0 and want != 0.0:
            entry = cl[i]
        if entry != 0.0 and want == 0.0:
            entry = 0.0
        if entry != 0.0 and want != 0.0:
            sign = 1.0 if want > 0 else -1.0
            if sl is not None:
                hit = (lo[i] <= entry * (1.0 - sl)) if sign > 0 else \
                      (hi[i] >= entry * (1.0 + sl))
                if hit:
                    out[i] = 0.0
                    entry = 0.0
                    blocked = True
                    continue
            if tp is not None:
                hit = (hi[i] >= entry * (1.0 + tp)) if sign > 0 else \
                      (lo[i] <= entry * (1.0 - tp))
                if hit:
                    out[i] = 0.0
                    entry = 0.0
                    blocked = True
                    continue
    return pd.Series(out, index=idx).fillna(0.0)
