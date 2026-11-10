"""组合实验室：把多个低相关策略合成一个更高夏普的组合。

为什么这是提升夏普唯一可靠的路径
------------------------------
单个 H1 策略扣掉点差后夏普约 0.5，这是这类经典技术指标的天花板。
但组合夏普取决于**成分间的相关性**：

    SR_p = (Σ w_i SR_i) / sqrt(ΣΣ w_i w_j ρ_ij σ_i σ_j)

两个 SR=0.5 的策略：ρ=1 → 0.5；ρ=0 → 0.71；ρ=-0.5 → 0.87。
**调参数改变不了 SR_i，但选低相关成分能直接抬 SR_p。**

三种权重方案
------------
- 等权：最简单，不容易过拟合
- 逆波动：低波动成分权重更高（风险平价的最简版）
- 波动率目标：按组合滚动波动缩放到目标年化波动（改变风险水平，不改变夏普，
  但对实盘资金曲线更友好）

纪律：权重只用 **IS（样本内）** 估计，OOS 只用于评估 —— 不允许用 OOS 调权重。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gold.backtest import spread_fee_bps
from gold.factors import build_features_cached
from gold.forge import Genome, build_position
from gold.mining import split_is_oos

BARS_PER_YEAR = {"M1": 252 * 24 * 60, "M5": 252 * 24 * 12,
                 "M15": 252 * 24 * 4, "M30": 252 * 24 * 2,
                 "H1": 252 * 24, "H4": 252 * 6, "D1": 252, "W1": 52}


def bars_per_year(tf: str) -> float:
    return float(BARS_PER_YEAR.get(str(tf).upper(), 252))


def ann_sharpe(rets: pd.Series, tf: str = "H1") -> float:
    r = pd.Series(rets).dropna()
    if len(r) < 10:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(bars_per_year(tf)))


def strategy_returns(df: pd.DataFrame, g: Genome, tf: str = "H1",
                     spread_usd: float = 0.33,
                     is_ratio: float = 0.70,
                     feat=None,
                     mask_cache: dict | None = None
                     ) -> tuple[pd.Series, pd.Series]:
    """返回 (IS 段收益, OOS 段收益)，已扣点差。

    feat/mask_cache 可传入复用：多个成分共用同一份特征与过滤掩码，
    避免每个成分重算 0.9 秒的滚动分位。
    """
    if feat is None:
        feat = build_features_cached(df)
    pos = build_position(df, feat, g, mask_cache if mask_cache is not None else {})
    pos = pd.Series(pos).reindex(df.index).fillna(0.0)

    ret = df["close"].astype(float).pct_change(fill_method=None)
    fee = spread_fee_bps(float(df["close"].mean()), spread_usd) / 1e4
    held = pos.shift(1).fillna(0.0)
    turnover = held.diff().abs().fillna(held.abs())
    net = held * ret - turnover * fee
    net = net.fillna(0.0)

    is_idx, oos_idx = split_is_oos(df, is_ratio)
    return net.reindex(is_idx.index), net.reindex(oos_idx.index)


def _weights(kind: str, is_rets: pd.DataFrame) -> pd.Series:
    cols = list(is_rets.columns)
    if kind == "等权":
        w = pd.Series(1.0 / len(cols), index=cols)
    elif kind == "逆波动":
        vol = is_rets.std(ddof=1).replace(0.0, np.nan)
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        w = inv / inv.sum() if inv.sum() > 0 else pd.Series(1.0 / len(cols), index=cols)
    else:  # 波动率目标：先等权，再对组合整体缩放
        w = pd.Series(1.0 / len(cols), index=cols)
    return w


def greedy_pick(oos: pd.DataFrame, k: int, tf: str = "H1") -> list[str]:
    """贪心选 k 个成分：每次加入后使**等权组合**的 OOS 夏普最大。

    只用 OOS 做挑选会乐观；这里仅用于展示"组合能到多高"，
    实际落地应改用 IS 选成分、OOS 只做验证（见 docstring 说明）。
    """
    cols = list(oos.columns)
    if k >= len(cols):
        return cols
    chosen: list[str] = []
    while len(chosen) < k:
        cand_best, cand_sr = None, -1e18
        for c in cols:
            if c in chosen:
                continue
            trial = chosen + [c]
            sr = ann_sharpe(oos[trial].mean(axis=1), tf)
            if sr == sr and sr > cand_sr:
                cand_best, cand_sr = c, sr
        if cand_best is None:
            break
        chosen.append(cand_best)
    return chosen


def build_portfolio(df: pd.DataFrame, genomes: dict[str, Genome], tf: str = "H1",
                    kind: str = "等权", spread_usd: float = 0.33,
                    is_ratio: float = 0.70,
                    vol_target: float | None = None) -> dict:
    """构建组合并给出单策略 vs 组合的对比。"""
    from gold.factors import build_features_cached as _bf

    feat = _bf(df)
    mask_cache: dict = {}

    is_map, oos_map = {}, {}
    for name, g in genomes.items():
        try:
            a, b = strategy_returns(df, g, tf, spread_usd, is_ratio,
                                    feat=feat, mask_cache=mask_cache)
        except Exception:  # noqa: BLE001
            continue
        is_map[name] = a
        oos_map[name] = b
    if not is_map:
        return {"ok": False, "note": "没有可用成分"}

    is_df = pd.DataFrame(is_map).fillna(0.0)
    oos_df = pd.DataFrame(oos_map).fillna(0.0)

    w = _weights(kind, is_df)
    port_is = (is_df * w).sum(axis=1)
    port_oos = (oos_df * w).sum(axis=1)

    scale = None
    if vol_target is not None:
        win = int(max(5, bars_per_year(tf) // 200))
        roll = port_oos.rolling(win, min_periods=5).std()
        ann_roll = roll * np.sqrt(bars_per_year(tf))
        scale = (float(vol_target) / ann_roll.replace(0.0, np.nan)).clip(0.0, 3.0)
        port_oos = port_oos * scale.fillna(1.0)

    singles = []
    for name in oos_df.columns:
        singles.append({
            "成分": name,
            "权重": float(w.get(name, 0.0)),
            "IS夏普": ann_sharpe(is_df[name], tf),
            "OOS夏普": ann_sharpe(oos_df[name], tf),
            "OOS年化": float(oos_df[name].mean() * bars_per_year(tf)),
        })
    single_df = pd.DataFrame(singles).sort_values("OOS夏普", ascending=False)

    corr = oos_df.corr()
    avg_corr = float(corr.where(~np.eye(len(corr), dtype=bool)).stack().mean()) \
        if len(corr) > 1 else float("nan")

    eq = (1.0 + port_oos.fillna(0.0)).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else float("nan")

    return {
        "ok": True,
        "成分": single_df,
        "权重": {k: float(v) for k, v in w.items()},
        "相关性": corr,
        "平均相关性": avg_corr,
        "组合IS夏普": ann_sharpe(port_is, tf),
        "组合OOS夏普": ann_sharpe(port_oos, tf),
        "组合OOS年化": float(port_oos.mean() * bars_per_year(tf)),
        "组合最大回撤": dd,
        "最佳单策略夏普": float(single_df["OOS夏普"].max()) if len(single_df) else float("nan"),
        "OOS曲线": eq,
        "note": (f"{len(single_df)} 个成分 · {kind}"
                 + (f" · 波动率目标 {vol_target:.0%}" if vol_target else "")
                 + f" · 平均相关性 {avg_corr:.2f}"),
    }
