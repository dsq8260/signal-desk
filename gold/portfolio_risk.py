"""组合设计：相关性去重、权重分配、暴露上限、压力测试。

为什么需要这一层
----------------
单条 EA 成绩好 ≠ 组合能用。真实账户会同时跑多个 EA，于是出现：
- **同类策略叠仓**：3 个双均线变体一起做多，等于 3 倍单边风险；
- **同向暴露过大**：全部只多黄金，金价一跌全亏；
- **保证金挤爆**：每个 EA 都占保证金，总占用超过账户承受力。

本模块只做四件事，全部可测：
1. `correlation()` / `pick_uncorrelated()`：相关性矩阵 + 按阈值挑互不重复的 EA；
2. `weights()`：等权 / 逆波动 / 风险平价（带单 EA 上限）；
3. `exposure()`：单品种、单方向、总保证金三项上限检查；
4. `stress_test()`：点差翻倍、滑点追加、跳空、连亏四种冲击下的组合表现。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 相关性
def correlation(returns: dict[str, pd.Series],
                method: str = "pearson") -> pd.DataFrame:
    """多 EA 收益的相关性矩阵（按索引对齐，缺失填 0）。"""
    if not returns:
        return pd.DataFrame()
    df = pd.DataFrame(returns).fillna(0.0)
    if df.shape[1] == 1:
        return pd.DataFrame([[1.0]], index=df.columns, columns=df.columns)
    return df.corr(method=method)


def pick_uncorrelated(corr: pd.DataFrame, thresh: float = 0.7,
                      keep_order: list[str] | None = None) -> list[str]:
    """按顺序挑出互相关低于阈值的 EA（贪心）。

    ``keep_order`` 决定优先级（一般按质量排序：夏普高的优先保留）。
    """
    if corr.empty:
        return []
    order = [c for c in (keep_order or list(corr.columns)) if c in corr.columns]
    picked: list[str] = []
    for name in order:
        if all(abs(float(corr.loc[name, p])) < float(thresh) for p in picked):
            picked.append(name)
    return picked


# ---------------------------------------------------------------- 权重
def weights(kind: str = "equal", vol: dict[str, float] | None = None,
            max_weight: float = 0.4) -> pd.Series:
    """权重分配：等权 / 逆波动 / 风险平价（迭代求解，等价于 ERC）。

    - ``equal``：1/N；
    - ``inverse_vol``：按 1/波动 归一；
    - ``risk_parity``：使每个 EA 的**风险贡献**相等（逆波动是它的近似）；
    最后统一施加单 EA 上限并重新归一。
    """
    if not vol:
        raise ValueError("必须提供各 EA 的波动率")
    names = list(vol)
    v = np.array([max(1e-12, float(vol[n])) for n in names])
    k = str(kind).lower()

    if k == "equal":
        w = np.ones(len(names))
    elif k == "inverse_vol":
        w = 1.0 / v
    elif k == "risk_parity":
        w = 1.0 / v
        for _ in range(200):                     # 简单迭代逼近 ERC
            rc = w * v
            target = rc.mean()
            w = w * (target / np.maximum(rc, 1e-12)) ** 0.05
    else:
        raise ValueError(f"未知权重方式: {kind}")

    w = w / w.sum()
    # 单 EA 上限：超限的封顶，剩下的按比例回填
    cap = float(max_weight)
    if cap * len(names) < 1.0 - 1e-9:
        raise ValueError(
            f"单 EA 上限 {cap:.0%} × {len(names)} 个 EA < 100%，"
            f"权重无法分配（上限至少要 {1 / len(names):.0%}）")
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        w[over] = cap
        free = ~over
        if not free.any():
            break
        rest = 1.0 - w[over].sum()
        w[free] = w[free] / w[free].sum() * max(0.0, rest)
    w = w / w.sum()
    return pd.Series(w, index=names, name="权重")


def risk_contribution(w: pd.Series, vol: dict[str, float]) -> pd.Series:
    """各 EA 的风险贡献占比（诊断"是不是某一个 EA 独占风险"）。"""
    v = pd.Series({k: max(1e-12, float(vol[k])) for k in w.index})
    rc = (w * v).abs()
    total = float(rc.sum()) or 1.0
    return (rc / total).rename("风险贡献")


# ---------------------------------------------------------------- 暴露
@dataclass
class Exposure:
    """组合的暴露与限额。金额单位与账户一致（美元）。"""
    equity: float
    # 每个 EA：(品种, 方向, 名义价值)
    positions: dict[str, tuple[str, str, float]] = field(default_factory=dict)
    leverage: float = 100.0
    max_symbol_pct: float = 0.5      # 单品种名义价值 / 权益
    max_direction_pct: float = 0.8   # 单方向名义价值 / 权益
    max_margin_pct: float = 0.5      # 总保证金占用 / 权益

    def detail(self) -> pd.DataFrame:
        rows = []
        for name, (sym, side, notional) in self.positions.items():
            notional = float(notional)
            rows.append({
                "EA": name, "品种": sym, "方向": side,
                "名义价值": notional,
                "占权益": notional / self.equity if self.equity > 0 else np.nan,
                "保证金": notional / self.leverage if self.leverage > 0 else notional,
            })
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        d = self.detail()
        if d.empty:
            return {"品种最大占比": 0.0, "方向最大占比": 0.0, "总保证金占比": 0.0}
        by_sym = d.groupby("品种")["名义价值"].sum() / self.equity
        by_side = d.groupby("方向")["名义价值"].sum() / self.equity
        margin = float(d["保证金"].sum()) / self.equity if self.equity > 0 else np.inf
        return {"品种最大占比": float(by_sym.max()),
                "方向最大占比": float(by_side.max()),
                "总保证金占比": float(margin)}

    def breaches(self) -> list[str]:
        s = self.summary()
        out = []
        if s["品种最大占比"] > self.max_symbol_pct:
            out.append(f"单品种名义 {s['品种最大占比']:.0%} > "
                       f"上限 {self.max_symbol_pct:.0%}")
        if s["方向最大占比"] > self.max_direction_pct:
            out.append(f"单方向名义 {s['方向最大占比']:.0%} > "
                       f"上限 {self.max_direction_pct:.0%}")
        if s["总保证金占比"] > self.max_margin_pct:
            out.append(f"总保证金 {s['总保证金占比']:.0%} > "
                       f"上限 {self.max_margin_pct:.0%}")
        return out


# ---------------------------------------------------------------- 压力测试
def stress_test(returns: pd.Series, weights_: pd.Series | None = None,
                *, spread_mult: float = 2.0, extra_slip_bps: float = 10.0,
                gap_pct: float = 0.03, n_loss_streak: int = 5,
                trades_per_year: float = 100.0,
                cost_bps_base: float = 1.3) -> pd.DataFrame:
    """四种冲击下组合收益的变化（不含未来信息，纯成本/缺口重算）。"""
    r = pd.Series(returns, dtype=float).fillna(0.0)
    base_cost = float(cost_bps_base) * float(trades_per_year) / 10000.0

    def _cost(mult: float, extra: float) -> float:
        return (float(cost_bps_base) * mult + float(extra)) \
            * float(trades_per_year) / 10000.0

    rows = []
    base_total = float((1 + r).prod() - 1)
    for name, cost in (
        ("基准", base_cost),
        (f"点差 ×{spread_mult:g}", _cost(spread_mult, 0.0)),
        (f"滑点 +{extra_slip_bps:g} bp", _cost(1.0, extra_slip_bps)),
        ("两者叠加", _cost(spread_mult, extra_slip_bps)),
    ):
        net = (1 + r).prod() * (1 - cost) - 1
        rows.append({"情景": name, "成本": cost, "组合收益": net,
                     "相对基准": net - base_total})
    # 跳空冲击：一次性 −gap_pct
    gap = (1 + base_total) * (1 - float(gap_pct)) - 1
    rows.append({"情景": f"跳空 −{gap_pct:.0%}", "成本": base_cost,
                 "组合收益": gap, "相对基准": gap - base_total})
    # 连亏冲击：连续 n 笔最大单笔亏损
    if len(r) > 0:
        worst = float(r.min())
        if worst < 0:
            streak = (1 + base_total) * (1 + worst) ** int(n_loss_streak) - 1
            rows.append({"情景": f"连亏 {n_loss_streak} 笔(用最差单笔)",
                         "成本": base_cost, "组合收益": streak,
                         "相对基准": streak - base_total})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 组合回测
def combine(returns: dict[str, pd.Series], w: pd.Series) -> pd.Series:
    """按权重合成组合收益（缺失期间不参与，避免把 0 当收益）。"""
    df = pd.DataFrame({k: returns[k] for k in w.index if k in returns})
    if df.empty:
        return pd.Series(dtype=float)
    ww = w.reindex(df.columns).fillna(0.0)
    return (df.fillna(0.0) * ww).sum(axis=1)


@dataclass
class PortfolioPlan:
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    weights: pd.Series = field(default_factory=pd.Series)
    corr: pd.DataFrame = field(default_factory=pd.DataFrame)
    exposure: dict = field(default_factory=dict)
    breaches: list = field(default_factory=list)
    stress: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = ""


def build_plan(returns: dict[str, pd.Series],
               vols: dict[str, float],
               equity: float,
               positions: dict[str, tuple[str, str, float]],
               *, kind: str = "risk_parity", max_weight: float | None = None,
               corr_thresh: float = 0.7, priority: list[str] | None = None,
               leverage: float = 100.0) -> PortfolioPlan:
    """一步生成组合方案：去重 → 权重 → 暴露检查 → 压力测试。

    ``max_weight=None`` 时按 EA 个数自适应（3 个及以下放宽到等权，避免上限不可行）。
    """
    corr = correlation({k: returns[k] for k in vols if k in returns})
    picked = pick_uncorrelated(corr, corr_thresh, priority or list(vols))
    if not picked:
        return PortfolioPlan(note="没有可用 EA")
    if max_weight is None:
        # 上限必须可行：至少 1/N，再多给 10% 余量
        max_weight = max(1.0 / len(picked) * 1.1, 0.34)
        max_weight = min(max_weight, 1.0)
    w = weights(kind, {k: vols[k] for k in picked}, max_weight)
    pos = {k: positions[k] for k in picked if k in positions}
    exp = Exposure(equity=equity, positions=pos, leverage=leverage)
    comb = combine({k: returns[k] for k in picked if k in returns}, w)
    stress = stress_test(comb) if len(comb) else pd.DataFrame()
    table = pd.DataFrame({
        "EA": w.index,
        "权重": w.values,
        "波动": [vols[k] for k in w.index],
        "风险贡献": risk_contribution(w, {k: vols[k] for k in w.index}).values,
    })
    return PortfolioPlan(
        table=table, weights=w, corr=corr, exposure=exp.summary(),
        breaches=exp.breaches(), stress=stress,
        note=(f"相关性去重 {len(vols)} → {len(picked)} 个 · "
              f"{kind} · 单 EA 上限 {max_weight:.0%}"),
    )
