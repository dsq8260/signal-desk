"""小资金适配：算"这个品种/止损下，本金至少多少才能把单笔风险压到目标"。

背景（实盘约束）
----------------
黄金 1 手 = 100 盎司，最小手数 0.01 手 = 1 盎司 → 价格波动 1 美元 = 1 美元盈亏。
2% 止损（金价 4000）= 80 美元/笔。所以 200~1000 美元的账户，
**只下最小手数就已经是 8%~40% 的单笔风险**，任何"0.5% 风险"的说法都是假的。

用法：
    min_equity_for_risk("GOLD", sl_pct=0.02, risk_pct=0.005)  # -> 16000
    feasibility("GOLD", equity=1000, sl_pct=0.02, risk_pct=0.005)  # -> 实际 8%
"""

from __future__ import annotations

from dataclasses import dataclass

# 品种 -> (合约规模, 最小手数, 手数步长)；来自 MT5 品种规格
SPECS = {
    "GOLD": (100.0, 0.01, 0.01),
    "XAGUSD": (5000.0, 0.01, 0.01),
    "EURUSD": (100000.0, 0.01, 0.01),
    "GBPUSD": (100000.0, 0.01, 0.01),
    "USDJPY": (100000.0, 0.01, 0.01),
    "USDCHF": (100000.0, 0.01, 0.01),
}


def _key(symbol: str | None) -> str:
    return str(symbol or "").upper().replace("#", "").strip()


def money_per_lot(symbol: str, price: float, sl_pct: float) -> float:
    """按给定止损百分比，1 手会亏多少钱（价格单位 × 合约规模）。"""
    contract = SPECS.get(_key(symbol), (100.0, 0.01, 0.01))[0]
    return float(price) * float(sl_pct) * contract


def min_equity_for_risk(symbol: str, price: float, sl_pct: float,
                        risk_pct: float) -> float:
    """要让"最小手数"的风险不超过 risk_pct，需要多少本金。"""
    contract, min_lot, _ = SPECS.get(_key(symbol), (100.0, 0.01, 0.01))
    loss = money_per_lot(symbol, price, sl_pct) * min_lot
    if risk_pct <= 0:
        return float("inf")
    return loss / float(risk_pct)


@dataclass
class Feasibility:
    symbol: str
    equity: float
    sl_pct: float
    target_risk_pct: float
    actual_risk_pct: float
    min_equity: float
    lots: float
    ok: bool
    advice: str


def feasibility(symbol: str, equity: float, price: float, sl_pct: float,
                risk_pct: float) -> Feasibility:
    """给定账户规模，判断能否达成目标单笔风险，并给出可执行建议。"""
    contract, min_lot, step = SPECS.get(_key(symbol), (100.0, 0.01, 0.01))
    loss_per_lot = money_per_lot(symbol, price, sl_pct)
    want_lots = (float(equity) * float(risk_pct)) / loss_per_lot \
        if loss_per_lot > 0 else 0.0
    lots = max(min_lot, round(want_lots / step) * step)
    actual = loss_per_lot * lots / float(equity) if equity > 0 else float("inf")
    need = min_equity_for_risk(symbol, price, sl_pct, risk_pct)
    ok = actual <= float(risk_pct) + 1e-9

    if ok:
        advice = f"可达成：{lots:.2f} 手，单笔风险 {actual:.2%}"
    elif float(equity) < need:
        gap = need - float(equity)
        # 反推：要让最小手数满足目标风险，止损最多能放多少
        sl_max = (float(equity) * float(risk_pct)
                  / (float(price) * contract * min_lot)) if price else 0.0
        advice = (f"最小手数 {min_lot} 手已占 {actual:.1%} 风险；"
                  f"要压到 {risk_pct:.1%} 需本金 ≥ {need:,.0f}"
                  f"（还差 {gap:,.0f}）。等值做法："
                  f"① 美分账户（合约按 1/100 计，最小手数风险同步缩小 100 倍）"
                  f"② 把止损收到 {sl_max:.2%} 以内"
                  f"③ 换更小合约的品种")
    else:
        advice = f"手数可调：{lots:.2f} 手，单笔风险 {actual:.2%}"
    return Feasibility(_key(symbol), float(equity), float(sl_pct),
                       float(risk_pct), actual, need, lots, ok, advice)


def plan_table(symbol: str, price: float, sl_pcts=(0.02, 0.01, 0.005),
               equities=(200, 500, 1000, 5000, 16000)) -> list[dict]:
    """给界面/文档用的小资金可行性表。"""
    rows = []
    for eq in equities:
        for sl in sl_pcts:
            f = feasibility(symbol, eq, price, sl, 0.005)
            rows.append({"本金": eq, "止损": f"{sl:.1%}",
                         "最小手数风险": f"{f.actual_risk_pct:.1%}",
                         "0.5%风险可行": "可行" if f.ok else "不可行",
                         "所需本金": round(f.min_equity)})
    return rows
