"""按 XM 的账户类型算最小可行风险（用户实测：XM 属综合适中的经纪商）。

XM 账户类型（2026-11 截图）
--------------------------
| 类型 | 1 手 = | 最小交易量 | 点差 |
|---|---|---|---|
| Standard | 100,000 | 0.01 手 | 低至 1.6 |
| Micro | 1,000 | 0.1 手 | 低至 1.6 |
| Ultra Low Standard | 100,000 | 0.01 手 | 低至 0.8 |
| Ultra Low Micro | 1,000 | 0.1 手 | 低至 0.8 |

关键结论（实测口径）
-------------------
Micro 的合约是标准账户的 1/100，最小手数却是 0.1（标准 0.01），
两者相乘后最小暴露是标准账户的 **1/10**：

    Standard  0.01 × 100,000 = 1,000 单位
    Micro     0.1  ×   1,000 =   100 单位      ← 只有 1/10

对黄金（标准 1 手 = 100 盎司，Micro 1 手 = 1 盎司）：

    Standard  0.01 × 100 oz = 1.0 oz
    Micro     0.1  ×   1 oz = 0.1 oz          ← 小 10 倍

所以 **Micro 能显著降低最小风险**（是 1/10，不是 1/100）：
    500 美元账户 + 黄金 + 2% 止损：
      Standard → 80 美元/笔 = 16.0%  ✗
      Micro    →  8 美元/笔 =  1.6%  ✓
把黄金做到 0.5% 风险：标准账户需 16,000，Micro 只需 1,600。
外汇 Micro 最小 100 单位 → 2% 止损约 2.2 美元，500 账户仅 0.44%。

"""

from __future__ import annotations

from dataclasses import dataclass

# 品种 -> (标准账户 1 手对应的"标的单位"量)
#   黄金：100 盎司/手；白银：5000 盎司/手；外汇：100,000 基础货币/手
UNITS_PER_LOT = {
    "GOLD": 100.0, "XAUUSD": 100.0,
    "XAGUSD": 5000.0, "SILVER": 5000.0,
    "EURUSD": 100000.0, "GBPUSD": 100000.0,
    "USDJPY": 100000.0, "USDCHF": 100000.0,
}

# 账户类型 -> (合约缩放, 最小手数, 点差描述)
ACCOUNT_TYPES = {
    "standard": (1.0, 0.01, "点差低至 1.6"),
    "micro": (0.01, 0.1, "点差低至 1.6"),
    "ultra_standard": (1.0, 0.01, "点差低至 0.8"),
    "ultra_micro": (0.01, 0.1, "点差低至 0.8"),
    "cent": (0.01, 0.01, "美分账户：入金 100 美分 = 1 美元"),
}


def _key(s: str | None) -> str:
    return str(s or "").upper().replace("#", "").strip()


def units_per_lot(symbol: str, account: str = "standard") -> float:
    base = UNITS_PER_LOT.get(_key(symbol), 100.0)
    return base * ACCOUNT_TYPES.get(account, ACCOUNT_TYPES["standard"])[0]


def min_units(symbol: str, account: str = "standard") -> float:
    """最小可交易暴露（标的单位）：黄金是盎司，外汇是基础货币。"""
    _, min_lot, _ = ACCOUNT_TYPES.get(account, ACCOUNT_TYPES["standard"])
    return units_per_lot(symbol, account) * min_lot


@dataclass
class Risk:
    symbol: str
    account: str
    min_units: float
    risk_usd: float          # 最小手数 + 该止损时，一笔亏多少美元
    equity: float
    risk_pct: float
    need_equity_for_half: float
    viable: bool


def min_risk(symbol: str, price: float, sl_pct: float, equity: float,
             account: str = "standard") -> Risk:
    """最小手数下的真实单笔风险（美元与占权益比例）。"""
    u = min_units(symbol, account)
    risk = u * float(price) * float(sl_pct)
    pct = risk / float(equity) if equity else float("inf")
    need = risk / 0.005 if risk > 0 else 0.0     # 要让这笔占到 0.5% 所需本金
    return Risk(_key(symbol), account, u, risk, float(equity), pct, need,
                pct <= 0.02)


def compare_table(symbol: str, price: float, sl_pct: float = 0.02,
                  equity: float = 500) -> list[dict]:
    rows = []
    for acc in ACCOUNT_TYPES:
        r = min_risk(symbol, price, sl_pct, equity, acc)
        rows.append({
            "账户": acc, "最小暴露": f"{r.min_units:,.1f}",
            "单笔风险$": round(r.risk_usd, 2),
            "占权益": f"{r.risk_pct:.1%}",
            "0.5%所需本金": f"{r.need_equity_for_half:,.0f}",
            "≤2%可行": "是" if r.viable else "否",
        })
    return rows
