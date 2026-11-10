"""账户类型自动识别 + 一键方案：用户只给「资金」，其余系统自己判定。

识别依据（都不是猜的，全部来自 MT5 品种规格）
--------------------------------------------
| 账户 | 黄金 1 手 | 最小手数 | 判定 |
|---|---|---|---|
| Standard / Ultra Low Standard | 100 盎司 | 0.01 | contract=100 且 min=0.01 |
| Micro / Ultra Low Micro | 1 盎司 | 0.1 | contract=1 且 min=0.1 |
| 美分账户（部分经纪商） | 按 1/100 计 | 0.01 | 货币后缀 USc/USDc 或 contract=1 且 min=0.01 |
| Shares | 股票 CFD | 1 手 | contract 不是 100 的整数族 |

"Ultra Low" 与普通版只差点差，所以用**实测点差**区分：
黄金点差 ≤ 20 点视为 Ultra Low，否则普通版。
"""

from __future__ import annotations

from dataclasses import dataclass

from gold.accounts import (ACCOUNT_TYPES, min_risk, min_units,
                           units_per_lot)

ULTRA_SPREAD_POINTS = 20.0     # 黄金点差 ≤20 点 → Ultra Low

CLASSES = ("standard", "micro", "ultra_standard", "ultra_micro", "cent", "shares")


@dataclass
class AccountInfo:
    account_class: str          # standard / micro / ultra_standard / ...
    label: str                  # 中文名
    contract_size: float
    min_lot: float
    spread_points: float | None
    cent: bool
    confidence: str             # high / medium / low

    def as_row(self) -> dict:
        return {"账户类型": self.label, "1手合约": self.contract_size,
                "最小手数": self.min_lot, "点差(点)": self.spread_points,
                "置信度": self.confidence}


def detect_account(contract_size: float | None,
                   min_lot: float | None,
                   spread_points: float | None = None,
                   currency: str | None = None) -> AccountInfo:
    """从 MT5 品种规格推断账户类型（用于全自动，不需要用户选）。"""
    cs = float(contract_size or 0)
    ml = float(min_lot or 0.01)
    sp = float(spread_points) if spread_points is not None else None
    cur = str(currency or "").upper()
    cent = cur.endswith("C") and cur not in ("USD", "EUR") or "USC" in cur

    if cs <= 0:
        return AccountInfo("standard", "标准（未识别，按标准处理）", cs, ml, sp,
                           cent, "low")
    if cent:
        return AccountInfo("cent", "美分账户", cs, ml, sp, True, "high")
    # 黄金 / 白银族：contract 100（标准）或 1（Micro）
    if ml >= 0.5:                     # 最小手数 1 手 → 股票 CFD
        cls = "shares"
    elif ml <= 0.011 and cs >= 50:
        cls = "ultra_standard" if (sp is not None and sp <= ULTRA_SPREAD_POINTS) \
            else "standard"
    elif ml <= 0.11 and cs < 50:
        cls = "ultra_micro" if (sp is not None and sp <= ULTRA_SPREAD_POINTS) \
            else "micro"
    else:
        cls = "standard"
    label = {
        "standard": "Standard（标准）", "micro": "Micro（微型）",
        "ultra_standard": "Ultra Low Standard",
        "ultra_micro": "Ultra Low Micro", "cent": "美分账户",
        "shares": "Shares（股票 CFD）",
    }[cls]
    conf = "high" if sp is not None else "medium"
    return AccountInfo(cls, label, cs, ml, sp, cent, conf)


@dataclass
class Plan:
    symbol: str
    account: str
    risk_pct: float
    risk_usd: float
    viable: bool
    need_equity: float
    suggestion: str


def auto_plan(equity: float, *, price: float = 4000.0, sl_pct: float = 0.02,
              target_risk: float = 0.005,
              account: str = "standard") -> dict:
    """一键方案：给定资金 + 已识别的账户类型，直接给出"能不能做、做什么、用什么手数"。

    返回 dict：可行/建议/各品种明细，可直接喂给生成器与 EA 默认值。
    """
    out = []
    for sym in ("GOLD", "EURUSD"):
        px = price if sym == "GOLD" else 1.10
        r = min_risk(sym, px, sl_pct, float(equity), account)
        out.append({"品种": sym, "最小暴露": round(r.min_units, 3),
                    "单笔风险$": round(r.risk_usd, 2),
                    "占权益": round(r.risk_pct, 4),
                    "可行": bool(r.risk_pct <= 0.02),
                    "0.5%所需本金": round(r.need_equity_for_half)})
    best = min(out, key=lambda x: x["占权益"])
    feasible = [x for x in out if x["可行"]]
    pick = min(feasible, key=lambda x: x["占权益"]) if feasible else None
    if pick:
        sugg = (f"用 {pick['品种']}（{account}）：最小手数风险 "
                f"{pick['占权益']:.2%}，可直接生成")
    else:
        sugg = (f"当前资金 {equity:.0f} 在任何品种上都超过 2% 风险；"
                f"建议：① 换 Micro/美分账户 ② 资金至少加到 "
                f"{min(x['0.5%所需本金'] for x in out):,.0f}")
    return {"可行": bool(feasible), "建议": sugg,
            "推荐品种": pick["品种"] if pick else None,
            "账户类型": account, "明细": out, "最佳": best}


def ea_defaults(plan: dict, equity: float) -> dict:
    """把方案翻译成 EA 的输入默认值（生成器直接写进 .mq5）。"""
    risk = 0.5
    if plan.get("明细"):
        r = plan["明细"][0]["占权益"]
        risk = max(0.2, min(2.0, round(r * 100, 2)))
    return {
        "InpLotMode": "LOT_PERCENT",
        "InpRiskPct": risk,
        "InpMaxRiskPct": 2.0,
        "InpMaxSpreadPoints": 40,
        "InpMaxDailyLossPct": 5.0,
        "InpOnePerBar": True,
        "InpMaxLots": round(max(0.01, equity / 20000.0), 2),
    }
