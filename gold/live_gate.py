"""实盘就绪判定（Live Readiness）：把"敢不敢上实盘"变成可计算的门槛。

为什么单列一套
--------------
现有的 12 条门槛回答"这个策略有没有统计上的边际"（研究口径）；
实盘还要多问三件研究口径不管的事：

1. **活不活得下去**：最差一段的最大回撤是否在可承受范围（决定会不会被强平/停手）；
2. **成本扛不扛得住**：把滑点也加进去还剩多少（回测的点差 ≠ 实盘成交价）；
3. **样本外守不守得住**：锁定样本外必须为正，而不是"平均为正"。

判定分三档：`可上模拟盘` / `再观察` / `不可实盘`。
**任何一项未测（None）都不得判"可上"** —— 与主门槛同一原则。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 实盘就绪的默认标准（可按账户风险偏好改，但改之前先想清楚）
DEFAULTS = {
    "min_oos_months": 6,        # 锁定样本外至少 6 个月
    "max_drawdown": 0.20,       # 最大回撤 ≤20%
    "min_folds_positive": 2,    # 滚动 3 折至少 2 折为正
    "min_worst_fold": -0.05,    # 最差一折亏损不超过 5%
    "min_slippage_surplus": 0.0,  # 加滑点后收益仍 >0
    "max_rounds_year": 300.0,   # 换手别太高（成本与噪声）
    "min_rounds_year": 20.0,    # 太低的换手样本不足，统计不可信
    "max_pbo": 0.5,             # 过拟合概率
}


@dataclass
class LiveCheck:
    name: str
    need: str
    value: str
    status: str            # pass / fail / unmeasured

    @property
    def ok(self) -> bool:
        return self.status == "pass"


@dataclass
class LiveVerdict:
    verdict: str                       # 可上模拟盘 / 再观察 / 不可实盘
    checks: list[LiveCheck] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict == "可上模拟盘"


def _c(name, need, value, cond):
    if value is None:
        return LiveCheck(name, need, "未测", "unmeasured")
    return LiveCheck(name, need, str(value), "pass" if cond else "fail")


def live_readiness(*, oos_months: int | None = None,
                   max_drawdown: float | None = None,
                   folds_positive: int | None = None,
                   folds_total: int | None = None,
                   worst_fold_return: float | None = None,
                   return_with_slippage: float | None = None,
                   rounds_per_year: float | None = None,
                   pbo: float | None = None,
                   std: dict | None = None) -> LiveVerdict:
    """按实盘口径逐条判定。

    参数全是"实测值"，缺哪个就传 None（会判未测，而不是猜）。
    """
    s = {**DEFAULTS, **(std or {})}
    checks = [
        _c("锁定样本外时长", f"≥{s['min_oos_months']} 个月", oos_months,
           oos_months is not None and oos_months >= s["min_oos_months"]),
        _c("最大回撤", f"≤{s['max_drawdown']:.0%}",
           (f"{max_drawdown:.1%}" if max_drawdown is not None else None),
           max_drawdown is not None and max_drawdown >= -s["max_drawdown"]),
        _c("滚动折为正", f"≥{s['min_folds_positive']}/{folds_total or 3} 折",
           (f"{folds_positive}/{folds_total}"
            if folds_positive is not None and folds_total else None),
           folds_positive is not None and folds_positive >= s["min_folds_positive"]),
        _c("最差一折收益", f">={s['min_worst_fold']:.0%}",
           (f"{worst_fold_return:+.1%}" if worst_fold_return is not None else None),
           worst_fold_return is not None
           and worst_fold_return >= s["min_worst_fold"]),
        _c("加滑点后仍为正", ">0",
           (f"{return_with_slippage:+.1%}"
            if return_with_slippage is not None else None),
           return_with_slippage is not None
           and return_with_slippage > s["min_slippage_surplus"]),
        _c("每年回合", f"{s['min_rounds_year']:g}~{s['max_rounds_year']:g}",
           (f"{rounds_per_year:.0f}" if rounds_per_year is not None else None),
           rounds_per_year is not None
           and s["min_rounds_year"] <= rounds_per_year <= s["max_rounds_year"]),
        _c("CSCV-PBO", f"≤{s['max_pbo']:g}",
           (f"{pbo:.3f}" if pbo is not None else None),
           pbo is not None and pbo <= s["max_pbo"]),
    ]

    fails = [c for c in checks if c.status == "fail"]
    unknown = [c for c in checks if c.status == "unmeasured"]
    reasons = [f"{c.name}[{c.need} 实际 {c.value}]" for c in fails]
    reasons += [f"{c.name}[未测]" for c in unknown]

    if fails:
        verdict = "不可实盘"
    elif unknown:
        verdict = "再观察"       # 缺数据不等于过，只能继续观察
    else:
        verdict = "可上模拟盘"
    return LiveVerdict(verdict=verdict, checks=checks, reasons=reasons)


def robust_score(fold_returns: list[float],
                 penalty: float = 0.5) -> float:
    """稳健性目标：中位数 − λ×离散度 − λ×最差折。

    用来替换"只看平均夏普"的选参方式 —— 平均好但忽上忽下的策略，
    在实盘里会先把人熬死。返回一个越大越好的分数。
    """
    vals = [float(v) for v in fold_returns if v is not None]
    if not vals:
        return float("-inf")
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    median = (vals_sorted[n // 2] if n % 2
              else 0.5 * (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]))
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = var ** 0.5
    worst = vals_sorted[0]
    return float(median - penalty * std + 0.5 * penalty * worst)
