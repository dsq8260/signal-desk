"""四条硬门槛与红黄绿灯（阶段 2）。

四条门槛（AGENTS.md 里写死的那四条）：
    1. 方向准确率 >= 55%
    2. 跑赢基准
    3. 多标的有效占比 >= 60%
    4. 块置换 p < 0.05

灯色规则：
    4 条通过            -> 绿灯
    2~3 条通过          -> 黄灯
    <=1 条通过          -> 红灯
    >=3 条样本不足      -> 灰灯（无法判定）

红灯时结论必须写明"未发现优于基准的证据"，不允许使用"有效"这类词。
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIN_DIRECTION_N = 30
MIN_ASSETS = 2

DIRECTION_THRESHOLD = 0.55
MULTI_ASSET_THRESHOLD = 0.60
PERMUTATION_ALPHA = 0.05

LIGHT_TEXT = {"green": "绿灯", "yellow": "黄灯", "red": "红灯", "gray": "灰灯"}


@dataclass
class Gate:
    """单条门槛的判定结果。passed=None 表示样本不足。"""

    name: str
    threshold: str
    value: str
    passed: bool | None
    note: str = ""


@dataclass
class GateReport:
    gates: list[Gate] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    insufficient: int = 0
    light: str = "gray"
    conclusion: str = ""

    def as_dict(self) -> dict:
        return {
            "gates": [g.__dict__ for g in self.gates],
            "passed": self.passed,
            "failed": self.failed,
            "insufficient": self.insufficient,
            "light": self.light,
            "light_text": LIGHT_TEXT[self.light],
            "conclusion": self.conclusion,
        }


def judge_gates(
    *,
    direction_accuracy: float | None = None,
    direction_n: int = 0,
    strategy_return: float | None = None,
    benchmark_return: float | None = None,
    multi_asset_ratio: float | None = None,
    multi_asset_n: int = 0,
    permutation_p: float | None = None,
) -> GateReport:
    """按四条硬门槛判定红黄绿灯。"""
    gates: list[Gate] = []

    # 1. 方向准确率
    if direction_accuracy is None or direction_n < MIN_DIRECTION_N:
        gates.append(
            Gate(
                "方向准确率", f">={DIRECTION_THRESHOLD:.0%}", "样本不足",
                None, f"有效方向样本 {direction_n} < {MIN_DIRECTION_N}",
            )
        )
    else:
        ok = direction_accuracy >= DIRECTION_THRESHOLD
        gates.append(
            Gate(
                "方向准确率", f">={DIRECTION_THRESHOLD:.0%}",
                f"{direction_accuracy:.2%}（n={direction_n}）", ok,
            )
        )

    # 2. 跑赢基准
    if strategy_return is None or benchmark_return is None:
        gates.append(Gate("跑赢基准", "策略收益 > 基准收益", "样本不足", None, "缺收益"));
    else:
        ok = strategy_return > benchmark_return
        gates.append(
            Gate(
                "跑赢基准", "策略收益 > 基准收益",
                f"{strategy_return:+.2%} vs {benchmark_return:+.2%}", ok,
            )
        )

    # 3. 多标的有效占比
    if multi_asset_ratio is None or multi_asset_n < MIN_ASSETS:
        gates.append(
            Gate(
                "多标的有效占比", f">={MULTI_ASSET_THRESHOLD:.0%}", "样本不足",
                None, f"有效标的数 {multi_asset_n} < {MIN_ASSETS}",
            )
        )
    else:
        ok = multi_asset_ratio >= MULTI_ASSET_THRESHOLD
        gates.append(
            Gate(
                "多标的有效占比", f">={MULTI_ASSET_THRESHOLD:.0%}",
                f"{multi_asset_ratio:.0%}（{multi_asset_n} 个标的）", ok,
            )
        )

    # 4. 块置换
    if permutation_p is None:
        gates.append(Gate("块置换 p 值", "<0.05", "样本不足", None, "未执行置换检验"))
    else:
        ok = permutation_p < PERMUTATION_ALPHA
        gates.append(
            Gate("块置换 p 值", "<0.05", f"p={permutation_p:.4f}", ok)
        )

    passed = sum(1 for g in gates if g.passed is True)
    failed = sum(1 for g in gates if g.passed is False)
    insufficient = sum(1 for g in gates if g.passed is None)

    if insufficient >= 3:
        light = "gray"
        conclusion = "证据不足：4 条硬门槛中有 {} 条样本不足，无法判定有效性。".format(insufficient)
    elif passed == 4:
        light = "green"
        conclusion = "四条硬门槛全部通过，样本内证据支持该规则。"
    elif passed >= 2:
        light = "yellow"
        conclusion = "部分门槛未通过，证据不充分，暂不宣称有效。"
    else:
        light = "red"
        conclusion = "未发现优于基准的证据。"

    return GateReport(
        gates=gates, passed=passed, failed=failed,
        insufficient=insufficient, light=light, conclusion=conclusion,
    )


def format_gates(report: GateReport) -> str:
    """打印用文本表。"""
    lines = [f"红黄绿灯：{LIGHT_TEXT[report.light]}"
             f"（通过 {report.passed} / 未通过 {report.failed} / 样本不足 {report.insufficient}）"]
    for g in report.gates:
        if g.passed is True:
            mark = "[通过]"
        elif g.passed is False:
            mark = "[未过]"
        else:
            mark = "[不足]"
        lines.append(f"  {mark} {g.name:<12} 门槛 {g.threshold:<18} 实测 {g.value} {g.note}")
    lines.append(f"结论：{report.conclusion}")
    return "\n".join(lines)
