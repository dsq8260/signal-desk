"""给每个参数配人话标签与说明，避免变量名（n/m/upper）直接暴露给用户。

界面用 `param_meta(strategy, key)` 取 (中文名, 一句话说明)。
"""

from __future__ import annotations

# 每个策略的参数：key -> (中文名, 说明)
PARAM_DOCS: dict[str, dict[str, tuple[str, str]]] = {
    "双均线": {
        "fast": ("快线周期", "短期均线的天数。越小越敏感，假信号越多；常用 10-30"),
        "slow": ("慢线周期", "长期均线的天数。越大越稳但越滞后；常用 60-120"),
    },
    "RSRS择时": {
        "n": ("回归窗口", "用最近多少根 K 线去算「高点对低点」的斜率，常用 18"),
        "m": ("标准化窗口",
              "把斜率放进最近多少根里比大小得标准分。250 约等于一年交易日"),
        "upper": ("做多阈值", "标准分高于这个值才做多，常用 0.7"),
        "lower": ("离场/做空阈值", "标准分低于这个值离场；双向模式下改为做空，常用 -0.7"),
    },
    "唐奇安通道突破": {
        "entry": ("突破周期", "收盘价突破最近 N 根的最高价就做多，常用 20"),
        "exit_": ("离场周期", "跌破最近 M 根的最低价就平仓，常用 10"),
    },
    "MACD动量": {
        "fast": ("快线EMA", "快速指数均线的天数，常用 12"),
        "slow": ("慢线EMA", "慢速指数均线的天数，常用 26"),
        "signal": ("信号线", "对 DIF 再做平滑的天数，常用 9"),
    },
    "RSI反转": {
        "n": ("RSI周期", "算 RSI 的天数，常用 14"),
        "buy": ("超卖线", "RSI 低于这个值算超卖、做多，常用 30"),
        "sell": ("多头离场线", "RSI 回到这个值以上平掉多单，常用 55"),
    },
    "布林带回归": {
        "n": ("均线周期", "中轨均线的天数，常用 20"),
        "k": ("带宽倍数", "上下轨 = 均线 ± k × 标准差，常用 2.0"),
    },
    "ATR跟踪止损": {
        "n": ("ATR周期", "算平均真实波幅的天数，常用 14"),
        "mult": ("跟踪倍数", "从最高点回落 k × ATR 就平仓，常用 3.0"),
        "entry_ma": ("进场均线", "收盘站上这条均线才做多，常用 20"),
    },
    "LLT上穿": {
        "n": ("平滑周期", "低延迟趋势线的平滑参数，越大越平滑，常用 60"),
    },
    # 策略在 STRATEGIES 里的正式名字
    "低延迟趋势线LLT": {
        "n": ("平滑周期", "低延迟趋势线的平滑参数，越大越平滑，常用 60"),
    },
}

# 全局参数（各页通用）
GLOBAL_DOCS: dict[str, tuple[str, str]] = {
    "spread": ("点差（美元）", "每笔交易的成本，XM 的 GOLD# 实测约 0.27-0.33"),
    "n_perm": ("置换检验次数",
               "把收益顺序打乱多少次来对比，判断是不是运气。越大越可信但越慢"),
    "is_ratio": ("样本内占比",
                 "用前多少比例的数据来挑参数，剩下的做样本外验证。0.7 表示前 70%"),
    "max_cand": ("候选上限", "最多评估多少组组合。越多越慢，但统计校正也越严格"),
}


def param_meta(strategy: str, key: str,
               default: tuple[str, str] | None = None) -> tuple[str, str]:
    """取某个策略参数的中文名与说明；没有登记就退回变量名本身。"""
    doc = PARAM_DOCS.get(strategy, {})
    if key in doc:
        return doc[key]
    return default or (key, "")


def global_meta(key: str) -> tuple[str, str]:
    return GLOBAL_DOCS.get(key, (key, ""))


def describe_axes(strategy: str, axes: dict[str, list]) -> str:
    """把 {'fast': [5, 10], ...} 变成人话，用于参数挖掘页。"""
    total = 1
    parts = []
    for k, values in axes.items():
        total *= max(1, len(values))
        label, _ = param_meta(strategy, k)
        vals = " / ".join(str(v) for v in values)
        parts.append(f"{label}试 {vals}")
    return "；".join(parts) + f"　→ 共 {total} 种组合"
