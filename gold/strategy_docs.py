"""策略库元数据：出处、公式、参数含义、MQL5 支持情况。

界面用 `catalog()` 拿全部信息；`mq5_name()` 处理策略名与 MQL5 模板名不一致的情况。
"""

from __future__ import annotations

# 策略名 -> (出处, 一句话原理, 核心公式)
CATALOG = {
    "RSRS择时": (
        "光大证券《RSRS 阻力支撑相对强度》2017",
        "用「最高价对最低价」的回归斜率衡量支撑阻力强弱，再做标准分择时",
        "beta = cov(high, low)/var(low)；score = beta × R²；z = 标准分(score, m)；z>upper 做多",
    ),
    "低延迟趋势线LLT": (
        "广发证券《低延迟趋势线》2013",
        "二阶递推平滑，比普通均线滞后更小，价格在线上持多",
        "LLT[i] = (a-a²/4)·P[i] + (a²/2)·P[i-1] - (a-3a²/4)·P[i-2] + 2(1-a)·LLT[i-1] - (1-a)²·LLT[i-2]，a=2/(n+1)",
    ),
    "唐奇安通道突破": (
        "海龟交易法则（Richard Dennis）",
        "突破 N 日最高价做多，跌破 M 日最低价平仓",
        "close > max(high, entry) 做多；close < min(low, exit) 平仓",
    ),
    "双均线": (
        "基础趋势跟踪（无特定出处）",
        "快线在慢线上方持多，反之离场（双向模式做空）",
        "SMA(fast) > SMA(slow) 持多",
    ),
    "ATR跟踪止损": (
        "海龟止损 + Wilder ATR",
        "站上均线开仓，从最高点回落 k×ATR 平仓",
        "入场：close > SMA(entry_ma)；出场：close < 持仓期最高 - mult×ATR(n)",
    ),
    "布林带回归": (
        "John Bollinger (1980s)",
        "跌破下轨做多（超卖），回到中轨平仓；双向模式突破上轨做空",
        "mid = SMA(n)；band = k×std；close < mid-k·sd 买；close > mid 平",
    ),
    "MACD动量": (
        "Gerald Appel (1979)",
        "DIF 在 DEA 上方持多",
        "DIF = EMA(fast) - EMA(slow)；DEA = EMA(DIF, signal)；DIF > DEA 持多",
    ),
    "RSI反转": (
        "J. Welles Wilder (1978)",
        "超卖做多、回到中位平仓；双向模式超买做空",
        "RSI < buy 买；RSI > sell 平；双向：RSI > 100-buy 做空",
    ),
}

# 策略名 -> MQL5 模板名（两者不完全一致）
MQ5_NAME = {
    "RSRS择时": "RSRS择时",
    "低延迟趋势线LLT": "LLT上穿",
    "唐奇安通道突破": "唐奇安通道突破",
    "双均线": "双均线",
    "ATR跟踪止损": "ATR跟踪止损",
    "布林带回归": "布林带回归",
    "MACD动量": "MACD金叉",
    "RSI反转": "RSI反转",
}

# 尚未实现、但可移植的单序列策略（来自 QuantsPlaybook 等公开研报）
PLANNED = [
    ("时序动量 TSMOM", "Moskowitz/Ooi/Pedersen 2012",
     "N 期收益为正则持多，为负则做空（最简趋势）"),
    ("时变夏普", "光大证券",
     "滚动窗口收益/波动，比值高时持多"),
    ("相对强弱单向波动差", "国泰君安",
     "上涨波动与下跌波动之差，衡量多空力量"),
    ("ICU 均线", "中信建投",
     "对高低价分别做包络，突破包络持多"),
    ("鳄鱼线", "Bill Williams",
     "三条不同周期的平滑均线，发散方向定趋势"),
    ("QRS 择时", "东方证券",
     "用高频/低频分量之比判断趋势强弱"),
]


def catalog(strategies: dict | None = None, mq5_specs: dict | None = None) -> list[dict]:
    """合并出策略库表格数据。"""
    rows = []
    keys = list(strategies or CATALOG)
    for name in keys:
        src, idea, formula = CATALOG.get(name, ("—", "—", "—"))
        params = (strategies or {}).get(name, {}).get("params", {})
        mq5 = MQ5_NAME.get(name, name)
        has_mq5 = mq5 in (mq5_specs or {})
        rows.append({
            "策略": name,
            "出处": src,
            "原理": idea,
            "公式": formula,
            "参数": params,
            "MQL5 名称": mq5 if has_mq5 else "—",
            "可导出 EA": "是" if has_mq5 else "否",
        })
    return rows


def mq5_name(strategy: str) -> str:
    return MQ5_NAME.get(strategy, strategy)
