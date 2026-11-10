"""验证与防过拟合套件（阶段 2）。

    from validation import rank_ic, oos_decay, judge_gates, event_study
"""

from __future__ import annotations

from .benchmarks import (
    beaten_by_random,
    buy_and_hold_returns,
    market_index_returns,
    random_signal_distribution,
)
from .event_study import MIN_SAMPLE, event_study, top_events
from .gates import Gate, GateReport, format_gates, judge_gates
from .metrics import (
    count_trades,
    position_composition,
    PERIODS_PER_YEAR,
    hit_rate,
    icir,
    layered_monotonicity,
    net_returns,
    performance_stats,
    rank_ic,
    rank_ic_series,
    summarize_strategy,
    turnover,
)
from .risk import (
    drawdown_series,
    drawdown_table,
    extended_stats,
    format_metrics,
    trade_stats,
)
from .overfit import (
    benjamini_hochberg,
    block_permutation_test,
    block_shuffle,
    oos_decay,
    parameter_sensitivity,
)

__all__ = [
    # metrics
    "rank_ic", "rank_ic_series", "icir", "layered_monotonicity",
    "net_returns", "turnover", "hit_rate", "performance_stats",
    "summarize_strategy", "position_composition", "count_trades", "PERIODS_PER_YEAR",
    # overfit
    "oos_decay", "parameter_sensitivity", "block_permutation_test",
    "block_shuffle", "benjamini_hochberg",
    # benchmarks
    "buy_and_hold_returns", "market_index_returns",
    "random_signal_distribution", "beaten_by_random",
    # event study
    "event_study", "top_events", "MIN_SAMPLE",
    # gates
    "Gate", "GateReport", "judge_gates", "format_gates",
    # risk
    "drawdown_series", "drawdown_table", "trade_stats",
    "extended_stats", "format_metrics",
]
