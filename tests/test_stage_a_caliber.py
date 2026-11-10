"""阶段 A 验收测试：外汇点差 / 周期年化 / 真实置换 / 禁止跨运行拼接。

这四条是"不伪造"的地基：
- 给 EURUSD 用黄金点差（0.33）会让每笔成本 30%，回测直接 -100%；
- H1 按 D1 年化会把夏普低估约 4.9 倍；
- forge 曾经用 0.5-0.5·tanh(收益) 反推 p（伪造）；
- 累计榜曾经把多次运行的最值拼成一条记录。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.forge import forge
from gold.freqs import freq_key
from gold.mining import mine
from gold.symbols import spread_for
from validation import metrics as M


def _fx_bars(n: int = 900, seed: int = 3, price: float = 1.10,
             vol: float = 0.0012) -> pd.DataFrame:
    """合成外汇行情（EURUSD 量级的价格与波动）。"""
    rng = np.random.default_rng(seed)
    close = price * np.cumprod(1 + rng.normal(0.00002, vol, n))
    idx = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.0008,
                         "low": close * 0.9992, "close": close,
                         "volume": 0.0}, index=idx)


# ---------------------------------------------------------------- 点差
def test_spread_for_matches_spec():
    """点差优先取 MT5 实测（data/market/index.json），没有实测才用静态表。

    2026-11-01 起 GOLD# 实测 16 点 = 0.16 美元（静态表保守写 0.33）。
    这里断言两条不变式：量级正确、外汇绝不能用黄金的点差；
    实测/兜底/未知品种三种来源的细节由 test_spread_model.py 覆盖。
    """
    from gold.symbols import spread_source

    gold = spread_for("GOLD")
    assert 0.10 <= gold <= 0.35, gold              # 美元/盎司量级
    if spread_source("GOLD") == "measured":
        assert gold == pytest.approx(0.16, abs=1e-9)
    for sym in ("EURUSD", "GBPUSD", "USDCHF"):
        v = spread_for(sym)
        assert 0.0 < v < 0.001, (sym, v)           # 1e-4 量级
    assert 0.0 < spread_for("USDJPY") < 0.05


def test_eurusd_with_correct_spread_differs_from_gold_spread():
    """同一份 EURUSD 数据：正确点差 vs 黄金点差，结果必须显著不同。

    把黄金点差（实测 0.16 美元/盎司，旧的静态值 0.33）用到 1.10 的汇价上，
    等于每笔 15%~30% 的成本 —— 必须把收益打到灾难级，这就是要守住的坑。
    """
    df = _fx_bars()
    right = mine(df, "双均线", {"fast": [10], "slow": [40]},
                 spread_usd=spread_for("EURUSD"), n_perm=20, tf="H1")
    wrong = mine(df, "双均线", {"fast": [10], "slow": [40]},
                 spread_usd=spread_for("GOLD"), n_perm=20, tf="H1")
    assert not right.table.empty and not wrong.table.empty
    right_ret = float(right.table["OOS收益"].iloc[0])
    wrong_ret = float(wrong.table["OOS收益"].iloc[0])
    # 用黄金点差必须把收益打到灾难级（实测 0.16 时约 -45%，旧静态 0.33 时更惨）
    assert wrong_ret < -0.3, wrong_ret
    # 正确点差下不可能这么惨
    assert right_ret > wrong_ret + 0.2
    # 成本口径本身也要在合理区间（不是 30%/笔）
    assert right.table["p"].iloc[0] >= 0


def test_fee_bps_scales_with_symbol_price():
    """spread_fee_bps 的换算必须按各自价格：外汇的点差 bps 不能等于黄金的。"""
    from gold.backtest import spread_fee_bps

    gold_fee = spread_fee_bps(2600.0, spread_for("GOLD"))
    fx_fee = spread_fee_bps(1.10, spread_for("EURUSD"))
    assert 0.5 < gold_fee < 3.0      # 黄金约 1.3 bp
    assert 0.5 < fx_fee < 3.0        # 欧元约 1.1 bp
    assert abs(gold_fee - fx_fee) < 1.0


# ---------------------------------------------------------------- 年化
def test_h1_sharpe_is_not_annualised_as_daily():
    """同一份数据：H1 口径的夏普应是 D1 口径的 √(6048/252) ≈ 4.9 倍。"""
    df = _fx_bars()
    h1 = mine(df, "双均线", {"fast": [10], "slow": [40]},
              spread_usd=spread_for("EURUSD"), n_perm=20, tf="H1")
    d1 = mine(df, "双均线", {"fast": [10], "slow": [40]},
              spread_usd=spread_for("EURUSD"), n_perm=20, tf="D1")
    sh1 = abs(float(h1.table["OOS夏普"].iloc[0]))
    sd1 = abs(float(d1.table["OOS夏普"].iloc[0]))
    if sd1 < 1e-9:
        pytest.skip("夏普接近 0，比例无意义")
    ratio = sh1 / sd1
    assert 3.5 < ratio < 6.5, f"H1/D1 夏普比 {ratio:.2f}，应接近 4.9"


def test_freq_key_registers_periods():
    # freq_key 是"注册后再返回"的懒加载，必须先调用才会写进表
    keys = [freq_key(tf) for tf in ("M15", "H4", "D1", "H1")]
    assert keys == ["M15", "H4", "D1", "H1"]
    assert M.PERIODS_PER_YEAR["H1"] == 6048
    assert (M.PERIODS_PER_YEAR["M15"] > M.PERIODS_PER_YEAR["H4"]
            > M.PERIODS_PER_YEAR["D1"])


# ---------------------------------------------------------------- 置换
def test_forge_p_values_come_from_real_permutation():
    """forge 的 p 必须来自块置换：只能是 1/(n_perm+1) 的整数倍。"""
    df = _fx_bars(600)
    n_perm = 30
    r = forge(df, n_candidates=12, spread_usd=spread_for("EURUSD"),
              n_perm=n_perm, tf="H1")
    if r.table.empty:
        pytest.skip("合成数据未产生有效候选")
    ps = pd.to_numeric(r.table["块置换p"], errors="coerce").dropna()
    assert len(ps) > 0, "没有任何置换 p 值 —— 说明置换没跑"
    step = 1.0 / (n_perm + 1)
    for p in ps:
        k = p / step
        assert abs(k - round(k)) < 1e-6, f"p={p} 不是 1/(n_perm+1) 的整数倍"
    assert (ps > 0).all() and (ps <= 1.0).all()


def test_forge_p_is_not_a_tanh_heuristic():
    """回归：不能再出现「按收益反推 p」。对同一批候选，tanh 公式不可能复现真实 p。"""
    df = _fx_bars(600)
    r = forge(df, n_candidates=12, spread_usd=spread_for("EURUSD"),
              n_perm=30, tf="H1")
    if r.table.empty:
        pytest.skip("无候选")
    t = r.table
    fake = [min(1.0, max(0.0, 0.5 - 0.5 * np.tanh(float(net) * 4)))
            for net in t["OOS收益"]]
    real = pd.to_numeric(t["块置换p"], errors="coerce").tolist()
    # 至少有明显不同的行（如果全等，说明又退回启发式了）
    diff = sum(1 for a, b in zip(fake, real) if abs(a - b) > 1e-9)
    assert diff > 0, "真实 p 与 tanh 启发式完全一致 —— 可疑"


def test_forge_records_bh_q_and_benchmark():
    df = _fx_bars(600)
    r = forge(df, n_candidates=12, spread_usd=spread_for("EURUSD"),
              n_perm=30, tf="H1")
    if r.table.empty:
        pytest.skip("无候选")
    for c in ("块置换p", "q", "基准收益", "跑赢基准", "随机检验"):
        assert c in r.table.columns
    q = pd.to_numeric(r.table["q"], errors="coerce").dropna()
    assert ((q >= 0) & (q <= 1)).all()


def test_forge_survivors_require_permutation_significance():
    """过门槛的候选必须块置换 p<0.05；p 不显著的不能进 survivors。"""
    df = _fx_bars(600)
    r = forge(df, n_candidates=30, spread_usd=spread_for("EURUSD"),
              n_perm=50, tf="H1")
    if len(r.survivors) == 0:
        return
    ps = pd.to_numeric(r.survivors["块置换p"], errors="coerce")
    assert (ps < 0.05).all(), ps.to_dict()
