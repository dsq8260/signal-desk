"""多品种检验：同一条 EA 在多个品种上是否都"有效"。

为什么要按**共同重叠区间**比
--------------------------
实测（2026-11-01，XM）：GOLD# 有 33,545 根 H1（2021-03 起），但
EURUSD# 只有 15,778 根（2024-04 起）、USDJPY# 13,402 根（2024-08 起）、
GBPUSD# 只有 680 根、SILVER# 没有 H1 数据。
如果各品种各用各的区间，"多品种有效"就变成了拿 5.7 年黄金和 0.1 年英镑比 ——
所以这里只取所有参与品种的**时间交集**，并在结果里写明用了哪段。

"有效"的口径（可配置，默认与单品种核心门槛一致）：
方向准确率 ≥55% 且 超额收益 >0（跑赢该品种 buy&hold）且 盈利因子 ≥1.0。
**不看夏普**：跨品种波动差异太大，夏普不可直接比。
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

from gold.backtest import spread_fee_bps
from gold.factors import build_features
from gold.forge import Genome, build_position
from gold.symbols import spread_for
from validation import metrics as M

MARKET_DIR = pathlib.Path("data/market")
INDEX_FILE = MARKET_DIR / "index.json"

# 我们的品种名 → 文件名前缀（CSV 落盘时去掉了 #）
MIN_BARS = 2000          # 少于这个根数的品种不参与（GBPUSD 只有 680 根）


def load_index() -> dict:
    if not INDEX_FILE.exists():
        return {}
    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def available(min_bars: int | None = None) -> dict[str, dict]:
    """可参与多品种检验的符号（按根数过滤）。

    ``min_bars=None`` 时用模块级 ``MIN_BARS``（**运行时**读取，
    否则改常量不会生效 —— 默认参数在定义时就绑定了）。
    """
    th = int(MIN_BARS if min_bars is None else min_bars)
    idx = load_index().get("symbols") or {}
    return {k: v for k, v in idx.items() if int(v.get("rows") or 0) >= th}


def load_bars(symbol: str, market_dir: pathlib.Path | None = None) -> pd.DataFrame:
    idx = available()
    meta = idx.get(symbol)
    if not meta:
        return pd.DataFrame()
    d = pathlib.Path(market_dir or MARKET_DIR)
    path = d / str(meta["file"])
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    tcol = "time" if "time" in raw.columns else raw.columns[0]
    raw[tcol] = pd.to_datetime(raw[tcol], errors="coerce")
    raw = raw.dropna(subset=[tcol]).drop_duplicates(tcol)
    return raw.set_index(tcol).sort_index()


def common_window(symbols: list[str] | None = None,
                  market_dir: pathlib.Path | None = None) -> tuple[pd.Timestamp,
                                                                    pd.Timestamp, list[str]]:
    """所有参与品种的时间交集。返回 (start, end, 实际参与的符号)。"""
    idx = available()
    syms = [s for s in (symbols or list(idx)) if s in idx]
    if not syms:
        return pd.NaT, pd.NaT, []
    starts = [pd.Timestamp(idx[s]["start"]) for s in syms]
    ends = [pd.Timestamp(idx[s]["end"]) for s in syms]
    start, end = max(starts), min(ends)
    if start >= end:
        return pd.NaT, pd.NaT, []
    return start, end, syms


def _metrics_on(df: pd.DataFrame, g: Genome, tf: str,
                spread_usd: float) -> dict:
    feat = build_features(df)
    pos = build_position(df, feat, g).shift(1).fillna(0.0)
    fwd = df["close"].astype(float).pct_change().shift(-1)
    fee = spread_fee_bps(float(df["close"].mean()), float(spread_usd))
    st = M.summarize_strategy(fwd, pos, fee_bps=fee, freq="H1",
                              allow_short=g.allow_short)
    rt = st.get("roundtrip", {}) or {}
    bench = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0)
    ret = float(st["net"].get("total_return", float("nan")))
    hit = float(st["hit"].get("hit_rate", float("nan")))
    pf = float(rt.get("pf", float("nan")))
    return {"方向": hit, "收益": ret, "基准": bench, "超额": ret - bench,
            "盈利因子": pf, "回合": int(rt.get("trades", 0)),
            "笔数": int(st["trades"])}


def check(genome: Genome, *, symbols: list[str] | None = None,
          min_dir: float = 0.55, min_pf: float = 1.0,
          bar: str = "方向≥55% 且 超额>0 且 盈利因子≥1.0",
          market_dir: pathlib.Path | None = None) -> dict:
    """在共同重叠区间上逐品种检验，返回通过数与明细。

    ``n_symbols_passed`` / ``n_symbols_tested`` / ``ratio`` 可直接喂给
    `verify.make_gates` 的"多品种 ≥60%"那条门槛。
    """
    start, end, syms = common_window(symbols, market_dir)
    if not syms or pd.isna(start):
        return {"ok": False, "reason": "no_common_window",
                "n_symbols_passed": 0, "n_symbols_tested": 0,
                "ratio": float("nan"), "detail": [], "bar": bar,
                "window": None}

    detail = []
    for sym in syms:
        df = load_bars(sym, market_dir)
        if df.empty:
            continue
        win = df.loc[(df.index >= start) & (df.index <= end)]
        if len(win) < 200:
            continue
        try:
            m = _metrics_on(win, genome, "H1", spread_for(sym.replace("#", "")))
        except Exception:  # noqa: BLE001
            continue
        passed = (m["方向"] == m["方向"] and m["方向"] >= min_dir
                  and m["超额"] == m["超额"] and m["超额"] > 0
                  and m["盈利因子"] == m["盈利因子"] and m["盈利因子"] >= min_pf)
        detail.append({"品种": sym, "通过": bool(passed), **m})

    tested = len(detail)
    passed_n = sum(1 for d in detail if d["通过"])
    return {"ok": tested > 0, "n_symbols_passed": passed_n,
            "n_symbols_tested": tested,
            "ratio": (passed_n / tested) if tested else float("nan"),
            "detail": detail, "bar": bar,
            "window": [str(start), str(end)],
            "reach_60": bool(tested >= 2 and passed_n / tested >= 0.6)}
