"""信号台数据快照生成器（只读，不改动现有系统任何文件）。

数据来源全部是项目里已有的真实文件：
  data/market/GOLD_H1.csv        MT5 数据桥导出的真实 H1 行情（含点差、成交量）
  data/forge_runs/*.json         101 次真实挖掘运行的全局 Deflated p
  data/market/index.json         数据源元信息
  项目交接与结论_20261104.md       MT5 真 tick 验收数字（人工录入，见 CONST 段）

输出 snapshot.js，供 index.html 在 file:// 下直接引用（不引 CDN、不需要服务）。

口径声明：本脚本重算的通道 / 信号 / 净值曲线只用于界面可视化，
不是验收结果，也不能替代 MT5 真 tick 回测。所有验收数字单独标注来源。
"""

from __future__ import annotations

import glob
import hashlib
import json
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "market" / "GOLD_H1.csv"
OUT = pathlib.Path(__file__).resolve().parent / "snapshot.js"

WIN = 1200          # 价格图展示最近 1200 根 H1（约 2.5 个月）
MA_N, ATR_M, K = 20, 14, 1.8

# 人工录入：来自「项目交接与结论_20261104.md」与「交接文档.md」的实测数字。
# 这些不是本脚本算出来的，界面上会标来源，避免与本地重算混淆。
MT5_VERIFY = {
    "ea": "GoldEA_ATR_k18 （ATR 通道突破 n=20 m=14 k=1.8 confirm=1）",
    "period": "2025-09-11 → 2026-10-30 · MT5 真 tick · GOLD# H1 · 0.01 手 / 10,000 本金",
    "return": 0.0727, "pf": 1.3951, "dd": -0.0391,
    "trades": 102, "sharpe": 2.46, "verdict": "未通过",
    "verdict_why": "PF 1.3951 差门槛 1.4 仅 0.005 · 方向 51.1% < 55% · BH q 与全局 p 未过 · 多品种 0/3",
    "source": "项目交接与结论_20261104.md（MT5 真 tick 实测）",
}
TUNE_ROUNDS = [
    {"name": "基准", "desc": "190σ / SL2% / TP5%", "pf": 1.2616},
    {"name": "A 止盈6.5%", "desc": "放宽止盈", "pf": 1.2695},
    {"name": "B 止损1.6%", "desc": "收紧止损", "pf": 1.0071},
    {"name": "C 入场210σ", "desc": "更保守入场", "pf": 1.1950},
    {"name": "D 出场1.0σ", "desc": "提前出场", "pf": 1.2233},
    {"name": "E 时间止损20", "desc": "加快周转", "pf": 1.0817},
    {"name": "F D+E", "desc": "两项叠加", "pf": 0.9486},
]
COMBO = {
    "rows": [
        {"name": "均值回归单项", "hit": 0.579, "ret": 0.796, "dd": -0.052, "sharpe": 2.48},
        {"name": "ATR 单项", "hit": 0.511, "ret": 0.847, "dd": -0.073, "sharpe": 1.31},
        {"name": "70/30 组合", "hit": 0.522, "ret": 0.825, "dd": -0.035, "sharpe": 2.71},
    ],
    "folds": [0.108, 0.009, 0.075], "pbo": 0.117,
    "source": "项目交接与结论_20261104.md · Python 全样本 2021-03 ~ 2026-10",
}
# 12 条门槛：只填项目里有据可查的实际值；查不到的一律 None（界面显示「未测」），绝不猜数。
GATES = [
    {"name": "方向准确率", "need": "≥55%", "value": 0.511, "fmt": "51.1%", "ok": False, "scale": [0.45, 0.60], "mark": 0.55},
    {"name": "样本外跑赢基准", "need": ">0", "value": None, "fmt": "未测", "ok": None, "scale": None, "mark": None},
    {"name": "块置换 p", "need": "<0.05", "value": None, "fmt": "未测", "ok": None, "scale": None, "mark": None},
    {"name": "BH 校正 q", "need": "<0.05", "value": None, "fmt": "未测", "ok": None, "scale": None, "mark": None},
    {"name": "Deflated p", "need": "<0.05", "value": None, "fmt": "未测（全局 0.934）", "ok": None, "scale": None, "mark": None},
    {"name": "最大回撤", "need": "≥ −20%", "value": -0.0391, "fmt": "−3.91%", "ok": True, "scale": [-0.25, 0], "mark": -0.20},
    {"name": "盈利因子", "need": "≥1.40", "value": 1.3951, "fmt": "1.3951", "ok": False, "scale": [0.8, 1.8], "mark": 1.4},
    {"name": "每年回合", "need": "30~300", "value": 90.0, "fmt": "≈90", "ok": True, "scale": [0, 320], "mark": 30},
    {"name": "多品种 ≥60%", "need": "≥2 品种中 ≥60% 有效", "value": 0.0, "fmt": "0/3 通过", "ok": False, "scale": [0, 1], "mark": 0.6},
    {"name": "滚动 3 段 ≥2 段为正", "need": "≥2 段", "value": None, "fmt": "未测（组合 3/3）", "ok": None, "scale": None, "mark": None},
    {"name": "参数平台期", "need": "同号率 ≥60%", "value": None, "fmt": "未测", "ok": None, "scale": None, "mark": None},
    {"name": "CSCV-PBO", "need": "≤0.50", "value": 0.117, "fmt": "0.117（组合）", "ok": True, "scale": [0, 1], "mark": 0.5, "lower_better": True},
]


def fnum(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def main():
    raw = CSV.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()[:16]

    df = pd.read_csv(CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    vol = df["tick_volume"].to_numpy(float)
    spr = df["spread"].to_numpy(float)
    times = df["time"]

    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr = pd.Series(tr).rolling(ATR_M, min_periods=ATR_M).mean().to_numpy()
    ma = pd.Series(c).rolling(MA_N, min_periods=MA_N).mean().to_numpy()
    upper = ma + K * atr
    lower = ma - K * atr

    # 信号：收盘突破上轨持多，回落中轨离场；只做多；仓位 shift(1) 执行。
    n = len(c)
    state = np.zeros(n, dtype=int)      # bar i 收盘后的目标仓位
    sig_idx = []
    pos = 0
    for i in range(MA_N, n):
        if not np.isfinite(atr[i]):
            continue
        if pos == 0 and c[i] > upper[i]:
            sig_idx.append((i, "entry"))
            pos = 1
        elif pos == 1 and c[i] < ma[i]:
            sig_idx.append((i, "exit"))
            pos = 0
        state[i] = pos

    pos_at_bar = np.r_[0, state[:-1]]              # t 收盘的信号，t+1 才生效
    bar_ret = np.r_[0.0, c[1:] / c[:-1] - 1.0]
    # 每次开 / 平仓扣一次真实点差（CSV spread 为 points，GOLD# 1 point ≈ 0.01 USD）
    cost_rate = spr * 0.01 / np.maximum(c, 1.0)
    trade_change = np.zeros(n, dtype=bool)
    trade_change[1:] = pos_at_bar[1:] != pos_at_bar[:-1]
    cost = np.zeros(n)
    cost[1:] = np.where(trade_change[1:], cost_rate[1:], 0.0)
    eq = np.cumprod(1.0 + pos_at_bar * bar_ret - cost)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0

    # 全周期净值按日重采样
    eqs = pd.Series(eq, index=times)
    daily = eqs.resample("1D").last().dropna()
    equity_all = [[t.strftime("%Y-%m-%d"), fnum(v, 5)] for t, v in daily.items()]
    dds = pd.Series(dd, index=times).resample("1D").min().reindex(daily.index).fillna(0.0)
    dd_all = [[t.strftime("%Y-%m-%d"), fnum(v, 5)] for t, v in dds.items()]

    # 最近 WIN 根：价格 + 通道
    s = max(0, n - WIN)
    bars = []
    for i in range(s, n):
        bars.append([
            times.iloc[i].strftime("%Y-%m-%d %H:%M"),
            fnum(o[i]), fnum(h[i]), fnum(l[i]), fnum(c[i]),
            int(vol[i]), fnum(spr[i], 1),
            fnum(ma[i]), fnum(upper[i]), fnum(lower[i]), fnum(atr[i]),
        ])
    signals = [
        {"t": times.iloc[i].strftime("%Y-%m-%d %H:%M"), "type": t,
         "price": fnum(c[i]), "idx": i - s}
        for i, t in sig_idx if i >= s
    ]

    # 当前状态（全部由最后一根 bar 的真实数据算出）
    last = n - 1
    in_pos = int(state[last])
    dist_up = upper[last] - c[last]
    dist_mid = c[last] - ma[last]
    look = 1000
    lo = max(0, n - look)
    vol_pct = float((atr[last] / c[last] <= (atr[lo:] / c[lo:])).mean() * 100)
    slope = np.abs(ma[last] / ma[last - 10] - 1.0) if np.isfinite(ma[last - 10]) else 0.0
    ma_shift10 = np.r_[np.full(10, np.nan), ma[:-10]]
    slope_hist = np.abs(ma[lo:] / ma_shift10[lo:] - 1.0)
    trend_pct = float((slope <= slope_hist[np.isfinite(slope_hist)]).mean() * 100)
    spr_pct = float((spr[last] <= spr[lo:]).mean() * 100)
    trend_dir = 1 if (np.isfinite(ma[last - 10]) and ma[last] > ma[last - 10]) else -1

    # 时段活跃度：最近 3000 根按小时的真实 tick_volume 均值
    hh = max(0, n - 3000)
    hv = pd.DataFrame({"hour": times.iloc[hh:].dt.hour, "v": vol[hh:]}).groupby("hour")["v"].mean()
    hourly = [fnum(hv.get(k, 0.0), 0) for k in range(24)]
    hv_max = max(hourly) or 1.0

    # 信号漏斗：真实可统计的层；H4 大周期确认需要 EA 侧日志，明确留空。
    spread_ok = int((spr[s:] <= 40).sum())
    win_sigs = [(i, t) for i, t in sig_idx if i >= s]
    entries = sum(1 for _, t in win_sigs if t == "entry")
    rounds, open_round = 0, False
    for _, t in win_sigs:
        if t == "entry":
            open_round = True
        elif t == "exit" and open_round:
            rounds += 1
            open_round = False
    funnel = [
        {"label": "新 bar 到达", "n": n - s, "real": True},
        {"label": "点差 ≤ 40 points", "n": spread_ok, "real": True,
         "note": "用行情文件里的真实 spread 列统计"},
        {"label": "大周期 H4 同向", "n": None, "real": False,
         "note": "EA 侧过滤，需决策日志才能统计"},
        {"label": "收盘突破上轨（做多信号）", "n": entries, "real": True},
        {"label": "完成的回合（入场→离场）", "n": rounds, "real": True},
    ]

    # 101 次真实挖掘运行的全局 Deflated p
    p_hist = []
    for p in sorted(glob.glob(str(ROOT / "data" / "forge_runs" / "*.json"))):
        try:
            d = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
            if "global_p" in d and d.get("ts"):
                p_hist.append({"t": d["ts"][:16].replace("T", " "),
                               "p": fnum(d["global_p"], 4),
                               "n": d.get("n_sampled"), "pass": d.get("n_pass")})
        except Exception:
            continue
    p_hist.sort(key=lambda x: x["t"])

    data = {
        "meta": {
            "symbol": "GOLD#", "timeframe": "H1",
            "rows": int(n),
            "start": times.iloc[0].strftime("%Y-%m-%d %H:%M"),
            "end": times.iloc[last].strftime("%Y-%m-%d %H:%M"),
            "sha256": sha, "win": len(bars), "signals_total": len(sig_idx),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "data/market/GOLD_H1.csv（MT5 GoldBridge 导出）",
            "params": {"ma": MA_N, "atr": ATR_M, "k": K},
            "disclaimer": "通道 / 信号 / 净值曲线为本地 H1 重算，仅用于可视化，不是验收结果；验收数字见「真 tick 成绩单」。",
        },
        "bars": bars, "signals": signals,
        "equity_all": equity_all, "dd_all": dd_all,
        "state": {
            "time": times.iloc[last].strftime("%Y-%m-%d %H:%M"),
            "close": fnum(c[last]), "ma": fnum(ma[last]),
            "upper": fnum(upper[last]), "lower": fnum(lower[last]), "atr": fnum(atr[last]),
            "spread": fnum(spr[last], 1), "in_pos": in_pos,
            "dist_up": fnum(dist_up), "dist_up_atr": fnum(dist_up / atr[last]),
            "dist_mid": fnum(dist_mid),
            "vol_pct": fnum(vol_pct, 0), "trend_pct": fnum(trend_pct, 0),
            "spr_pct": fnum(spr_pct, 0), "trend_dir": trend_dir,
            "hourly": hourly, "hourly_max": fnum(hv_max, 0),
            "vol_regime": "高波动" if vol_pct >= 70 else ("低波动" if vol_pct <= 30 else "常态波动"),
            "trend_regime": "强趋势" if trend_pct >= 70 else ("弱趋势" if trend_pct <= 30 else "常态趋势"),
            "trend_label": "上行" if trend_dir > 0 else "下行",
            "regime_rule": "规则判定：ATR/价格 与 MA20 斜率在过去 1000 根 H1 中的分位，非预测、非新闻情绪",
        },
        "funnel": funnel,
        "mt5": MT5_VERIFY, "tune": TUNE_ROUNDS, "combo": COMBO,
        "gates": GATES, "p_hist": p_hist,
        "missing": [
            {"name": "新闻 / 事件情绪", "why": "项目无新闻数据源", "how": "接免费财经日历 API，或 MT5 经济日历"},
            {"name": "COT 持仓 / 多空比", "why": "项目无持仓报告数据", "how": "CFTC COT 周报（免费），或经纪商多空比"},
            {"name": "恐慌指数 / ETF 流向", "why": "项目无跨市场数据", "how": "AkShare / 公开 CSV，日频即可"},
            {"name": "EA 决策日志", "why": "EA 尚未写逐 bar 闸门判定", "how": "在 mq5.py 的 ReportState 扩 5 列即可"},
        ],
    }

    OUT.write_text("window.DESK = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"OK  snapshot.js  {OUT.stat().st_size/1024:.0f} KB")
    print(f"    bars={n}  window={len(bars)}  signals_total={len(sig_idx)}  window_signals={len(signals)}")
    print(f"    last={data['state']['time']}  close={data['state']['close']}  "
          f"upper={data['state']['upper']}  ma={data['state']['ma']}  in_pos={in_pos}")
    print(f"    vol_pct={data['state']['vol_pct']}  trend_pct={data['state']['trend_pct']}  "
          f"spr_pct={data['state']['spr_pct']}  p_runs={len(p_hist)}  equity_days={len(equity_all)}")


if __name__ == "__main__":
    main()
