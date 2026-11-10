"""策略工作台（界面背后的纯逻辑，可单测）。

对应界面：`定义你的策略` → `生成候选` → `候选评测` → `策略预览 / 执行规则 / 导出`。

纪律：
- 生成与评测**只有一个实现**（复用 forge.genome_space / evaluate_genome）；
- 结论只来自 `verify.make_gates` + `verdict_of`，界面不得自己判；
- 净值曲线是**真实回测序列**，取不到就不画（不生成随机走势）。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

import pandas as pd

from gold.backtest import spread_fee_bps
from gold.factors import build_features
from gold.forge import Genome, build_position, evaluate_genome, genome_space
from gold.mining import deflated_p
from gold.verify import (
    gates_to_reasons,
    make_gates,
    plateau_check,
    reasons_text,
    verdict_of,
)


# ---------------------------------------------------------------- 请求
@dataclass
class Request:
    """界面左侧"定义你的策略"的输入。"""

    symbol: str = "GOLD"
    tf: str = "H1"
    entries: list[str] = field(default_factory=list)     # 空=全部
    filters: list[str] = field(default_factory=list)     # 空=全部
    mode: str = "sample"          # sample / full / evolve
    n: int = 60
    sl: float | None = None       # 单笔风险（止损比例）
    tp: float | None = None       # 止盈比例
    allow_short: bool = False
    seed: int = 7
    n_perm: int = 30

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "tf": self.tf, "entries": self.entries,
                "filters": self.filters, "mode": self.mode, "n": self.n,
                "sl": self.sl, "tp": self.tp, "allow_short": self.allow_short,
                "seed": self.seed, "n_perm": self.n_perm}


def build_genomes(req: Request) -> list[Genome]:
    """按请求构造候选基因组（全枚举或按 seed 采样，均确定性）。"""
    space = genome_space((False, True) if req.allow_short else (False,))
    # 额外过滤（大周期确认 / 欧美时段）不在 genome_space 里，需要手动展开，
    # 否则用户只勾这两类过滤时空间为空 → 界面"生成不出东西"。
    try:
        from gold.combo import EXTRA_FILTERS as _EXTRA

        _wanted = [f for f in (req.filters or []) if f in _EXTRA]
        if _wanted:
            _base = list(space)
            for _f in _wanted:
                for _g in _base:
                    space.append(Genome(_g.entry, dict(_g.params or {}), _f,
                                        _g.allow_short, _g.sl, _g.tp))
    except Exception:  # noqa: BLE001
        pass
    if req.entries:
        space = [g for g in space if g.entry in set(req.entries)]
    if req.filters:
        space = [g for g in space if g.filter in set(req.filters)]
    if not space:
        return []
    if req.mode == "full" or int(req.n) >= len(space):
        picked = space
    else:
        rng = random.Random(int(req.seed))
        picked = rng.sample(space, int(req.n))
    # 风控是"覆盖"而不是"筛选"：用户给了单笔风险，就用它替换每个基因组的止损止盈
    if req.sl is not None or req.tp is not None:
        picked = [Genome(g.entry, dict(g.params), g.filter, g.allow_short,
                         req.sl, req.tp) for g in picked]
    return picked


def _same_stop(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < 1e-12


def space_size(req: Request) -> int:
    """该请求对应的完整空间大小（用于界面显示"全枚举 N 个"）。"""
    return len(build_genomes(Request(**{**req.as_dict(), "mode": "full"})))


# ---------------------------------------------------------------- 评测
def _annual_rounds(rounds: int, oos_bars: int, tf: str) -> float:
    from gold.freqs import BARS_PER_YEAR

    per_year = float(BARS_PER_YEAR.get(str(tf).upper(), 252) or 252)
    years = max(1e-9, float(oos_bars) / per_year)
    return float(rounds) / years


def evaluate_candidates(df: pd.DataFrame, genomes: list[Genome], *,
                        tf: str = "H1", symbol: str = "GOLD",
                        spread_usd: float = 0.33, n_perm: int = 30,
                        fold=None, plateau_top: int = 6,
                        min_dir: float = 0.55, min_pf: float = 1.4,
                        max_dd: float = 0.20,
                        min_rounds_year: float = 30.0,
                        max_rounds_year: float = 300.0,
                        alpha: float = 0.05,
                        multi_symbol: bool = True,
                        pbo: bool = True) -> pd.DataFrame:
    """批量评测候选，返回界面表格 + 结论 + 拒绝原因。

    ``fold`` 由验证器给定（`oos.outward_fold`）；不给就退回唯一合法切分
    `mining.split_is_oos`。**不允许在这里自己切。**
    """
    from gold import oos as _oos
    from gold.mining import split_is_oos

    if df is None or len(df) < 400 or not genomes:
        return pd.DataFrame()
    # 默认样本外 = 最后 20%（约 6700 根 H1 ≈ 9 个月）。
    # 原来用 outward_fold（仅 ~130 根 bar），低频策略在里面只有 2~5 笔，
    # 胜率和夏普都没有统计意义。
    fold = fold if fold is not None else _oos.fixed_fold(df, 0.8)
    is_df, oos_df = _oos.slice_fold(df, fold)
    feat = build_features(df)
    fee = spread_fee_bps(float(df["close"].mean()), float(spread_usd))

    rows = []
    for g in genomes:
        # 低频策略（ATR 通道、均值回归）在短样本外段可能只有 2~4 笔，
        # 之前 min_trades=5 会把它们全部丢掉 → 界面显示"没有候选"。
        ev = evaluate_genome(df, feat, is_df, oos_df, fee, g, min_trades=2,
                             tf=tf, n_perm=int(n_perm))
        if ev is None:
            continue
        rows.append({
            "EA": ev["label"], "genome": g,
            "进场": g.entry,
            "参数": ",".join(f"{k}={v}" for k, v in sorted(g.params.items()))
                    or "默认",
            "过滤": g.filter,
            "方向": "双向" if g.allow_short else "只多",
            "止损": g.sl, "止盈": g.tp,
            "夏普": float(ev["OOS夏普"]),
            "收益": float(ev["OOS收益"]),
            "回撤": float(ev["OOS回撤"]),
            "方向准确率": float(ev["OOS方向"]),
            "回合": int(ev["回合"]),
            "每年回合": _annual_rounds(int(ev["回合"]), len(oos_df), tf),
            "盈利因子": float(ev["盈利因子"]),
            "块置换p": float(ev["块置换p"]),
            "基准": float(ev["基准收益"]),
            "OOS已用": 0,
        })
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("夏普", ascending=False) \
        .reset_index(drop=True)

    # 平台期只算前 N 个（每个要跑一组邻域，代价高）
    same_sign = [float("nan")] * len(out)
    for i in range(min(int(plateau_top), len(out))):
        try:
            same_sign[i] = float(plateau_check(out.at[i, "genome"], df, tf=tf,
                                               spread_usd=spread_usd,
                                               fold=fold)["同号率"])
        except Exception:  # noqa: BLE001
            same_sign[i] = float("nan")

    # ---- CSCV-PBO：这批候选的过拟合概率 ----
    # 需要"逐个配置的逐期净收益"矩阵，口径与 evaluate_genome 一致
    # （OOS 段、shift(1) 生效、按仓位变动扣点差）。
    pbo_res: dict = {}
    if pbo and len(out) >= 3:
        try:
            from gold import pbo as _pbo

            fwd_oos = oos_df["close"].astype(float).pct_change().shift(-1)
            fee_frac = float(fee) / 10000.0
            feat_oos = feat.loc[oos_df.index]
            cols = {}
            for i in range(len(out)):
                try:
                    pos = build_position(oos_df, feat_oos,
                                         out.at[i, "genome"]).shift(1).fillna(0.0)
                    net = (pos * fwd_oos.fillna(0.0)
                           - fee_frac * pos.diff().abs().fillna(0.0))
                    cols[str(i)] = net
                except Exception:  # noqa: BLE001
                    continue
            if len(cols) >= 3:
                pbo_res = _pbo.pbo_from_returns_by_config(cols, n_blocks=12)
        except Exception as exc:  # noqa: BLE001
            pbo_res = {"ok": False, "reason": str(exc), "pbo": float("nan")}
    _pbo_val = pbo_res.get("pbo") if pbo_res.get("ok") else float("nan")

    # ---- 多品种检验（只对前 N 个做，代价高）----
    multi: dict[int, dict] = {}
    if multi_symbol:
        try:
            from gold import multisymbol as _ms

            if _ms.available():
                for i in range(min(int(plateau_top), len(out))):
                    try:
                        multi[i] = _ms.check(out.at[i, "genome"])
                    except Exception:  # noqa: BLE001
                        continue
        except Exception:  # noqa: BLE001
            multi = {}

    # 多重检验：K = 本次评测的候选数
    k = max(1, len(out))
    q_list = []
    from gold.mining import bh_qvalues

    q_list = bh_qvalues([float(v) for v in out["块置换p"]])

    verdicts, reasons, texts = [], [], []
    for i, row in out.iterrows():
        gates = make_gates({
            "方向": row["方向准确率"],
            "超额": float(row["收益"]) - float(row["基准"]),
            "回撤": row["回撤"], "盈利因子": row["盈利因子"],
            "每年回合": row["每年回合"], "块置换p": row["块置换p"],
            "q": q_list[i] if i < len(q_list) else None,
            "deflated_p": deflated_p(float(row["夏普"]), k),
        }, min_dir=min_dir, max_dd=max_dd, min_pf=min_pf,
            min_rounds_year=min_rounds_year,
            max_rounds_year=max_rounds_year, alpha=alpha,
            n_symbols_passed=(multi[i]["n_symbols_passed"]
                              if i in multi else None),
            n_symbols_tested=(multi[i]["n_symbols_tested"]
                              if i in multi else None),
            wf_positive=None, wf_total=None,
            same_sign_rate=(None if same_sign[i] != same_sign[i]
                            else same_sign[i]),
            pbo=(None if _pbo_val != _pbo_val else _pbo_val))
        v, _why = verdict_of(gates)
        rs = gates_to_reasons(gates)
        verdicts.append(v)
        reasons.append(rs)
        texts.append(reasons_text(rs))

    out["q"] = q_list
    out["PBO"] = _pbo_val
    out.attrs["pbo_detail"] = pbo_res
    out["多品种通过"] = [f"{multi[i]['n_symbols_passed']}/"
                        f"{multi[i]['n_symbols_tested']}"
                        if i in multi else "未测"
                        for i in range(len(out))]
    out["多品种比例"] = [multi[i]["ratio"] if i in multi else float("nan")
                        for i in range(len(out))]
    out["多品种达标"] = [bool(multi[i].get("reach_60")) if i in multi else None
                        for i in range(len(out))]
    out.attrs["multi_window"] = (multi[0]["window"] if 0 in multi else None)
    out["同号率"] = same_sign
    out["结论"] = verdicts
    out["reject_reason"] = reasons
    out["拒绝原因"] = texts
    out["入池"] = [v == "通过" for v in verdicts]
    out.attrs["k_tests"] = k
    out.attrs["fold"] = fold.as_dict() if fold is not None else {}
    return out


# ---------------------------------------------------------------- 展示用
def equity_curve(df: pd.DataFrame, g: Genome, *, tf: str = "H1",
                 spread_usd: float = 0.33, points: int = 120) -> list[float]:
    """该基因组的**真实净值曲线**（按点差扣费，抽取到 points 个点）。"""
    if df is None or len(df) < 10:
        return []
    try:
        feat = build_features(df)
        pos = build_position(df, feat, g).shift(1).fillna(0.0)
    except Exception:  # noqa: BLE001
        return []
    fee = spread_fee_bps(float(df["close"].mean()), float(spread_usd))
    fwd = df["close"].astype(float).pct_change().shift(-1)
    pnl = pos * fwd - pos.diff().abs().fillna(0.0) * (fee / 10000.0)
    eq = (1.0 + pnl.fillna(0.0)).cumprod()
    if len(eq) == 0:
        return []
    step = max(1, len(eq) // int(max(10, points)))
    vals = [float(v) for v in eq.iloc[::step].tolist()]
    if vals and vals[-1] != float(eq.iloc[-1]):
        vals.append(float(eq.iloc[-1]))
    return vals


_RULES = {
    "SMA金叉": "快线 SMA({fast}) 上穿慢线 SMA({slow}) 时持多，跌破则离场",
    "唐奇安突破": "收盘突破前 {entry} 根最高价开多，跌破前 {exit_} 根最低价平仓",
    "RSI反转": "RSI(14) 跌破 {buy} 开多，回到 55 上方平仓",
    "MACD金叉": "DIF(12,26) 上穿 DEA({signal}) 持多，死叉离场",
    "LLT上穿": "收盘价站上 LLT({n}) 持多，跌破离场",
    "时序动量TSMOM": "{n} 期收益 > {thr} 时持多",
    "时变夏普": "{n} 期滚动夏普 > {thr} 时持多",
    "鳄鱼线": "鳄鱼线多头排列（lips {lips} > teeth {teeth} > jaw {jaw}，位移 {shift}）时持多",
    # 2026-11-02 新增，必须在这里登记，否则生成器下拉里选不到
    "均线偏离回归": "偏离 MA({ma_n}) 达 {entry_thr}‰ 个标准差后反向做，回到 {exit_thr} 就了结",
    "ATR通道突破": "收盘突破 MA({n})±{k}×ATR({m}) 上轨持多，回落中轨离场（confirm={confirm}）",
}


def rule_text(g: Genome) -> list[dict]:
    """把基因组翻译成三条人话规则（入场 / 保护 / 退出）。"""
    tpl = _RULES.get(g.entry)
    params = dict(g.params or {})
    entry_txt = None
    if tpl:
        try:
            # 只能用**参数本身**做占位符。唐奇安那个参数就叫 entry，
            # 若把 "entry" 映射成策略名，会串成"突破前 唐奇安突破 根"。
            entry_txt = tpl.format(**params)
        except (KeyError, IndexError, ValueError):
            entry_txt = None
    if not entry_txt:
        p_txt = (",".join(f"{k}={v}" for k, v in sorted(params.items()))
                 or "默认")
        entry_txt = f"{g.entry}({p_txt}) 满足条件时持多"
    if g.filter == "低波动":
        entry_txt += "；且只在波动分位 ≤70% 时交易"
    elif g.filter == "强趋势":
        entry_txt += "；且只在趋势分位 ≥50% 时交易"

    risk = []
    if g.sl is not None:
        risk.append(f"浮亏达 {g.sl:.0%} 止损离场")
    if g.tp is not None:
        risk.append(f"浮盈达 {g.tp:.0%} 止盈离场")
    protect = "；".join(risk) if risk else "不带固定止损止盈（靠信号反向离场）"

    exit_txt = ("信号转空即平仓（只多模式不反手）" if not g.allow_short
                else "信号转空反手做空（同一时刻只持一个方向）")
    return [{"n": "01", "标题": "入场 · 做多" if not g.allow_short
             else "入场 · 双向", "内容": entry_txt},
            {"n": "02", "标题": "保护 · 止损止盈", "内容": protect},
            {"n": "03", "标题": "退出", "内容": exit_txt}]


def export_json(g: Genome, meta: dict | None = None) -> str:
    """把基因组导成 JSON（可直接再喂回 `Genome(...)`）。"""
    payload = {"genome": g.as_dict(), "label": g.label(),
               "rules": rule_text(g), "meta": meta or {}}
    return json.dumps(payload, ensure_ascii=False, indent=2)


_ASCII = {
    "SMA金叉": "MA", "唐奇安突破": "Donchian", "MACD金叉": "MACD",
    "RSI反转": "RSI", "LLT上穿": "LLT", "时序动量TSMOM": "TSMOM",
    "时变夏普": "TVS", "鳄鱼线": "Alligator",
    "均线偏离回归": "MADev", "ATR通道突破": "ATR",
}
_FILTER_ASCII = {"低波动": "Vol", "强趋势": "Trend"}


def mql5_name(g: Genome) -> str:
    """给 EA 起一个纯 ASCII 的文件名（MT5 与 Windows 都安全）。

    例：`GoldEA_Donchian_entry20_exit10_Trend_SL2_TP4`
    """
    import re as _re

    bits = [_ASCII.get(g.entry, "EA")]
    for k, v in sorted((g.params or {}).items()):
        bits.append(f"{k}{v}")
    if g.filter in _FILTER_ASCII:
        bits.append(_FILTER_ASCII[g.filter])
    if g.allow_short:
        bits.append("Both")
    if g.sl:
        bits.append(f"SL{int(round(float(g.sl) * 100))}")
    if g.tp:
        bits.append(f"TP{int(round(float(g.tp) * 100))}")
    clean = [_re.sub(r"[^0-9A-Za-z_]", "", str(b)) or "x" for b in bits]
    return ("GoldEA_" + "_".join(clean))[:60]


def summary_line(row: pd.Series) -> str:
    """一句话概括某个候选（界面卡片用）。"""
    return (f"{row['进场']} · {row['过滤']} · {row['方向']} ｜ "
            f"夏普 {row['夏普']:.2f} · 收益 {row['收益']:+.1%} · "
            f"回撤 {row['回撤']:.1%} · PF {row['盈利因子']:.2f}")
