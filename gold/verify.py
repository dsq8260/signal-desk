"""统一验证器：三段切分（50/25/25）+ walk-forward + 十条硬门槛 + 方向诊断。

为什么需要一个"验证器"而不是散落在各处
--------------------------------------
`mine/forge/evolve/strata/tune_ea` 过去各自算自己的指标、各自定门槛，
于是出现了"这里看夏普、那里看收益"的口径分裂。本模块只做一件事：
给定行情与基因组，输出**一份完整、可比较、可审计的成绩单**，并给出
`通过 / 继续观察 / 未通过` 三档结论。

铁的纪律
--------
- **测试集（最后 25%）只在最后投票使用**，绝不参与选参、去重、排序；
- 参数选择与排名只看 训练段+验证段；
- 块置换在**验证段**上做（测试段只做最终检验，避免"用测试集找显著性"）；
- 单品种时「多品种≥60%」这条判为 `未测`，**不能算通过**。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gold.backtest import block_permutation, spread_fee_bps
from gold.factors import build_features
from gold.freqs import freq_key
from gold.forge import Genome, build_position
from gold.mining import bh_qvalues, deflated_p, emax_sharpe
from validation import metrics as M

# ---------------------------------------------------------------- 切分
def split_train_val_test(df: pd.DataFrame,
                         ratios: tuple[float, float, float] = (0.5, 0.25, 0.25)
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按时间顺序切成 训练 / 验证 / 测试 三段（不允许重叠、不允许打乱）。"""
    r = [float(x) for x in ratios]
    if len(r) != 3 or abs(sum(r) - 1.0) > 1e-9:
        raise ValueError(f"三段比例必须和为 1：{ratios}")
    if min(r) <= 0:
        raise ValueError("每段比例必须为正")
    n = len(df)
    n_tr = int(n * r[0])
    n_va = int(n * r[1])
    return (df.iloc[:n_tr], df.iloc[n_tr:n_tr + n_va], df.iloc[n_tr + n_va:])


def walk_forward_windows(df: pd.DataFrame, n_windows: int = 3,
                         train_frac: float = 0.6
                         ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """滚动窗口：每次用"前 60%"拟合、后 40% 验证，窗口整体向前推。

    返回 [(训练段, 验证段), ...]，只在**不含测试集**的数据上调用。
    """
    n = len(df)
    if n_windows < 1 or n < 60:
        return []
    usable = int(n * 0.95)                      # 留一点余量给最早/最晚的指标热身
    span = usable // int(n_windows)
    out = []
    for i in range(int(n_windows)):
        start = i * span
        end = min(n, start + span)
        seg = df.iloc[start:end]
        if len(seg) < 40:
            continue
        cut = max(10, int(len(seg) * train_frac))
        out.append((seg.iloc[:cut], seg.iloc[cut:]))
    return out


# ---------------------------------------------------------------- 单段指标
def segment_metrics(seg: pd.DataFrame, pos_full: pd.Series, tf: str,
                    fee_bps: float, allow_short: bool = False) -> dict:
    """在给定段上算"执行后"指标（pos shift(1) 次日执行，含点差）。"""
    if len(seg) < 2:
        return {}
    pos = pos_full.shift(1).reindex(seg.index).fillna(0.0)
    fwd = seg["close"].astype(float).pct_change().shift(-1)
    st = M.summarize_strategy(fwd, pos, fee_bps=fee_bps, freq=freq_key(tf),
                              allow_short=allow_short)
    net = st["net"]
    rt = st.get("roundtrip", {}) or {}
    bench = float(seg["close"].iloc[-1] / seg["close"].iloc[0] - 1.0)
    return {
        "收益": float(net.get("total_return", float("nan"))),
        "夏普": float(net.get("sharpe", float("nan"))),
        "回撤": float(net.get("max_drawdown", float("nan"))),
        "方向": float(st["hit"].get("hit_rate", float("nan"))),
        "笔数": int(st["trades"]),
        "回合": int(rt.get("trades", 0)),
        "盈利因子": float(rt.get("pf", float("nan"))),
        "基准": bench,
        "超额": float(net.get("total_return", float("nan"))) - bench,
        "在场": float(st.get("composition", {}).get("long_ratio", float("nan"))),
        "_pos": pos, "_fwd": fwd, "_fee": fee_bps,
    }


# ---------------------------------------------------------------- 方向诊断
def direction_diagnostic(df: pd.DataFrame, g: Genome, tf: str,
                         spread_usd: float, by: str = "year") -> pd.DataFrame:
    """把方向准确率拆开看：按年份 / 按波动分位 / 按小时。

    只回答一个问题：这个策略"判方向"的能力是真的低于 55%，还是被某几段拖累。
    """
    feat = build_features(df)
    fee = spread_fee_bps(float(df["close"].mean()), spread_usd)
    pos = build_position(df, feat, g).shift(1).fillna(0.0)
    hit = M.hit_rate(pos, df["close"].astype(float).pct_change().shift(-1))
    base = {"整体": {"方向": hit.get("hit_rate"), "样本": hit.get("n", 0)}}

    def bucket_stats(key: pd.Series, name: str) -> pd.DataFrame:
        rows = []
        for k, idx in df.groupby(key, observed=True).groups.items():
            seg = df.loc[idx]
            if len(seg) < 30:
                continue
            st = segment_metrics(seg, build_position(df, feat, g), tf, fee)
            if not st:
                continue
            rows.append({name: str(k), "方向": st["方向"], "笔数": st["笔数"],
                         "收益": st["收益"], "样本": len(seg)})
        return pd.DataFrame(rows)

    if by == "year":
        out = bucket_stats(pd.Series(df.index.year, index=df.index), "年份")
    elif by == "hour":
        out = bucket_stats(pd.Series(df.index.hour, index=df.index), "小时")
    elif by == "vol":
        vol = feat["vol_pct"]
        terc = pd.cut(vol, [-0.01, 0.33, 0.67, 1.01],
                      labels=["低波动", "中波动", "高波动"])
        out = bucket_stats(terc, "波动档")
    else:
        raise ValueError(f"未知分桶方式: {by}")
    out.attrs["overall"] = base["整体"]
    return out


# ---------------------------------------------------------------- 门槛
@dataclass
class Gate:
    name: str
    need: str
    value: str
    status: str          # pass / fail / unmeasured

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def make_gates(rec: dict, *, min_dir: float = 0.55, max_dd: float = 0.20,
               min_pf: float = 1.4, min_rounds_year: float = 30.0,
               max_rounds_year: float = 300.0, alpha: float = 0.05,
               n_symbols_passed: int | None = None,
               n_symbols_tested: int | None = None,
               wf_positive: int | None = None, wf_total: int | None = None,
               same_sign_rate: float | None = None,
               min_same_sign: float = 0.6,
               pbo: float | None = None,
               max_pbo: float = 0.5) -> list[Gate]:
    """按规格书的十条硬门槛逐条判定（缺数据一律 unmeasured，不算通过）。"""
    def g(name, need, value, cond):
        if value is None:
            return Gate(name, need, "未测", "unmeasured")
        return Gate(name, need, str(value), "pass" if cond else "fail")

    dd = rec.get("回撤")
    pf = rec.get("盈利因子")
    rpy = rec.get("每年回合")
    p = rec.get("块置换p", rec.get("p"))
    q = rec.get("q")
    dp = rec.get("deflated_p")
    return [
        g("方向准确率", f"≥{min_dir:.0%}", rec.get("方向"),
          rec.get("方向") is not None and rec["方向"] >= min_dir),
        g("样本外跑赢基准", ">0", rec.get("超额"),
          rec.get("超额") is not None and rec["超额"] > 0),
        g("块置换 p", f"<{alpha}", p, p is not None and p < alpha),
        g("BH 校正 q", f"<{alpha}", q, q is not None and q < alpha),
        g("Deflated p", f"<{alpha}", dp, dp is not None and dp < alpha),
        g("最大回撤", f"≤{max_dd:.0%}", dd,
          dd is not None and dd >= -max_dd),
        g("盈利因子", f"≥{min_pf}", pf, pf is not None and pf >= min_pf),
        g("每年回合", f"{min_rounds_year:.0f}~{max_rounds_year:.0f}", rpy,
          rpy is not None and min_rounds_year <= rpy <= max_rounds_year),
        # 单品种环境没有"多品种有效性"这回事 → 判 N/A（未测），既不算过也不算不过
        (Gate("多品种 ≥60%", "≥2 个品种中 ≥60% 有效",
              "单品种 N/A" if (n_symbols_tested or 1) < 2
              else f"{n_symbols_passed}/{n_symbols_tested}",
              "unmeasured" if (n_symbols_tested or 1) < 2
              else ("pass" if (n_symbols_passed or 0) / max(1, n_symbols_tested)
                    >= 0.6 else "fail"))),
        g("滚动 3 段中 ≥2 段为正", "≥2 段", wf_positive,
          (wf_positive or 0) >= 2 and (wf_total or 0) >= 3),
        # 参数平台期：邻域里与中心同号的比例（拒绝"孤立尖峰"最优）
        g("参数平台期", f"同号率≥{min_same_sign:.0%}", same_sign_rate,
          same_sign_rate is not None and same_sign_rate >= min_same_sign),
        # CSCV-PBO：样本内挑最优、样本外还能排进前半的概率（越低越稳）
        g("CSCV-PBO", f"≤{max_pbo:g}",
          (f"{pbo:.3f}" if pbo is not None else None),
          (pbo is not None and pbo <= max_pbo)),
    ]


def verdict_of(gates: list[Gate]) -> tuple[str, str]:
    """三档结论：通过 / 继续观察 / 未通过（未测不能算通过）。"""
    fails = [x for x in gates if x.status == "fail"]
    unmeasured = [x for x in gates if x.status == "unmeasured"]
    if not fails and not unmeasured:
        return "通过", "十条全过"
    if fails:
        return "未通过", "未过：" + "、".join(x.name for x in fails)
    return "继续观察", "缺证据：" + "、".join(x.name for x in unmeasured)


@dataclass
class VerifyResult:
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    segments: dict = field(default_factory=dict)
    k_tests: int = 0
    n_perm: int = 0
    note: str = ""


class Verifier:
    """统一验证器：四段、十条门槛、三档结论，四模式共用。"""

    def __init__(self, df: pd.DataFrame, tf: str = "H1", symbol: str = "GOLD",
                 spread_usd: float = 0.33,
                 ratios: tuple[float, float, float] = (0.5, 0.25, 0.25),
                 n_windows: int = 3, alpha: float = 0.05,
                 max_dd: float = 0.20, min_pf: float = 1.4,
                 min_dir: float = 0.55):
        self.df = df
        self.tf = tf
        self.symbol = symbol
        self.spread_usd = float(spread_usd)
        self.ratios = ratios
        self.n_windows = int(n_windows)
        self.alpha = float(alpha)
        self.max_dd = float(max_dd)
        self.min_pf = float(min_pf)
        self.min_dir = float(min_dir)
        self.train, self.val, self.test = split_train_val_test(df, ratios)
        self.feat = build_features(df)
        self.fee = spread_fee_bps(float(df["close"].mean()), self.spread_usd)
        self._fit = pd.concat([self.train, self.val])   # 只给"选择"用的部分

    # ---- 选择阶段：只看 train+val ----
    def select_metrics(self, g: Genome) -> dict:
        pos = build_position(self.df, self.feat, g)
        tr = segment_metrics(self.train, pos, self.tf, self.fee,
                             g.allow_short)
        va = segment_metrics(self.val, pos, self.tf, self.fee, g.allow_short)
        return {"train": tr, "val": va, "pos": pos}

    # ---- 最终投票：测试集只在这里出现 ----
    def test_metrics(self, g: Genome) -> dict:
        pos = build_position(self.df, self.feat, g)
        return segment_metrics(self.test, pos, self.tf, self.fee,
                               g.allow_short)

    def walk_forward(self, g: Genome) -> list[dict]:
        """滚动窗口只在 train+val 上做，测试集不参与。"""
        pos = build_position(self.df, self.feat, g)
        out = []
        for i, (tr, va) in enumerate(walk_forward_windows(self._fit,
                                                          self.n_windows)):
            st = segment_metrics(va, pos, self.tf, self.fee, g.allow_short)
            if not st:
                continue
            out.append({"窗口": i + 1,
                        "起": str(va.index[0].date()),
                        "止": str(va.index[-1].date()),
                        "收益": st["收益"], "夏普": st["夏普"],
                        "正收益": bool(st["收益"] > 0)})
        return out

    def run(self, genomes: list[Genome], n_perm: int = 200,
            min_trades: int = 20) -> VerifyResult:
        """跑完整流程：选参 → 验证段置换 → 排序去重 → 测试段投票 → 十门槛。"""
        if not genomes:
            return VerifyResult(note="没有候选")
        k_tests = len(genomes)
        rows = []
        for g in genomes:
            sel = self.select_metrics(g)
            tr, va = sel["train"], sel["val"]
            if not tr or not va:
                continue
            if tr["笔数"] < min_trades or va["笔数"] < min_trades // 2:
                continue
            perm = block_permutation(va["_fwd"], va["_pos"], n_perm=int(n_perm),
                                     block=20, fee_bps=self.fee,
                                     seed=7, allow_short=g.allow_short)
            rows.append({
                "EA": g.label(), "进场": g.entry,
                "参数": ",".join(f"{k}={v}" for k, v in sorted(g.params.items()))
                        or "默认",
                "过滤": g.filter, "方向": "双向" if g.allow_short else "只多",
                "训练收益": tr["收益"], "训练夏普": tr["夏普"],
                "验证收益": va["收益"], "验证夏普": va["夏普"],
                "验证方向": va["方向"], "验证笔数": va["笔数"],
                "块置换p": float(perm["p_value"]),
                "_g": g,
            })
        if not rows:
            return VerifyResult(note="没有候选通过最低成交要求")

        tbl = pd.DataFrame(rows)
        # 排序/去重只看验证段（绝不看测试集）
        tbl = tbl.sort_values("验证夏普", ascending=False).reset_index(drop=True)
        tbl["q"] = bh_qvalues([float(v) for v in tbl["块置换p"]])
        best_sr = float(tbl["验证夏普"].max())
        dp = deflated_p(best_sr, k_tests)
        tbl["deflated_p"] = dp

        # 测试段只在最后投票
        tests = []
        for _, r in tbl.iterrows():
            st = self.test_metrics(r["_g"])
            tests.append(st)
        for key in ("收益", "夏普", "回撤", "方向", "笔数", "回合", "盈利因子",
                    "基准", "超额"):
            tbl[f"测试{key}"] = [t.get(key, float("nan")) for t in tests]
        yrs = max(1e-9, len(self.test) / 6000.0) if self.tf.upper() == "H1" \
            else max(1e-9, len(self.test) / 252.0)
        tbl["测试每年回合"] = tbl["测试回合"] / yrs

        # 十门槛 + 结论（单品种：多品种门槛判未测）
        gates_list, verdicts, reasons = [], [], []
        same_signs: list = []
        reason_rows: list = []
        for _, r in tbl.iterrows():
            rec = {"方向": r["测试方向"], "超额": r["测试超额"],
                   "回撤": r["测试回撤"], "盈利因子": r["测试盈利因子"],
                   "每年回合": r["测试每年回合"],
                   "块置换p": r["块置换p"], "q": r["q"], "deflated_p": dp}
            wf = self.walk_forward(r["_g"])
            pos_n = sum(1 for w in wf if w["正收益"])
            # 参数平台期：在训练段邻域上判"是不是孤立尖峰"
            try:
                same_sign = float(plateau_check(r["_g"], self.df, tf=self.tf,
                                                spread_usd=self.spread_usd,
                                                fold=None)["同号率"])
            except Exception:  # noqa: BLE001
                same_sign = None
            gates = make_gates(rec, min_dir=self.min_dir, max_dd=self.max_dd,
                              min_pf=self.min_pf, alpha=self.alpha,
                              n_symbols_passed=1, n_symbols_tested=1,
                              wf_positive=pos_n, wf_total=len(wf),
                              same_sign_rate=same_sign)
            v, why = verdict_of(gates)
            same_signs.append(same_sign)
            reason_rows.append(gates_to_reasons(gates))
            gates_list.append([{"name": x.name, "need": x.need,
                                "value": x.value, "status": x.status}
                               for x in gates])
            verdicts.append(v)
            reasons.append(why)
        tbl["门槛明细"] = gates_list
        tbl["结论"] = verdicts
        tbl["原因"] = reasons
        tbl["同号率"] = same_signs
        tbl["reject_reason"] = reason_rows
        tbl = tbl.drop(columns=["_g"])

        return VerifyResult(
            table=tbl,
            segments={"训练": (str(self.train.index[0].date()),
                             str(self.train.index[-1].date()), len(self.train)),
                      "验证": (str(self.val.index[0].date()),
                             str(self.val.index[-1].date()), len(self.val)),
                      "测试": (str(self.test.index[0].date()),
                             str(self.test.index[-1].date()), len(self.test))},
            k_tests=k_tests, n_perm=int(n_perm),
            note=(f"K={k_tests} · 每候选验证段置换 {int(n_perm)} 次 · "
                  f"最优验证夏普 {best_sr:.2f} · Deflated p={dp:.3f} · "
                  f"通过 {verdicts.count('通过')} · "
                  f"继续观察 {verdicts.count('继续观察')} · "
                  f"未通过 {verdicts.count('未通过')}"),
        )


# ---------------------------------------------------------------- 四种模式的适配层
def _genome_from_dict(g) -> Genome | None:
    if not isinstance(g, dict) or not g.get("entry"):
        return None
    return Genome(str(g["entry"]), dict(g.get("params") or {}),
                  g.get("filter") or "无过滤",
                  bool(g.get("allow_short")), g.get("sl"), g.get("tp"))


def _rows_of(res):
    """兼容两种结果对象：`.table`（采样式）与 `.best`（进化式）。"""
    for attr in ("table", "best"):
        t = getattr(res, attr, None)
        if t is not None and len(t):
            return t
    return None


def genomes_from_result(res) -> list[Genome]:
    """把任意挖掘结果（采样/枚举/进化/分层/微调）转成基因组列表。"""
    tbl = _rows_of(res)
    if tbl is None:
        return []
    out = []
    if "_genome" in getattr(tbl, "columns", []):
        for g in tbl["_genome"]:
            gn = _genome_from_dict(g)
            if gn is not None:
                out.append(gn)
        return out
    # 分层搜索的结果没有 _genome，只有 进场/过滤/参数 三列
    if "进场" in getattr(tbl, "columns", []):
        import ast as _ast

        for _, r in tbl.iterrows():
            params = r.get("参数")
            if isinstance(params, str):
                try:
                    params = _ast.literal_eval(params)
                except (ValueError, SyntaxError):
                    params = {}
            out.append(Genome(str(r.get("进场")),
                              dict(params or {}),
                              r.get("过滤") or "无过滤", False))
    return out


def run_verified(df: pd.DataFrame, genomes: list[Genome], *, tf: str = "H1",
                 symbol: str = "GOLD", spread_usd: float = 0.33,
                 n_perm: int = 200, n_windows: int = 3,
                 max_dd: float = 0.20, min_pf: float = 1.4,
                 min_dir: float = 0.55) -> VerifyResult:
    """**所有挖掘模式共用的唯一入口**：同一套三段切分、置换、十门槛、三档结论。"""
    v = Verifier(df, tf=tf, symbol=symbol, spread_usd=spread_usd,
                 n_windows=n_windows, max_dd=max_dd, min_pf=min_pf,
                 min_dir=min_dir)
    return v.run(list(genomes), n_perm=n_perm)


# ---------------------------------------------------------------- 严格验证（滚动 OOS）
def run_strict(df, genomes, *, tf: str = "H1", symbol: str = "GOLD",
               spread_usd: float | None = None, n_folds: int = 4,
               oos_frac: float = 0.2, n_perm: int = 50,
               run_id: str | None = None, alpha: float = 0.05,
               purpose: str = "selection") -> pd.DataFrame:
    """所有入口的严格验证：滚动 OOS + 账本 + 复用惩罚。

    纪律：基因组由调用方给定（生成阶段），**切分与评测只在这里发生**；
    每个 fold 在训练段上算 IS 指标、在样本外段算 OOS 指标；每折记一次账；
    最后按最后一折的复用次数 N 做多重检验惩罚。

    ``purpose`` 默认 "selection"：输出的表会被用来挑 EA，
    这本身就等于「用了这段 OOS」—— 按最保守的口径记账。
    """
    from gold import oos as _oos
    from gold.backtest import spread_fee_bps
    from gold.factors import build_features
    from gold.forge import Genome, evaluate_genome
    from gold.symbols import spread_for

    if spread_usd is None:
        spread_usd = spread_for(symbol)
    folds = _oos.rolling_folds(df, n_folds=n_folds, oos_frac=oos_frac)
    feat = build_features(df)
    fee = spread_fee_bps(float(df["close"].mean()), float(spread_usd))
    gs = [g if isinstance(g, Genome)
          else Genome(**(g or {})) if isinstance(g, dict) else g
          for g in genomes]

    rows = []
    for fold in folds:
        train, oos_df = _oos.slice_fold(df, fold)
        n_uses = 0
        if run_id:
            try:
                rec = _oos.record_use(run_id=run_id, symbol=symbol,
                                      timeframe=tf, scheme="rolling",
                                      fold=fold.fold, oos_start=fold.oos_start,
                                      oos_end=fold.oos_end, purpose=purpose,
                                      n_candidates=len(gs))
                n_uses = int(rec["n_uses_at_use"])
            except (OSError, ValueError):
                n_uses = 0
        for g in gs:
            ev = evaluate_genome(df, feat, train, oos_df, fee, g,
                                 min_trades=5, tf=tf, n_perm=n_perm)
            if ev is None:
                continue
            rows.append({"fold": fold.fold,
                         "样本外起": str(fold.oos_start.date()),
                         "样本外止": str(fold.oos_end.date()),
                         "EA": ev["label"],
                         "IS夏普": ev["IS夏普"], "OOS夏普": ev["OOS夏普"],
                         "OOS收益": ev["OOS收益"], "OOS回撤": ev["OOS回撤"],
                         "OOS方向": ev["OOS方向"], "回合": ev["回合"],
                         "盈利因子": ev["盈利因子"], "块置换p": ev["块置换p"],
                         "基准": ev["基准收益"], "OOS已用": n_uses})
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    agg = out.groupby("EA").agg(
        折数=("fold", "nunique"),
        正收益折=("OOS收益", lambda s: int((s > 0).sum())),
        OOS夏普均值=("OOS夏普", "mean"),
        OOS夏普最差=("OOS夏普", "min"),
        方向均值=("OOS方向", "mean"),
        块置换p均值=("块置换p", "mean"),
    ).reset_index()

    last = out[out["fold"] == out["fold"].max()]
    n_used = int(last["OOS已用"].max()) or 1
    k = max(1, len(gs))
    pen = _oos.penalty(k_tests=k, n_uses=n_used,
                       best_sharpe=float(last["OOS夏普"].max()), alpha=alpha)
    agg["最后一折夏普"] = agg["EA"].map(dict(zip(last["EA"], last["OOS夏普"])))
    agg["K_eff"] = pen["k_eff"]
    agg["deflated_p_adj"] = [
        _oos.penalty(k_tests=k, n_uses=n_used, best_sharpe=float(s),
                     alpha=alpha)["deflated_p_adj"] if s == s else float("nan")
        for s in agg["最后一折夏普"]]
    agg["复用判定"] = _oos.reuse_verdict(n_used)

    # 平台期 + 十门槛结论（对每个基因组只在最后一折上判定一次）
    same_sign, verdict, why = [], [], []
    for name in agg["EA"]:
        row = last[last["EA"] == name]
        ss = None
        if len(row):
            try:
                gg = [g for g in gs if g.label() == name]
                if gg:
                    ss = float(plateau_check(gg[0], df, tf=tf,
                                             spread_usd=spread_usd,
                                             fold=folds[-1])["同号率"])
            except Exception:  # noqa: BLE001
                ss = None
        r0 = row.iloc[0] if len(row) else {}
        rec = {
            "方向": float(r0["OOS方向"]) if len(row) else None,
            "超额": (float(r0["OOS收益"]) - float(r0["基准"])) if len(row) else None,
            "回撤": float(r0["OOS回撤"]) if len(row) else None,
            "盈利因子": float(r0["盈利因子"]) if len(row) else None,
            "每年回合": None,
            "块置换p": float(r0["块置换p"]) if len(row) else None,
            "q": None,
            "deflated_p": float(pen["deflated_p_adj"]),
        }
        gates = make_gates(rec, alpha=alpha, n_symbols_passed=1,
                           n_symbols_tested=1,
                           wf_positive=int(agg.loc[agg["EA"] == name,
                                                   "正收益折"].iloc[0]),
                           wf_total=int(agg.loc[agg["EA"] == name,
                                                "折数"].iloc[0]),
                           same_sign_rate=ss)
        vv, _why = verdict_of(gates)
        same_sign.append(ss)
        verdict.append(vv)
        why.append(gates_to_reasons(gates))
    agg["同号率"] = same_sign
    agg["结论"] = verdict
    agg["reject_reason"] = why
    agg.attrs["penalty"] = pen
    agg.attrs["note"] = pen["note"]
    agg.attrs["folds"] = [f.as_dict() for f in folds]
    return agg.sort_values(["正收益折", "OOS夏普均值"],
                           ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- 参数平台期
def plateau_check(genome, df, *, tf: str = "H1", spread_usd: float | None = None,
                  fold=None, steps: int = 1, max_variants: int = 9,
                  min_trades: int = 5) -> dict:
    """参数平台期检验：围绕该基因组取参数邻域，看表现是否"同号"。

    判定用**训练段**（不碰样本外）：参数稳健性属于"选参阶段"该看的东西，
    用样本外去判平台期等于又消费了一次 OOS。

    返回值：
    - ``同号率``：邻域里与中心**同为盈利符号**的比例（0~1）；
    - ``cv`` / ``spread``：借用 `validation.overfit.parameter_sensitivity`；
    - ``ok``：同号率 ≥ 0.6；
    - ``detail``：每个邻居的参数与夏普（便于人工复核）。
    """
    from gold import oos as _oos
    from gold import tune_ea
    from gold.backtest import spread_fee_bps
    from gold.factors import build_features
    from gold.forge import Genome, build_position
    from gold.symbols import spread_for
    from validation.overfit import parameter_sensitivity

    if spread_usd is None:
        spread_usd = spread_for("GOLD")
    if fold is None:
        fold = _oos.outward_fold(df)
    train, _oos_df = _oos.slice_fold(df, fold)
    feat = build_features(df)
    fee = spread_fee_bps(float(df["close"].mean()), float(spread_usd))
    g = genome if isinstance(genome, Genome) else tune_ea.coerce_genome(genome)
    if g is None:
        return {"同号率": float("nan"), "n": 0, "ok": False, "detail": [],
                "cv": float("nan"), "spread": float("nan"),
                "note": "基因组无法解析"}

    combos = tune_ea.variant_params(g.entry, g.params, steps=steps,
                                    max_combos=max_variants)
    sh: dict[str, float] = {}
    detail = []
    for p in combos:
        gg = Genome(g.entry, dict(p), g.filter, g.allow_short, g.sl, g.tp)
        # 只用训练段：平台期属于选参阶段的稳健性，不该消费样本外
        try:
            pos = build_position(df, feat, gg)
            st = segment_metrics(train, pos, tf, fee, gg.allow_short)
        except Exception:  # noqa: BLE001
            continue
        if not st or int(st.get("笔数", 0)) < int(min_trades):
            continue
        sh_s = float(st["夏普"])
        key = ",".join(f"{k}={v}" for k, v in sorted(p.items())) or "默认"
        sh[key] = sh_s
        detail.append({"参数": key, "IS夏普": sh_s})
    if not sh:
        return {"同号率": float("nan"), "n": 0, "ok": False, "detail": [],
                "cv": float("nan"), "spread": float("nan"),
                "note": "邻域里没有可评测的组合"}

    base_key = ",".join(f"{k}={v}" for k, v in sorted(g.params.items())) or "默认"
    center = sh.get(base_key)
    if center is None:
        center = list(sh.values())[0]
    same = sum(1 for v in sh.values() if (v > 0) == (center > 0))
    rate = same / len(sh)
    sens = parameter_sensitivity(sh)
    return {"同号率": float(rate), "n": int(len(sh)), "ok": bool(rate >= 0.6),
            "detail": detail, "center_sharpe": float(center),
            "cv": float(sens.get("cv", float("nan"))),
            "spread": float(sens.get("spread", float("nan"))),
            "note": f"邻域 {len(sh)} 组，同号 {same} 组（{rate:.0%}）"}


def gates_to_reasons(gates) -> list[dict]:
    """把未通过/未测的门槛整理成结构化 reject_reason（哪条、实际值、阈值）。"""
    out = []
    for g in gates:
        if g.status == "pass":
            continue
        out.append({"门槛": g.name, "要求": g.need, "实际": g.value,
                    "状态": g.status})
    return out


def reasons_text(reasons: list[dict]) -> str:
    """一句话版本，供界面/日志直接打印。

    未测的门槛不逐条罗列（旧记录常缺 6~7 列，全是"未测"噪声），
    只把**未通过**的写清楚，未测的归并成"缺 N 项数据"。
    """
    if not reasons:
        return "无（全部门槛通过）"
    fails = [r for r in reasons if r.get("状态") == "fail"]
    unknown = [r for r in reasons if r.get("状态") != "fail"]
    parts = [f"{r['门槛']}[未通过 实际 {r['实际']} / 阈值 {r['要求']}]"
             for r in fails]
    if unknown:
        names = "、".join(str(r["门槛"]) for r in unknown[:3])
        more = "" if len(unknown) <= 3 else f" 等 {len(unknown)} 项"
        parts.append(f"缺数据：{names}{more}")
    return "；".join(parts)


# ---------------------------------------------------------------- 池子排名（唯一实现）
def rank_pool(board: pd.DataFrame, *, min_sharpe: float = 0.5,
              min_rounds: float | None = 30.0,
              max_rounds: float | None = None,
              min_pf: float | None = 1.4, max_dd: float = 0.10,
              min_dir: float = 0.55, alpha: float = 0.05,
              min_trades: int = 30,
              same_sign_rate: float | None = None,
              n_symbols_tested: int = 1,
              wf_by_ea: dict | None = None) -> pd.DataFrame:
    """给 EA 池排名 —— 全系统**唯一**的入池实现（app 不再自己写一套）。

    输入是 `leaderboard.cumulative()` 的表（含 `最新*` 列）；输出在原表上加：
    - `结论`：通过 / 继续观察 / 未通过（`verify.verdict_of` 的三档）；
    - `reject_reason`：结构化拒绝原因（哪条门槛 / 实际值 / 阈值）；
    - `入池`：布尔（只有"通过"才为 True）。

    排序键：先按 `正收益折数`（若有），再按 `最新夏普` —— 不再用单一最值。
    """
    if board is None or len(board) == 0:
        return pd.DataFrame()
    p = board.copy()
    num = ("最新夏普", "最新收益", "最新回撤", "最新笔数", "最新回合",
           "最新每年回合", "最新方向", "最新基准", "最新盈利因子",
           "最新每笔期望", "最新块置换p", "最新q")
    for c in num:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")

    def _v(row, col, default=None):
        if col not in p.columns:
            return default
        v = row.get(col)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return None if f != f else f

    verdicts, reasons, reasons_txt, inpool = [], [], [], []
    for _, row in p.iterrows():
        sharpe = _v(row, "最新夏普")
        dd = _v(row, "最新回撤")
        pf = _v(row, "最新盈利因子")
        rpy = _v(row, "最新每年回合")
        trades = _v(row, "最新笔数")
        hit = _v(row, "最新方向")
        bench = _v(row, "最新基准")
        ret = _v(row, "最新收益")
        pval = _v(row, "最新块置换p")
        # 历史记录的 q 是按 tanh 伪 p 算的；没有真 p 就不认这个 q（不继承伪造值）
        q = _v(row, "最新q") if pval is not None else None
        excess = (ret - bench) if (ret is not None and bench is not None) else None

        wf_pos, wf_tot = None, None
        if wf_by_ea and row.get("EA") in wf_by_ea:
            wf_pos, wf_tot = wf_by_ea[row.get("EA")]

        rec = {
            "方向": hit, "超额": excess, "回撤": dd, "盈利因子": pf,
            "每年回合": rpy, "块置换p": pval, "q": q,
            "deflated_p": None,
        }
        gates = make_gates(rec, min_dir=min_dir, max_dd=max_dd,
                           min_pf=(min_pf if min_pf is not None else 1.4),
                           min_rounds_year=(min_rounds or 0.0),
                           max_rounds_year=(max_rounds if max_rounds
                                            and max_rounds > 0 else 300.0),
                           alpha=alpha, n_symbols_passed=1,
                           n_symbols_tested=n_symbols_tested,
                           wf_positive=wf_pos, wf_total=wf_tot,
                           same_sign_rate=same_sign_rate)
        # 入池的"初筛"条件：夏普/回撤/笔数（其余交给 11 条门槛）
        if sharpe is None or sharpe <= min_sharpe:
            gates.append(Gate("夏普", f">{min_sharpe}", str(sharpe), "fail"))
        if dd is not None and dd < -max_dd:
            gates.append(Gate("回撤上限", f"≤{max_dd:.0%}", str(dd), "fail"))
        if trades is not None and trades < min_trades:
            gates.append(Gate("最少笔数", f"≥{min_trades}", str(trades), "fail"))
        v, _why = verdict_of(gates)
        rs = gates_to_reasons(gates)
        verdicts.append(v)
        reasons.append(rs)
        reasons_txt.append(reasons_text(rs))
        inpool.append(v == "通过")

    p["结论"] = verdicts
    p["reject_reason"] = reasons
    p["拒绝原因"] = reasons_txt
    p["入池"] = inpool

    sort_cols = [c for c in ("正收益折", "最新夏普", "夏普中位")
                 if c in p.columns]
    if sort_cols:
        p = p.sort_values(sort_cols, ascending=False, na_position="last")
    return p.reset_index(drop=True)
