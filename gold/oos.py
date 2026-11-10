"""样本外（OOS）使用账本 + 滚动切分 + 复用惩罚。

为什么需要账本
--------------
"最终样本外只能跑一次"这条纪律，靠人自觉是守不住的 —— 必须记账。
只要某一段 OOS 被用来选过一次参数，它就**不再是样本外**了。

本模块做三件事：
1. `rolling_folds` / `fixed_fold`：按时间切出训练段与 OOS 段（OOS 永远在训练段之后）；
2. `record_use` / `count_uses`：把每次"用 OOS 选参"永久记账（只增，没有删除 API）；
3. `penalty`：按该区间已被使用次数 N 做多重检验惩罚（K_eff = K × N）。

滚动 OOS 天然占优：每个 fold 的 OOS 区间不同 → 各桶 N 都是 1；
固定 OOS 反复用同一区间 → N 快速累加。这是设计意图，不是 bug。
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib

import pandas as pd

LEDGER_PATH = pathlib.Path("data/oos_ledger.json")
SCHEMA_VERSION = 1

# 系统里最长的指标回看窗口：RSRS 的 m=250、强趋势过滤 250、时变夏普 120、
# 双均线慢线最高 300。取 250 作为默认 purge，保证训练段边界那几根不会
# 把指标"看得见的未来"带进样本外。
PURGE_LOOKBACK = 250


def auto_purge_bars(n_bars: int, lookback: int = PURGE_LOOKBACK) -> int:
    """按数据长度自动定 purge 根数：默认 250，短数据按 10% 收缩（至少 10 根）。"""
    n = max(1, int(n_bars))
    return int(min(int(lookback), max(10, int(n * 0.1))))


# ---------------------------------------------------------------- 切分
def _fmt(ts) -> str:
    return str(pd.Timestamp(ts))[:19]


class OosFold:
    """一段训练/样本外切分（时间索引，含 purge 预留字段）。"""

    __slots__ = ("fold", "train_start", "train_end", "oos_start", "oos_end",
                 "purge_bars")

    def __init__(self, fold: int, train_start, train_end, oos_start, oos_end,
                 purge_bars: int = 0):
        self.fold = int(fold)
        self.train_start = pd.Timestamp(train_start)
        self.train_end = pd.Timestamp(train_end)
        self.oos_start = pd.Timestamp(oos_start)
        self.oos_end = pd.Timestamp(oos_end)
        self.purge_bars = int(purge_bars)

    @property
    def gap_bars(self) -> int:
        """训练段末尾到样本外开头之间被切掉的根数（purge gap）。"""
        return int(self.purge_bars)

    def as_dict(self) -> dict:
        return {"fold": self.fold,
                "train": [_fmt(self.train_start), _fmt(self.train_end)],
                "oos": [_fmt(self.oos_start), _fmt(self.oos_end)],
                "purge_bars": self.purge_bars}

    def __repr__(self) -> str:
        return (f"OosFold(fold={self.fold}, train={_fmt(self.train_start)}.."
                f"{_fmt(self.train_end)}, oos={_fmt(self.oos_start)}.."
                f"{_fmt(self.oos_end)}, purge={self.purge_bars})")


def rolling_folds(df: pd.DataFrame, n_folds: int = 4, oos_frac: float = 0.2,
                  min_train_frac: float = 0.4, scheme: str = "rolling",
                  purge_bars: int | None = None) -> list[OosFold]:
    """按时间顺序切出若干 (train, oos) 段，OOS 永远在训练段之后。

    - ``scheme="rolling"``：训练窗口长度固定，整体前移；
    - ``scheme="anchored"``：训练起点固定，终点扩张；
    - ``purge_bars``：训练段末尾切掉的根数（第一步只透传，第四步实现真实 gap）。
    """
    n = len(df)
    if n < 50:
        raise ValueError(f"数据太短（{n} 根），无法切分")
    if purge_bars is None:
        purge_bars = auto_purge_bars(n)
    purge_bars = int(purge_bars)
    n_folds = max(1, int(n_folds))
    n_oos = max(10, int(n * float(oos_frac)))
    oos_len = max(5, n_oos // n_folds)
    # rolling：训练窗口长度**恒定**（= 总长 − 样本外总长）；anchored：从第 0 根起扩张
    train_len = max(int(n * float(min_train_frac)), n - n_oos)

    folds: list[OosFold] = []
    for i in range(n_folds):
        oos_end_idx = n - (n_folds - 1 - i) * oos_len
        oos_start_idx = max(0, oos_end_idx - oos_len)
        train_end_idx = oos_start_idx - int(purge_bars)
        train_start_idx = (0 if scheme == "anchored"
                           else max(0, train_end_idx - train_len))
        if train_end_idx - train_start_idx < 20 or oos_end_idx - oos_start_idx < 5:
            continue
        folds.append(OosFold(
            fold=len(folds) + 1,
            train_start=df.index[train_start_idx],
            train_end=df.index[train_end_idx - 1],
            oos_start=df.index[oos_start_idx],
            oos_end=df.index[oos_end_idx - 1],
            purge_bars=int(purge_bars),
        ))
    if not folds:
        raise ValueError("切分后没有可用 fold（数据太短或参数过严）")
    return folds


def fixed_fold(df: pd.DataFrame, is_ratio: float = 0.7,
               purge_bars: int | None = None) -> OosFold:
    """兼容现状的 70/30 一刀切（固定 OOS）。"""
    n = len(df)
    if purge_bars is None:
        purge_bars = auto_purge_bars(n)
    purge_bars = int(purge_bars)
    cut = int(n * float(is_ratio))
    if cut < 20 or n - cut < 5:
        raise ValueError(f"数据太短（{n} 根），无法按 {is_ratio} 切分")
    train_end = cut - 1 - int(purge_bars)
    if train_end < 19:
        raise ValueError("purge_bars 太大，训练段不够")
    return OosFold(fold=1, train_start=df.index[0],
                   train_end=df.index[train_end],
                   oos_start=df.index[cut], oos_end=df.index[n - 1],
                   purge_bars=int(purge_bars))


def outward_fold(df: pd.DataFrame, n_folds: int = 4, oos_frac: float = 0.2,
                 purge_bars: int | None = None) -> OosFold:
    """取"对外那一折"：滚动切分里**最后一个** fold。

    所有挖掘入口（forge/evolve/strata/mine/tune）都用它，禁止自己再切 70/30。
    """
    folds = rolling_folds(df, n_folds=n_folds, oos_frac=oos_frac,
                          purge_bars=purge_bars)
    return folds[-1]


def slice_fold(df: pd.DataFrame, fold: OosFold
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 fold 切出 (训练段, 样本外段)。两端都是闭区间。"""
    train = df.loc[(df.index >= fold.train_start) & (df.index <= fold.train_end)]
    oos = df.loc[(df.index >= fold.oos_start) & (df.index <= fold.oos_end)]
    if len(train) and len(oos):
        if train.index[-1] >= oos.index[0]:
            raise ValueError(
                f"训练段与样本外段重叠：train_end={train.index[-1]} "
                f">= oos_start={oos.index[0]}")
        # purge gap 必须真的存在：两段之间的 bar 数 ≥ 声明的 purge_bars
        gap = int(((df.index > train.index[-1]) & (df.index < oos.index[0])).sum())
        if gap < int(fold.purge_bars):
            raise ValueError(
                f"purge gap 不足：声明 {fold.purge_bars} 根，实际 {gap} 根")
    return train, oos


# ---------------------------------------------------------------- 账本
def _key(symbol: str, timeframe: str, oos_start, oos_end) -> str:
    return f"{symbol}|{timeframe}|{_fmt(oos_start)}|{_fmt(oos_end)}"


def _empty() -> dict:
    return {"version": SCHEMA_VERSION, "entries": [], "counts": {}}


def load_ledger(path: str | pathlib.Path | None = None) -> dict:
    p = pathlib.Path(LEDGER_PATH if path is None else path)
    if not p.exists():
        return _empty()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"账本文件损坏，拒绝继续：{p}（{exc}）") from exc
    data.setdefault("version", SCHEMA_VERSION)
    data.setdefault("entries", [])
    data.setdefault("counts", {})
    return data


def _save(ledger: dict, path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, p)
    return p


def count_uses(symbol: str, timeframe: str, oos_start, oos_end,
               purpose: str = "selection",
               path: str | pathlib.Path | None = None) -> int:
    """这段 OOS 已被"用于选参"的次数（report_only 不计）。"""
    path = LEDGER_PATH if path is None else path
    ledger = load_ledger(path)
    k = _key(symbol, timeframe, oos_start, oos_end)
    if purpose == "selection":
        return int((ledger.get("counts") or {}).get(k, 0))
    return sum(1 for e in ledger["entries"]
               if e.get("bucket") == k and e.get("purpose") == purpose)


def record_use(*, run_id: str, symbol: str, timeframe: str, scheme: str,
               fold: int, oos_start, oos_end, purpose: str = "selection",
               n_candidates: int = 0, seed: int = 0, data_sha256: str = "",
               selected: dict | None = None,
               path: str | pathlib.Path | None = None) -> dict:
    """把一次 OOS 使用写入账本，返回该条记录（含 n_uses_at_use）。

    - 只增：没有删除/清空接口；
    - `use_id` 重复直接报错（防止同一件事被记两次或记账代码重入）；
    - 写失败会抛异常，调用方必须如实标注，不能假装成功。
    """
    path = LEDGER_PATH if path is None else path
    ledger = load_ledger(path)
    bucket = _key(symbol, timeframe, oos_start, oos_end)
    use_id = f"{run_id}_f{int(fold)}"
    for e in ledger["entries"]:
        if e.get("use_id") == use_id:
            raise ValueError(f"账本里已有 use_id={use_id}，拒绝重复记账")

    counts = ledger.setdefault("counts", {})
    n_before = int(counts.get(bucket, 0))
    if purpose == "selection":
        counts[bucket] = n_before + 1

    entry = {
        "use_id": use_id,
        "run_id": str(run_id),
        "symbol": str(symbol),
        "timeframe": str(timeframe),
        "scheme": str(scheme),
        "fold": int(fold),
        "oos_start": _fmt(oos_start),
        "oos_end": _fmt(oos_end),
        "purpose": str(purpose),
        "n_candidates": int(n_candidates),
        "seed": int(seed),
        "data_sha256": str(data_sha256),
        "selected": selected or {},
        "n_uses_at_use": (n_before + 1) if purpose == "selection" else n_before,
        "bucket": bucket,
        "created_at": _dt.datetime.now().astimezone().isoformat(
            timespec="seconds"),
    }
    ledger["entries"].append(entry)
    _save(ledger, path)
    return entry


def verify_ledger(path: str | pathlib.Path | None = None) -> dict:
    """完整性自检：use_id 唯一、创建时间单调不减、counts 与条目一致。"""
    path = LEDGER_PATH if path is None else path
    ledger = load_ledger(path)
    problems: list[str] = []
    seen: set[str] = set()
    last_ts = ""
    recount: dict[str, int] = {}
    for e in ledger["entries"]:
        uid = e.get("use_id")
        if not uid:
            problems.append("存在没有 use_id 的条目")
        elif uid in seen:
            problems.append(f"use_id 重复: {uid}")
        else:
            seen.add(uid)
        ts = str(e.get("created_at") or "")
        if ts and ts < last_ts:
            problems.append(f"created_at 倒退: {uid} ({ts} < {last_ts})")
        last_ts = max(last_ts, ts)
        if e.get("purpose") == "selection":
            b = e.get("bucket")
            recount[b] = recount.get(b, 0) + 1
    for k, v in recount.items():
        if int((ledger.get("counts") or {}).get(k, 0)) != v:
            problems.append(f"计数不一致: {k} 记 {ledger['counts'].get(k)} 实际 {v}")
    return {"ok": not problems, "problems": problems,
            "n_entries": len(ledger["entries"]),
            "n_buckets": len(ledger.get("counts") or {})}


# ---------------------------------------------------------------- 复用惩罚
def penalty(*, k_tests: int, n_uses: int, best_sharpe: float,
            alpha: float = 0.05) -> dict:
    """按"这段 OOS 已被用 N 次"做多重检验惩罚。

    K_eff = K × N：每多用一次 OOS ≈ 多一轮选择，直接进多重检验。
    同时返回 Bonferroni 上界 p_bonf = min(1, p×N) 作为保守参照。
    """
    from gold.mining import deflated_p, emax_sharpe

    k = max(1, int(k_tests))
    n = max(1, int(n_uses))
    k_eff = k * n
    raw = float(deflated_p(float(best_sharpe), k))
    adj = float(deflated_p(float(best_sharpe), k_eff))
    return {
        "k_tests": k,
        "n_uses": n,
        "k_eff": k_eff,
        "emax_sharpe": float(emax_sharpe(k_eff)),
        "deflated_p_raw": raw,
        "deflated_p_adj": adj,
        "alpha": float(alpha),
        "note": (f"该 OOS 已被使用 {n} 次，置信度已下调："
                 f"deflated p {raw:.3f} → {adj:.3f}"
                 f"（K_eff = {k}×{n} = {k_eff}）"),
    }


def bonferroni(p_value: float, n_uses: int) -> float:
    """保守上界：p × N（截断到 1）。"""
    return min(1.0, max(0.0, float(p_value)) * max(1, int(n_uses)))


def reuse_verdict(n_uses: int, pass_max: int = 3,
                  fail_min: int = 7) -> str:
    """复用次数的三档判定（**只会更严**，不会为了让结果好看而放宽）。"""
    n = int(n_uses)
    if n <= int(pass_max):
        return "pass"
    if n < int(fail_min):
        return "degrade"
    return "fail"
