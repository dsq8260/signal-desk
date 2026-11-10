"""挖矿台账：把所有参数挖掘记录存档，用于跨策略的全局多重检验。

为什么需要：
单个策略内校正（挖 16 组 → 按 16 次判）还不够狠——你一共试过多少个策略、
多少组参数，才是真实的试验次数。挖了 500 组之后挑出一个 p=0.03 的策略，
按 500 次校正几乎必然不显著。
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import asdict, dataclass, field

DEFAULT_PATH = pathlib.Path("data/mining_ledger.json")


def emax_sharpe(n_trials: int) -> float:
    k = max(1, int(n_trials))
    return 0.0 if k == 1 else math.sqrt(2.0 * math.log(k))


def global_deflated_p(best_sharpe: float, total_trials: int,
                      sr_std: float = 1.0) -> float:
    """跨全部试验次数的校正 p（挖了多少组，就按多少组算）。"""
    if best_sharpe != best_sharpe:
        return 1.0
    z = (float(best_sharpe) - emax_sharpe(total_trials)) / max(1e-9, float(sr_std))
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


@dataclass
class Record:
    ts: str = ""
    strategy: str = ""
    params: dict = field(default_factory=dict)
    n_trials: int = 0
    oos_return: float = float("nan")
    oos_sharpe: float = float("nan")
    raw_p: float = float("nan")
    q: float = float("nan")
    verdict: str = ""
    data_note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def load(path: pathlib.Path | str = DEFAULT_PATH) -> list[dict]:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def append(rec: Record, path: pathlib.Path | str = DEFAULT_PATH) -> list[dict]:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = load(p)
    rows.append(rec.as_dict())
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def summary(path: pathlib.Path | str = DEFAULT_PATH) -> dict:
    """累计试验次数 = 所有记录的 n_trials 之和（真实"我试过多少次"）。"""
    rows = load(path)
    total = int(sum(int(r.get("n_trials", 0) or 0) for r in rows))
    sharpes = [r.get("oos_sharpe") for r in rows
               if isinstance(r.get("oos_sharpe"), (int, float))
               and r["oos_sharpe"] == r["oos_sharpe"]]
    best = max(sharpes) if sharpes else float("nan")
    return {
        "runs": len(rows),
        "total_trials": total,
        "best_oos_sharpe": best,
        "emax": emax_sharpe(total) if total else 0.0,
        "global_p": global_deflated_p(best, total) if total and best == best else 1.0,
        "strategies": sorted({str(r.get("strategy", "")) for r in rows}),
    }


def clear(path: pathlib.Path | str = DEFAULT_PATH) -> None:
    p = pathlib.Path(path)
    if p.exists():
        p.unlink()
