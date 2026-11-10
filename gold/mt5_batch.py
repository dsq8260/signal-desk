"""MT5 测试器批量验证：把候选 EA 逐个丢进策略测试器，结果落盘。

为什么要这一层
--------------
`mt5_mcp.run_tester` 一次只能跑一个 EA，而且失败原因有五种
（`not_indexed` / `prepare_failed` / `run_failed` / `no_run_id` / `no_mcp`）。
批量验证最怕两件事：

1. **缺索引的 EA 被静默跳过** —— 看起来"跑完了"，其实一个都没跑；
2. **单个 EA 抛异常把整批带崩** —— 后面的候选再没机会验。

所以这里的规定是：
- 缺索引 → 进 `skipped` 并带"刷新导航器"的指引，绝不当作成功；
- 单项异常 → 进 `failed`，继续跑下一项；
- 只有真正拿到报告的项才落盘 `out_dir/<EA名>.json`（含 report 原文），
  失败项不写任何"看起来像结果"的文件，避免伪造数据。

安全边界：本模块只调用 `mt5_mcp` 的只读/测试器工具，
**不调用任何 `trade_*`**，不会下单。
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re

from gold import mt5_mcp, runlog

log = logging.getLogger(__name__)

#: 上一次 `candidates_from_runs()` 没取到候选时的原因（CLI 用它打印中文提示）。
LAST_REASON = ""

#: 清单里可能承载"胜者"的字段名（runlog 的 outcome 与顶层都试一遍）。
_WINNER_KEYS = ("winners", "top", "top_candidates", "candidates",
                "survivors", "rows", "items", "results", "best")
#: 候选对象里可能直接写着 EA 名的字段。
_NAME_KEYS = ("name", "ea_name", "mql5_name", "ex5_name", "ex5",
              "program", "expert", "file")
#: 候选对象里可能藏着 genome 的字段（能用 workbench.mql5_name 换算）。
_GENOME_KEYS = ("genome", "_genome")

_REFRESH_HINT = ("在 MT5 导航器里右键 → 刷新（或按 F5），确认该 EA 已编译并出现在"
                 "「智能交易系统 / Experts」下，然后重跑本命令。")
_BUSY_HINT = ("MT5 测试器可能被上一次运行占住（界面显示「停止 / 正在运行」）："
              "先在 MT5 里点停止，或等它结束后重试。")


# ---------------------------------------------------------------- 小工具
def _safe_name(name: str) -> str:
    """EA 名 → 安全的文件名（保留 ASCII 字母数字与 _-.，其余换成 _）。"""
    s = re.sub(r"[^0-9A-Za-z_.\-]+", "_", str(name)).strip("._")
    return s or "unnamed"


def _name_from_genome(g) -> str:
    """用 workbench.mql5_name 把 genome 字典换成 EA 文件名；不合法就返回空串。"""
    if not isinstance(g, dict) or not str(g.get("entry") or "").strip():
        return ""
    try:
        from gold.forge import Genome
        from gold.workbench import mql5_name

        fields = ("entry", "params", "filter", "allow_short", "sl", "tp")
        return str(mql5_name(Genome(**{k: g[k] for k in fields if k in g}))).strip()
    except Exception as exc:  # genome 残缺/接口变动都不该拖垮候选抽取
        log.warning("genome 换算 EA 名失败：%s", exc)
        return ""


def _names_from(obj, depth: int = 0) -> list[str]:
    """从任意嵌套结构里抽出 EA 名（递归深度有限，避免坏数据套娃）。"""
    if depth > 4:
        return []
    if isinstance(obj, str):
        s = obj.strip()
        return [s] if s else []
    if isinstance(obj, (list, tuple)):
        out: list[str] = []
        for item in obj:
            out.extend(_names_from(item, depth + 1))
        return out
    if not isinstance(obj, dict):
        return []

    out = []
    for key in _NAME_KEYS:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            out.append(val.strip())
    for key in _GENOME_KEYS:
        name = _name_from_genome(obj.get(key))
        if name:
            out.append(name)
    for key in _WINNER_KEYS:
        if key in obj:
            out.extend(_names_from(obj[key], depth + 1))
    return out


def _load_manifests(run_dir=None) -> list[dict]:
    """按时间倒序读运行清单；run_dir 可指向别的目录（测试用）。"""
    if run_dir is None:
        return runlog.list_runs()
    base = pathlib.Path(run_dir)
    out: list[dict] = []
    if not base.exists():
        return out
    for p in sorted(base.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            out.append({"run_id": p.stem, "status": "unreadable"})
    return out


# ---------------------------------------------------------------- 候选抽取
def candidates_from_runs(limit: int = 5, run_dir=None) -> list[dict]:
    """从 runs/*.json 的最近运行里抽候选 EA 名。

    认这些来源（按清单内出现顺序）：
    - `outcome.winners` / `outcome.top` / 顶层的 `winners` / `top` 等列表；
    - 列表项里的 `name` / `ea_name` / `mql5_name` / `ex5_name` 等字段；
    - 列表项里的 `genome` / `_genome` → `workbench.mql5_name(genome)`。

    返回 `[{"name": str, "run_id": str, "source": str}, ...]`（去重，最多 limit 条）。
    抽不到时返回 `[]`，原因写进模块级 `LAST_REASON`（也打 warning）。
    """
    global LAST_REASON

    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = 5

    try:
        manifests = _load_manifests(run_dir)
    except Exception as exc:
        LAST_REASON = f"读取运行清单失败：{exc}"
        log.warning(LAST_REASON)
        return []

    if not manifests:
        where = run_dir if run_dir is not None else runlog.RUNS_DIR
        LAST_REASON = (f"{where} 下没有任何运行清单（*.json）；"
                       "先跑一次锻造/进化，或把已有的胜者清单放进去。")
        log.warning(LAST_REASON)
        return []

    found: list[dict] = []
    seen: set[str] = set()
    for manifest in manifests:
        if not isinstance(manifest, dict):
            continue
        run_id = str(manifest.get("run_id") or "")
        names: list[str] = []
        for key in _WINNER_KEYS:
            if key in manifest and key != "rows":
                names.extend(_names_from(manifest[key]))
        outcome = manifest.get("outcome")
        if isinstance(outcome, dict):
            for key in _WINNER_KEYS:
                if key in outcome:
                    names.extend(_names_from(outcome[key]))
        for raw in names:
            name = str(raw).strip()
            if name.lower().endswith(".ex5"):
                name = name[:-4]
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            found.append({"name": name, "run_id": run_id, "source": "manifest"})
            if len(found) >= limit:
                LAST_REASON = ""
                return found

    if not found:
        LAST_REASON = (f"读了最近 {len(manifests)} 份运行清单，但没有 winners/top 字段，"
                       "也找不到可换算的 genome；无法自动确定候选 EA。"
                       "请先确认锻造结果里保存了胜者（outcome.winners / outcome.rows），"
                       "或手工把 EA 名写进清单后重试。")
        log.warning(LAST_REASON)
        return []
    LAST_REASON = ""
    return found


# ---------------------------------------------------------------- 索引状态
def index_status(cli, names: list[str]) -> dict:
    """把名字分成「终端索引里已有」和「缺索引」两组。

    单个名字查询抛异常时按缺失处理（宁可提醒刷新，也不要假装能跑）。
    """
    indexed: list[str] = []
    missing: list[str] = []
    for raw in names or []:
        name = str(raw or "").strip()
        if not name:
            continue
        try:
            hit = cli.expert_indexed(name)
        except Exception as exc:
            log.warning("查索引失败 %s：%s", name, exc)
            hit = None
        (indexed if hit else missing).append(name)
    return {"indexed": indexed, "missing": missing}


# ---------------------------------------------------------------- 落盘
def _write_json(out_dir: pathlib.Path, name: str, payload: dict) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_safe_name(name)}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- 批量执行
def _normalize_items(items) -> list[str]:
    out: list[str] = []
    for raw in items or []:
        if isinstance(raw, dict):
            val = (raw.get("name") or raw.get("ea_name")
                   or raw.get("mql5_name") or "")
        else:
            val = raw
        name = str(val or "").strip()
        if name.lower().endswith(".ex5"):
            name = name[:-4]
        if name:
            out.append(name)
    return out


def run_batch(items, *, symbol="GOLD#", timeframe="H1",
              from_date, to_date, model="Every tick", deposit=1000,
              leverage=100, timeout_sec=600, client=None,
              out_dir="runs/tester") -> dict:
    """逐个跑 MT5 测试器；单点失败不影响整批。

    返回 `{"ok", "ran", "skipped", "failed", "results", "connected", ...}`：

    - `ran`：**真正拿到报告并成功落盘**的个数；
    - `skipped`：缺索引等"压根没跑"的项，每项带 `hint` 指引；
    - `failed`：跑了但失败/异常/落盘失败的项，带 `reason` + `error` 原文；
    - `results`：成功项，每项含 `report` 原文与落盘 `path`。

    `ok=True` 仅当 `ran > 0` 且 `skipped`、`failed` 都为空。
    """
    names = _normalize_items(items)
    out_path = pathlib.Path(out_dir)
    batch: dict = {
        "ok": False,
        "ran": 0,
        "skipped": [],
        "failed": [],
        "results": [],
        "connected": False,
        "out_dir": str(out_path),
        "symbol": symbol,
        "timeframe": timeframe,
        "model": model,
        "from_date": from_date,
        "to_date": to_date,
    }
    if not names:
        batch["error"] = "没有可跑的 EA（候选列表为空）"
        return batch

    cli = client if client is not None else mt5_mcp.Mt5Mcp(timeout=120)
    try:
        connected = bool(cli.handshake())
    except Exception as exc:
        connected = False
        batch["error"] = f"连不上 MT5 MCP：{exc}"
    if not connected:
        batch["reason"] = "no_mcp"
        batch.setdefault("error", str(getattr(cli, "last_error", "")
                                      or "连不上 MT5 MCP（握手失败）"))
        log.warning("run_batch 未连上 MCP：%s", batch["error"])
        return batch
    batch["connected"] = True

    for name in names:
        try:
            res = cli.run_tester(
                name, symbol, timeframe, from_date, to_date,
                model=model, deposit=deposit, leverage=leverage,
                timeout_sec=timeout_sec)
        except Exception as exc:  # 单项异常不能带崩整批
            batch["failed"].append({
                "name": name, "reason": "exception",
                "error": f"{type(exc).__name__}: {exc}",
                "hint": _BUSY_HINT,
            })
            log.warning("tester 异常 %s：%s", name, exc)
            continue

        if not isinstance(res, dict):
            batch["failed"].append({
                "name": name, "reason": "bad_response",
                "error": f"run_tester 返回了 {type(res).__name__}，不是 dict",
            })
            continue

        if not res.get("ok"):
            reason = str(res.get("reason") or "unknown")
            err = str(res.get("error") or "")
            if reason == "not_indexed":
                batch["skipped"].append({
                    "name": name, "reason": "not_indexed", "error": err,
                    "hint": _REFRESH_HINT,
                })
            else:
                hint = _BUSY_HINT if "still started" in err.lower() else ""
                batch["failed"].append({
                    "name": name, "reason": reason, "error": err,
                    **({"hint": hint} if hint else {}),
                })
            continue

        batch["ran"] += 1
        payload = {
            "name": name,
            "ok": True,
            "run_id": res.get("run_id"),
            "symbol": symbol,
            "timeframe": timeframe,
            "model": model,
            "from_date": from_date,
            "to_date": to_date,
            "deposit": deposit,
            "leverage": leverage,
            "ini": res.get("ini"),
            "error": res.get("error") or "",
            "report": res.get("report"),
        }
        entry = {"name": name, "ok": True, "run_id": res.get("run_id"),
                 "report": res.get("report")}
        try:
            path = _write_json(out_path, name, payload)
            entry["path"] = str(path)
        except Exception as exc:  # 落盘失败必须如实计入 failed
            batch["failed"].append({
                "name": name, "reason": "write_failed",
                "error": f"测试器跑完了但结果落盘失败：{exc}",
            })
            log.warning("落盘失败 %s：%s", name, exc)
            continue
        batch["results"].append(entry)

    batch["ok"] = bool(batch["ran"] and not batch["skipped"]
                       and not batch["failed"])
    return batch


# ---------------------------------------------------------------- 汇总
#: 指标名 → report 里可能出现的字段（大小写不敏感，多来源兜底）。
_METRICS = (
    ("净利润", ("net_profit", "netprofit", "net profit", "profit",
                "total_net_profit", "净利润")),
    ("盈利因子", ("profit_factor", "profitfactor", "profit factor", "pf",
                  "盈利因子")),
    ("成交数", ("trades", "total_trades", "trade_count", "deals", "orders",
                "成交数", "交易数")),
    ("最大回撤", ("max_drawdown_equity", "max_drawdown_balance",
                  "maxdrawdown", "max_drawdown", "max drawdown", "equity_dd",
                  "drawdown", "dd", "最大回撤")),
    ("夏普", ("sharpe", "sharpe_ratio", "sharperatio", "sharpe ratio",
              "夏普")),
)
_NESTED = ("metrics", "report", "summary", "stats", "result", "results", "data")


def _pick(report, keys) -> object | None:
    """在 report（含有限层嵌套字典）里找第一个存在的字段；找不到返回 None。

    只做查找，不做换算 —— 取不到就如实是 None，绝不编数字。
    """
    if not isinstance(report, dict):
        return None
    want = [str(k).lower() for k in keys]
    queue = [(report, 0)]
    seen = 0
    while queue and seen < 50:
        node, depth = queue.pop(0)
        seen += 1
        lowered = {}
        for k, v in node.items():
            lowered[str(k).lower()] = v
        for key in want:
            if key in lowered and lowered[key] is not None:
                return lowered[key]
        if depth >= 3:
            continue
        for k, v in node.items():
            if isinstance(v, dict) and str(k).lower() in _NESTED:
                queue.append((v, depth + 1))
    return None


def summary_table(batch: dict) -> list[dict]:
    """把 batch 里每个成功项的 report 压成一行关键指标。

    字段名不确定时逐级兜底；report 缺字段就写 None（不编数字）。
    """
    rows: list[dict] = []
    results = (batch or {}).get("results") or []
    for item in results:
        if not isinstance(item, dict):
            continue
        report = item.get("report")
        row = {"EA": str(item.get("name") or ""),
               "run_id": item.get("run_id")}
        for label, keys in _METRICS:
            row[label] = _pick(report, keys)
        rows.append(row)
    return rows
