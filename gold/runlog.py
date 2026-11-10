"""运行清单（run manifest）：让每一次运行都能被追溯与复现。

为什么需要这个
--------------
过去 51 次运行只留下 `data/forge_runs/run_<时间戳>.json`，里面没有
seed、没有数据指纹、没有点差、没有门槛参数 —— 于是无法回答
"这个 1.98 的夏普是用哪份数据、什么参数跑出来的"。

本模块只做一件事：把一次运行的**输入**完整落盘（`runs/<run_id>.json`），
再在结束时补上**输出**。字段设计见 `TEMPLATE`。

约定
----
- 开始写 `status="running"`，正常结束原子覆盖为 `"done"`；
- 进程崩溃会留下 `"running"` —— 这是审计信息，不伪装成成功；
- 数据指纹优先用**文件 sha256**（可证明"同一份数据"），取不到文件时退化为
  `hash_scope="derived"`（行数+首末时间+首末收盘价），并如实标注。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys

RUNS_DIR = pathlib.Path("runs")

# ---------------------------------------------------------------- 基础信息
def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def new_run_id(seed: int, data_fp: str) -> str:
    """`YYMMDD_HHMMSS_<seed>_<指纹前 6 位>_<4 位随机>`。

    末尾的随机后缀是必要的：同一秒内跑两次同样 seed、同样数据的运行，
    光靠时间戳会得到**同一个 run_id**，清单互相覆盖（实测在测试里撞过）。
    """
    ts = _dt.datetime.now().strftime("%y%m%d_%H%M%S")
    tail = (data_fp or "nofp")[:6]
    uniq = os.urandom(2).hex()
    return f"{ts}_{int(seed)}_{tail}_{uniq}"


def git_describe() -> str:
    """有 git 就记 branch@sha（含 dirty 标记），没有就如实写 no-git。"""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if sha.returncode != 0:
            return "no-git"
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        flag = "-dirty" if (dirty.stdout or "").strip() else ""
        br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True, timeout=5)
        branch = (br.stdout or "").strip() or "?"
        return f"{branch}@{(sha.stdout or '').strip()}{flag}"
    except (OSError, subprocess.SubprocessError):
        return "no-git"


def versions() -> dict:
    """依赖版本：只记会影响数值结果的包。"""
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "scipy"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "?")
        except ImportError:
            out[mod] = "absent"
    return out


# ---------------------------------------------------------------- 数据指纹
def file_sha256(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_fingerprint(symbol: str, timeframe: str,
                     path: str | pathlib.Path | None = None,
                     df=None) -> dict:
    """返回数据指纹。

    - 给了 ``path`` 且文件存在 → 文件 sha256（最强证据）；
    - 只给 ``df`` → 用「行数 + 首末时间 + 首末收盘价」派生指纹；
    两种情况都在 ``hash_scope`` 里写明，绝不把派生值冒充成文件哈希。
    """
    base = {"symbol": str(symbol), "timeframe": str(timeframe)}
    if path is not None:
        p = pathlib.Path(path)
        if p.exists():
            base.update({"path": str(p), "sha256": file_sha256(p),
                         "hash_scope": "file", "bytes": p.stat().st_size})
            if df is not None and len(df):
                base.update({"rows": int(len(df)),
                             "start": str(df.index[0]),
                             "end": str(df.index[-1])})
            return base
        base["path"] = str(p)
        base["path_missing"] = True

    if df is None or len(df) == 0:
        base.update({"sha256": "", "hash_scope": "none",
                     "rows": 0, "start": None, "end": None})
        return base

    first, last = df.index[0], df.index[-1]
    payload = (f"{len(df)}|{first}|{last}|"
               f"{float(df['close'].iloc[0]):.6f}|{float(df['close'].iloc[-1]):.6f}")
    base.update({
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "hash_scope": "derived",
        "rows": int(len(df)),
        "start": str(first),
        "end": str(last),
    })
    return base


# ---------------------------------------------------------------- 清单读写
def _atomic_write(path: pathlib.Path, obj: dict) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def start_run(*, entry: str, reproduce_cmd: str, data: dict, cost: dict,
              search: dict, gates: dict, oos: dict,
              seed: int) -> dict:
    """开跑：写 status=running 的清单，返回清单本体。"""
    run_id = new_run_id(seed, data.get("sha256", ""))
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
        "entry": entry,
        "reproduce_cmd": reproduce_cmd,
        "seed": int(seed),
        "code": {"git": git_describe(),
                 "python": platform.python_version(),
                 "platform": platform.platform()},
        "data": data,
        "cost": cost,
        "search": search,
        "oos": oos,
        "gates": gates,
        "versions": versions(),
        "outcome": None,
        "ledger_write": None,
    }
    write_manifest(manifest)
    return manifest


def write_manifest(manifest: dict) -> pathlib.Path:
    run_id = manifest["run_id"]
    return _atomic_write(RUNS_DIR / f"{run_id}.json", manifest)


def load_manifest(run_id: str) -> dict:
    return json.loads((RUNS_DIR / f"{run_id}.json").read_text(encoding="utf-8"))


def finish_run(manifest: dict, *, outcome: dict,
               ledger_write: str | None = None) -> pathlib.Path:
    """正常结束：补 outcome 并置 status=done。"""
    m = dict(manifest)
    m["status"] = "done"
    m["finished_at"] = _now()
    m["outcome"] = outcome
    if ledger_write is not None:
        m["ledger_write"] = ledger_write
    return write_manifest(m)


def fail_run(manifest: dict, *, error: str) -> pathlib.Path:
    """异常结束：status=failed，保留错误原文（不静默吞掉）。"""
    m = dict(manifest)
    m["status"] = "failed"
    m["finished_at"] = _now()
    m["error"] = str(error)[:2000]
    return write_manifest(m)


def list_runs() -> list[dict]:
    """按时间倒序列出所有清单（缺文件/坏 JSON 不阻塞）。"""
    out = []
    if not RUNS_DIR.exists():
        return out
    for p in sorted(RUNS_DIR.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            out.append({"run_id": p.stem, "status": "unreadable"})
    return out


if __name__ == "__main__":  # 手工检查用
    for r in list_runs()[:10]:
        print(r.get("run_id"), r.get("status"), r.get("entry", "")[:60])
    sys.exit(0)
