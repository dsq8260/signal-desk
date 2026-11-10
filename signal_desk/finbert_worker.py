"""FinBERT 情绪计算的独立进程（不放在 8777 服务进程里）。

为什么独立进程：
  * EA 工厂常驻挖掘是纯 Python 回测循环，长期占 GIL；
  * 在它的后台线程里 import torch / 载入 438MB 权重，Windows 下会与
    原生 DLL 载入锁互相卡住，辅助页永远等不到结果；
  * 独立进程各跑各的，HTTP 服务只读 data/finbert_cache.json，永远不阻塞。

职责边界：只给「英文新闻文字偏正面/负面/中性」打标签，不预测黄金涨跌。
用法：python finbert_worker.py [--once] [--ppid <服务进程PID>]
"""
from __future__ import annotations

import argparse
import ctypes
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import en_news            # noqa: E402
import finbert_sent as fb  # noqa: E402

INTERVAL = 300.0


def parent_alive(pid: int) -> bool:
    """父进程（8777 服务）不在了就自己退出，避免残留孤儿进程占内存。"""
    if not pid:
        return True
    try:
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            ok = k.GetExitCodeProcess(h, ctypes.byref(code))
            return bool(ok) and code.value == 259   # STILL_ACTIVE
        finally:
            k.CloseHandle(h)
    except Exception:  # noqa: BLE001
        return True


def nap(seconds: float, ppid: int) -> bool:
    """分段睡，方便及时感知父进程退出；返回 False 表示应当结束。"""
    end = time.time() + seconds
    while time.time() < end:
        if not parent_alive(ppid):
            return False
        time.sleep(min(5.0, max(0.1, end - time.time())))
    return True


def once() -> dict:
    t0 = time.time()
    if not fb.status().get("enabled"):
        r = fb.fallback("本地 FinBERT 权重未就绪，逐条信号沿用关键词规则。")
    else:
        en = en_news.digest(force=True)
        titles = [str(x.get("title") or "").strip()
                  for x in (en.get("items") or [])[:12]]
        print(f"[worker] 英文标题 {len(titles)} 条 · 源 {en.get('source')} · "
              f"ok={en.get('ok')}", flush=True)
        r = fb.score(titles)
        r["feed"] = {
            "ok": bool(en.get("ok")), "source": en.get("source") or "",
            "n": int(en.get("n") or 0), "stale": bool(en.get("stale")),
            "error": en.get("error") or "",
        }
    fb.write_cache(r)
    print(f"[worker] 已落盘：{r.get('n_scored')} 条 · {r.get('label')} · "
          f"用时 {time.time() - t0:.1f}s", flush=True)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="只算一轮就退出（调试用）")
    ap.add_argument("--ppid", type=int, default=0, help="父进程 PID，退出后本进程自动收工")
    a = ap.parse_args()
    print("[worker] FinBERT 独立进程启动", flush=True)
    while True:
        try:
            once()
        except Exception as e:  # noqa: BLE001
            print(f"[worker] 本轮失败：{type(e).__name__}: {e}", flush=True)
            try:
                fb.write_cache(fb.fallback(f"情绪计算失败：{type(e).__name__}: {e}"))
            except Exception:  # noqa: BLE001
                pass
        if a.once:
            return
        if not nap(INTERVAL, a.ppid):
            print("[worker] 服务已退出，worker 收工", flush=True)
            return


if __name__ == "__main__":
    main()
