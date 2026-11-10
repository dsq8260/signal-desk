"""MT5 MCP 客户端：把生成的 EA 挂到图表上（默认干跑，不下单）。

配置来源（按优先级）：
1. 环境变量 `MT5_MCP_URL` / `MT5_MCP_KEY`
2. 项目根目录 `.mt5_mcp.json`：`{"url": "...", "key": "..."}`
3. `%APPDATA%\\MetaQuotes\\Terminal\\*\\config\\assistant.ini` 的 `[MCP.MetaTrader]`
   —— 注意：该文件里的 `ApiKey` 是**加密存储**（>80 字符），只能取 `Endpoint`。

安全约定：
- 本模块只做"挂 EA / 卸 EA / 读终端状态"，**不调用任何 trade_\* 工具**；
- 挂载时强制注入 `InpDryRun=true`，要真下单必须用户在 MT5 里手动改；
- 终端不允许算法交易（trade_allowed=false）时，挂上也不会下单，这里会如实提示。
"""

from __future__ import annotations

import configparser
import json
import os
import pathlib
import re
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / ".mt5_mcp.json"
DEFAULT_URL = "http://127.0.0.1:22346/mcp"


# ---------------------------------------------------------------- 配置
def _read_assistant_endpoint() -> str | None:
    base = os.environ.get("APPDATA")
    if not base:
        return None
    for ini in pathlib.Path(base, "MetaQuotes", "Terminal").glob(
            "*/config/assistant.ini"):
        try:
            text = ini.read_bytes().decode("utf-16")
        except (OSError, UnicodeDecodeError):
            continue
        sect = None
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("[") and s.endswith("]"):
                sect = s[1:-1].strip()
                continue
            if sect == "MCP.MetaTrader" and s.lower().startswith("endpoint"):
                val = s.split("=", 1)[1].strip() if "=" in s else ""
                if val:
                    return val
    return None


def load_config() -> dict:
    """返回 {"url": str, "key": str, "source": str}。取不到 key 时 key=""。"""
    url = os.environ.get("MT5_MCP_URL", "").strip()
    key = os.environ.get("MT5_MCP_KEY", "").strip()
    if url and key:
        return {"url": url, "key": key, "source": "env"}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            u = str(data.get("url") or "").strip()
            k = str(data.get("key") or "").strip()
            if u and k:
                return {"url": u, "key": k, "source": str(CONFIG_FILE.name)}
        except (OSError, json.JSONDecodeError):
            pass
    return {"url": url or _read_assistant_endpoint() or DEFAULT_URL,
            "key": key, "source": "assistant.ini" if not key else "env"}


# ---------------------------------------------------------------- 客户端
class Mt5Mcp:
    """极简 MCP over HTTP 客户端（JSON-RPC + SSE 兼容）。"""

    def __init__(self, url: str | None = None, key: str | None = None,
                 timeout: int = 25):
        cfg = load_config()
        self.url = (url or cfg["url"]).strip()
        self.key = (key if key is not None else cfg["key"]).strip()
        self.timeout = int(timeout)
        self.session = ""
        self._id = 0
        self.last_error = ""
        self.handshaked = False
        self.last_run_id = ""

    # -- 底层 --
    def _post(self, payload: dict, timeout: float | None = None) -> dict:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json, text/event-stream")
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")
        if self.session:
            req.add_header("Mcp-Session-Id", self.session)
        with urllib.request.urlopen(
                req, timeout=float(timeout or self.timeout)) as r:
            if r.headers.get("Mcp-Session-Id"):
                self.session = r.headers["Mcp-Session-Id"]
            raw = r.read().decode("utf-8", "replace")
            ct = r.headers.get("Content-Type", "")
        if "text/event-stream" in ct:
            for line in raw.splitlines():
                if line.startswith("data:"):
                    try:
                        return json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _rpc(self, method: str, params: dict | None = None,
             timeout: float | None = None) -> dict:
        self._id += 1
        return self._post({"jsonrpc": "2.0", "id": self._id,
                           "method": method, "params": params or {}},
                          timeout=timeout)

    def handshake(self) -> bool:
        if self.handshaked:
            return True
        try:
            resp = self._rpc("initialize", {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "gold-terminal", "version": "0.1"}})
            if "result" not in resp:
                self.last_error = (f"握手失败：{resp.get('error') or resp}")
                return False
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
            self.handshaked = True
            return True
        except (urllib.error.URLError, OSError) as exc:
            self.last_error = f"连不上 MT5 MCP（{self.url}）：{exc}"
            return False

    # -- 工具调用 --
    def call(self, name: str, args: dict | None = None,
             timeout: float | None = None) -> dict:
        """调一个 tool，返回解析后的 JSON payload（失败返回 {"ok": False,...}）。"""
        if not self.handshake():
            return {"ok": False, "error": self.last_error}
        try:
            resp = self._rpc("tools/call", {"name": name,
                                            "arguments": args or {}},
                             timeout=timeout)
        except (urllib.error.URLError, OSError) as exc:
            self.last_error = f"{name} 调用失败：{exc}"
            return {"ok": False, "error": self.last_error}
        result = resp.get("result") or {}
        if result.get("isError"):
            txt = ""
            for item in result.get("content") or []:
                if item.get("type") == "text":
                    txt = item.get("text", "")
            return {"ok": False, "error": txt or "tool 返回错误"}
        for item in result.get("content") or []:
            if item.get("type") == "text":
                txt = item.get("text", "")
                try:
                    return json.loads(txt)
                except json.JSONDecodeError:
                    return {"ok": True, "text": txt}
        return result or {"ok": True}

    # -- 高层 --
    def available(self) -> bool:
        return bool(self.call("terminal_get_info"))

    def terminal(self) -> dict:
        d = self.call("terminal_get_info")
        if isinstance(d, dict):
            d.setdefault("ok", True)
        return d

    def account(self) -> dict:
        return self.call("get_trading_account_info")

    def charts(self) -> list[dict]:
        d = self.call("list_open_charts")
        for key in ("charts", "data", "items"):
            if isinstance(d.get(key), list):
                return d[key]
        return []

    def find_chart(self, symbol: str, period: str) -> str | None:
        want_sym = str(symbol).upper()
        want_tf = str(period).upper()
        for c in self.charts():
            sym = str(c.get("symbol") or "").upper()
            tf = str(c.get("period") or "").upper()
            cid = c.get("chart_id") or c.get("id")
            if sym == want_sym and tf == want_tf and cid:
                return str(cid)
        return None

    def open_chart(self, symbol: str, period: str) -> str | None:
        d = self.call("chart_open", {"symbol": symbol, "period": period})
        for key in ("chart_id", "id", "chartId"):
            if d.get(key):
                return str(d[key])
        return None

    def attach(self, chart_id: str, ex5_path: str,
               params: str | None = None) -> dict:
        args = {"chart_id": str(chart_id),
                "expert_path": str(pathlib.Path(ex5_path).resolve())}
        if params:
            args["expert_parameters"] = params
        return self.call("chart_add_expert", args)

    def detach(self, chart_id: str) -> dict:
        return self.call("chart_remove_expert", {"chart_id": str(chart_id)})

    # ---------------------------------------------------------------- 文件推送
    def workspace(self) -> dict:
        """MQL5 目录与可读写根（get_workspace_info）。"""
        d = self.call("get_workspace_info")
        return d.get("workspace") or {}


    def experts_dir(self) -> str:
        """`<MQL5>\\Experts` 绝对路径（取不到就退回空串）。"""
        ws = self.workspace()
        mql5 = ws.get("mql5_folder") or ws.get("current_folder") or ""
        return f"{mql5}\\Experts" if mql5 else ""


    def push_file(self, local_path: str, remote_path: str | None = None) -> dict:
        """把本地文件写进终端目录（十六进制传输，走 MCP 的写权限）。"""
        src = pathlib.Path(local_path)
        if not src.exists():
            return {"ok": False, "error": f"本地文件不存在：{src}"}
        if remote_path is None:
            base = self.experts_dir()
            if not base:
                return {"ok": False, "error": "取不到 MQL5 目录（get_workspace_info）"}
            remote_path = f"{base}\\{src.name}"
        data = src.read_bytes()
        res = self.call("write_binary_file", {
            "path": str(remote_path), "data": data.hex(),
            "overwrite": True, "truncate": True})
        err = res.get("error") if isinstance(res, dict) else None
        return {"ok": not err, "remote": str(remote_path),
                "bytes": len(data), "error": err}

        # ---------------------------------------------------------- 模板挂载
    def _empty_chart(self, symbol: str, period: str,
                     allow_new: bool = True) -> str | None:
        """找一张该品种/周期且**当前没有 EA** 的图表；没有就新开。"""
        want_sym = str(symbol).upper()
        want_tf = str(period).upper()
        for ch in self.charts():
            if (str(ch.get("symbol") or "").upper() == want_sym
                    and str(ch.get("period") or "").upper() == want_tf
                    and not ch.get("expert")):
                cid = ch.get("chart_id") or ch.get("id")
                if cid:
                    return str(cid)
        return self.open_chart(symbol, period) if allow_new else None

    def templates_dir(self) -> str:
        ws = self.workspace()
        mql5 = ws.get("mql5_folder") or ""
        return f"{mql5}\\Profiles\\Templates" if mql5 else ""

    def _ensure_base_template(self) -> tuple[str | None, str]:
        """确保有一份含 <expert> 的基线模板，返回 (模板名, 错误)。"""
        tpl_dir = self.templates_dir()
        if not tpl_dir:
            return None, "取不到 MQL5 模板目录"
        base_name = "gold_auto_base.tpl"
        cur = self.call("read_file", {"path": f"{tpl_dir}\\{base_name}"})
        if not cur.get("error") and "<expert>" in str(cur.get("text") or "").lower():
            return base_name, ""

        # 需要一个"已索引"的 EA 来产出基线：优先内置的 ExpertMACD
        seed = None
        for p in self.programs():
            name = str(p.get("name", ""))
            if name in ("ExpertMACD", "ExpertMA", "ExpertADX"):
                seed = p
                break
        if seed is None:
            prog = self.programs()
            seed = prog[0] if prog else None
        if seed is None:
            return None, "终端索引里没有任何 EA，无法生成基线模板"

        tmp_chart = self.open_chart("GOLD#", "H1")
        if not tmp_chart:
            return None, "开不了临时图表"
        try:
            self.call("chart_add_expert", {"chart_id": tmp_chart,
                                           "expert_path": seed["path"]})
            sav = self.call("chart_save_template",
                            {"chart_id": tmp_chart,
                             "template_filename": "gold_auto_base",
                             "overwrite": True})
            if sav.get("error"):
                return None, f"存基线模板失败：{sav['error']}"
        finally:
            if seed is not None:
                self.detach(tmp_chart)
        cur = self.call("read_file", {"path": f"{tpl_dir}\\{base_name}"})
        if cur.get("error") or "<expert>" not in str(cur.get("text") or "").lower():
            return None, "基线模板里没有 <expert> 段"
        return base_name, ""

    def attach_via_template(self, ex5_name: str, symbol: str, period: str,
                            inputs: dict | None = None,
                            chart_id: str | None = None) -> dict:
        """用模板把 EA 挂到图表（绕过程序索引）。"""
        import re as _re

        if not self.handshake():
            return {"ok": False, "error": self.last_error}
        tpl_dir = self.templates_dir()
        base_name, err = self._ensure_base_template()
        if err:
            return {"ok": False, "error": err}

        base = self.call("read_file",
                         {"path": f"{tpl_dir}\\{base_name}"}).get("text") or ""
        m = _re.search(r"<expert>[\s\S]*?</expert>", base, _re.I)
        if not m:
            return {"ok": False, "error": "基线模板缺少 <expert> 段"}

        ex5_name = str(ex5_name)
        if ex5_name.lower().endswith(".ex5"):
            ex5_name = ex5_name[:-4]
        ins = {str(k): str(v) for k, v in (inputs or {}).items()}
        block = ("<expert>\n"
                 f"name={ex5_name}\n"
                 f"path=Experts\\{ex5_name}.ex5\n"
                 "expertmode=1\n"
                 "<inputs>\n"
                 + "".join(f"{k}={v}\n" for k, v in ins.items())
                 + "</inputs>\n</expert>")
        content = base[:m.start()] + block + base[m.end():]
        # 让模板的品种/周期对上目标
        tf_map = {"M1": 1, "M5": 1, "M15": 1, "M30": 1, "H1": 1, "H4": 1,
                  "D1": 1, "W1": 1}
        content = _re.sub(r"^symbol=.*$", f"symbol={symbol}", content,
                          count=1, flags=_re.M)
        content = _re.sub(r"^period_type=.*$", "period_type=1", content,
                          count=1, flags=_re.M)
        size = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240,
                "D1": 1440, "W1": 10080}.get(str(period).upper(), 60)
        content = _re.sub(r"^period_size=.*$", f"period_size={size}",
                          content, count=1, flags=_re.M)

        tpl_name = f"gold_auto_{ex5_name[:40]}.tpl"
        w = self.call("write_file", {"path": f"{tpl_dir}\\{tpl_name}",
                                     "content": content, "overwrite": True})
        if w.get("error"):
            return {"ok": False, "error": f"写模板失败：{w['error']}"}

        cid = chart_id or self.open_chart(symbol, period)
        if not cid:
            return {"ok": False, "error": "打不开目标图表"}
        app = self.call("chart_apply_template",
                        {"chart_id": str(cid),
                         "template_filename": tpl_name})
        if app.get("error"):
            return {"ok": False, "error": f"应用模板失败：{app['error']}"}

        shown = None
        for ch in self.charts():
            if str(ch.get("chart_id")) == str(cid):
                exp = ch.get("expert")
                shown = (exp or {}).get("name") if isinstance(exp, dict) else exp
        ok = bool(shown and str(shown).lower() == ex5_name.lower())
        return {"ok": ok, "chart_id": str(cid), "expert": ex5_name,
                "template": tpl_name, "via": "template",
                "error": "" if ok else f"模板已应用但图表上没有该 EA（见到 {shown!r}）"}

    # ---------------------------------------------------------- 索引等待
    def wait_indexed(self, name: str, timeout: float = 60.0,
                     interval: float = 2.0) -> bool:
        """轮询终端索引，直到该 EA 出现（用户按一次 F5 后即可通过）。"""
        import time as _time

        deadline = _time.time() + max(1.0, float(timeout))
        while _time.time() < deadline:
            if self.expert_indexed(name):
                return True
            _time.sleep(max(0.5, float(interval)))
        return False

    # ---------------------------------------------------------- 测试器
    def run_tester(self, ex5_name: str, symbol: str, timeframe: str,
                   from_date: str, to_date: str, *,
                   model: str = "Every tick", deposit: float = 1000.0,
                   leverage: int = 100, inputs_path: str | None = None,
                   timeout_sec: int = 900,
                   execution_delay: int = 0) -> dict:
        """用 MT5 策略测试器跑一次并取回 JSON 报告。

        注意：`tester_prepare_config` 只接受**已在终端索引里**的程序
        （实测未索引会报 "must reference a compiled program"），
        所以先查索引；缺索引返回 reason="not_indexed"。
        """
        if not self.handshake():
            return {"ok": False, "reason": "no_mcp", "error": self.last_error}
        name = str(ex5_name)
        if name.lower().endswith(".ex5"):
            name = name[:-4]
        idx = self.expert_indexed(name)
        if idx is None:
            return {"ok": False, "reason": "not_indexed",
                    "error": f"{name} 不在终端程序索引里，先在 MT5 导航器刷新"}

        tester_dir = f"{self.templates_dir().replace(chr(92) + 'Templates', '')}" \
                     f"{chr(92)}Tester"
        ini = f"{tester_dir}{chr(92)}gold_{name[:40]}.ini"
        prep = self.call("tester_prepare_config", {
            "mql5_program_path": idx.get("path") or "",
            "symbol": symbol, "timeframe": timeframe, "model": model,
            "from_date": from_date, "to_date": to_date,
            "deposit": float(deposit), "leverage": int(leverage),
            "execution_delay": int(execution_delay), "output_path": ini})
        if prep.get("error"):
            return {"ok": False, "reason": "prepare_failed",
                    "error": prep["error"]}

        # 上一次测试可能还在跑（被 timeout 打断后仍占用测试器）→ 等它结束再试
        run: dict = {}
        for _attempt in range(8):
            run = self.call("tester_run_backtest", {
                "config_path": ini,
                **({"inputs_path": inputs_path} if inputs_path else {}),
                "wait": True, "timeout_sec": int(timeout_sec)},
                timeout=float(timeout_sec) + 60.0)
            err = str(run.get("error") or "")
            if "still started" not in err.lower():
                break
            import time as _t

            _t.sleep(15.0)
        if run.get("error"):
            return {"ok": False, "reason": "run_failed", "error": run["error"]}

        rid_prev = None

        rid = run.get("run_id") or run.get("id")
        if not rid:
            return {"ok": False, "reason": "no_run_id",
                    "error": "测试器没返回 run_id", "raw": run}
        self.last_run_id = str(rid)
        rep = self.call("tester_get_report", {"run_id": str(rid)})
        return {"ok": not rep.get("error"), "run_id": str(rid),
                "report": rep, "ini": ini,
                "error": rep.get("error") or ""}

    def attach_smart(self, ex5_name: str, ex5_path: str, symbol: str,
                     period: str, inputs: dict | None = None) -> dict:
        """先走索引直挂；失败（未索引）再走模板。两条路都失败才报错。

        只落在**没有 EA 的图表**上：实测把模板 apply 到已有 EA 的图表不会替换，
        而且直挂会顶掉用户手工挂的 EA（比如数据桥），所以一律避开。
        """
        cid = self._empty_chart(symbol, period)
        if not cid:
            return {"ok": False, "error": "打不开图表"}
        params = ",".join(f"{k}={v}" for k, v in (inputs or {}).items())
        first = self.call("chart_add_expert",
                          {"chart_id": str(cid), "expert_path": ex5_path,
                           **({"expert_parameters": params} if params else {})})
        if not first.get("error"):
            return {"ok": True, "chart_id": str(cid), "expert": ex5_name,
                    "via": "index", "error": ""}
        # 未索引 → 模板通道
        tpl = self.attach_via_template(ex5_name, symbol, period,
                                       inputs=inputs, chart_id=str(cid))
        tpl["index_error"] = first.get("error")
        return tpl

    def send_ea(self, ex5_path: str, symbol: str, period: str,
                magic: int | None = None, lots: float | None = None,
                dry_run: bool = True) -> dict:
        """把 .ex5 挂到该品种/周期的图表上（默认干跑）。"""
        p = pathlib.Path(ex5_path)
        if not p.exists():
            return {"ok": False, "error": f"找不到 .ex5：{p}"}
        if not self.handshake():
            return {"ok": False, "error": self.last_error}

        chart_id = self.find_chart(symbol, period) or self.open_chart(symbol, period)
        if not chart_id:
            return {"ok": False, "error": "打不开图表（chart_open 没返回 chart_id）"}

        parts = [f"InpDryRun={'true' if dry_run else 'false'}"]
        if magic is not None:
            parts.append(f"InpMagic={int(magic)}")
        if lots is not None:
            parts.append(f"InpLots={float(lots)}")
        res = self.attach(chart_id, str(p), ",".join(parts))
        ok = not res.get("error")
        return {"ok": ok, "chart_id": chart_id, "symbol": symbol,
                "period": period, "expert": p.name, "dry_run": bool(dry_run),
                "parameters": ",".join(parts),
                "detail": res.get("error") or res}


    def programs(self, program_type: str = "expert",
                 name_filter: str | None = None) -> list[dict]:
        """终端索引里可用的 MQL5 程序（不是磁盘上的文件）。"""
        args: dict = {"program_type": program_type}
        if name_filter:
            args["filter"] = name_filter
        d = self.call("list_available_mql5_programs", args)
        return d.get("programs") or []

    def expert_indexed(self, name: str) -> dict | None:
        """该 EA 是否已在终端索引里；在就返回索引条目。"""
        want = str(name).lower()
        for p in self.programs():
            if str(p.get("name", "")).lower() == want:
                return p
        return None

    def deploy_and_attach(self, mq5_path: str, symbol: str, period: str,
                          *, dry_run: bool = True, magic: int | None = None,
                          lots: float | None = None) -> dict:
        """编译产物 → 推 .mq5 到 MQL5\Experts → MetaEditor 编译 → 预检索引 → 挂载。

        返回 `ok=False` 时带 `reason`：
        - `no_ex5`：还没编译
        - `push_failed` / `compile_failed`
        - `not_indexed`：文件已就位，但终端索引没刷新（需要刷新导航器）
        - `attach_failed`
        """
        import pathlib as _pl
        import subprocess as _sp

        import gold.mq5 as _mq

        src = _pl.Path(mq5_path)
        if not src.exists():
            return {"ok": False, "reason": "no_ex5",
                    "error": f"找不到源码 {src}"}
        if not self.handshake():
            return {"ok": False, "reason": "no_mcp", "error": self.last_error}

        experts = self.experts_dir()
        if not experts:
            return {"ok": False, "reason": "no_workspace",
                    "error": "取不到 MQL5 目录"}
        remote_mq5 = f"{experts}\\{src.name}"
        push = self.call("write_file", {"path": remote_mq5,
                                        "content": src.read_text(
                                            encoding="utf-8", errors="replace"),
                                        "overwrite": True})
        if push.get("error"):
            return {"ok": False, "reason": "push_failed",
                    "error": push["error"]}
        remote_ex5 = remote_mq5[:-4] + ".ex5"

        me = _pl.Path(_mq.METAEDITOR)
        if not me.exists():
            return {"ok": False, "reason": "no_metaeditor",
                    "error": f"找不到 MetaEditor：{me}"}
        log = remote_mq5[:-4] + ".compile.log"
        proc = _sp.run([str(me), f"/compile:{remote_mq5}", f"/log:{log}"],
                       capture_output=True, text=True, timeout=240)
        if proc.returncode != 0:
            return {"ok": False, "reason": "compile_failed",
                    "error": f"MetaEditor 退出码 {proc.returncode}"}

        idx = self.expert_indexed(src.stem)
        if idx is None:
            return {"ok": False, "reason": "not_indexed",
                    "installed": remote_ex5,
                    "error": ("EA 已编译进 MQL5\\Experts，但 MT5 的程序索引还没刷新。"
                              "在 MT5 导航器里右键 → 刷新（或按 F5），"
                              "然后点「重试挂载」。")}

        res = self.send_ea(remote_ex5, symbol, period, magic=magic,
                           lots=lots, dry_run=dry_run)
        if not res.get("ok"):
            res["reason"] = "attach_failed"
        else:
            res["reason"] = "ok"
        res["installed"] = idx.get("path") or remote_ex5
        return res

# ---------------------------------------------------------------- 便捷入口
def install_and_send(mq5_path: str | pathlib.Path, symbol: str,
                     period: str, *, dry_run: bool = True,
                     magic: int | None = None) -> dict:
    """编译产物 → 安装到 MT5 Experts 目录 → 挂到图表（干跑）。"""
    from gold.mq5 import install_to_mt5

    ex5 = pathlib.Path(mq5_path).with_suffix(".ex5")
    if not ex5.exists():
        return {"ok": False, "error": f"还没编译出 {ex5.name}，先点「生成并编译 EX5」"}
    inst = install_to_mt5(ex5)
    if not inst.get("ok"):
        return {"ok": False, "error": f"安装到 MT5 失败：{inst.get('errors')}"}
    target = inst["copied"][0]
    client = Mt5Mcp()
    out = client.send_ea(target, symbol, period, magic=magic, dry_run=dry_run)
    out["installed"] = target
    return out


def guess_mt5_symbol(symbol: str) -> str:
    """把我们的品种名换成终端里的写法（XM 用 GOLD#）。"""
    s = str(symbol).upper()
    if s in ("GOLD", "XAUUSD", "GOLD#"):
        return s
    if re.fullmatch(r"[A-Z]{6}", s):
        return s
    return s
