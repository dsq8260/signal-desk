"""导入 MT5 策略测试器的报告（.xml / .html），统一汇总排名。

用途
----
MT5 市场里的 EA 是二进制 .ex5，读不出逻辑，但**可以在 MT5 自带的策略测试器里
跑出报告**（测试器 → 右键 → 保存为报告 → XML 或 HTML）。本模块解析这些报告，
把多个 EA 的结果汇总成一张可比对的表。

为什么不用 Python 重跑
--------------------
.ex5 是编译产物，无法知道它的逻辑；反编译既不可行也不合规。
所以"评估已有 EA"只有两条合法路径：
1. 让 MT5 自己回测 → 导入报告（本模块）
2. 让 EA 真实跑 → 按 magic 归因（gold/attrib.py）
"""

from __future__ import annotations

import pathlib
import re

import pandas as pd

# 报告里可能出现的字段名（中英文都支持，MT5 随界面语言变化）
LABELS = {
    "净盈利": ["总净盈利", "Total Net Profit", "TotalNetProfit", "净盈利"],
    "毛利": ["毛利", "Gross Profit", "GrossProfit"],
    "毛损": ["毛损", "Gross Loss", "GrossLoss"],
    "盈利因子": ["盈利因子", "Profit Factor", "ProfitFactor"],
    "预期收益": ["预期收益", "Expected Payoff", "ExpectedPayoff"],
    "恢复因子": ["恢复因子", "Recovery Factor", "RecoveryFactor"],
    "夏普": ["夏普比率", "Sharpe Ratio", "SharpeRatio"],
    "交易总计": ["交易总计", "Total Trades", "TotalTrades", "总成交"],
    "最大结余亏损": ["最大结余亏损", "Maximal drawdown (balance)",
                     "Balance Drawdown Maximal", "BalanceDrawdownMaximal",
                     "最大结余回撤"],
    "相对结余亏损": ["相对结余亏损", "Relative Drawdown", "RelativeDrawdown",
                     "相对回撤"],
    "初始入金": ["初始入金", "Initial Deposit", "InitialDeposit"],
    "符号": ["交易品种", "Symbol"],
    "周期": ["周期", "Period", "时间周期"],
}

_FILE_EXTS = {".xml", ".htm", ".html"}
_TAG_RE = re.compile(r"<[^>]+>")
_NUM_RE = re.compile(r"[-+]?\d[\d\s\u00a0]*(?:[.,]\d+)?")
_SPACES = " \t\r\n\u00a0"


def _strip_tags(text: str) -> str:
    """去标签、合并空白，保留可读文本。"""
    t = _TAG_RE.sub(" ", text)
    t = t.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(rf"[{_SPACES}]+", " ", t)


def _to_num(raw: str) -> float | None:
    """把 '1 234.56' / '-306.48' / '0.41' / '(-0.07%)' 解析成 float。"""
    if raw is None:
        return None
    s = raw.replace("\u00a0", " ").strip()
    s = s.strip("()% ")
    s = s.replace(" ", "").replace(",", "")
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(" ", "").replace(",", ""))
    except ValueError:
        return None


def _grab_text(text: str, names: list[str], window: int = 40) -> str | None:
    """抓标签后面紧跟的**文本**（如品种 GOLD#、周期 H1），不是数字。"""
    for name in names:
        for m in re.finditer(re.escape(name), text):
            tail = text[m.end(): m.end() + window].strip()
            tok = re.match(r"([A-Za-z][A-Za-z0-9_#.\-]{1,19})", tail)
            if tok:
                return tok.group(1)
    return None


def _xml_pairs(raw: str) -> dict:
    """提取 <Tag>value</Tag> 形式的键值对（MT5 的 XML 报告就是这种结构）。

    注意：XML 里标签名本身就是字段名，所以不能先剥标签——
    _strip_tags 会把 <TotalNetProfit> 一起删掉，只剩数字。
    """
    out: dict[str, str] = {}
    for m in re.finditer(r"<([A-Za-z_][\w.\-]*)>\s*([^<>]{1,120}?)\s*</\1>", raw):
        key = m.group(1).lower().replace("_", "")
        out.setdefault(key, m.group(2).strip())
    return out


def _grab(text: str, names: list[str], window: int = 60) -> str | None:
    """在文本里找某个标签后面紧跟的数值片段。"""
    for name in names:
        for m in re.finditer(re.escape(name), text):
            tail = text[m.end(): m.end() + window]
            hit = _NUM_RE.search(tail)
            if hit:
                return hit.group(0)
    return None


def parse_report(path: str | pathlib.Path) -> dict:
    """解析一个 MT5 测试器报告，返回统一字段。"""
    p = pathlib.Path(path)
    out = {"文件": p.name, "EA": p.stem, "符号": "", "周期": ""}
    for k in ("净盈利", "毛利", "毛损", "盈利因子", "预期收益", "恢复因子",
              "夏普", "交易总计", "最大结余亏损", "相对结余亏损", "初始入金"):
        out[k] = None
    if not p.exists():
        return out

    raw = None
    for enc in ("utf-8", "utf-16", "gbk", "cp1252"):
        try:
            raw = p.read_text(encoding=enc, errors="strict")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if raw is None:
        raw = p.read_text(encoding="utf-8", errors="ignore")

    text = _strip_tags(raw)
    pairs = _xml_pairs(raw)

    def lookup(names: list[str]) -> str | None:
        for n in names:
            key = n.lower().replace(" ", "").replace("_", "")
            if key in pairs:
                return pairs[key]
        return None

    for key, names in LABELS.items():
        if key in ("符号", "周期"):
            continue
        val = _to_num(lookup(names))
        if val is None:
            val = _to_num(_grab(text, names))
        if val is not None:
            out[key] = val

    # 品种/周期
    sym_raw = lookup(LABELS["符号"]) or _grab_text(text, LABELS["符号"])
    if sym_raw:
        out["符号"] = str(sym_raw).strip()
    per_raw = lookup(LABELS["周期"]) or _grab_text(text, LABELS["周期"])
    if per_raw:
        out["周期"] = str(per_raw).strip()

    return out


def find_reports(dirs: list[str | pathlib.Path] | str | pathlib.Path | None = None
                 ) -> list[pathlib.Path]:
    """在给定目录（默认 reports/）里找测试器报告。"""
    if dirs is None:
        targets = [pathlib.Path("reports")]
    elif isinstance(dirs, (str, pathlib.Path)):
        targets = [pathlib.Path(dirs)]
    else:
        targets = [pathlib.Path(x) for x in dirs]

    found: list[pathlib.Path] = []
    for d in targets:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in _FILE_EXTS:
                # 只收看起来像测试器报告的（含关键字段）
                try:
                    head = p.read_text(encoding="utf-8", errors="ignore")[:200000]
                except Exception:  # noqa: BLE001
                    continue
                if any(k in head for k in ("总净盈利", "Total Net Profit",
                                            "TotalNetProfit", "盈利因子",
                                            "Profit Factor", "ProfitFactor")):
                    found.append(p)
    return found


def rank_reports(dirs=None) -> pd.DataFrame:
    """汇总所有报告并排名（按净盈利）。"""
    rows = [parse_report(p) for p in find_reports(dirs)]
    cols = ["文件", "EA", "符号", "周期", "净盈利", "盈利因子", "交易总计",
            "最大结余亏损", "相对结余亏损", "夏普", "恢复因子", "预期收益",
            "毛利", "毛损", "初始入金"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    return df.sort_values("净盈利", ascending=False,
                          na_position="last").reset_index(drop=True)
