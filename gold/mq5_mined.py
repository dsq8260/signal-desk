"""把「04 策略挖掘」的候选 spec 直接生成 MQL5 EA 源码。

与 gold/mq5.py 的模板生成器不同：那边是固定模板 + 少量参数；这边候选的
规则是逐条组合出来的（突破 / ADX / EMA 相对 / RSI / 布林 / 动量 / 大周期
过滤，ATR 止损止盈 + 通道离场 + 限时离场），所以直接把 spec **逐条翻译**
成 MQL5 代码——生成出来的 EA 和 Python 回测跑的是同一套语义：

  · t 根收盘出信号、t+1 根开盘市价执行；
  · SL/TP 按「信号根的 ATR14（SMA 口径）× 倍数」挂到订单上，由服务器
    盘中触发（回测里同根先判止损的悲观口径，实盘即真实盘中成交）；
  · 通道离场：收盘跌破/突破前 N 根极值 → 次根开盘市价平仓；
  · 限时离场：持有满 N 根 → 平仓；
  · 手数 = 权益 × 风险% ÷ 止损距离，向下取整到步长，上限 max_lots；
  · 日内亏损停机（默认 2% 权益，与回测 daily_stop 一致）；
  · 默认 InpDryRun=true：只打印与自报成绩，不下单。

不支持的条件类型（口述策略里的 cross/ma/consec/hour 等）会明确报错，
绝不静默丢弃——丢一条条件就不是被验证过的那个策略了。
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field

SUPPORTED = ("brk", "adx", "marel", "rsi", "boll", "mom", "htf")
TF_ENUM = {"M1": "PERIOD_M1", "M5": "PERIOD_M5", "M15": "PERIOD_M15",
           "M30": "PERIOD_M30", "H1": "PERIOD_H1", "H4": "PERIOD_H4",
           "D1": "PERIOD_D1", "W1": "PERIOD_W1"}


@dataclass
class MinedResult:
    ok: bool = False
    mq5_path: str = ""
    source: str = ""
    error: str = ""
    unsupported: list = field(default_factory=list)


# ---------------------------------------------------------------- 输入注册表
class _Reg:
    """收集要暴露成 input 的策略参数，重名自动加序号。"""

    def __init__(self):
        self.items: list[tuple[str, str, object, str]] = []
        self._cnt: dict[str, int] = {}

    def add(self, base: str, mtype: str, default, comment: str) -> str:
        k = self._cnt.get(base, 0) + 1
        self._cnt[base] = k
        name = f"Inp{base}" if k == 1 else f"Inp{base}{k}"
        self.items.append((mtype, name, default, comment))
        return name


def _fmt(v) -> str:
    if isinstance(v, float):
        s = f"{v:g}"
        return s if ("." in s or "e" in s) else s + ".0"
    return str(v)


# ---------------------------------------------------------------- 条件翻译
def _cond_block(c: dict, reg: _Reg) -> tuple[str, list[str]]:
    """一条条件 → (MQL5 代码块, 需要的句柄列表)。"""
    f = str(c.get("f"))
    if f == "brk":
        n = reg.add("BrkN", "int", int(c.get("n", 20)), "突破回看根数")
        if str(c.get("dir", "up")) == "up":
            return (f"""   {{
      double hh = HiBefore(1, {n});
      if(hh < 0.0 || !(CloseAt(1) > hh)) return false;
   }}""", [])
        return (f"""   {{
      double ll = LoBefore(1, {n});
      if(ll < 0.0 || !(CloseAt(1) < ll)) return false;
   }}""", [])
    if f == "adx":
        v = reg.add("AdxV", "double", float(c.get("v", 25)), "ADX 门槛")
        op = ">" if str(c.get("op", ">")) == ">" else "<"
        return (f"""   {{
      double v[]; ArraySetAsSeries(v, true);
      if(CopyBuffer(hAdx, 0, 1, 1, v) != 1) return false;
      if(!(v[0] {op} {v})) return false;
   }}""", ["adx"])
    if f == "marel":
        if str(c.get("ma", "ema")).lower() != "ema":
            raise ValueError("marel 条件只支持 EMA（挖掘家族只用 EMA）")
        fa = reg.add("EmaFast", "int", int(c.get("fast", 20)), "快线周期")
        sb = reg.add("EmaSlow", "int", int(c.get("slow", 50)), "慢线周期")
        op = ">" if str(c.get("op", ">")) == ">" else "<"
        return (f"""   {{
      double f[], s[];
      ArraySetAsSeries(f, true); ArraySetAsSeries(s, true);
      if(CopyBuffer(hEmaF, 0, 1, 1, f) != 1) return false;
      if(CopyBuffer(hEmaS, 0, 1, 1, s) != 1) return false;
      if(!(f[0] {op} s[0])) return false;
   }}""", ["marel"])
    if f == "rsi":
        v = reg.add("RsiV", "double", float(c.get("v", 60)), "RSI 门槛")
        op = ">" if str(c.get("op", ">")) == ">" else "<"
        return (f"""   {{
      double v[]; ArraySetAsSeries(v, true);
      if(CopyBuffer(hRsi, 0, 1, 1, v) != 1) return false;
      if(!(v[0] {op} {v})) return false;
   }}""", [])
    if f == "boll":
        v = reg.add("BollV", "double", float(c.get("v", 0.8)), "布林位置门槛(0-1)")
        op = ">" if str(c.get("op", ">")) == ">" else "<"
        return (f"""   {{
      double ma = SmaAt(1, 20), atr = AtrSma(1);
      if(ma <= 0.0 || atr <= 0.0) return false;
      double up = ma + 1.8 * atr, dn = ma - 1.8 * atr;
      if(up <= dn) return false;
      double pos = (CloseAt(1) - dn) / (up - dn);
      if(!(pos {op} {v})) return false;
   }}""", [])
    if f == "mom":
        n = reg.add("MomN", "int", int(c.get("n", 3)), "动量回看根数")
        v = reg.add("MomV", "double", float(c.get("v", 0)), "动量门槛(ATR)")
        op = ">" if str(c.get("op", ">")) == ">" else "<"
        return (f"""   {{
      double atr = AtrSma(1);
      double c0 = CloseAt(1), cn = CloseAt(1 + {n});
      if(atr <= 0.0 || c0 <= 0.0 || cn <= 0.0) return false;
      if(!((c0 - cn) / atr {op} {v})) return false;
   }}""", [])
    if f == "htf":
        tf = str(c.get("tf", "H4")).upper()
        if tf not in TF_ENUM:
            raise ValueError(f"大周期 {tf} 不支持")
        n = reg.add("HtfN", "int", int(c.get("n", 50)), "大周期 EMA 周期")
        op = ">" if str(c.get("op", ">")) == ">" else "<"
        return (f"""   {{
      double e = HtfEmaVal();
      if(e <= 0.0) return false;
      if(!(CloseAt(1) {op} e)) return false;
   }}""", [("htf", tf, int(c.get("n", 50)))])
    raise ValueError(f"不支持的条件类型 {f}")


def _side_fn(name: str, conds: list, reg: _Reg, labels: list[str]):
    handles: list = []
    blocks = []
    for c in conds:
        blk, hs = _cond_block(c, reg)
        blocks.append(blk)
        handles.extend(hs)
        labels.append(str(c))
    body = "\n".join(blocks) if blocks else "   return false; // 该方向没有入场条件"
    code = f"bool {name}()\n{{\n{body}\n   return true;\n}}\n"
    return code, handles


# ---------------------------------------------------------------- 主生成器
def generate(spec: dict, name: str = "MinedEA", out_dir="out",
             magic: int = 20261106, meta: dict | None = None,
             doc_lines: list[str] | None = None) -> MinedResult:
    meta = meta or {}
    try:
        direction = str(spec.get("direction", "long"))
        ex = spec.get("exit") or {}
        rk = spec.get("risk") or {}
        conds_l = list((spec.get("entry") or {}).get("long") or [])
        conds_s = list((spec.get("entry") or {}).get("short") or [])
        bad = sorted({str(c.get("f")) for c in conds_l + conds_s
                      if str(c.get("f")) not in SUPPORTED})
        if bad:
            return MinedResult(ok=False, error="候选含不支持生成 EA 的条件：" + "、".join(bad),
                               unsupported=bad)

        reg = _Reg()
        fn_l, hl = _side_fn("CondLong", conds_l, reg, [])
        fn_s, hs = _side_fn("CondShort", conds_s, reg, [])
        use_adx = "adx" in hl
        use_marel = "marel" in hl
        use_rsi = _needs_rsi(conds_l, conds_s)
        htf_list = [h for h in hl + hs if isinstance(h, tuple)]
        use_htf = bool(htf_list)
        if use_htf and len({(t, n) for _, t, n in htf_list}) > 1:
            return MinedResult(ok=False, error="多个不同参数的大周期过滤，暂不支持")

        sl_atr = float(ex.get("sl_atr", 2.0) or 0)
        tp_atr = float(ex.get("tp_atr", 4.0) or 0)
        chn = int(ex.get("channel_n", 0) or 0)
        mb = int(ex.get("max_bars", 0) or 0)
        risk_pct = float(rk.get("risk_pct", 1.0) or 1.0)
        max_lots = float(rk.get("max_lots", 0.05) or 0.05)
        allow_long = direction in ("long", "both")
        allow_short = direction in ("short", "both")

        strat_inputs = "\n".join(
            f"input {t} {n} = {_fmt(v)}; // {c}" for t, n, v, c in reg.items)
        if strat_inputs:
            strat_inputs = "\n" + strat_inputs

        # 最小 bar 数：60 起步（回测同口径）+ 最大的回看窗口
        wins = [int(c.get("n", 20)) for c in conds_l + conds_s
                if str(c.get("f")) in ("brk", "mom")]
        wins += [int(c.get("slow", 50)) for c in conds_l + conds_s
                 if str(c.get("f")) == "marel"]
        min_bars = 80 + (max(wins) if wins else 0) + (60 if use_htf else 0)

        # ADX / marel 参数（句柄用）：取第一条出现的那组；出现不同参数则拒绝（句柄只有一套）
        _adxs = {(int(c.get("n", 14)), str(c.get("op", ">"))) for c in conds_l + conds_s
                 if str(c.get("f")) == "adx"}
        if len(_adxs) > 1:
            return MinedResult(ok=False, error="多个不同参数的 ADX 条件，暂不支持")
        _marels = {(int(c.get("fast", 20)), int(c.get("slow", 50)))
                   for c in conds_l + conds_s if str(c.get("f")) == "marel"}
        if len(_marels) > 1:
            return MinedResult(ok=False, error="多个不同参数的均线相对条件，暂不支持")
        adx_n = next((int(c.get("n", 14)) for c in conds_l + conds_s
                      if str(c.get("f")) == "adx"), 14)
        ema_f = next((int(c.get("fast", 20)) for c in conds_l + conds_s
                      if str(c.get("f")) == "marel"), 20)
        ema_s = next((int(c.get("slow", 50)) for c in conds_l + conds_s
                      if str(c.get("f")) == "marel"), 50)
        # EMA 参数作为 input 已在 reg 里；句柄要用同名 input
        f_in = next((n for t, n, v, c in reg.items if n.startswith("InpEmaFast")), str(ema_f))
        s_in = next((n for t, n, v, c in reg.items if n.startswith("InpEmaSlow")), str(ema_s))
        htf_tf = TF_ENUM[htf_list[0][1]] if use_htf else "PERIOD_H4"
        htf_n_in = (next((n for t, n, v, c in reg.items if n.startswith("InpHtfN")), "50")
                    if use_htf else "50")

        # 句柄声明 + OnInit 创建段：只用到的指标才生成（Python 端拼，不用 MQL5 #if）
        decls, oninit = [], []
        if use_adx:
            decls.append("int hAdx = INVALID_HANDLE;")
            oninit.append(f"   hAdx = iADX(_Symbol, _Period, {adx_n});")
            oninit.append("   if(hAdx == INVALID_HANDLE) return INIT_FAILED;")
        if use_marel:
            decls.append("int hEmaF = INVALID_HANDLE;")
            decls.append("int hEmaS = INVALID_HANDLE;")
            oninit.append(f"   hEmaF = iMA(_Symbol, _Period, {f_in}, 0, MODE_EMA, PRICE_CLOSE);")
            oninit.append(f"   hEmaS = iMA(_Symbol, _Period, {s_in}, 0, MODE_EMA, PRICE_CLOSE);")
            oninit.append("   if(hEmaF == INVALID_HANDLE || hEmaS == INVALID_HANDLE) return INIT_FAILED;")
        if use_rsi:
            decls.append("int hRsi = INVALID_HANDLE;")
            oninit.append("   hRsi = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);")
            oninit.append("   if(hRsi == INVALID_HANDLE) return INIT_FAILED;")
        if use_htf:
            decls.append("int hHtf = INVALID_HANDLE;")
            oninit.append(f"   hHtf = iMA(_Symbol, {htf_tf}, {htf_n_in}, 0, MODE_EMA, PRICE_CLOSE);")
            oninit.append("   if(hHtf == INVALID_HANDLE) return INIT_FAILED;")
        if not oninit:
            oninit.append("   // 本策略不需要指标句柄")

        rule_doc = "".join(f"//|  {i + 1}. {l}\n"
                           for i, l in enumerate(doc_lines or []))
        bt_note = meta.get("bt_note", "")
        grade = meta.get("grade", "")
        cand = meta.get("id", "")

        src = _SRC.substitute(
            name=name, magic=int(magic), rule_doc=rule_doc, cand=cand, grade=grade,
            symbol=spec.get("symbol", ""), tf=spec.get("tf", ""),
            family=meta.get("family", ""), bt_note=bt_note,
            allow_long=str(allow_long).lower(), allow_short=str(allow_short).lower(),
            risk_pct=_fmt(risk_pct), max_lots=_fmt(max_lots),
            sl_atr=_fmt(sl_atr), tp_atr=_fmt(tp_atr), chn=str(chn), mb=str(mb),
            min_bars=str(min_bars), strat_inputs=strat_inputs,
            handles_decl="\n".join(decls), oninit="\n".join(oninit),
            htf_tf=htf_tf,
            cond_long=fn_l, cond_short=fn_s,
        )
        if not use_htf:   # htf 函数引用 hHtf 句柄，未启用时整段裁掉
            src = re.sub(r"// 大周期 EMA：[^\n]*\n// 与回测[^\n]*\ndouble HtfEmaVal\(\)\n\{.*?\n\}\n\n",
                         "", src, flags=re.S)

        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{name}.mq5"
        path.write_text(src, encoding="utf-8")
        return MinedResult(ok=True, mq5_path=str(path), source=src)
    except Exception as e:  # noqa: BLE001
        return MinedResult(ok=False, error=f"{type(e).__name__}: {e}")


def _needs_rsi(conds_l, conds_s) -> bool:
    return any(str(c.get("f")) == "rsi" for c in list(conds_l) + list(conds_s))


# ---------------------------------------------------------------- MQL5 模板
from string import Template  # noqa: E402

_SRC = Template(r"""//+------------------------------------------------------------------+
//| $name.mq5 — 04 策略挖掘候选 $cand · 评级 $grade
//| 家族：$family · 标的：$symbol $tf（请挂到该品种/周期的空图表上）
//| 口径：t 收盘出信号、t+1 开盘市价执行；SL/TP 挂单由服务器盘中触发。
//| 出场：SL ${sl_atr}×ATR14 / TP ${tp_atr}×ATR14 / 通道${chn}根 / 限时${mb}根
//| 仓位：单笔风险 ${risk_pct}% 权益反推手数，上限 ${max_lots} 手
//|
//| ==== 规则（与 Python 回测同源，人工核对用）====
$rule_doc//|
//| $bt_note
//| 提示：回测不代表未来；先干跑（InpDryRun=true）再小仓；改动任何参数
//| 都等于换了一个策略，必须重新回测+重新跟踪。
//+------------------------------------------------------------------+
#property copyright "signal_desk miner"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

input group "执行保护"
input bool   InpDryRun     = true;     // 干跑：只打印与自报，不下单
input int    InpMagic      = $magic;   // 魔术号（防撞，改前查占用）
input int    InpSlippage   = 30;       // 滑点(点)
input int    InpMaxSpreadPoints = 40;  // 点差上限(点)，超过不开新仓
input double InpMaxDailyLossPct = 2.0; // 日内亏损停机(%权益)
input bool   InpReport     = true;     // 自报成绩单 CSV
input string InpReportTag  = "$name";

input group "仓位与风控"
input double InpRiskPct    = ${risk_pct}; // 单笔风险 % 权益
input double InpMaxLots    = ${max_lots}; // 手数上限
input bool   InpAllowLong  = ${allow_long};
input bool   InpAllowShort = ${allow_short};

input group "出场（默认=验证时取值）"
input double InpSlAtr      = ${sl_atr}; // 止损 × ATR14（0=不挂）
input double InpTpAtr      = ${tp_atr}; // 止盈 × ATR14（0=不挂）
input int    InpChanN      = ${chn};    // 通道离场根数（0=关）
input int    InpMaxBars    = ${mb};     // 限时离场根数（0=关）
$strat_inputs

$handles_decl
datetime gSigBar = 0;

// ---- 基础取值 ----
double CloseAt(int shift)
{
   double c[]; ArraySetAsSeries(c, true);
   if(CopyClose(_Symbol, _Period, shift, 1, c) != 1) return 0.0;
   return c[0];
}
// bar shift 之前 n 根（不含它自己）的最高/最低；失败返回 -1
double HiBefore(int shift, int n)
{
   if(n <= 0) return -1.0;
   double h[]; ArraySetAsSeries(h, true);
   if(CopyHigh(_Symbol, _Period, shift + 1, n, h) < n) return -1.0;
   return h[ArrayMaximum(h)];
}
double LoBefore(int shift, int n)
{
   if(n <= 0) return -1.0;
   double l[]; ArraySetAsSeries(l, true);
   if(CopyLow(_Symbol, _Period, shift + 1, n, l) < n) return -1.0;
   return l[ArrayMinimum(l)];
}
double SmaAt(int shift, int n)
{
   if(n <= 0) return 0.0;
   double c[]; ArraySetAsSeries(c, true);
   if(CopyClose(_Symbol, _Period, shift, n, c) < n) return 0.0;
   double s = 0.0;
   for(int i = 0; i < n; i++) s += c[i];
   return s / n;
}
// ATR14（SMA 口径，与回测数据桥一致；MT5 自带 iATR 是 Wilder 口径，别混用）
double AtrSma(int shift)
{
   int n = 14;
   double h[], l[], c[];
   ArraySetAsSeries(h, true); ArraySetAsSeries(l, true); ArraySetAsSeries(c, true);
   if(CopyHigh(_Symbol, _Period, shift, n + 1, h) < n + 1) return 0.0;
   if(CopyLow(_Symbol, _Period, shift, n + 1, l) < n + 1) return 0.0;
   if(CopyClose(_Symbol, _Period, shift, n + 1, c) < n + 1) return 0.0;
   double s = 0.0;
   for(int k = 0; k < n; k++)
   {
      double tr = MathMax(h[k] - l[k],
                  MathMax(MathAbs(h[k] - c[k + 1]), MathAbs(l[k] - c[k + 1])));
      s += tr;
   }
   return s / n;
}
// 大周期 EMA：取「决策时刻（新 bar 开盘）已收盘的最近大周期 bar」。
// 与回测 htf_trend 修正后的口径一致，无未来函数。
double HtfEmaVal()
{
   datetime T = iTime(_Symbol, _Period, 0);
   int k = iBarShift(_Symbol, $htf_tf, T, false);
   if(k < 0) return 0.0;
   double e[]; ArraySetAsSeries(e, true);
   if(CopyBuffer(hHtf, 0, k + 1, 1, e) != 1) return 0.0;
   return e[0];
}

// ---- 自报成绩单：ea_<tag>_<magic>_equity.csv（MT5 公共文件夹）----
void ReportState()
{
   if(!InpReport) return;
   static datetime lastBar = 0;
   datetime bt = iTime(_Symbol, _Period, 0);
   if(bt == lastBar) return;
   lastBar = bt;
   string fname = "ea_" + InpReportTag + "_" + IntegerToString(InpMagic) + "_equity.csv";
   static bool firstRun = true;
   if(firstRun) { firstRun = false; if(MQLInfoInteger(MQL_TESTER)) FileDelete(fname); }
   static int h = INVALID_HANDLE;
   if(h == INVALID_HANDLE)
   {
      h = FileOpen(fname, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI);
      if(h == INVALID_HANDLE)
         h = FileOpen(fname, FILE_COMMON | FILE_WRITE | FILE_TXT | FILE_ANSI);
      if(h == INVALID_HANDLE) return;
   }
   bool fresh = (FileSize(h) == 0);
   if(!fresh) FileSeek(h, 0, SEEK_END);
   int dir = 0; double lots = 0.0, price = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != (long)InpMagic) continue;
      dir = (int)PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : -1;
      lots = PositionGetDouble(POSITION_VOLUME);
      price = PositionGetDouble(POSITION_PRICE_CURRENT);
      break;
   }
   if(fresh) FileWrite(h, "time,equity,balance,position,lots,price");
   FileWrite(h, StringFormat("%s,%.2f,%.2f,%d,%.2f,%.2f",
             TimeToString(iTime(_Symbol, _Period, 0), TIME_DATE | TIME_MINUTES),
             AccountInfoDouble(ACCOUNT_EQUITY),
             AccountInfoDouble(ACCOUNT_BALANCE), dir, lots, price));
   FileClose(h);
}

bool HasPosition(int &dir)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != (long)InpMagic) continue;
      dir = (int)PositionGetInteger(POSITION_TYPE);
      return true;
   }
   dir = -1;
   return false;
}

bool SpreadOK()
{
   if(InpMaxSpreadPoints <= 0) return true;
   long sp = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   return sp <= InpMaxSpreadPoints;
}

// 日内亏损停机：与回测 daily_stop（-2% 当日不再开新仓）同口径
bool DailyLossOK()
{
   static datetime dayStart = 0;
   static double   eqStart  = 0.0;
   datetime today = iTime(_Symbol, PERIOD_D1, 0);
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(today != dayStart || eqStart <= 0.0)
   {
      dayStart = today; eqStart = eq;
      return true;
   }
   if(InpMaxDailyLossPct <= 0.0) return true;
   double drop = (eqStart - eq) / eqStart * 100.0;
   if(drop >= InpMaxDailyLossPct)
   {
      static datetime warned = 0;
      if(warned != today)
      {
         warned = today;
         PrintFormat("[风控] 日内亏损 %.2f%% 达上限 %.2f%%，今日停止开仓",
                     drop, InpMaxDailyLossPct);
      }
      return false;
   }
   return true;
}

// 从持仓注释恢复信号 bar 时间（"sd<时间戳"）；丢了就按入场时间倒推一根
datetime SigBarTime()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != (long)InpMagic) continue;
      string c = PositionGetString(POSITION_COMMENT);
      if(StringFind(c, "sd") == 0)
         return (datetime)StringToInteger(StringSubstr(c, 2));
      datetime pt = (datetime)PositionGetInteger(POSITION_TIME);
      return pt - PeriodSeconds(_Period);
   }
   return 0;
}

void CloseAll(string reason)
{
   if(InpDryRun)
   {
      PrintFormat("[dry-run] %s 信号成立，但 InpDryRun=true，未平仓", reason);
      return;
   }
   CTrade trade;
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippage);
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != (long)InpMagic) continue;
      trade.PositionClose(ticket);
      PrintFormat("[离场] %s", reason);
   }
}

// 持仓管理：通道离场 + 限时离场（SL/TP 由服务器盘中触发，不在这里管）
bool ManageExits(int dir)
{
   if(InpChanN <= 0 && InpMaxBars <= 0) return false;
   datetime sig = (gSigBar != 0) ? gSigBar : SigBarTime();
   if(sig == 0) return false;
   int held = (int)((iTime(_Symbol, _Period, 1) - sig) / PeriodSeconds(_Period));
   if(held < 0) held = 0;
   double c1 = CloseAt(1);
   if(InpChanN > 0 && held >= InpChanN && c1 > 0.0)
   {
      if(dir == POSITION_TYPE_BUY)
      {
         double ll = LoBefore(1, InpChanN);
         if(ll > 0.0 && c1 < ll) { CloseAll("通道离场"); return true; }
      }
      else
      {
         double hh = HiBefore(1, InpChanN);
         if(hh > 0.0 && c1 > hh) { CloseAll("通道离场"); return true; }
      }
   }
   if(InpMaxBars > 0 && held >= InpMaxBars)
   {
      CloseAll("限时离场");
      return true;
   }
   return false;
}

// ---- 入场条件（由候选规则逐条翻译，与 Python check() 同语义）----
$cond_long
$cond_short

void OpenTrade(int side, double atr)
{
   if(InpDryRun)
   {
      PrintFormat("[dry-run] %s 信号成立（ATR=%.2f），但 InpDryRun=true，未下单",
                  side > 0 ? "做多" : "做空", atr);
      return;
   }
   if(!SpreadOK()) { Print("[风控] 点差过大，跳过"); return; }
   double sl_dist = (InpSlAtr > 0.0) ? atr * InpSlAtr : 0.0;
   double tp_dist = (InpTpAtr > 0.0) ? atr * InpTpAtr : 0.0;
   double sizing  = (sl_dist > 0.0) ? sl_dist : atr * 3.0;   // 无止损时按 3ATR 估手数（与回测一致）
   double tick_val  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick_val <= 0.0 || tick_size <= 0.0 || sizing <= 0.0) return;
   double loss_per_lot = sizing / tick_size * tick_val;
   if(loss_per_lot <= 0.0) return;
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double mn   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(step <= 0.0) step = 0.01;
   if(mn   <= 0.0) mn = 0.01;
   double lots = MathFloor((eq * InpRiskPct / 100.0) / loss_per_lot / step) * step;
   if(lots > InpMaxLots) lots = MathFloor(InpMaxLots / step) * step;
   if(lots < mn)
   {
      Print("[风控] 风险预算算出的手数不足最小手，跳过");
      return;
   }
   CTrade trade;
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippage);
   double px = (side > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                          : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl = 0.0, tp = 0.0;
   if(sl_dist > 0.0) sl = (side > 0) ? px - sl_dist : px + sl_dist;
   if(tp_dist > 0.0) tp = (side > 0) ? px + tp_dist : px - tp_dist;
   gSigBar = iTime(_Symbol, _Period, 1);
   string cmt = "sd" + IntegerToString((long)gSigBar);
   if(side > 0) trade.Buy(lots, _Symbol, 0.0, sl, tp, cmt);
   else         trade.Sell(lots, _Symbol, 0.0, sl, tp, cmt);
   PrintFormat("[开仓] %s %.2f 手 @%.2f SL=%.2f TP=%.2f（信号根 ATR=%.2f）",
               side > 0 ? "买入" : "卖出", lots, px, sl, tp, atr);
}

//+------------------------------------------------------------------+
int OnInit()
{
   if(!MQLInfoInteger(MQL_TESTER) && !TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
      Print("[提示] 终端未开启算法交易，信号不会下单");
$oninit
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { }

void OnTick()
{
   if(Bars(_Symbol, _Period) < $min_bars) return;
   static datetime lastBar = 0;
   datetime bt = iTime(_Symbol, _Period, 0);
   if(bt == lastBar) return;          // 只在新 bar 上判定（bar 级口径）
   lastBar = bt;

   int dir;
   bool has = HasPosition(dir);
   if(has)
   {
      if(ManageExits(dir))
      {
         ReportState();
         return;                       // 本根刚离场：下一根再看新信号（与回测一致）
      }
   }

   int d2;
   if(!HasPosition(d2) && DailyLossOK())
   {
      double atr = AtrSma(1);          // 信号根（shift=1）的 ATR14
      if(atr > 0.0)
      {
         int side = 0;
         if(InpAllowLong && CondLong())       side = 1;    // 多头优先（与回测 entry_side 一致）
         else if(InpAllowShort && CondShort()) side = -1;
         if(side != 0) OpenTrade(side, atr);
      }
   }
   ReportState();
}
//+------------------------------------------------------------------+
""")
