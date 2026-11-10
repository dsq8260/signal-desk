"""把挖出来的策略生成 MQL5 EA 源码，并可调用 MetaEditor 编译成 .ex5。

三个必须遵守的 MQL5 事实（写错就编不过 / 跑不对）：
1. `iMA()` 等返回**句柄**，参数 6 个（无 shift），取值必须 `CopyBuffer(handle,0,shift,1,buf)`；
2. `iHighest/iLowest/iClose` 这些 MQL4 函数在 MQL5 里不存在，用 `CopyHigh/CopyLow/CopyClose`；
3. 句柄在 OnInit 建一次，OnTick 只 CopyBuffer（否则每 tick 建句柄会泄漏）。

本版本新增（与 Python 侧对齐）：
- **状态过滤层**：波动率分位过滤，默认关闭；
- **做空语义分离**：趋势类"信号反转即做空"，均值回归类"超买才做空"。
"""

from __future__ import annotations

import pathlib
import re
import subprocess
from dataclasses import dataclass

METAEDITOR = r"C:\Program Files\XM Global MT5\metaeditor64.exe"
EXPERTS_DIRS = [
    r"C:\Users\ych\AppData\Roaming\MetaQuotes\Terminal\BB16F565FAAA6B23A20C26C49416FF05\MQL5\Experts",
]

PREAMBLE = r"""
// --- 波动率分位过滤：当前 ATR 在过去 InpVolLookback 根里的百分位 <= InpVolMaxPct ---
bool VolFilterOK()
{
   if(!InpUseVolFilter) return true;
   // 每个 bar 只算一次：以前每个 tick 都 CopyBuffer(250)+循环，慢到不可用
   static datetime cachedBar = 0;
   static bool cachedOK = true;
   datetime bt = iTime(_Symbol, _Period, 0);
   if(bt == cachedBar) return cachedOK;
   cachedBar = bt;
   cachedOK = VolFilterCompute();
   return cachedOK;
}

bool VolFilterCompute()
{
   int h = iATR(_Symbol, _Period, 14);
   if(h == INVALID_HANDLE) return true;
   double buf[];
   int n = CopyBuffer(h, 0, 1, InpVolLookback, buf);
   if(n < 50) return true;
   double cur = buf[0];
   int cnt = 0;
   for(int i = 0; i < n; i++)
      if(buf[i] <= cur) cnt++;
   return (((double)cnt / (double)n) <= InpVolMaxPct);
}

// --- 强趋势过滤：|n 期收益| / (n 期平均 ATR%) 在过去 lookback 根里的分位 >= 阈值 ---
// 与 Python 侧 factors.trend_strength + regime_mask("trend_pct", 0.5, "above") 同口径：
//   trend = |close/close[n]-1| / mean(ATR14/close, n)，再取 rolling 250 根的分位。
// 只在**新 bar** 上重算（251×60 次算术，每根一次，可接受）。
bool TrendFilterOK()
{
   if(!InpUseTrendFilter) return true;
   static datetime cachedBar = 0;
   static bool cachedOK = true;
   datetime bt = iTime(_Symbol, _Period, 0);
   if(bt == cachedBar) return cachedOK;
   cachedBar = bt;

   int n   = MathMax(5, InpTrendN);            // 收益率回看（默认 60）
   int lb  = MathMax(20, InpTrendLookback);    // 分位回看（默认 250）
   int an  = MathMax(2, InpTrendAtrN);         // ATR 周期（默认 14）
   int need = lb + n + an + 2;

   double c[], a[];
   if(CopyClose(_Symbol, _Period, 1, need, c) != need) { cachedOK = true; return true; }
   int h = iATR(_Symbol, _Period, an);
   if(h == INVALID_HANDLE) { cachedOK = true; return true; }
   if(CopyBuffer(h, 0, 1, need, a) != need) { cachedOK = true; return true; }

   double t0 = 0.0;
   int cnt = 0, tot = 0;
   // 与 Python _pct_rank 对齐：窗口正好 lb 根（含当前），有效样本要求 >= lb/4
   for(int k = 0; k < lb; k++)
   {
      double base = c[k + n];
      if(base <= 0.0 || c[k + n - 1] <= 0.0) continue;
      double ret = MathAbs(c[k] / base - 1.0);
      double sum = 0.0;
      int m = 0;
      for(int j = k; j < k + n; j++)
      {
         double px = c[j];
         if(px <= 0.0 || a[j] <= 0.0) continue;
         sum += a[j] / px;      // ATR 占价格比例
         m++;
      }
      if(m == 0) continue;
      double trend = ret / (sum / (double)m);
      if(k == 0) t0 = trend;
      tot++;
      if(trend <= t0) cnt++;
   }
   if(tot < MathMax(20, lb / 4)) { cachedOK = true; return true; }
   cachedOK = (((double)cnt / (double)tot) >= InpTrendMinPct);
   return cachedOK;
}

bool FiltersOK() { return VolFilterOK() && TrendFilterOK(); }

// --- 自报成绩单：ea_<tag>_equity.csv ---
void ReportState()
{
   if(!InpReport) return;
   // 每根 bar 最多写一行；句柄只在第一次打开（以前每 tick 都 open/close）
   static datetime lastBar = 0;
   datetime bt = iTime(_Symbol, _Period, 0);
   if(bt == lastBar) return;
   lastBar = bt;

   // 文件名带 magic（多个 EA 不互相覆盖）；测试器里每次运行先清空，
   // 否则会跨回测追加，导致回撤/净值被算成脏数据。
   string fname = "ea_" + InpReportTag + "_" + IntegerToString(InpMagic)
                  + "_equity.csv";
   static bool firstRun = true;
   if(firstRun)
   {
      firstRun = false;
      if(MQLInfoInteger(MQL_TESTER)) FileDelete(fname);
   }
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

   int dir = 0;
   double lots = 0.0, price = 0.0;
   // 只统计自己 magic 的持仓（否则会把别的 EA / 手工单算进自己的净值曲线）
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != (long)InpMagic) continue;
      dir   = (int)PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : -1;
      lots  = PositionGetDouble(POSITION_VOLUME);
      price = PositionGetDouble(POSITION_PRICE_CURRENT);
      break;
   }

   if(fresh)
      FileWrite(h, "time,equity,balance,position,lots,price");

   FileWrite(h, StringFormat("%s,%.2f,%.2f,%d,%.2f,%.2f",
             TimeToString(iTime(_Symbol, _Period, 0), TIME_DATE | TIME_MINUTES),
             AccountInfoDouble(ACCOUNT_EQUITY),
             AccountInfoDouble(ACCOUNT_BALANCE),
             dir, lots, price));
   FileClose(h);
}

bool HasPosition(int &dir)
{
   // 只认**自己 magic** 的持仓：否则会把别的 EA / 用户手工单当成自己的，
   // 信号一翻就平掉别人的仓，界面上就出现"持仓一闪一闪"。
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

void CloseAll()
{
   if(InpDryRun)
   {
      Print("[dry-run] 想平仓，但 InpDryRun=true，未下单");
      ReportState();
      return;
   }
   CTrade trade;
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippage);
   // 只平自己的单，绝不碰别人的仓位
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != (long)InpMagic) continue;
      trade.PositionClose(ticket);
   }
   ReportState();
}

// 按百分比换算成绝对价位。点值用 SYMBOL_POINT，避免五位数报价的坑。
double PctToPrice(double price, double pct, int dir)
{
   if(pct <= 0.0) return 0.0;
   return NormalizeDouble(price * (1.0 + dir * pct), _Digits);
}


// 大周期确认：H4 收盘相对 span 根前是否同向（多头要求上行，空头要求下行）
bool HtfOK(int dir)
{
   if(!InpUseHtfFilter) return true;
   int span = MathMax(2, InpHtfSpan);
   double now  = iClose(_Symbol, PERIOD_H4, 1);
   double past = iClose(_Symbol, PERIOD_H4, 1 + span);
   if(now <= 0.0 || past <= 0.0) return true;      // 数据不足时不拦（与 Python 的 ffill 口径接近）
   bool up = (now > past);
   return (dir == POSITION_TYPE_BUY) ? up : (!up);
}

// 时间止损：持仓超过 N 根 bar 仍未回归就平仓（避免"扛太久"）
void ManageTimeStop()
{
   if(InpTimeStopBars <= 0) return;
   static datetime entryBar = 0;
   int dir;
   if(!HasPosition(dir)) { entryBar = 0; return; }     // 没仓就清零
   datetime nowBar = iTime(_Symbol, _Period, 0);
   if(entryBar == 0) { entryBar = nowBar; return; }
   int heldBars = (int)((nowBar - entryBar) / PeriodSeconds(_Period));
   if(heldBars >= InpTimeStopBars)
   {
      PrintFormat("[止损] 时间止损：持仓 %d 根 ≥ %d，平仓", heldBars,
                  InpTimeStopBars);
      CloseAll();
      entryBar = 0;
   }
}

// 移动止损：多头看持仓期间最高价，回撤 InpTrailPct 就平仓（空头反之）
void ManageTrailing()
{
   if(InpTrailPct <= 0.0) return;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != (long)InpMagic) continue;

      bool isBuy = (int)PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
      double open = PositionGetDouble(POSITION_PRICE_OPEN);
      double cur  = isBuy ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                          : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      // 用持仓期内的极值做基准（M1 高频下用当前 bar 的 high/low 近似）
      double ext = isBuy ? MathMax(cur, iHigh(_Symbol, _Period, 0))
                         : MathMin(cur, iLow(_Symbol, _Period, 0));
      static double bestUp[], bestDn[];
      int n = (int)PositionsTotal();
      if(ArraySize(bestUp) != n) { ArrayResize(bestUp, n); ArrayResize(bestDn, n); }
      if(isBuy)
      {
         if(bestUp[i] < ext) bestUp[i] = ext;
         if(bestUp[i] > 0 && cur <= bestUp[i] * (1.0 - InpTrailPct / 100.0))
         {
            CTrade tr; tr.SetExpertMagicNumber(InpMagic);
            tr.PositionClose(t);
            PrintFormat("[止损] 移动止损触发：最高 %.2f → 现价 %.2f", bestUp[i], cur);
         }
      }
      else
      {
         if(bestDn[i] <= 0 || bestDn[i] > ext) bestDn[i] = ext;
         if(bestDn[i] > 0 && cur >= bestDn[i] * (1.0 + InpTrailPct / 100.0))
         {
            CTrade tr; tr.SetExpertMagicNumber(InpMagic);
            tr.PositionClose(t);
            PrintFormat("[止损] 移动止损触发：最低 %.2f → 现价 %.2f", bestDn[i], cur);
         }
      }
   }
}

// 当前点差（点）；点差过大时不开新仓（新闻/换月时段常见）
bool SpreadOK()
{
   if(InpMaxSpreadPoints <= 0) return true;
   long sp = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   return sp <= InpMaxSpreadPoints;
}

// 按本次实际手数估算风险占比；超过上限直接拒单（保护小账户）
bool RiskOK(double lots, double price, double slPct)
{
   if(InpMaxRiskPct <= 0.0 || slPct <= 0.0) return true;
   double tick_val  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick_val <= 0.0 || tick_size <= 0.0) return true;
   double loss = (price * slPct) / tick_size * tick_val * lots;
   double eq   = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq <= 0.0) return true;
   double pct = loss / eq * 100.0;
   if(pct > InpMaxRiskPct)
   {
      PrintFormat("[风控] 拒单：单笔风险 %.2f%% > 上限 %.2f%%（手数 %.2f）",
                  pct, InpMaxRiskPct, lots);
      return false;
   }
   return true;
}

// 日内亏损停机：当天权益回撤超过阈值后，当日不再开新仓
bool DailyLossOK()
{
   static datetime dayStart = 0;
   static double  eqStart  = 0.0;
   datetime today = iTime(_Symbol, PERIOD_D1, 0);
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(today != dayStart || eqStart <= 0.0)
   {
      dayStart = today;
      eqStart  = eq;
      return true;
   }
   if(InpMaxDailyLossPct <= 0.0 || eqStart <= 0.0) return true;
   double drop = (eqStart - eq) / eqStart * 100.0;
   if(drop >= InpMaxDailyLossPct)
   {
      static datetime warned = 0;
      if(warned != today)
      {
         warned = today;
         PrintFormat("[风控] 日内亏损 %.2f%% 已达上限 %.2f%%，今日停止开仓",
                     drop, InpMaxDailyLossPct);
      }
      return false;
   }
   return true;
}

// 仓位换算：固定手数，或按"单笔风险% ÷ 止损距离"反推手数。
// 关键点：风险恒定 → 回撤不随账户规模变化（固定手数时，小账户会被 8%/笔 打爆）。
double LotsForTrade(double price, double slPct)
{
   if(InpLotMode == LOT_FIXED || slPct <= 0.0)
      return NormalizeLots(InpLots);

   double eq   = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk = eq * InpRiskPct / 100.0;             // 想亏多少钱
   double dist = price * slPct;                       // 止损距离（价格）
   double tick_val  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick_val <= 0.0 || tick_size <= 0.0 || dist <= 0.0)
      return NormalizeLots(InpLots);

   double loss_per_lot = dist / tick_size * tick_val; // 1 手亏多少
   if(loss_per_lot <= 0.0)
      return NormalizeLots(InpLots);
   return NormalizeLots(risk / loss_per_lot);
}

// 按品种的 min/step/max 规整手数，并夹在 [min, InpMaxLots]
double NormalizeLots(double lots)
{
   double mn   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double mx   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0.0) step = 0.01;
   if(mn  <= 0.0) mn = 0.01;
   if(mx  <= 0.0) mx = 100.0;
   double cap = MathMin(mx, InpMaxLots > 0.0 ? InpMaxLots : mx);
   double v = MathRound(lots / step) * step;
   if(v < mn) v = mn;
   if(v > cap) v = cap;
   return NormalizeDouble(v, 2);
}

void OpenTrade(int dir)
{
   if(InpDryRun)
   {
      PrintFormat("[dry-run] 想开仓 dir=%d，但 InpDryRun=true，未下单", dir);
      ReportState();
      return;
   }
   CTrade trade;
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippage);

   double px = (dir == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                          : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   // 同 bar 只开一次（防抖：被止损后同一根 bar 立刻重进）
   static datetime lastOpenBar = 0;
   datetime curBar = iTime(_Symbol, _Period, 0);
   if(InpOnePerBar && curBar == lastOpenBar)
   {
      Print("[风控] 同一根 bar 已开过仓，跳过");
      return;
   }
   if(!SpreadOK())   { Print("[风控] 点差过大，跳过");  return; }
   if(!HtfOK(dir))   { Print("[过滤] 大周期不同向，跳过"); return; }
   if(!DailyLossOK()){ return; }                    // 已在函数内打印

   double lots = LotsForTrade(px, InpSLPct);
   if(!RiskOK(lots, px, InpSLPct)) return;
   lastOpenBar = curBar;
   // 多头：止损在下方、止盈在上方；空头相反
   int sgn = (dir == POSITION_TYPE_BUY) ? -1 : 1;
   double sl = (InpSLPct > 0.0) ? PctToPrice(px, InpSLPct, sgn) : 0.0;
   double tp = (InpTPPct > 0.0) ? PctToPrice(px, InpTPPct, -sgn) : 0.0;

   if(dir == POSITION_TYPE_BUY)
      trade.Buy(lots, _Symbol, 0.0, sl, tp, "gold-ea");
   else
      trade.Sell(lots, _Symbol, 0.0, sl, tp, "gold-ea");
   ReportState();
}

// wantLong / wantShort 由模板给出，这里只管执行
void ApplySignal(bool wantLong, bool wantShort)
{
   int dir;
   bool has = HasPosition(dir);

   if(!FiltersOK())                 // 过滤不过：只平仓，不开新仓
   {
      if(has) CloseAll();
      return;
   }
   if(wantLong)
   {
      if(has && dir == POSITION_TYPE_SELL) CloseAll();
      if(InpAllowLong && (!has || dir != POSITION_TYPE_BUY)) OpenTrade(POSITION_TYPE_BUY);
      return;
   }
   if(wantShort && InpAllowShort)
   {
      if(has && dir == POSITION_TYPE_BUY) CloseAll();
      if(!has || dir != POSITION_TYPE_SELL) OpenTrade(POSITION_TYPE_SELL);
      return;
   }
   if(has) CloseAll();
}
"""

HEADER = """//+------------------------------------------------------------------+
//| {name}.mq5  —  由「黄金 EA 终端」自动生成
//| 策略：{strategy_cn}
//| 参数：{params_desc}
//| 口径：t 收盘出信号、下一根开盘执行。
//|
//| ==== 策略逻辑（自动生成，便于人工核对）====
{rule_doc}//|
//| 提示：回测不代表未来表现，请先在模拟账户试跑。
//+------------------------------------------------------------------+
#property copyright "gold-ea-terminal"
#property version   "1.20"
#property strict

#include <Trade\\Trade.mqh>

// 仓位模式：FIXED = 固定手数；PERCENT = 按权益百分比风险反推手数
enum LotMode {{ LOT_FIXED = 0, LOT_PERCENT = 1 }};
input LotMode InpLotMode   = LOT_FIXED; // 仓位模式
input double InpLots       = 0.01;      // 固定手数（InpLotMode=FIXED 时生效）
input double InpRiskPct    = 0.5;       // 单笔风险%（InpLotMode=PERCENT 时生效）
input double InpMaxLots    = 5.0;       // 手数上限（防爆）
// ---- 实盘保护（0 = 关闭该项）----
input int    InpMaxSpreadPoints   = 40;   // 点差上限(点)，超过不开新仓
input double InpMaxRiskPct        = 2.0;  // 单笔风险上限(%权益)，超过拒单
input double InpMaxDailyLossPct   = 5.0;  // 日内亏损停机(%权益)
input double InpTrailPct         = 0.0;
input int    InpTimeStopBars     = 0;    // 时间止损：持仓超过 N 根 bar 未回归就平仓（0=关闭）

input bool   InpUseHtfFilter      = false; // 大周期确认（H4 同向）
input int    InpHtfSpan           = 10;    // H4 跨度（根）
  // 移动止损(%): 0=关闭，从最高点回撤该比例即平仓

input bool   InpOnePerBar         = true; // 同一根 bar 只开一次
input double InpSLPct      = 0.0;       // 止损（百分比，0=不用）
input double InpTPPct      = 0.0;       // 止盈（百分比，0=不用）
input bool   InpDryRun     = true;      // 干跑：只记录与自报成绩，不下单
input int    InpMagic      = 20261028;  // 魔术号
input int    InpSlippage   = 30;        // 滑点(点)
input bool   InpAllowLong  = true;      // 允许做多
input bool   InpAllowShort = false;     // 允许做空
{strategy_inputs}

// ---- 状态过滤（默认全关）----
input bool   InpUseVolFilter = false;  // 启用波动率分位过滤
input int    InpVolLookback  = 250;    // 分位回看根数
input double InpVolMaxPct    = 0.70;   // 只在该分位以下才交易

input bool   InpUseTrendFilter = false; // 启用强趋势过滤
input int    InpTrendN         = 60;    // 趋势强度回看根数
input int    InpTrendLookback  = 250;   // 趋势分位回看根数
input int    InpTrendAtrN      = 14;    // ATR 周期
input double InpTrendMinPct    = 0.50;  // 趋势分位 >= 才交易

// ---- 自报成绩单（供系统评比，写进 MT5 公共文件夹）----
input bool   InpReport      = true;     // 写成绩单
input string InpReportTag   = "{report_tag}"; // 成绩单文件名标签
input int    InpReportSecs  = 60;       // 多少秒写一行

{globals}
{preamble}
//+------------------------------------------------------------------+
int OnInit()
{{
   if(!MQLInfoInteger(MQL_TESTER) && !TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
      Print("[提示] 终端未开启算法交易，信号不会下单");
   EventSetTimer(5);
   ReportState();
{oninit}
   return(INIT_SUCCEEDED);
}}

void OnDeinit(const int reason) {{ EventKillTimer(); }}
void OnTimer() {{ }}
void OnTick()
{{
   if(Bars(_Symbol, _Period) < {min_bars}) return;
   ManageTrailing();               // 移动止损每 tick 检查（不受新 bar 门限制）
   // 只在**新 bar** 上算信号：与 Python 的 bar 级回测同口径，
   // 也避免"每个即时价"模型下每 tick 都重算指标（这正是回测慢的主因）。
   static datetime lastBar = 0;
   datetime bt = iTime(_Symbol, _Period, 0);
   if(bt == lastBar) return;
   lastBar = bt;
{ontick}
   ApplySignal(wantLong, wantShort);
   ManageTimeStop();
   if(InpReport) ReportState();     // 每 bar 记一行，不随 tick 增长
}}
//+------------------------------------------------------------------+
//| 做空语义：{short_doc}
//+------------------------------------------------------------------+
"""

# ==========================================================================
# 合并模块：数据桥 + 图表可视化（with_bridge / visual 时注入生成源码）
# 写盘文件名与 GoldBridge.mq5 完全同口径，Python 端 gold/bridge.py 无需改动；
# 挂了合并版 EA 的图表可以不再单独挂 GoldBridge。
# ==========================================================================
MERGED_INPUTS = """
// ---- 合并：数据桥 + 可视化（挂了本 EA 可不再单独挂 GoldBridge）----
input bool   InpBridgeOn     = false;  // 合并数据桥：tick+K线+账户/持仓/成交 落盘
input int    InpBrgDumpSecs    = 30;   // K 线重写周期（秒）
input bool   InpBrgWriteTick   = true; // 写实时 tick
input string InpBrgSymbols     = "";   // 额外导出品种（逗号分隔；留空=仅图表品种）
input int    InpBrgAccountSecs = 5;    // 账户/持仓/成交 写盘周期（秒）
input int    InpBrgDealDays    = 90;   // 成交历史回看天数
input bool   InpVisualOn     = false;  // 图表可视化：信号面板 + 买卖箭头
"""

MERGED_CODE = r"""
//+------------------------------------------------------------------+
//| 合并模块 A：数据桥（与 GoldBridge.mq5 同口径）                     |
//+------------------------------------------------------------------+
const int BRG_PAGE_BARS = 20000;
const int BRG_PAGES     = 6;
int      gBrgTicks    = 0;
datetime gBrgLastBars = 0;
datetime gBrgLastAcc  = 0;

string BrgSafeName(string s)
{
   string out = "";
   int len = StringLen(s);
   for(int i = 0; i < len; i++)
   {
      ushort ch = StringGetCharacter(s, i);
      bool ok = ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
                 (ch >= '0' && ch <= '9') || ch == '_');
      if(ok) out += ShortToString(ch);
   }
   return (out == "") ? "SYM" : out;
}

int BrgParseSymbols(string &list[])
{
   string parts[];
   int n = StringSplit(InpBrgSymbols, ',', parts);
   ArrayResize(list, 0);
   for(int i = 0; i < n; i++)
   {
      string s = parts[i];
      StringTrimLeft(s);
      StringTrimRight(s);
      if(StringLen(s) == 0) continue;
      int idx = ArraySize(list);
      ArrayResize(list, idx + 1);
      list[idx] = s;
   }
   if(ArraySize(list) == 0)
   {
      ArrayResize(list, 1);
      list[0] = _Symbol;
   }
   return ArraySize(list);
}

void BrgWriteTick(string sym)
{
   if(!InpBridgeOn || !InpBrgWriteTick) return;
   MqlTick t;
   if(!SymbolInfoTick(sym, t)) return;
   string fname = "gold_tick_" + BrgSafeName(sym) + ".csv";
   int h = FileOpen(fname, FILE_COMMON | FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE)
   {
      h = FileOpen(fname, FILE_COMMON | FILE_WRITE | FILE_TXT | FILE_ANSI);
      if(h == INVALID_HANDLE) return;
   }
   if(FileSize(h) > 4 * 1024 * 1024)
   {
      FileClose(h);
      FileDelete(fname, FILE_COMMON);
      h = FileOpen(fname, FILE_COMMON | FILE_WRITE | FILE_TXT | FILE_ANSI);
      if(h == INVALID_HANDLE) return;
   }
   else
      FileSeek(h, 0, SEEK_END);
   FileWrite(h, StringFormat("%I64d,%.5f,%.5f", (long)t.time_msc, t.bid, t.ask));
   FileClose(h);
}

void BrgDumpAccount()
{
   int h = FileOpen("mt5_account.csv", FILE_COMMON | FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE) return;
   FileWrite(h, "time,login,server,company,currency,balance,equity,margin,margin_free,margin_level,profit,leverage");
   FileWrite(h, StringFormat("%s,%I64d,%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%I64d",
      TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES),
      AccountInfoInteger(ACCOUNT_LOGIN),
      AccountInfoString(ACCOUNT_SERVER),
      AccountInfoString(ACCOUNT_COMPANY),
      AccountInfoString(ACCOUNT_CURRENCY),
      AccountInfoDouble(ACCOUNT_BALANCE),
      AccountInfoDouble(ACCOUNT_EQUITY),
      AccountInfoDouble(ACCOUNT_MARGIN),
      AccountInfoDouble(ACCOUNT_MARGIN_FREE),
      AccountInfoDouble(ACCOUNT_MARGIN_LEVEL),
      AccountInfoDouble(ACCOUNT_PROFIT),
      AccountInfoInteger(ACCOUNT_LEVERAGE)));
   FileClose(h);
}

void BrgDumpPositions()
{
   int h = FileOpen("mt5_positions.csv", FILE_COMMON | FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE) return;
   FileWrite(h, "ticket,symbol,type,volume,price_open,price_current,profit,swap,magic,comment");
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(!PositionSelectByTicket(tk)) continue;
      FileWrite(h, StringFormat("%I64d,%s,%d,%.2f,%.5f,%.5f,%.2f,%.2f,%I64d,%s",
         (long)tk,
         PositionGetString(POSITION_SYMBOL),
         (int)PositionGetInteger(POSITION_TYPE),
         PositionGetDouble(POSITION_VOLUME),
         PositionGetDouble(POSITION_PRICE_OPEN),
         PositionGetDouble(POSITION_PRICE_CURRENT),
         PositionGetDouble(POSITION_PROFIT),
         PositionGetDouble(POSITION_SWAP),
         PositionGetInteger(POSITION_MAGIC),
         PositionGetString(POSITION_COMMENT)));
   }
   FileClose(h);
}

void BrgDumpDeals()
{
   datetime from = TimeCurrent() - (datetime)InpBrgDealDays * 86400;
   if(!HistorySelect(from, TimeCurrent())) return;
   int h = FileOpen("mt5_deals.csv", FILE_COMMON | FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE) return;
   FileWrite(h, "time,ticket,symbol,type,entry,volume,price,profit,swap,commission,magic,comment");
   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong tk = HistoryDealGetTicket(i);
      if(tk == 0) continue;
      FileWrite(h, StringFormat("%s,%I64d,%s,%d,%d,%.2f,%.5f,%.2f,%.2f,%.2f,%I64d,%s",
         TimeToString((datetime)HistoryDealGetInteger(tk, DEAL_TIME), TIME_DATE | TIME_MINUTES),
         (long)tk,
         HistoryDealGetString(tk, DEAL_SYMBOL),
         (int)HistoryDealGetInteger(tk, DEAL_TYPE),
         (int)HistoryDealGetInteger(tk, DEAL_ENTRY),
         HistoryDealGetDouble(tk, DEAL_VOLUME),
         HistoryDealGetDouble(tk, DEAL_PRICE),
         HistoryDealGetDouble(tk, DEAL_PROFIT),
         HistoryDealGetDouble(tk, DEAL_SWAP),
         HistoryDealGetDouble(tk, DEAL_COMMISSION),
         HistoryDealGetInteger(tk, DEAL_MAGIC),
         HistoryDealGetString(tk, DEAL_COMMENT)));
   }
   FileClose(h);
}

void BrgDumpAll()
{
   string syms[];
   int ns = BrgParseSymbols(syms);
   string tfs[];
   ENUM_TIMEFRAMES periods[];
   ArrayResize(tfs, 8);
   ArrayResize(periods, 8);
   tfs[0]="M1";  periods[0]=PERIOD_M1;
   tfs[1]="M5";  periods[1]=PERIOD_M5;
   tfs[2]="M15"; periods[2]=PERIOD_M15;
   tfs[3]="M30"; periods[3]=PERIOD_M30;
   tfs[4]="H1";  periods[4]=PERIOD_H1;
   tfs[5]="H4";  periods[5]=PERIOD_H4;
   tfs[6]="D1";  periods[6]=PERIOD_D1;
   tfs[7]="W1";  periods[7]=PERIOD_W1;

   for(int s = 0; s < ns; s++)
   {
      string sym = syms[s];
      if(!SymbolSelect(sym, true)) continue;
      int total = 0;
      for(int i = 0; i < ArraySize(tfs); i++)
      {
         datetime endT = TimeCurrent() + 60;
         long secs = (long)PeriodSeconds(periods[i]);
         for(int pg = 0; pg < BRG_PAGES; pg++)
         {
            MqlRates r[];
            int n = 0;
            if(pg == 0)
               n = CopyRates(sym, periods[i], 0, BRG_PAGE_BARS, r);
            else
            {
               datetime startT = endT - (datetime)(secs * BRG_PAGE_BARS);
               n = CopyRates(sym, periods[i], startT, endT, r);
            }
            if(n <= 0) break;
            string fname = "gold_bars_" + BrgSafeName(sym) + "_" + tfs[i]
                           + ((pg == 0) ? "" : "_p" + IntegerToString(pg)) + ".csv";
            int h = FileOpen(fname, FILE_COMMON | FILE_WRITE | FILE_TXT | FILE_ANSI);
            if(h == INVALID_HANDLE) break;
            FileWrite(h, "time,open,high,low,close,tick_volume");
            for(int k = 0; k < n; k++)
               FileWrite(h, StringFormat("%I64d,%.5f,%.5f,%.5f,%.5f,%I64d",
                        (long)r[k].time, r[k].open, r[k].high,
                        r[k].low, r[k].close, (long)r[k].tick_volume));
            FileClose(h);
            total++;
            endT = r[0].time - 1;
            if(n < BRG_PAGE_BARS - 1) break;
         }
      }
      PrintFormat("[Brg] %s → %d 个周期已落盘", sym, total);
   }
}

void BrgInit()
{
   if(!InpBridgeOn) return;
   BrgDumpAll();
   BrgDumpAccount();
   BrgDumpPositions();
   BrgDumpDeals();
   gBrgLastBars = TimeCurrent();
   gBrgLastAcc  = TimeCurrent();
   PrintFormat("[Brg] 合并数据桥已启动：图表=%s 额外品种=[%s]", _Symbol, InpBrgSymbols);
}

void BrgTimer()
{
   if(!InpBridgeOn) return;
   gBrgTicks++;
   if(gBrgTicks % MathMax(1, InpBrgDumpSecs) == 0)
   {
      BrgDumpAll();
      gBrgLastBars = TimeCurrent();
   }
   if(gBrgTicks % MathMax(1, InpBrgAccountSecs) == 0)
   {
      BrgDumpAccount();
      BrgDumpPositions();
      BrgDumpDeals();
      gBrgLastAcc = TimeCurrent();
   }
}

void BrgDeinit()
{
   if(InpBridgeOn) Print("[Brg] 合并数据桥停止");
}

//+------------------------------------------------------------------+
//| 合并模块 B：图表可视化（Comment 信号面板 + 买卖箭头）              |
//+------------------------------------------------------------------+
bool     gVisWantLong  = false;
bool     gVisWantShort = false;
string   gVisLastSig   = "还没有信号";

void VisInit()
{
   if(!InpVisualOn) return;
   VisPanel();
}

void VisTimer()
{
   if(InpVisualOn) VisPanel();   // 1s 定时刷新盘口/权益
}

void VisDeinit()
{
   if(!InpVisualOn) return;
   Comment("");
   ObjectsDeleteAll(0, "GVS_");
}

// 开仓意图（含干跑）打一枚箭头，肉眼回看信号位置
void VisMark(int dir)
{
   if(!InpVisualOn) return;
   datetime t = TimeCurrent();
   double px = (dir == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                          : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   string name = "GVS_" + IntegerToString(InpMagic) + "_" + IntegerToString((long)t);
   if(ObjectFind(0, name) < 0)
   {
      ObjectCreate(0, name, (dir == POSITION_TYPE_BUY) ? OBJ_ARROW_UP : OBJ_ARROW_DOWN, 0, t, px);
      // 中国习惯：做多=红（涨）、做空=绿（跌），与 SIGNAL DESK 页面配色一致
      ObjectSetInteger(0, name, OBJPROP_COLOR,
                       (dir == POSITION_TYPE_BUY) ? clrOrangeRed : clrSpringGreen);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
   }
   gVisLastSig = ((dir == POSITION_TYPE_BUY) ? "▲ 做多信号 @ " : "▼ 做空信号 @ ")
                 + DoubleToString(px, 2) + "   " + TimeToString(t, TIME_DATE | TIME_MINUTES);
}

// 面板配色（与 SIGNAL DESK 主题一致）
#define GVS_BG        C'16,21,28'
#define GVS_BORDER    C'39,49,61'
#define GVS_TITLE     C'255,200,80'
#define GVS_TXT       C'230,232,235'
#define GVS_DIM       C'120,128,140'
#define GVS_GOOD      C'63,201,138'
#define GVS_BAD       C'240,113,107'

void VisBg(int rows)
{
   string name = "GVS_bg";
   if(ObjectFind(0, name) < 0)
   {
      ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, name, OBJPROP_BGCOLOR, GVS_BG);
      ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, name, OBJPROP_COLOR, GVS_BORDER);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
      ObjectSetInteger(0, name, OBJPROP_BACK, false);
      ObjectSetInteger(0, name, OBJPROP_ZORDER, 0);
   }
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, 10);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, 344);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, rows * 24 + 12);
}

// 一行文字：idx 行号（0=标题）、clr 颜色、size 字号
void VisRow(int idx, string txt, color clr, int size)
{
   string name = "GVS_r" + IntegerToString(idx);
   if(ObjectFind(0, name) < 0)
   {
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
      ObjectSetString(0, name, OBJPROP_FONT, "Microsoft YaHei UI");
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
      ObjectSetInteger(0, name, OBJPROP_ZORDER, 1);
   }
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 24);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, 20 + idx * 24);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, size);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetString(0, name, OBJPROP_TEXT, txt);
}

void VisPanel()
{
   if(!InpVisualOn) return;
   /* 底板必须先于文字行创建：MT5 图表对象按创建顺序绘制，后建的在上层。
      若底板后建会把整版文字盖住（只见黑框不见字）——2026-11-10 修复 */
   VisBg(7);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   long   spr = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);

   // 只认自己 magic 的持仓（与 HasPosition 同口径）
   bool has = false; int dir = 0;
   double vol = 0, opn = 0, prof = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      has = true;
      dir = (int)PositionGetInteger(POSITION_TYPE);
      vol = PositionGetDouble(POSITION_VOLUME);
      opn = PositionGetDouble(POSITION_PRICE_OPEN);
      prof = PositionGetDouble(POSITION_PROFIT);
      break;
   }

   // 行 0：标题
   VisRow(0, "◆ " + MQLInfoString(MQL_PROGRAM_NAME), GVS_TITLE, 11);

   // 行 1：本 bar 信号（有信号高亮，无信号灰）
   string sig = "";
   color  sigc = GVS_DIM;
   if(gVisWantLong)  { sig += "▲做多 "; sigc = GVS_GOOD; }
   if(gVisWantShort) { sig += "▼做空 "; sigc = (sigc == GVS_GOOD) ? GVS_TXT : GVS_BAD; }
   if(sig == "") sig = "○ 观望";
   VisRow(1, "信号  " + sig + "    允许 多" + (InpAllowLong ? "✓" : "✗")
            + " 空" + (InpAllowShort ? "✓" : "✗"), sigc, 10);

   // 行 2：持仓（按浮动盈亏配色）
   if(has)
   {
      color pc = (prof >= 0) ? GVS_GOOD : GVS_BAD;
      VisRow(2, ((dir == POSITION_TYPE_BUY) ? "多单 " : "空单 ")
            + DoubleToString(vol, 2) + " 手 @ " + DoubleToString(opn, 2)
            + "   浮盈 " + (prof >= 0 ? "+" : "") + DoubleToString(prof, 2), pc, 10);
   }
   else
      VisRow(2, "持仓  无", GVS_DIM, 10);

   // 行 3：盘口
   VisRow(3, "买 " + DoubleToString(bid, 2) + " / 卖 " + DoubleToString(ask, 2)
            + " · 点差 " + IntegerToString((int)spr) + " 点", GVS_TXT, 10);

   // 行 4：账户（按浮动配色）
   double fl = AccountInfoDouble(ACCOUNT_PROFIT);
   color  fc = (fl >= 0) ? GVS_GOOD : GVS_BAD;
   VisRow(4, "权益 " + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2)
            + " · 余额 " + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2)
            + " · 浮动 " + (fl >= 0 ? "+" : "") + DoubleToString(fl, 2), fc, 10);

   // 行 5：最近信号
   VisRow(5, "最近 " + gVisLastSig, GVS_DIM, 9);

   // 行 6：运行状态
   VisRow(6, (InpDryRun ? "干跑 ✓ 只自报不下单" : "实盘 ✗ 真实下单")
            + " · 数据桥 " + (InpBridgeOn ? "✓" : "✗")
            + " · K线@" + TimeToString(gBrgLastBars, TIME_MINUTES)
            + " · 账户@" + TimeToString(gBrgLastAcc, TIME_MINUTES), GVS_DIM, 9);

}
"""


@dataclass
class GenResult:
    ok: bool = False
    mq5_path: str = ""
    ex5_path: str = ""
    source: str = ""
    error: str = ""
    log: str = ""


def _inputs(params: dict, mapping: dict[str, str]) -> str:
    lines = []
    for k, v in params.items():
        mql = mapping.get(k, k)
        if isinstance(v, float):
            lines.append(f"input double Inp{mql} = {v:.2f};   // {k}")
        else:
            lines.append(f"input int    Inp{mql} = {int(v)};      // {k}")
    return "\n".join(lines)


SPECS = {
    "双均线": {
        "params": {"fast": ("FastMA", 20), "slow": ("SlowMA", 60)},
        "globals": "int hFast = INVALID_HANDLE;\nint hSlow = INVALID_HANDLE;",
        "oninit": """   hFast = iMA(_Symbol, _Period, InpFastMA, 0, MODE_SMA, PRICE_CLOSE);
   hSlow = iMA(_Symbol, _Period, InpSlowMA, 0, MODE_SMA, PRICE_CLOSE);
   if(hFast == INVALID_HANDLE || hSlow == INVALID_HANDLE) return(INIT_FAILED);""",
        "ontick": """   double f[], s[];
   if(CopyBuffer(hFast, 0, 1, 1, f) != 1) return;
   if(CopyBuffer(hSlow, 0, 1, 1, s) != 1) return;
   bool wantLong  = (f[0] > s[0]);
   bool wantShort = (f[0] < s[0]);""",
        "min_bars": "{slow} + 5",
        "short_doc": "趋势类：快线在慢线下方即做空（与 Python both=True 一致）",
    },
    "唐奇安通道突破": {
        "params": {"entry": ("Entry", 20), "exit_": ("Exit", 10)},
        "globals": "bool gInPos = false;\nbool gInShort = false;",
        "oninit": "   gInPos = false; gInShort = false;",
        "ontick": """   double hi[], lo[], cl[];
   // 从 shift=2 取：窗口必须**排除**当前判断的 bar1，否则
   // max(high) 含 bar1 自己，close>max 永远不成立 → EA 一单都不开。
   if(CopyHigh(_Symbol, _Period, 2, InpEntry, hi) < InpEntry) return;
   if(CopyLow(_Symbol, _Period, 2, InpExit, lo) < InpExit) return;
   if(CopyClose(_Symbol, _Period, 1, 1, cl) != 1) return;
   double hh = hi[ArrayMaximum(hi)];
   double ll = lo[ArrayMinimum(lo)];
   if(!gInPos && cl[0] > hh)       { gInPos = true;   gInShort = false; }
   else if(!gInShort && cl[0] < ll){ gInShort = true; gInPos = false; }
   bool wantLong  = gInPos;
   bool wantShort = gInShort;""",
        "min_bars": "{entry} + 5",
        "short_doc": "趋势类：跌破 exit_ 日低点即做空",
    },
    "ATR通道突破": {
        "params": {"n": ("N", 20), "m": ("M", 14), "k": ("K", 2.0),
                   "confirm": ("Confirm", 1)},
        "globals": "int hMaCh = INVALID_HANDLE;\nint hAtrCh = INVALID_HANDLE;\nbool gChPos = false;",
        "oninit": """   hMaCh = iMA(_Symbol, _Period, InpN, 0, MODE_SMA, PRICE_CLOSE);
   hAtrCh = iATR(_Symbol, _Period, InpM);
   gChPos = false;""",
        "ontick": """   int cnf = (int)MathMax(1, InpConfirm);
   if(cnf > 8) cnf = 8;
   double md[], tr[], cl[];
   if(CopyBuffer(hMaCh, 0, 1, cnf, md) != cnf) return;
   if(CopyBuffer(hAtrCh, 0, 1, cnf, tr) != cnf) return;
   if(CopyClose(_Symbol, _Period, 1, cnf, cl) != cnf) return;
   // 连续 cnf 根都收在上轨之外才算突破（confirm=1 时退化为单根判断）
   bool allAbove = true, allBelow = true;
   for(int i = 0; i < cnf; i++)
   {
      double up = md[i] + InpK * tr[i];
      double dn = md[i] - InpK * tr[i];
      if(!(cl[i] > up)) allAbove = false;
      if(!(cl[i] < dn)) allBelow = false;
   }
   double midNow = md[0];
   if(!gChPos && allAbove) gChPos = true;            // 连续站上上轨 → 做多
   else if(gChPos && cl[0] < midNow) gChPos = false; // 回到中轨 → 出场
   bool wantLong  = gChPos;
   bool wantShort = false;""",
        "min_bars": "{n} + {m} + 5",
        "short_doc": "顺势突破：ATR 自适应通道，站上上轨做多、回到中轨出场",
    },
    "均线偏离回归": {
        "params": {"ma_n": ("MaN", 10), "entry_thr": ("EntryThr", 100.0),
                   "exit_thr": ("ExitThr", 100.0)},
        "globals": "int hMaDev = INVALID_HANDLE;\nint hSdDev = INVALID_HANDLE;",
        "oninit": """   hMaDev = iMA(_Symbol, _Period, InpMaN, 0, MODE_SMA, PRICE_CLOSE);
   hSdDev = iStdDev(_Symbol, _Period, InpMaN, 0, MODE_SMA, PRICE_CLOSE);""",
        "ontick": """   double m[], sd[], cl[];
   if(CopyBuffer(hMaDev, 0, 1, 1, m) != 1) return;
   if(CopyBuffer(hSdDev, 0, 1, 1, sd) != 1) return;
   if(CopyClose(_Symbol, _Period, 1, 2, cl) != 2) return;
   if(sd[0] <= 0.0) return;
   double dev = (cl[0] - m[0]) / sd[0] * 100.0;   // 偏离度（以标准差为单位）
   double devPrev = (cl[1] - m[0]) / sd[0] * 100.0;
   static bool inPos = false;
   if(!inPos && devPrev < -InpEntryThr && dev >= -InpEntryThr) inPos = true;
   else if(inPos && dev >= InpExitThr) inPos = false;
   bool wantLong  = inPos;
   bool wantShort = false;""",
        "min_bars": "{ma_n} + 5",
        "short_doc": "均值回归：偏离均线 1σ 以上后反向做，回到均值了结",
    },
    "MACD金叉": {
        "params": {"fast": ("FastEMA", 12), "slow": ("SlowEMA", 26),
                   "signal": ("SignalSMA", 9)},
        "globals": "int hMacd = INVALID_HANDLE;",
        "oninit": """   hMacd = iMACD(_Symbol, _Period, InpFastEMA, InpSlowEMA,
                 InpSignalSMA, PRICE_CLOSE);
   if(hMacd == INVALID_HANDLE) return(INIT_FAILED);""",
        "ontick": """   double m[], sg[];
   if(CopyBuffer(hMacd, 0, 1, 1, m) != 1) return;
   if(CopyBuffer(hMacd, 1, 1, 1, sg) != 1) return;
   bool wantLong  = (m[0] > sg[0]);
   bool wantShort = (m[0] < sg[0]);""",
        "min_bars": "{slow} + 10",
        "short_doc": "趋势类：DIF 在 DEA 下方即做空",
    },
    "RSI反转": {
        "params": {"n": ("RsiN", 14), "buy": ("BuyLevel", 30.0),
                   "sell": ("SellLevel", 55.0)},
        "globals": "int hRsi = INVALID_HANDLE;\nbool gInPos = false;\nbool gInShort = false;",
        "oninit": """   hRsi = iRSI(_Symbol, _Period, InpRsiN, PRICE_CLOSE);
   if(hRsi == INVALID_HANDLE) return(INIT_FAILED);
   gInPos = false; gInShort = false;""",
        "ontick": """   double r[];
   if(CopyBuffer(hRsi, 0, 1, 1, r) != 1) return;
   double overbought = 100.0 - InpBuyLevel;
   double cover      = 100.0 - InpSellLevel;
   if(!gInPos && !gInShort)
   {
      if(r[0] < InpBuyLevel)     gInPos = true;
      else if(r[0] > overbought) gInShort = true;
   }
   else if(gInPos   && r[0] > InpSellLevel) gInPos = false;
   else if(gInShort && r[0] < cover)        gInShort = false;
   bool wantLong  = gInPos;
   bool wantShort = gInShort;""",
        "min_bars": "{n} + 10",
        "short_doc": "均值回归类：RSI 超过 70 才做空，回到 45 平仓（不是信号消失就做空）",
    },
    "布林带回归": {
        "params": {"n": ("BandsN", 20), "k": ("Deviation", 2.0)},
        "globals": "int hBands = INVALID_HANDLE;\nbool gInPos = false;\nbool gInShort = false;",
        "oninit": """   hBands = iBands(_Symbol, _Period, InpBandsN, 0, InpDeviation, PRICE_CLOSE);
   if(hBands == INVALID_HANDLE) return(INIT_FAILED);
   gInPos = false; gInShort = false;""",
        "ontick": """   double up[], lo[], mid[], cl[];
   if(CopyBuffer(hBands, 1, 1, 1, up)  != 1) return;
   if(CopyBuffer(hBands, 2, 1, 1, lo)  != 1) return;
   if(CopyBuffer(hBands, 0, 1, 1, mid) != 1) return;
   if(CopyClose(_Symbol, _Period, 1, 1, cl) != 1) return;
   if(!gInPos && !gInShort)
   {
      if(cl[0] < lo[0])      gInPos = true;
      else if(cl[0] > up[0]) gInShort = true;
   }
   else if(gInPos   && cl[0] > mid[0]) gInPos = false;
   else if(gInShort && cl[0] < mid[0]) gInShort = false;
   bool wantLong  = gInPos;
   bool wantShort = gInShort;""",
        "min_bars": "{n} + 10",
        "short_doc": "均值回归类：突破上轨才做空，回到中轨平仓",
    },
    "ATR跟踪止损": {
        "params": {"n": ("AtrN", 14), "mult": ("AtrMult", 3.0),
                   "entry_ma": ("EntryMA", 20)},
        "globals": ("int hAtr = INVALID_HANDLE;\nint hMa  = INVALID_HANDLE;\n"
                    "bool gInPos = false;\nbool gInShort = false;\n"
                    "double gPeak = 0.0;\ndouble gTrough = 0.0;"),
        "oninit": """   hAtr = iATR(_Symbol, _Period, InpAtrN);
   hMa  = iMA(_Symbol, _Period, InpEntryMA, 0, MODE_SMA, PRICE_CLOSE);
   if(hAtr == INVALID_HANDLE || hMa == INVALID_HANDLE) return(INIT_FAILED);
   gInPos = false; gInShort = false; gPeak = 0.0; gTrough = 0.0;""",
        "ontick": """   double a[], m[], cl[];
   if(CopyBuffer(hAtr, 0, 1, 1, a) != 1) return;
   if(CopyBuffer(hMa,  0, 1, 1, m) != 1) return;
   if(CopyClose(_Symbol, _Period, 1, 1, cl) != 1) return;
   double c = cl[0];
   if(!gInPos && !gInShort)
   {
      if(c > m[0])      { gInPos = true;   gPeak = c; }
      else if(c < m[0]) { gInShort = true; gTrough = c; }
   }
   else if(gInPos)
   {
      if(c > gPeak) gPeak = c;
      if(a[0] > 0.0 && c < gPeak - InpAtrMult * a[0]) gInPos = false;
   }
   else
   {
      if(c < gTrough) gTrough = c;
      if(a[0] > 0.0 && c > gTrough + InpAtrMult * a[0]) gInShort = false;
   }
   bool wantLong  = gInPos;
   bool wantShort = gInShort;""",
        "min_bars": "{entry_ma} + 20",
        "short_doc": "趋势类：收盘在均线下方开空，涨破 低点+k×ATR 平仓",
    },
    "LLT上穿": {
        "params": {"n": ("LltN", 60)},
        "globals": "datetime gLastBar = 0;\nbool gWantLong = false;\nbool gWantShort = false;",
        "oninit": "   gLastBar = 0; gWantLong = false; gWantShort = false;",
        "ontick": """   datetime t = iTime(_Symbol, _Period, 0);
   if(t != gLastBar)
   {
      gLastBar = t;
      int need = MathMax(InpLltN * 3, 120);
      double p[];
      if(CopyClose(_Symbol, _Period, 1, need, p) != need) return;
      double a  = 2.0 / (InpLltN + 1.0);
      double l0 = p[0], l1 = p[1];
      for(int i = 2; i < need; i++)
      {
         double v = (a - a*a/4.0) * p[i]
                  + (a*a/2.0) * p[i-1]
                  - (a - 3.0*a*a/4.0) * p[i-2]
                  + 2.0*(1.0-a) * l1
                  - (1.0-a)*(1.0-a) * l0;
         l0 = l1; l1 = v;
      }
      gWantLong  = (p[need-1] > l1);
      gWantShort = (p[need-1] < l1);
   }
   bool wantLong  = gWantLong;
   bool wantShort = gWantShort;""",
        "min_bars": "120",
        "short_doc": "趋势类：收盘在 LLT 线下方即做空",
    },
    "RSRS择时": {
        "params": {"n": ("RsrsN", 18), "m": ("RsrsM", 250),
                   "upper": ("Upper", 0.7), "lower": ("Lower", -0.7)},
        "globals": ("datetime gLastBar = 0;\nbool gWantLong = false;\n"
                    "bool gWantShort = false;\nbool gInPos = false;\nbool gInShort = false;"),
        "oninit": """   gLastBar = 0; gWantLong = false; gWantShort = false;
   gInPos = false; gInShort = false;""",
        "ontick": """   datetime t = iTime(_Symbol, _Period, 0);
   if(t != gLastBar)
   {
      gLastBar = t;
      int need = InpRsrsN + InpRsrsM + 2;
      double h[], l[];
      if(CopyHigh(_Symbol, _Period, 1, need, h) != need) return;
      if(CopyLow(_Symbol, _Period, 1, need, l)  != need) return;
      double rs[];
      ArrayResize(rs, need);
      for(int i = InpRsrsN - 1; i < need; i++)
      {
         double mx = 0.0, my = 0.0;
         for(int k = 0; k < InpRsrsN; k++) { mx += l[i-k]; my += h[i-k]; }
         mx /= InpRsrsN; my /= InpRsrsN;
         double cov = 0.0, vx = 0.0, vy = 0.0;
         for(int k = 0; k < InpRsrsN; k++)
         {
            double dx = l[i-k] - mx;
            double dy = h[i-k] - my;
            cov += dx*dy; vx += dx*dx; vy += dy*dy;
         }
         cov /= InpRsrsN; vx /= InpRsrsN; vy /= InpRsrsN;
         double beta = (vx > 0.0) ? cov/vx : 0.0;
         double r2   = (vx > 0.0 && vy > 0.0) ? (cov*cov)/(vx*vy) : 0.0;
         rs[i] = beta * r2;
      }
      int start = need - InpRsrsM;
      double mu = 0.0;
      for(int i = start; i < need; i++) mu += rs[i];
      mu /= InpRsrsM;
      double sd = 0.0;
      for(int i = start; i < need; i++) sd += (rs[i]-mu)*(rs[i]-mu);
      sd = MathSqrt(sd / InpRsrsM);
      double z = (sd > 0.0) ? (rs[need-1] - mu) / sd : 0.0;
      if(z > InpUpper)      { gInPos = true;  gInShort = false; }
      else if(z < InpLower) { gInShort = true; gInPos = false; }
      gWantLong  = gInPos;
      gWantShort = gInShort;
   }
   bool wantLong  = gWantLong;
   bool wantShort = gWantShort;""",
        "min_bars": "InpRsrsN + InpRsrsM + 10",
        "short_doc": "趋势类：标准分跌破 lower 即做空",
    },
    # ---------------- 扩展进场（与 forge 的 EXT_ENTRIES 一一对应）--------------
    "时序动量TSMOM": {
        "params": {"n": ("MomN", 60), "thr": ("MomThr", 0.0)},
        "globals": ("datetime gLastBar = 0;\nbool gWantLong = false;\n"
                    "bool gWantShort = false;"),
        "oninit": "   gLastBar = 0; gWantLong = false; gWantShort = false;",
        "ontick": """   datetime t = iTime(_Symbol, _Period, 0);
   if(t != gLastBar)
   {
      gLastBar = t;
      double p[];
      if(CopyClose(_Symbol, _Period, 1, InpMomN + 2, p) != InpMomN + 2) return;
      double base = p[InpMomN];
      if(base <= 0.0) return;
      double mom = p[0] / base - 1.0;
      gWantLong  = (mom >  InpMomThr);
      gWantShort = (mom < -InpMomThr);
   }
   bool wantLong  = gWantLong;
   bool wantShort = gWantShort;""",
        "min_bars": "InpMomN + 10",
        "short_doc": "趋势类：过去 n 期收益跌破 -thr 才做空（反向信号做空）",
    },
    "时变夏普": {
        "params": {"n": ("TvsN", 60), "thr": ("TvsThr", 0.0)},
        "globals": ("datetime gLastBar = 0;\nbool gWantLong = false;\n"
                    "bool gWantShort = false;"),
        "oninit": "   gLastBar = 0; gWantLong = false; gWantShort = false;",
        "ontick": """   datetime t = iTime(_Symbol, _Period, 0);
   if(t != gLastBar)
   {
      gLastBar = t;
      int n = MathMax(5, InpTvsN);
      double p[];
      if(CopyClose(_Symbol, _Period, 1, n + 2, p) != n + 2) return;
      // 滚动窗口内的收益均值 / 收益标准差（与 Python 侧 ddof=1 一致）
      double sum = 0.0, sum2 = 0.0;
      int m = 0;
      for(int i = 0; i < n; i++)
      {
         double prev = p[i + 1];
         if(prev <= 0.0) continue;
         double r = p[i] / prev - 1.0;
         sum += r; sum2 += r * r; m++;
      }
      double ratio = 0.0;
      if(m >= 3)
      {
         double mu = sum / (double)m;
         double var = (sum2 - (double)m * mu * mu) / (double)(m - 1);
         double sd = (var > 0.0) ? MathSqrt(var) : 0.0;
         if(sd > 0.0) ratio = mu / sd;
      }
      gWantLong  = (ratio >  InpTvsThr);
      gWantShort = (ratio < -InpTvsThr);
   }
   bool wantLong  = gWantLong;
   bool wantShort = gWantShort;""",
        "min_bars": "InpTvsN + 10",
        "short_doc": "趋势类：滚动夏普跌破 -thr 才做空（反向信号做空）",
    },
    "鳄鱼线": {
        "params": {"jaw": ("JawN", 13), "teeth": ("TeethN", 8),
                   "lips": ("LipsN", 5), "shift": ("AlliShift", 3)},
        "globals": ("datetime gLastBar = 0;\nbool gWantLong = false;\n"
                    "bool gWantShort = false;"),
        "oninit": "   gLastBar = 0; gWantLong = false; gWantShort = false;",
        "ontick": """   datetime t = iTime(_Symbol, _Period, 0);
   if(t != gLastBar)
   {
      gLastBar = t;
      int need = MathMax(InpJawN, MathMax(InpTeethN, InpLipsN))
                 + InpAlliShift + 2;
      double p[];
      if(CopyClose(_Symbol, _Period, 1, need, p) != need) return;
      int off = InpAlliShift;          // 三条线统一向前平移（与 Python 一致）
      double jaw = 0.0, teeth = 0.0, lips = 0.0;
      for(int i = 0; i < InpJawN; i++)   jaw   += p[off + i];
      for(int i = 0; i < InpTeethN; i++) teeth += p[off + i];
      for(int i = 0; i < InpLipsN; i++)  lips  += p[off + i];
      jaw /= (double)InpJawN; teeth /= (double)InpTeethN; lips /= (double)InpLipsN;
      double c = p[0];                 // 当前收盘（Python 侧是未平移的 close）
      bool aligned = (lips > teeth && teeth > jaw && c > lips);
      gWantLong  = aligned;
      gWantShort = false;              // 鳄鱼线原版只做多
   }
   bool wantLong  = gWantLong;
   bool wantShort = gWantShort;""",
        "min_bars": "InpJawN + InpAlliShift + 10",
        "short_doc": "趋势类：三线多头排列且收盘在唇线上方才做多（原版只做多）",
    },
}


#  Genome.entry（挖掘侧名字）→ SPECS 模板名。放这里而不是界面里，
#  这样"能不能导出"可以写测试守住：每个能挖的进场都必须能映射到模板。
ENTRY_ALIAS = {
    "SMA金叉": "双均线",
    "唐奇安突破": "唐奇安通道突破",
    "MACD金叉": "MACD金叉",
    "RSI反转": "RSI反转",
    "LLT上穿": "LLT上穿",
    "时序动量TSMOM": "时序动量TSMOM",
    "时变夏普": "时变夏普",
    "鳄鱼线": "鳄鱼线",
}


def spec_for(entry: str) -> str | None:
    """进场名 → 可用的 MQL5 模板名；没有模板返回 None。"""
    name = ENTRY_ALIAS.get(str(entry), str(entry))
    return name if name in SPECS else None


def generate(strategy: str, params: dict, name: str | None = None,
             out_dir: str | pathlib.Path = "mql5",
             vol_filter_pct: float | None = None,
             trend_filter_pct: float | None = None,
             allow_short: bool = False,
             sl: float | None = None,
             tp: float | None = None,
             test_range: str = "",
             dry_run: bool = True,
             with_bridge: bool = False,
             visual: bool = False,
             extra_inputs: dict | None = None) -> GenResult:
    """生成 EA 源码。

    ``vol_filter_pct``：波动率分位过滤阈值（0.7 = 仅当 ATR 分位 ≤70% 时交易）；
    ``trend_filter_pct``：强趋势过滤阈值（0.5 = 仅当趋势分位 ≥50% 时交易）；
    ``allow_short``：是否启用做空；``sl``/``tp``：止损止盈百分比。

    这些参数与 `gold.forge.Genome` 一一对应，保证"挖出来的"和"导出的"是同一个 EA。
    ``with_bridge``：合并 GoldBridge 数据桥（tick+8周期K线+账户/持仓/成交 落盘，
    文件名同口径，挂了本 EA 可不再单独挂 GoldBridge）；
    ``visual``：图表可视化（Comment 信号面板 + 买卖箭头）。
    历史上的「干支（日柱地支）」维度已移除（那是股票日频择日用的，见 combo.py）。
    """
    spec = SPECS.get(strategy)
    if spec is None:
        return GenResult(ok=False, error=f"暂未实现 {strategy} 的 MQL5 模板")

    tag = re.sub(r"[^0-9A-Za-z_]", "", strategy)
    tag = tag or "EA"          # 中文策略名会被全部过滤掉，别生成 "GoldEA_"
    name = name or f"GoldEA_{tag}"
    p = {**{k: v[1] for k, v in spec["params"].items()}, **(params or {})}

    strat_inputs = _inputs(p, {k: v[0] for k, v in spec["params"].items()})
    min_bars = spec["min_bars"]
    if "{" in min_bars:
        fmt = {k: v for k, v in p.items() if isinstance(v, (int, float))}
        fmt.setdefault("slow", 60)
        fmt.setdefault("entry", 20)
        min_bars = min_bars.format(**fmt)

    extra = []
    if vol_filter_pct is not None:
        extra.append(f"波动率分位≤{float(vol_filter_pct):.2f}")
    if with_bridge:
        extra.append("合并数据桥")
    if visual:
        extra.append("图表可视化")
    if trend_filter_pct is not None:
        extra.append(f"趋势分位≥{float(trend_filter_pct):.2f}")
    if allow_short:
        extra.append("允许做空")
    if sl is not None:
        extra.append(f"止损{sl * 100:.0f}%")
    if tp is not None:
        extra.append(f"止盈{tp * 100:.0f}%")

    # 策略逻辑注释（模板 + 参数直接拼，人工核对用）
    _flt = ("强趋势过滤（趋势分位 ≥ 阈值）" if trend_filter_pct else
            ("低波动过滤（波动分位 ≤ 阈值）" if vol_filter_pct else "无状态过滤"))
    _dir = "只能做多" if not allow_short else "可做多也可做空"
    _lines = [
        f"进场：{spec.get('short_doc', strategy)}",
        "参数：" + ", ".join(f"{k}={v}" for k, v in p.items()),
        f"过滤：{_flt}",
        f"方向：{_dir}",
        f"止损：{('%.2f%%' % (sl * 100)) if sl else '无'}"
        f"　止盈：{('%.2f%%' % (tp * 100)) if tp else '无'}",
        "执行：t 根收盘出信号，t+1 根开盘执行（无未来函数）",
        "风控：点差上限 / 单笔风险上限 / 日内亏损停机 / 同 bar 只开一次",
    ]
    _rule_doc = "".join(f"//|  {i + 1}. {t}\n" for i, t in enumerate(_lines))

    src = HEADER.format(
        name=name, strategy_cn=strategy, rule_doc=_rule_doc,
        params_desc=(" ".join(f"{k}={v}" for k, v in p.items())
                     + ("　|　" + "　".join(extra) if extra else "")),
        strategy_inputs=strat_inputs, globals=spec["globals"],
        report_tag=name,
        preamble=PREAMBLE, oninit=spec["oninit"], ontick=spec["ontick"],
        min_bars=min_bars, short_doc=spec["short_doc"])

    # 覆盖过滤层默认值（模板里默认全关）
    if vol_filter_pct is not None:
        src = src.replace("input bool   InpUseVolFilter = false;",
                          "input bool   InpUseVolFilter = true;")
        src = src.replace("input double InpVolMaxPct    = 0.70;",
                          f"input double InpVolMaxPct    = {float(vol_filter_pct):.2f};")
    if trend_filter_pct is not None:
        src = src.replace("input bool   InpUseTrendFilter = false;",
                          "input bool   InpUseTrendFilter = true;")
        src = src.replace("input double InpTrendMinPct    = 0.50;",
                          f"input double InpTrendMinPct    = {float(trend_filter_pct):.2f};")
    if allow_short:
        src = src.replace("input bool   InpAllowShort = false;",
                          "input bool   InpAllowShort = true;")
    if with_bridge or visual:
        _patches = [
            ("input int    InpReportSecs  = 60;       // 多少秒写一行",
             "input int    InpReportSecs  = 60;       // 多少秒写一行\n"
             + MERGED_INPUTS.strip("\n")),
            ("void OnDeinit(const int reason) { EventKillTimer(); }",
             "void OnDeinit(const int reason) { EventKillTimer(); BrgDeinit(); VisDeinit(); }"),
            ("void OnTimer() { }",
             "void OnTimer()\n{\n   BrgTimer();\n   VisTimer();\n}"),
            ("   EventSetTimer(5);\n   ReportState();",
             "   EventSetTimer(1);\n   ReportState();\n   BrgInit();\n   VisInit();"),
            ("void OnTick()\n{\n   if(Bars(_Symbol, _Period) <",
             "void OnTick()\n{\n   if(InpBridgeOn && InpBrgWriteTick) BrgWriteTick(_Symbol);\n"
             "   if(Bars(_Symbol, _Period) <"),
            ("   ApplySignal(wantLong, wantShort);",
             "   gVisWantLong = wantLong; gVisWantShort = wantShort;\n"
             "   ApplySignal(wantLong, wantShort);\n   VisPanel();"),
            ("void OpenTrade(int dir)\n{\n   if(InpDryRun)",
             "void OpenTrade(int dir)\n{\n   VisMark(dir);\n   if(InpDryRun)"),
        ]
        for _old, _new in _patches:
            if _old not in src:
                return GenResult(ok=False,
                                 error=f"合并模块注入失败：找不到锚点 {_old[:44]!r}")
            src = src.replace(_old, _new, 1)
        if with_bridge:
            src = src.replace("input bool   InpBridgeOn     = false;",
                              "input bool   InpBridgeOn     = true;")
        if visual:
            src = src.replace("input bool   InpVisualOn     = false;",
                              "input bool   InpVisualOn     = true;")
        src += MERGED_CODE
    if sl is not None:
        src = src.replace("input double InpSLPct      = 0.0;",
                          f"input double InpSLPct      = {float(sl):.4f};")
    if tp is not None:
        src = src.replace("input double InpTPPct      = 0.0;",
                          f"input double InpTPPct      = {float(tp):.4f};")
    if test_range:
        src = src.replace(
            "//| ",
            f"//| 建议回测区间：{test_range}（与系统挖掘口径一致）\n//| ", 1)

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.mq5"
    path.write_text(src, encoding="utf-8")
    if not dry_run:
        # 仅用于回测/测试器：把默认值翻成 false，让订单真的下出去。
        # 上盘用的 EA 不要这样做 —— 保留干跑默认更安全。
        _src2 = src.replace("input bool InpDryRun = true;",
                             "input bool InpDryRun = false;")
        path.write_text(_src2, encoding="utf-8")

    if extra_inputs:
        # 按输入名替换默认值：int/double/bool 都支持，找不到就跳过（不报错）
        for _k, _v in extra_inputs.items():
            _pat = (r"(input\s+(?:int|double|bool|LotMode)\s+" + str(_k)
                    + r"\s*=\s*)([^;]+)(;)")
            if isinstance(_v, bool):
                _val = "true" if _v else "false"
            else:
                _val = str(_v)
            path.write_text(
                __import__("re").sub(_pat, r"\g<1>" + _val + r"\g<3>",
                                     path.read_text(encoding="utf-8"),
                                     count=1),
                encoding="utf-8")

    return GenResult(ok=True, mq5_path=str(path), source=src)


def compile_mq5(mq5_path: str | pathlib.Path,
                metaeditor: str = METAEDITOR) -> GenResult:
    p = pathlib.Path(mq5_path).resolve()
    if not p.exists():
        return GenResult(ok=False, error="源文件不存在")
    if not pathlib.Path(metaeditor).exists():
        return GenResult(ok=False, mq5_path=str(p), error=f"找不到 MetaEditor: {metaeditor}")
    log = p.with_suffix(".compile.log")
    try:
        subprocess.run([metaeditor, f"/compile:{p}", f"/log:{log}"],
                       capture_output=True, timeout=180, check=False)
    except Exception as e:  # noqa: BLE001
        return GenResult(ok=False, mq5_path=str(p), error=f"编译调用失败：{e}")
    ex5 = p.with_suffix(".ex5")
    text = ""
    if log.exists():
        for enc in ("utf-16", "utf-8", "gbk"):
            try:
                text = log.read_text(encoding=enc)
                break
            except Exception:  # noqa: BLE001
                continue
    errors = re.findall(r"error \d+", text)
    ok = ex5.exists() and not errors
    return GenResult(ok=ok, mq5_path=str(p),
                     ex5_path=str(ex5) if ex5.exists() else "",
                     source=p.read_text(encoding="utf-8"), log=text[-3000:],
                     error="" if ok else (text[-800:] or "编译失败，无日志"))


def install_to_mt5(mq5_path: str | pathlib.Path,
                   dest_dirs: list[str] | None = None) -> dict:
    src = pathlib.Path(mq5_path)
    if not src.exists():
        return {"ok": False, "copied": [], "errors": ["源文件不存在"]}
    copied, errors = [], []
    for d in (dest_dirs or EXPERTS_DIRS):
        try:
            dp = pathlib.Path(d)
            dp.mkdir(parents=True, exist_ok=True)
            (dp / src.name).write_bytes(src.read_bytes())
            copied.append(str(dp / src.name))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{d}: {type(e).__name__}")
    return {"ok": bool(copied), "copied": copied, "errors": errors}
