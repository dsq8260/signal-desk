//+------------------------------------------------------------------+
//| GoldBridge.mq5  —  MT5 数据桥（多品种版）
//| 作用：把一批品种的
//|   1) 实时 tick   → gold_tick_<SYM>.csv
//|   2) 8 个周期历史 K 线 → gold_bars_<SYM>_<TF>.csv
//| 写进 MT5 的「公共文件夹」，供 Python 读取。
//|
//| 为什么要这个桥：
//|   MT5 的 Python API 里 copy_rates_* 在本机一直返回
//|   (-1, 'Terminal: Call failed')，拿不到历史 K 线；
//|   而 EA 里的 CopyRates() 走终端内部通道，能正常拿到。
//|   于是让 EA 把数据写到文件，Python 读文件 —— 绕开那个 bug。
//|
//| 用法：
//|   1. 把本文件放到 MQL5\Experts\，MetaEditor 里 F7 编译
//|   2. 打开任意品种的图表（建议 GOLD# H1）
//|   3. 导航器里把 GoldBridge 拖到图表上，允许算法交易
//|   4. 在 EA 属性里填「导出品种」列表（逗号分隔）
//+------------------------------------------------------------------+
#property copyright "gold-ea-terminal"
#property version   "1.50"
#property strict
#property description "多品种数据桥：tick + 8 周期 K 线 → 公共文件夹 CSV"

input int    InpDumpSeconds = 30;      // 历史 K 线重写周期（秒）
// 每页最多取多少根（MT5 单次上限约 2 万），以及往回取几页。
const int    BRIDGE_PAGE_BARS = 20000;
const int    BRIDGE_PAGES     = 6;
input bool   InpWriteTick   = true;    // 是否写实时 tick
input bool   InpVerbose     = true;    // 是否在日志里打印
input string InpSymbols     = "GOLD#,XAGUSD,EURUSD,USDJPY,GBPUSD,USDCHF"; // 导出品种（逗号分隔）
input int    InpAccountSecs  = 5;      // 账户/持仓/成交 写盘周期（秒）
input int    InpDealDays     = 90;     // 成交历史回看天数

int gTimerTicks = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   EventSetTimer(1);
   DumpAll();
   DumpAccount();
   DumpPositions();
   DumpDeals();
   if(InpVerbose)
      PrintFormat("[GoldBridge] 已启动：图表品种=%s 重写=%ds 最多=%d 根 列表=%s",
                  _Symbol, InpDumpSeconds, BRIDGE_PAGE_BARS, InpSymbols);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   if(InpVerbose) PrintFormat("[GoldBridge] 已停止，reason=%d", reason);
}

void OnTimer()
{
   gTimerTicks++;
   if(gTimerTicks % MathMax(1, InpDumpSeconds) == 0)
      DumpAll();
   if(gTimerTicks % MathMax(1, InpAccountSecs) == 0)
   {
      DumpAccount();
      DumpPositions();
      DumpDeals();
   }
}

void OnTick()
{
   if(InpWriteTick) WriteTick(_Symbol);
}

//+------------------------------------------------------------------+
//| 文件名安全化：去掉 # 等非字母数字字符                              |
//+------------------------------------------------------------------+
string SafeName(string s)
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

//+------------------------------------------------------------------+
//| 解析逗号分隔的品种列表                                             |
//+------------------------------------------------------------------+
int ParseSymbols(string &list[])
{
   string parts[];
   int n = StringSplit(InpSymbols, ',', parts);
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

//+------------------------------------------------------------------+
//| 实时 tick：追加一行到 gold_tick_<SYM>.csv                          |
//+------------------------------------------------------------------+
void WriteTick(string sym)
{
   MqlTick t;
   if(!SymbolInfoTick(sym, t)) return;

   string fname = "gold_tick_" + SafeName(sym) + ".csv";
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
   {
      FileSeek(h, 0, SEEK_END);
   }

   FileWrite(h, StringFormat("%I64d,%.5f,%.5f", (long)t.time_msc, t.bid, t.ask));
   FileClose(h);
}

//+------------------------------------------------------------------+
//| 账户快照 → mt5_account.csv（整份重写，一行数据）                    |
//+------------------------------------------------------------------+
void DumpAccount()
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

//+------------------------------------------------------------------+
//| 当前持仓 → mt5_positions.csv                                      |
//+------------------------------------------------------------------+
void DumpPositions()
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

//+------------------------------------------------------------------+
//| 成交历史 → mt5_deals.csv（按 magic 可区分不同 EA）                  |
//+------------------------------------------------------------------+
void DumpDeals()
{
   datetime from = TimeCurrent() - (datetime)InpDealDays * 86400;
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

//+------------------------------------------------------------------+
//| 所有品种 × 8 个周期的历史 K 线                                     |
//+------------------------------------------------------------------+
void DumpAll()
{
   string syms[];
   int ns = ParseSymbols(syms);

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
      if(!SymbolSelect(sym, true))
      {
         if(InpVerbose) PrintFormat("[GoldBridge] 品种不可用：%s", sym);
         continue;
      }

      int total = 0;
      for(int i = 0; i < ArraySize(tfs); i++)
      {
         // 逐页往回取：p0 最新，p1 更早……各写一个文件，由 Python 合并去重
         datetime endT = TimeCurrent() + 60;
         long secs = (long)PeriodSeconds(periods[i]);
         for(int pg = 0; pg < BRIDGE_PAGES; pg++)
         {
            MqlRates r[];
            int n = 0;
            if(pg == 0)
               n = CopyRates(sym, periods[i], 0, BRIDGE_PAGE_BARS, r);
            else
            {
               datetime startT = endT - (datetime)(secs * BRIDGE_PAGE_BARS);
               n = CopyRates(sym, periods[i], startT, endT, r);
            }
            if(n <= 0) break;

            string fname = "gold_bars_" + SafeName(sym) + "_" + tfs[i]
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

            // 下一页的结束时间 = 本页最早一根之前
            endT = r[0].time - 1;
            if(n < BRIDGE_PAGE_BARS - 1) break;   // 已经到底了
         }
      }
      if(InpVerbose)
         PrintFormat("[GoldBridge] %s → %d 个周期", sym, total);
   }
}
//+------------------------------------------------------------------+
