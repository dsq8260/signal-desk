@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 外汇副账本 · EURUSD/GBPUSD 仅影子记录（不影响主系统）
echo ============================================================
echo  外汇副账本（Multi-Pair Shadow Ledger）
echo  - 只记影子样本，不下单、不推送、不进主系统工单面板
echo  - 规则同源：5 条件 + min_rr=2.5 + 禁逆势
echo  - 数据写入 data\shadow_book_multi.json（黄金主账本完全不碰）
echo  - 关闭此窗口即停止副账本，主服务与主账本不受影响
echo ============================================================
echo.
"%~dp0..\.venv\Scripts\python.exe" "%~dp0shadow_multi.py" --daemon
echo.
echo 副账本已退出。
pause
