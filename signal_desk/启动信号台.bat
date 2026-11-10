@echo off
setlocal
chcp 65001 >nul
title SIGNAL DESK - 信号台
cd /d "%~dp0.."

set "URL=http://127.0.0.1:8777/"
set "EDGE=C:/Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not exist "%EDGE%" set "EDGE=C:/Program Files\Microsoft\Edge\Application\msedge.exe"

echo.
echo   ============================================================
echo    SIGNAL DESK  -  只读信息层 / EA 工厂 / 智能辅助
echo   ============================================================
echo    地址     %URL%
echo    页面     00 总览 / 01 EA 工厂 / 02 智能辅助 / 03 运行监控
echo    能力     EA 挖掘生成与交付；8 板块收盘 K 线辅助看板；账户与行情只读
echo    边界     自动盯盘已停用；不下单；信号仅观察，不构成交易入场建议
echo    停止     在本窗口按 Ctrl+C，或双击 停止信号台.bat
echo   ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo   [错误] 找不到 .venv\Scripts\python.exe
  echo   请先在项目根目录建虚拟环境：
  echo       python -m venv .venv
  echo       .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

netstat -ano ^| findstr ":8777" ^| findstr "LISTENING" >nul 2>&1
if not errorlevel 1 goto ALREADY

echo   正在启动服务（约 5 - 10 秒）...
if exist "%EDGE%" (
  start "" "%EDGE%" "%URL%"
) else (
  start "" "%URL%"
)
".venv\Scripts\python.exe" "signal_desk\serve.py"
set RC=%ERRORLEVEL%

echo.
if not "%RC%"=="0" (
  echo   服务异常退出，代码 %RC%
  echo   请把上面的报错信息截图发给助手。
) else (
  echo   服务已正常停止。
)
echo.
pause
exit /b 0

:ALREADY
echo   8777 已有服务在运行，直接打开页面。
echo   注意：地址必须是 http:// 开头，不要用 https://
echo.
if exist "%EDGE%" (
  start "" "%EDGE%" "%URL%"
) else (
  start "" "%URL%"
)
echo.
pause
exit /b 0
