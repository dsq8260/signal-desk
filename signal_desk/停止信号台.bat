@echo off
setlocal
chcp 65001 >nul
title SIGNAL DESK - 停止服务

echo.
echo   正在查找监听 8777 的服务进程...
set FOUND=0
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8777" ^| findstr "LISTENING"') do (
  set FOUND=1
  echo   停止进程 PID %%P
  taskkill /PID %%P /F >nul 2>&1
)

if "%FOUND%"=="1" (
  echo   已停止。
) else (
  echo   没有检测到正在运行的服务。
)
echo.
pause
