@echo off
cd /d D:\Project\AI\binance_auto_trader

echo ==========================================
echo         BINANCE TRADER STATUS
echo ==========================================
echo.

:: Check if running
echo [Process Status]
tasklist /fi "IMAGENAME eq python.exe" 2>nul | find "python" >nul
if %errorlevel%==0 (
    echo Status: RUNNING
) else (
    echo Status: STOPPED
)
echo.

:: Show balance/signal status
echo [Trading Status]
if exist "data\paper_trading_status.json" (
    type "data\paper_trading_status.json"
) else if exist "data\signal_status.json" (
    type "data\signal_status.json"
) else (
    echo No status file found.
)
echo.

:: Show latest log entry
echo [Latest Log]
for /f "delims=" %%i in ('dir /b /od logs\*.log 2^>nul') do set "latest=%%i"
if defined latest (
    echo File: logs\%latest%
    echo.
    powershell -Command "Get-Content 'logs\%latest%' -Tail 10"
) else (
    echo No log files found.
)

echo.
echo ==========================================
pause
