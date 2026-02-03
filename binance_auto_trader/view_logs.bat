@echo off
cd /d D:\Project\AI\binance_auto_trader\logs

:: Get latest log file
for /f "delims=" %%i in ('dir /b /od *.log 2^>nul') do set "latest=%%i"

if defined latest (
    echo === Latest Log: %latest% ===
    echo.
    type "%latest%"
    echo.
    echo === Press any key to follow (Ctrl+C to exit) ===
    pause >nul
    powershell -Command "Get-Content '%latest%' -Wait -Tail 50"
) else (
    echo No log files found.
    pause
)
