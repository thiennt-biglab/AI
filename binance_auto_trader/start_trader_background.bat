@echo off
cd /d D:\Project\AI\binance_auto_trader

:: Create logs directory
if not exist "logs" mkdir logs

:: Get current date for log filename
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\trader_%datetime:~0,8%.log

echo [%date% %time%] Starting Binance Trader... >> %logfile%

:: Activate virtual environment if exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: Run in background with logging (change paper_trading.py to run.py for live)
start /b python paper_trading.py >> %logfile% 2>&1

echo Trader started! Logs: %logfile%
