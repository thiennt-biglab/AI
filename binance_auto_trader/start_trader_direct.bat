@echo off
cd /d D:\Project\AI\binance_auto_trader
echo Starting Binance Trader...

:: Activate virtual environment if exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: Run paper trading (change to run.py for live)
python paper_trading.py

pause
