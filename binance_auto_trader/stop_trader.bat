@echo off
echo Stopping Binance Trader...

:: Find and kill python processes running our scripts
taskkill /f /fi "IMAGENAME eq python.exe" /fi "WINDOWTITLE eq *paper_trading*" 2>nul
taskkill /f /fi "IMAGENAME eq python.exe" /fi "WINDOWTITLE eq *run.py*" 2>nul

:: Kill all python if above doesn't work (be careful if running other python apps)
:: taskkill /f /im python.exe 2>nul

echo Trader stopped!
pause
