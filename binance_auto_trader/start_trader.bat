@echo off
echo Starting Binance Trader...

:: Wait for Docker to be ready
timeout /t 30 /nobreak

:: Stop and remove existing container if exists
docker stop binance-trader 2>nul
docker rm binance-trader 2>nul

:: Start container
docker run -d ^
  --name binance-trader ^
  --restart unless-stopped ^
  -v D:\Project\AI\binance_auto_trader\config.py:/app/config.py:ro ^
  -v D:\Project\AI\binance_auto_trader\data:/app/data ^
  thiennt810/binance-auto-trader:latest ^
  python paper_trading.py

echo Binance Trader started!
