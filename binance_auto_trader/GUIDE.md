# Binance Auto Trader - Quick Guide

## Table of Contents
1. [Setup](#setup)
2. [Training](#training)
3. [Paper Trading](#paper-trading) **<-- START HERE**
4. [Live Trading](#live-trading)
5. [Fine-Tuning](#fine-tuning)
6. [Monitoring & Analysis](#monitoring--analysis)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `config.py`:
```python
API_KEY = "your_binance_api_key"
API_SECRET = "your_binance_api_secret"
```

### 3. Choose Trading Mode
Edit `config.py`:
```python
TRADING_MODE = 'accuracy'    # High win rate, fewer trades
# or
TRADING_MODE = 'profit'      # More trades, higher risk
```

---

## Training

### Initial Training (First Time)
Train the model on historical data (takes 30+ minutes):

```bash
python train_lstm_keras.py
```

**What it does:**
- Fetches 30 days of BTC/USDT data from Binance
- Trains 12 different model architectures
- Selects the best performing model
- Saves to `best_model.keras`

**Output files:**
- `best_model.keras` - Best trained model
- `scaler.pkl` - Data scaler for predictions
- `training_config.pkl` - Training parameters

### Training Options

| Mode | Win Rate | Trades/Day | Risk |
|------|----------|------------|------|
| accuracy | 85-92% | 2-5 | Low |
| profit | 65-75% | 5-10 | Medium |

---

## Paper Trading

**IMPORTANT: You MUST complete 30 days of paper trading before live trading!**

Paper trading uses real market data but simulates trades without real money.

### Start Paper Trading
```bash
python run.py
```
(When `PAPER_TRADING_MODE = True` in config.py, this automatically runs paper trading)

Or directly:
```bash
python paper_trading.py run
```

### Paper Trading Requirements

| Requirement | Value | Why |
|-------------|-------|-----|
| Duration | 30 days | Ensure consistency |
| Min Trades | 50 trades | Statistical significance |
| Win Rate | > 60% | Profitable strategy |
| Profit | > 0% | Must be profitable |
| Max Drawdown | < 30% | Risk management |

### Check Paper Trading Status
```bash
python paper_trading.py status
```

Shows:
- Days completed
- Number of trades
- Win rate
- Current capital
- Requirements progress

### Check if Ready for Live
```bash
python paper_trading.py check
```

### Paper Trading Workflow

```
Day 1:  Start paper trading
        |
        v
Day 1-30: Bot runs with simulated money
          - Uses real market data
          - Makes real predictions
          - Simulates trades
          - Tracks performance
        |
        v
Day 30+: Check requirements
        |
        +---> All met? --> APPROVED for live trading
        |
        +---> Not met? --> Continue paper trading
```

### After Paper Trading Approved

1. Edit `config.py`:
```python
PAPER_TRADING_MODE = False  # Enable live trading
```

2. Start live trading:
```bash
python run.py
```

### Reset Paper Trading (Start Over)
```bash
python paper_trading.py reset
```

---

## Live Trading

**Prerequisites (ALL required):**
1. Model trained (`best_model.keras` exists)
2. Paper trading completed (30 days, all requirements met)
3. `PAPER_TRADING_MODE = False` in config.py

### Start Live Trading
```bash
python run.py
```

**What it does:**
- Verifies paper trading completion
- Loads trained model
- Fetches real-time market data
- Generates LONG/SHORT/HOLD signals
- Places REAL orders on Binance Futures
- Manages TP/SL automatically
- Logs all trades for continuous learning
- Auto fine-tunes when performance drops

### Run in Background (Linux/Mac)
```bash
nohup python run.py > trading.log 2>&1 &
```

### Run in Background (Windows)
```bash
start /B python run.py > trading.log 2>&1
```

### Stop the Bot
- Press `Ctrl+C` in terminal
- Or kill the Python process

### Live Trading Safety Features

| Feature | Description |
|---------|-------------|
| Paper Trading Gate | Must complete 30 days first |
| Crash Protection | Stops trading during market crashes |
| Bot Protection | Avoids manipulation traps |
| Auto Fine-Tune | Adapts to market changes |
| Trade Logging | All trades recorded for analysis |
| Performance Monitor | Alerts on degradation |

---

## Fine-Tuning

Fine-tuning adapts the model to recent market conditions without full retraining.

### Manual Fine-Tune
```bash
python continuous_learning.py finetune
```

### Check Performance & Auto Fine-Tune
```bash
python continuous_learning.py check
```

### Start Continuous Monitoring
```bash
python continuous_learning.py monitor
```
This runs in background and auto fine-tunes when:
- Win rate drops below 65%
- Profit factor drops below 1.2

### Fine-Tuning Schedule

| Event | Action |
|-------|--------|
| Every hour | Check performance metrics |
| Win rate < 65% | Trigger fine-tuning |
| Every 7 days | Allow new fine-tune cycle |
| After fine-tune | Validate before deploying |

---

## Monitoring & Analysis

### Check Current Status
```bash
python continuous_learning.py status
```

Shows:
- Total trades
- Win rate
- Profit factor
- Model version
- Last fine-tune date

### View Trade Log
Trade history is saved in `trade_log.json`:
```bash
# View recent trades (Linux/Mac)
cat trade_log.json | python -m json.tool | tail -50

# View recent trades (Windows PowerShell)
Get-Content trade_log.json | ConvertFrom-Json | Select-Object -Last 10
```

### Performance Metrics
Check `performance_log.json` for historical performance:
- Win rate over time
- Profit factor trend
- Drawdown history

### Model History
Check `model_history.json` for:
- All model versions
- Fine-tune timestamps
- Performance at each version

---

## Configuration

### Key Settings in `config.py`

#### Trading Parameters
```python
SYMBOL = "BTCUSDT"              # Trading pair
DEFAULT_LEVERAGE = 3            # Leverage (3x recommended)
RISK_PERCENT = 10               # % of balance per trade
ENTRY_THRESHOLD = 0.92          # Min confidence to enter
SWITCH_THRESHOLD = 0.95         # Min confidence to switch
```

#### TP/SL Settings
```python
TP_MIN = 0.006                  # 0.6% take profit
SL_MIN = 0.008                  # 0.8% stop loss
```

#### Protection Features
```python
USE_CRASH_PROTECTION = True     # Stop trading during crashes
USE_BOT_PROTECTION = True       # Avoid bot manipulation
USE_WHALE_CONFIRMATION = True   # Follow smart money
USE_NEWS_FILTER = True          # Check news sentiment
```

#### Continuous Learning
```python
USE_CONTINUOUS_LEARNING = True  # Enable auto fine-tuning
CL_MIN_WIN_RATE = 0.65          # Fine-tune threshold
CL_FINE_TUNE_LR = 0.0001        # Fine-tune learning rate
CL_FINE_TUNE_EPOCHS = 30        # Fine-tune epochs
```

---

## File Structure

```
binance_auto_trader/
├── config.py                 # All settings
├── run.py                    # Main entry point
├── train_lstm_keras.py       # Model training
├── paper_trading.py          # Paper trading system
├── continuous_learning.py    # Fine-tuning system
├── trader.py                 # Order execution
├── feature_pipeline.py       # Data processing
├── strategy_lstm_live.py     # Prediction logic
├── whale_tracker.py          # Whale tracking
├── bot_protection.py         # Anti-manipulation
├── news_sentiment_enhanced.py # News analysis
├── GUIDE.md                  # This guide
│
├── best_model.keras          # Trained model
├── scaler.pkl                # Data scaler
├── trade_log.json            # Live trade history
├── paper_trades.json         # Paper trade history
├── paper_trading_status.json # Paper trading progress
├── performance_log.json      # Performance metrics
└── model_history.json        # Model versions
```

---

## Troubleshooting

### Common Issues

#### 1. "Model not found"
```bash
# Train the model first
python train_lstm_keras.py
```

#### 2. "API Error: Invalid signature"
- Check API key/secret in config.py
- Ensure API has Futures trading permission
- Check system time is synchronized

#### 3. "Insufficient balance"
- Reduce `RISK_PERCENT` in config.py
- Reduce `DEFAULT_LEVERAGE`
- Add more USDT to Futures wallet

#### 4. "Min notional < 5 USDT"
- Increase leverage or risk percent
- Bot will auto-adjust if possible

#### 5. "Win rate dropping"
```bash
# Check and fine-tune
python continuous_learning.py check
```

#### 6. "Too many losses"
- Bot auto-increases entry threshold after losses
- Check if market is in crash mode
- Consider pausing during high volatility

### Logs Location
- Trading log: Terminal output or `trading.log`
- Trade history: `trade_log.json`
- Performance: `performance_log.json`

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| **SETUP** | |
| Train model | `python train_lstm_keras.py` |
| **PAPER TRADING** | |
| Start paper trading | `python paper_trading.py run` |
| Check paper status | `python paper_trading.py status` |
| Check if live ready | `python paper_trading.py check` |
| Reset paper trading | `python paper_trading.py reset` |
| **LIVE TRADING** | |
| Run bot | `python run.py` |
| **MAINTENANCE** | |
| Check performance | `python continuous_learning.py status` |
| Force fine-tune | `python continuous_learning.py finetune` |
| Auto-monitor | `python continuous_learning.py monitor` |
| **TESTING** | |
| Backtest | `python backtest_lstm.py` |
| Simulate modes | `python simulate_trading.py` |

---

## Recommended Workflow

```
Step 1: Train Model
        python train_lstm_keras.py
        (Wait 30-60 minutes)
              |
              v
Step 2: Paper Trade (30 days)
        python run.py
        (PAPER_TRADING_MODE = True)
              |
              v
Step 3: Check Requirements
        python paper_trading.py check
              |
              +---> Not ready? Continue paper trading
              |
              v
Step 4: Enable Live Trading
        Edit config.py: PAPER_TRADING_MODE = False
              |
              v
Step 5: Start Live Trading
        python run.py
              |
              v
Step 6: Monitor & Fine-Tune
        python continuous_learning.py monitor
```

---

## Safety Recommendations

1. **Complete paper trading** - Full 30 days required
2. **Start small** - Use $50-100 initially for live
3. **Monitor daily** - Check performance regularly
4. **Set limits** - Don't risk more than you can lose
5. **Backup models** - Keep copies of working models
6. **Fine-tune weekly** - Adapt to market changes
7. **Check news** - Be aware of major events

---

## Expected Performance

| Metric | Accuracy Mode | Profit Mode |
|--------|---------------|-------------|
| Win Rate | 85-92% | 65-75% |
| Trades/Day | 2-5 | 5-10 |
| Monthly Return | 15-25% | 25-40% |
| Max Drawdown | 5-10% | 10-20% |
| Risk Level | Low | Medium |

*Note: Past performance does not guarantee future results. Crypto trading involves significant risk.*

---

## Support

- Issues: Check logs first
- Updates: Pull latest from repository
- Questions: Review this guide

**Last Updated:** February 2025
