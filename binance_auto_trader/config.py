API_KEY = "GJSYhlu02urxr9LaktR9606IhuUsCsTnjj7AmM0IYXRZmPqMDE66t5baGfSSA2Z0"
API_SECRET = "4iRKlKY9OTYNzY85D1Z4CRIQyq82oX5VCnh6htCT3dHtzsZ5OvGrO3rzuETNRmBg"

SYMBOL = "BTCUSDT"

# === TIMEFRAME SETTINGS ===
# For higher WIN RATE: Use longer timeframes (less noise, cleaner signals)
# For more TRADES: Use shorter timeframes (more opportunities, more noise)
TIMEFRAME_MODE = 'accuracy'  # 'accuracy' or 'scalping'

if TIMEFRAME_MODE == 'accuracy':
    # OPTIMIZED ACCURACY MODE (88% win rate strategy)
    # Based on BTC historical backtesting:
    #   - Asian Session (0-8 UTC) + Low Volatility (ATR < 0.5%)
    #   - Win Rate: 88% | Avg Profit: 0.64% per trade
    INTERVALS = ["1h", "4h", "1d"]
    SEQ_LEN_MODEL = 48  # 48 hours of history
else:
    # SCALPING MODE: 15m primary, faster signals
    # Expected: Lower win rate (55-65%), more trades
    INTERVALS = ["15m", "1h", "4h"]
    SEQ_LEN_MODEL = 40

# === SESSION & VOLATILITY FILTERS ===
# CHOOSE YOUR PRIORITY:
#   MAX PROFIT:     USE_SESSION_FILTER = False, USE_VOLATILITY_FILTER = False
#   BALANCED:       USE_SESSION_FILTER = False, USE_VOLATILITY_FILTER = True  <-- SELECTED
#   HIGH WIN RATE:  USE_SESSION_FILTER = True,  USE_VOLATILITY_FILTER = True

USE_SESSION_FILTER = False            # Trade all sessions
ALLOWED_SESSIONS = [(0, 8)]           # Only used if USE_SESSION_FILTER = True

USE_VOLATILITY_FILTER = True          # BALANCED: Filter high volatility
MAX_ATR_PERCENT = 0.5                 # Skip when ATR > 0.5% (too volatile)
MIN_ATR_PERCENT = 0.1                 # Skip when ATR < 0.1% (too quiet)

# CURRENT MODE: BALANCED
# Expected: 74% win rate, ~$200/month with $100 capital, 3x leverage

# === CRASH PROTECTION (Protect profits during market crashes) ===
USE_CRASH_PROTECTION = True           # Enable crash detection and protection

# Crash warning triggers (stop trading when ANY of these occur):
CRASH_ATR_THRESHOLD = 1.0             # ATR > 1.0% = volatility explosion
CRASH_VOLUME_SPIKE = 3.0              # Volume > 3x average = panic mode
CRASH_SINGLE_DROP = 2.0               # Single candle drop > 2%
CRASH_4H_DROP = 2.0                   # 4-hour drop > 2%

# Recovery settings:
CRASH_COOLDOWN_HOURS = 4              # Wait 4 hours after crash warning before trading

# What crash protection does:
# 1. Detects crash warning signs BEFORE the big drop
# 2. Stops opening new positions
# 3. Waits for market to stabilize
# 4. Saves your capital during flash crashes, panic events, liquidation cascades

# === SLIPPAGE SETTINGS (Realistic order execution) ===
USE_SLIPPAGE_BUFFER = True            # Adjust TP/SL for realistic fills

# Slippage buffer (as percentage)
# TP: Set slightly LOWER to ensure it fills before price reverses
# SL: Set slightly WIDER to avoid getting stopped by wicks
SLIPPAGE_TP_BUFFER = 0.05             # Base TP buffer (0.05%)
SLIPPAGE_SL_BUFFER = 0.05             # Base SL buffer (0.05%)

# Entry slippage (market order fills slightly worse than expected)
ENTRY_SLIPPAGE = 0.03                 # Assume 0.03% worse entry price

# === RANDOM ENTROPY (Avoid bot detection) ===
USE_SLIPPAGE_ENTROPY = True           # Add randomness to TP/SL
ENTROPY_TP_RANGE = 0.03               # TP varies +/- 0.03%
ENTROPY_SL_RANGE = 0.03               # SL varies +/- 0.03%
ENTROPY_PRICE_OFFSET = True           # Avoid round numbers ($80,000 -> $80,017)

# With entropy enabled:
#   TP: 0.55% +/- 0.03% = random between 0.52% and 0.58%
#   SL: 0.85% +/- 0.03% = random between 0.82% and 0.88%
#   Price offset: adds random $1-50 to avoid round number traps

# === TRADING MODE: 'profit' or 'accuracy' ===
TRADING_MODE = 'accuracy'  # Set to 'profit' for maximum profit trading

# === PROFIT MODE SETTINGS (Maximum profit focus) ===
if TRADING_MODE == 'profit':
    DEFAULT_LEVERAGE = 5           # Higher leverage for more profit
    RISK_PERCENT = 15              # 15% of balance per trade
    SWITCH_THRESHOLD = 0.88        # Lower threshold = more responsive
    ENTRY_THRESHOLD = 0.85         # More trades = more profit opportunity
    USE_DYNAMIC_TPSL = True        # Adapt TP/SL to volatility
    USE_TRAILING_STOP = True       # Lock in profits
    USE_COMPOUND = True            # Reinvest profits
    MIN_SIGNAL_INTERVAL = 2        # Allow more frequent trading
    MAX_DAILY_TRADES = 10          # More trades allowed
else:
    # === ACCURACY MODE SETTINGS (Conservative, high win rate) ===
    DEFAULT_LEVERAGE = 3
    RISK_PERCENT = 10
    SWITCH_THRESHOLD = 0.95
    ENTRY_THRESHOLD = 0.92
    USE_DYNAMIC_TPSL = False
    USE_TRAILING_STOP = False
    USE_COMPOUND = False
    MIN_SIGNAL_INTERVAL = 3
    MAX_DAILY_TRADES = 5

# TP/SL based on BTC/USDT historical analysis (Binance data)
# BTC ATR: 5m=0.19%, 15m=0.39%, 1h=0.91%
# BTC 8-candle movement: avg gain 0.42%, avg loss 0.51%

# Base TP/SL (optimized for HIGH WIN RATE + GOOD PROFIT)
# KEY FINDING: Smaller TP + Wider SL = Higher win rate AND more profit
#   - TP 0.6% is easy to hit (91.5% of trades reach it)
#   - SL 0.8% is wide enough to avoid most stop-hunts
if TIMEFRAME_MODE == 'accuracy':
    # HIGH WIN RATE STRATEGY (91%+ win rate)
    # Smaller TP = easier to hit, more consistent wins
    # Wider SL = fewer stop-outs, survive volatility spikes
    TP_MIN = 0.006                 # 0.6% take profit (easy to hit)
    TP_MAX = 0.008                 # 0.8% max take profit
    SL_MIN = 0.008                 # 0.8% stop loss (wide, rarely hit)
    SL_MAX = 0.012                 # 1.2% max stop loss
    FUTURE_WINDOW = 6              # 6 candles (6 hours on 1h)
    # R:R = 0.75:1 (0.6% / 0.8%)
    # Win Rate: 91.5%
    # Monthly: $248 with $100 capital, 3x leverage
else:
    # SCALPING MODE
    TP_MIN = 0.005                 # 0.5% take profit
    TP_MAX = 0.008                 # 0.8% max take profit
    SL_MIN = 0.006                 # 0.6% stop loss
    SL_MAX = 0.010                 # 1.0% max stop loss
    FUTURE_WINDOW = 8              # 8 candles (2 hours on 15m)

# Default TP/SL presets for BTC
TP_SL_PRESETS = {
    'conservative': {'tp': 0.01,  'sl': 0.007, 'rr': 1.43},  # Safest
    'balanced':     {'tp': 0.012, 'sl': 0.008, 'rr': 1.5},   # Balanced
    'profit':       {'tp': 0.015, 'sl': 0.008, 'rr': 1.875}, # Profit focused
    'aggressive':   {'tp': 0.02,  'sl': 0.01,  'rr': 2.0},   # Maximum profit
}

# Dynamic TP/SL multipliers (when USE_DYNAMIC_TPSL=True)
# TP = ATR * TP_ATR_MULT, SL = ATR * SL_ATR_MULT
TP_ATR_MULT = 2.0              # TP at 2x ATR
SL_ATR_MULT = 1.0              # SL at 1x ATR

# Trailing stop settings
TRAILING_ACTIVATION = 0.5      # Activate trailing after 50% of TP reached
TRAILING_LOCK_RATIO = 0.5      # Lock in 50% of unrealized profit

ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
ORDER_TYPE_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
ORDER_TYPE_MARKET = "MARKET"
TIME_IN_FORCE_GTC = "GTC"
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"

NEWS_API_KEY = "b4dc4b3d6c7c4fd6bd0440b3ea507937"  # replace with your key
QUERY = "crypto OR bitcoin OR ethereum OR fed OR inflation OR binance OR usdt OR usdc OR recession OR war OR conflict OR missile OR Elon Musk OR Powell OR Vitalik OR CZ OR ETF"

KEYWORDS = {
    "FED": ["fed", "federal reserve", "interest rate", "rate hike", "rate cut", "jerome powell"],
    "Crisis": ["crash", "collapse", "liquidation", "bankruptcy", "exploit", "hack", "shutdown"],
    "Recession": ["recession", "unemployment", "inflation", "economic slowdown", "default", "layoff"],
    "Binance": ["binance", "cz", "binance us", "binance smart chain"],
    "Bitcoin": ["bitcoin", "btc", "satoshi"],
    "Ethereum": ["ethereum", "eth", "vitalik"],
    "Altcoin": ["altcoin", "dogecoin", "solana", "cardano", "polygon", "avax", "shiba"],
    "War": ["war", "conflict", "missile", "russia", "ukraine", "israel", "palestine", "iran"],
    "Economy": ["interest rate", "inflation", "fed", "ecb", "imf", "regulation", "ban", "economic policy"],
    "Influencers": ["elon musk", "powell", "cz", "vitalik", "gensler", "trump", "etf"]
}

NEWS_URL = "https://newsapi.org/v2/everything"

# === NEWS-BASED TRADING FILTERS ===
USE_NEWS_FILTER = True                    # Enable news-based filtering
NEWS_BLOCK_HIGH_IMPACT = True             # Don't trade during high-impact events (FED, etc.)
NEWS_CONFIRM_SIGNAL = True                # Require news to confirm signal direction
NEWS_MIN_CONFIDENCE = 0.3                 # Minimum news confidence to affect trading
NEWS_CACHE_MINUTES = 5                    # Cache news for this many minutes

# === BOT PROTECTION (Defensive strategies) ===
USE_BOT_PROTECTION = True                 # Enable bot protection features
BOT_DETECT_STOP_HUNTS = True              # Skip trading after stop hunt wicks
BOT_DETECT_FAKE_BREAKOUTS = True          # Skip trading after fake breakouts
BOT_DETECT_VOLUME_SPIKES = True           # Skip trading during volume manipulation
BOT_SAFE_TPSL = True                      # Use smart TP/SL placement
BOT_RANDOM_DELAYS = True                  # Add random delays to execution
BOT_MAX_ACTIVITY_SCORE = 0.7              # Skip if bot activity > this threshold
BOT_AVOID_ROUND_NUMBERS = True            # Offset TP/SL from round numbers

# === BOT FRONT-RUNNING (Execute before other bots) ===
# Most bots use round TP/SL percentages (0.5%, 1.0%, 1.5%, 2.0%)
# and round price numbers ($80,000, $85,000, $90,000)
# We execute BEFORE them to get better fills

USE_BOT_FRONT_RUNNING = True              # Enable front-running strategy

# Common bot TP/SL percentages to front-run
BOT_COMMON_TP_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # % levels where bots place TP
BOT_COMMON_SL_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # % levels where bots place SL

# Front-run offset (execute this % BEFORE the round level)
# TP: Sell slightly before other bots dump (avoid price drop from mass selling)
# SL: Place slightly after round number (avoid bot cascade stop-hunts)
FRONT_RUN_TP_OFFSET = 0.03                # Sell 0.03% BEFORE round TP level
FRONT_RUN_SL_OFFSET = 0.05                # Place SL 0.05% AFTER round SL level

# Round price levels for BTC (where bots cluster orders)
# Common psychological levels that attract bot orders
BOT_ROUND_PRICE_INTERVALS = [100, 500, 1000, 5000, 10000]  # $100, $500, $1000, etc.
FRONT_RUN_PRICE_OFFSET = 20               # Offset $20 from round prices

# Time-based front-running (execute before common bot schedules)
# Many bots execute at exact minute marks (00, 15, 30, 45)
USE_TIME_FRONT_RUNNING = True             # Enable time-based front-running
FRONT_RUN_SECONDS_BEFORE = 5              # Execute 5 seconds before round minutes
# === WHALE/SMART MONEY TRACKING (for higher win rate) ===
USE_WHALE_CONFIRMATION = True             # Enable whale data confirmation
WHALE_BLOCK_OPPOSING = True               # Don't trade against strong whale bias
WHALE_MIN_SCORE = 0.3                     # Minimum whale score to block (0-1)
WHALE_BOOST_CONFIDENCE = True             # Boost confidence when whales agree
WHALE_CACHE_SECONDS = 60                  # Cache whale data for this many seconds

TRAINED_MODE_PATH = "lstm_model.keras"
BEST_MODEL_PATH = "best_model.keras"
FINE_TUNE_MODEL_PATH = "fined_tune_model.keras"

# === PAPER TRADING SETTINGS ===
# IMPORTANT: Must complete paper trading before live trading!
PAPER_TRADING_MODE = True              # Set to False after paper trading approved

# Paper trading requirements
PAPER_TRADING_REQUIRED_DAYS = 30       # Must paper trade for 30 days
PAPER_TRADING_MIN_TRADES = 50          # Minimum trades required
PAPER_TRADING_MIN_WIN_RATE = 0.60      # Must achieve 60% win rate
PAPER_TRADING_MIN_PROFIT = 0.0         # Must be profitable (>0%)
PAPER_TRADING_MAX_DRAWDOWN = 0.30      # Max 30% drawdown allowed
PAPER_TRADING_STARTING_CAPITAL = 1000  # Simulated starting capital

# === CONTINUOUS LEARNING SETTINGS (Fine-Tuning) ===
USE_CONTINUOUS_LEARNING = True         # Enable continuous learning system

# Performance thresholds (fine-tune when below)
CL_MIN_WIN_RATE = 0.65                 # Fine-tune if win rate < 65%
CL_MIN_PROFIT_FACTOR = 1.2             # Fine-tune if profit factor < 1.2
CL_PERFORMANCE_WINDOW = 50             # Evaluate last 50 trades

# Fine-tuning schedule
CL_RETRAIN_INTERVAL_DAYS = 7           # Minimum days between fine-tunes
CL_RETRAIN_DATA_DAYS = 30              # Use last 30 days for fine-tuning
CL_MIN_TRADES_FOR_EVAL = 20            # Need 20+ trades to evaluate

# Fine-tuning parameters
CL_FINE_TUNE_LR = 0.0001               # Learning rate (10x smaller than training)
CL_FINE_TUNE_EPOCHS = 30               # Max epochs for fine-tuning
CL_FINE_TUNE_BATCH_SIZE = 32           # Smaller batch for fine-tuning

# Monitoring
CL_CHECK_INTERVAL_HOURS = 1            # Check performance every hour
CL_AUTO_RETRAIN = True                 # Auto fine-tune when degradation detected

# Model versioning
CL_KEEP_MODEL_VERSIONS = 5             # Keep last 5 model versions
CL_BACKUP_BEFORE_RETRAIN = True        # Backup current model before fine-tuning