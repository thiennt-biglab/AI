# config.example.py
# Copy this file to config.py and fill in your API keys
# NEVER commit config.py with real API keys!

# === API KEYS (Get from Binance) ===
API_KEY = "your_binance_api_key_here"
API_SECRET = "your_binance_api_secret_here"

# === TRADING SYMBOL ===
SYMBOL = "BTCUSDT"

# === TIMEFRAME SETTINGS ===
TIMEFRAME_MODE = 'accuracy'  # 'accuracy' or 'scalping'

if TIMEFRAME_MODE == 'accuracy':
    INTERVALS = ["1h"]   # 1h candles - cleaner signals
    SEQ_LEN_MODEL = 48   # 48 candles = 48 hours (2 days) lookback
else:
    INTERVALS = ["15m"]  # 15m for scalping
    SEQ_LEN_MODEL = 48

# === SESSION & VOLATILITY FILTERS ===
USE_SESSION_FILTER = False
ALLOWED_SESSIONS = [(0, 8)]
USE_VOLATILITY_FILTER = True
MAX_ATR_PERCENT = 0.5
MIN_ATR_PERCENT = 0.1

# === CRASH PROTECTION ===
USE_CRASH_PROTECTION = True
CRASH_ATR_THRESHOLD = 1.0
CRASH_VOLUME_SPIKE = 3.0
CRASH_SINGLE_DROP = 2.0
CRASH_4H_DROP = 2.0
CRASH_COOLDOWN_HOURS = 4

# === SLIPPAGE SETTINGS ===
USE_SLIPPAGE_BUFFER = True
SLIPPAGE_TP_BUFFER = 0.05
SLIPPAGE_SL_BUFFER = 0.05
ENTRY_SLIPPAGE = 0.03

# === RANDOM ENTROPY ===
USE_SLIPPAGE_ENTROPY = True
ENTROPY_TP_RANGE = 0.03
ENTROPY_SL_RANGE = 0.03
ENTROPY_PRICE_OFFSET = True

# === TRADING MODE ===
TRADING_MODE = 'accuracy'  # 'profit' or 'accuracy'

if TRADING_MODE == 'profit':
    DEFAULT_LEVERAGE = 5
    RISK_PERCENT = 15
    SWITCH_THRESHOLD = 0.88
    ENTRY_THRESHOLD = 0.60    # Lower threshold for more trades
    ENTROPY_THRESHOLD = 0.65  # Higher entropy allowed
    USE_DYNAMIC_TPSL = True
    USE_TRAILING_STOP = True
    USE_COMPOUND = True
    MIN_SIGNAL_INTERVAL = 2
    MAX_DAILY_TRADES = 10
else:
    DEFAULT_LEVERAGE = 3
    RISK_PERCENT = 10
    SWITCH_THRESHOLD = 0.95
    ENTRY_THRESHOLD = 0.80    # Very high confidence only
    ENTROPY_THRESHOLD = 0.45  # Very low entropy = very certain
    USE_DYNAMIC_TPSL = False
    USE_TRAILING_STOP = False
    USE_COMPOUND = False
    MIN_SIGNAL_INTERVAL = 3   # 3 hours between trades at 1h
    MAX_DAILY_TRADES = 3      # Max 3 trades per day

# === TP/SL SETTINGS (1h candles) ===
if TIMEFRAME_MODE == 'accuracy':
    TP_MIN = 0.012   # 1.2% TP for 1h
    TP_MAX = 0.020   # 2.0% max
    SL_MIN = 0.008   # 0.8% SL -> R:R = 1.5:1
    SL_MAX = 0.012   # 1.2% max
    FUTURE_WINDOW = 12  # 12 candles = 12 hours at 1h
else:
    TP_MIN = 0.006   # Scalping: tighter TP
    TP_MAX = 0.010
    SL_MIN = 0.004   # Tighter SL
    SL_MAX = 0.006
    FUTURE_WINDOW = 8

TP_SL_PRESETS = {
    'conservative': {'tp': 0.01,  'sl': 0.007, 'rr': 1.43},
    'balanced':     {'tp': 0.012, 'sl': 0.008, 'rr': 1.5},
    'profit':       {'tp': 0.015, 'sl': 0.008, 'rr': 1.875},
    'aggressive':   {'tp': 0.02,  'sl': 0.01,  'rr': 2.0},
}

TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.0
TRAILING_ACTIVATION = 0.5
TRAILING_LOCK_RATIO = 0.5

# === ORDER TYPES ===
ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
ORDER_TYPE_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
ORDER_TYPE_MARKET = "MARKET"
TIME_IN_FORCE_GTC = "GTC"
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"

# === NEWS API (Optional) ===
NEWS_API_KEY = "your_newsapi_key_here"  # Get from newsapi.org

# News query expanded for geopolitical events
QUERY = "bitcoin OR crypto OR federal reserve OR china trade OR war OR gold price OR oil price"

# Keywords are now defined in historical_news.py with expanded categories:
# - Monetary: FED, ECB, BOJ, inflation
# - Geopolitical: War, Russia, Iran, Israel, China, Trade
# - Commodities: Gold, Oil
# - Crisis: crash, recession, bankruptcy
# - Crypto: bullish, bearish, regulation, adoption

NEWS_URL = "https://newsapi.org/v2/everything"

# === NEWS FILTER ===
USE_NEWS_FILTER = True
NEWS_BLOCK_HIGH_IMPACT = True
NEWS_CONFIRM_SIGNAL = True
NEWS_MIN_CONFIDENCE = 0.3
NEWS_CACHE_MINUTES = 5

# === BOT PROTECTION ===
USE_BOT_PROTECTION = True
BOT_DETECT_STOP_HUNTS = True
BOT_DETECT_FAKE_BREAKOUTS = True
BOT_DETECT_VOLUME_SPIKES = True
BOT_SAFE_TPSL = True
BOT_RANDOM_DELAYS = True
BOT_MAX_ACTIVITY_SCORE = 0.7
BOT_AVOID_ROUND_NUMBERS = True

# === BOT FRONT-RUNNING ===
USE_BOT_FRONT_RUNNING = True
BOT_COMMON_TP_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
BOT_COMMON_SL_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
FRONT_RUN_TP_OFFSET = 0.03
FRONT_RUN_SL_OFFSET = 0.05
BOT_ROUND_PRICE_INTERVALS = [100, 500, 1000, 5000, 10000]
FRONT_RUN_PRICE_OFFSET = 20
USE_TIME_FRONT_RUNNING = True
FRONT_RUN_SECONDS_BEFORE = 5

# === WHALE TRACKING ===
USE_WHALE_CONFIRMATION = True
WHALE_BLOCK_OPPOSING = True
WHALE_MIN_SCORE = 0.3
WHALE_BOOST_CONFIDENCE = True
WHALE_CACHE_SECONDS = 60

# === MODEL PATHS ===
TRAINED_MODE_PATH = "lstm_model.keras"
BEST_MODEL_PATH = "best_model.keras"
FINE_TUNE_MODEL_PATH = "fined_tune_model.keras"

# === PAPER TRADING ===
PAPER_TRADING_MODE = True  # Set to False after paper trading approved
PAPER_TRADING_REQUIRED_DAYS = 30
PAPER_TRADING_MIN_TRADES = 50
PAPER_TRADING_MIN_WIN_RATE = 0.60
PAPER_TRADING_MIN_PROFIT = 0.0
PAPER_TRADING_MAX_DRAWDOWN = 0.30
PAPER_TRADING_STARTING_CAPITAL = 1000

# === CONTINUOUS LEARNING ===
USE_CONTINUOUS_LEARNING = True
CL_MIN_WIN_RATE = 0.65
CL_MIN_PROFIT_FACTOR = 1.2
CL_PERFORMANCE_WINDOW = 50
CL_RETRAIN_INTERVAL_DAYS = 7
CL_RETRAIN_DATA_DAYS = 30
CL_MIN_TRADES_FOR_EVAL = 20
CL_FINE_TUNE_LR = 0.0001
CL_FINE_TUNE_EPOCHS = 30
CL_FINE_TUNE_BATCH_SIZE = 32
CL_CHECK_INTERVAL_HOURS = 1
CL_AUTO_RETRAIN = True
CL_KEEP_MODEL_VERSIONS = 5
CL_BACKUP_BEFORE_RETRAIN = True
