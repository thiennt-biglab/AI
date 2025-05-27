API_KEY = "GJSYhlu02urxr9LaktR9606IhuUsCsTnjj7AmM0IYXRZmPqMDE66t5baGfSSA2Z0"
API_SECRET = "4iRKlKY9OTYNzY85D1Z4CRIQyq82oX5VCnh6htCT3dHtzsZ5OvGrO3rzuETNRmBg"

SYMBOL = "WIFUSDT"
INTERVALS = ["5m", "15m", "1h"]  # Multiple timeframes
DEFAULT_LEVERAGE = 5
RISK_PERCENT = 1.5  # % số dư

MIN_RSI_LONG = 28
MAX_RSI_SHORT = 72
SL_PERCENT = 1.0
TP_PERCENT = 1.5

SWITCH_THRESHOLD = 0.80
ENTRY_THRESHOLD = 0.70

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