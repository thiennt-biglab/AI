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
QUERY = "crypto OR bitcoin OR ethereum OR war OR inflation OR fed OR elon musk OR trump"

KEYWORDS = {
    "war": ["war", "conflict", "missile", "Russia", "Ukraine", "Israel", "Palestine", "Iran"],
    "economy": ["interest rate", "inflation", "Fed", "ECB", "IMF", "regulation", "ban", "economic policy"],
    "influencers": ["Elon Musk", "Powell", "CZ", "Vitalik", "Gensler", "Trump", "ETF"]
}

NEWS_URL = "https://newsapi.org/v2/everything"