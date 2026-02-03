# historical_news.py
# Fetch and align historical news with training data (1 month from NewsAPI)

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
import json
import os
import time

try:
    from config import NEWS_API_KEY
except ImportError:
    NEWS_API_KEY = ""

# Expanded query for crypto + macro events
QUERY = "bitcoin OR crypto OR BTC OR ethereum OR federal reserve OR gold OR china trade"

# Comprehensive keywords for market-moving events
KEYWORDS = {
    # Central Banks & Monetary Policy
    "FED": ["fed", "federal reserve", "interest rate", "fomc", "powell", "rate hike", "rate cut", "quantitative", "tapering"],
    "ECB": ["ecb", "european central bank", "lagarde"],
    "BOJ": ["bank of japan", "boj", "yen"],

    # Geopolitical - War & Conflicts
    "War": ["war", "military", "invasion", "missile", "attack", "troops", "conflict", "combat", "airstrike", "bomb"],
    "Russia": ["russia", "putin", "moscow", "kremlin", "ukraine", "russian"],
    "MiddleEast": ["iran", "israel", "gaza", "hamas", "hezbollah", "saudi", "iraq", "syria", "middle east", "tehran"],

    # China & Trade
    "China": ["china", "chinese", "beijing", "xi jinping", "ccp", "yuan", "renminbi", "hong kong", "taiwan"],
    "Trade": ["tariff", "trade war", "sanctions", "embargo", "export ban", "import", "trade deal"],

    # Safe Haven & Commodities
    "Gold": ["gold", "silver", "precious metal", "safe haven", "xau"],
    "Oil": ["oil", "crude", "opec", "petroleum", "gas price", "energy crisis", "brent", "wti"],

    # Financial Crisis Indicators
    "Crisis": ["crash", "collapse", "liquidation", "bankruptcy", "default", "recession", "depression", "crisis", "bailout", "bank run"],
    "Inflation": ["inflation", "cpi", "consumer price", "hyperinflation", "deflation", "stagflation"],

    # Crypto Specific
    "Bullish": ["rally", "surge", "breakout", "bullish", "moon", "pump", "all-time high", "ath", "accumulation", "institutional buying"],
    "Bearish": ["dump", "crash", "bearish", "selloff", "plunge", "capitulation", "liquidations", "whale sell"],
    "Regulation": ["sec", "regulation", "ban crypto", "lawsuit", "etf approved", "etf rejected", "gensler", "cftc"],
    "Adoption": ["adoption", "accept bitcoin", "institutional", "blackrock", "fidelity", "microstrategy", "el salvador"],
}

NEWS_CACHE_FILE = "news_history_cache.json"


def fetch_historical_news(days_back=30):
    """
    Fetch historical news from NewsAPI (free tier: 1 month max).
    Returns list of articles with timestamps and sentiment.
    """
    if not NEWS_API_KEY or NEWS_API_KEY == "your_newsapi_key_here":
        print("[WARN] NEWS_API_KEY not configured, skipping historical news")
        return []

    articles = []

    # NewsAPI free tier: can fetch up to 1 month back
    end_date = datetime.now()
    start_date = end_date - timedelta(days=min(days_back, 30))

    url = "https://newsapi.org/v2/everything"

    # Fetch in chunks to avoid rate limits
    current_date = start_date
    while current_date < end_date:
        chunk_end = min(current_date + timedelta(days=7), end_date)

        params = {
            "q": QUERY,
            "from": current_date.strftime("%Y-%m-%d"),
            "to": chunk_end.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": NEWS_API_KEY
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if data.get("status") == "ok":
                for article in data.get("articles", []):
                    pub_date = article.get("publishedAt", "")
                    if pub_date:
                        articles.append({
                            "timestamp": pub_date,
                            "title": article.get("title", ""),
                            "description": article.get("description", ""),
                            "source": article.get("source", {}).get("name", "")
                        })
                print(f"[INFO] Fetched {len(data.get('articles', []))} articles for {current_date.strftime('%Y-%m-%d')}")
            else:
                print(f"[WARN] NewsAPI error: {data.get('message', 'Unknown error')}")

        except Exception as e:
            print(f"[ERROR] Failed to fetch news: {e}")

        current_date = chunk_end
        time.sleep(1)  # Rate limit: 1 request per second

    print(f"[INFO] Total articles fetched: {len(articles)}")
    return articles


def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob."""
    if not text:
        return 0.0
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity  # -1 to 1
    except:
        return 0.0


def detect_keywords(text, keywords_dict):
    """Detect important keywords in text."""
    if not text:
        return {}

    text_lower = text.lower()
    detected = {}

    for category, words in keywords_dict.items():
        for word in words:
            if word.lower() in text_lower:
                detected[category] = detected.get(category, 0) + 1

    return detected


def process_articles(articles):
    """Process articles to extract sentiment and keyword features."""
    processed = []

    for article in articles:
        title = article.get("title", "") or ""
        desc = article.get("description", "") or ""
        full_text = f"{title} {desc}"

        # Sentiment analysis
        sentiment = analyze_sentiment(full_text)

        # Keyword detection
        keywords = detect_keywords(full_text, KEYWORDS)

        # Parse timestamp
        try:
            ts = pd.to_datetime(article["timestamp"])
        except:
            continue

        processed.append({
            "timestamp": ts,
            "sentiment": sentiment,
            # Monetary policy
            "fed_mentions": keywords.get("FED", 0) + keywords.get("ECB", 0) + keywords.get("BOJ", 0),
            "inflation_mentions": keywords.get("Inflation", 0),
            # Geopolitical
            "war_mentions": keywords.get("War", 0) + keywords.get("Russia", 0) + keywords.get("MiddleEast", 0),
            "china_mentions": keywords.get("China", 0) + keywords.get("Trade", 0),
            # Safe haven / commodities
            "gold_mentions": keywords.get("Gold", 0),
            "oil_mentions": keywords.get("Oil", 0),
            # Crisis
            "crisis_mentions": keywords.get("Crisis", 0),
            # Crypto specific
            "bullish_mentions": keywords.get("Bullish", 0) + keywords.get("Adoption", 0),
            "bearish_mentions": keywords.get("Bearish", 0),
            "regulation_mentions": keywords.get("Regulation", 0),
        })

    return processed


def aggregate_news_by_hour(processed_articles):
    """
    Aggregate news features by hour for alignment with 1h candles.
    Returns DataFrame with hourly news features.
    """
    if not processed_articles:
        return pd.DataFrame()

    df = pd.DataFrame(processed_articles)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.floor("H")

    # Aggregate by hour
    hourly = df.groupby("hour").agg({
        "sentiment": ["mean", "std", "count"],
        "fed_mentions": "sum",
        "inflation_mentions": "sum",
        "war_mentions": "sum",
        "china_mentions": "sum",
        "gold_mentions": "sum",
        "oil_mentions": "sum",
        "crisis_mentions": "sum",
        "bullish_mentions": "sum",
        "bearish_mentions": "sum",
        "regulation_mentions": "sum",
    }).reset_index()

    # Flatten column names
    hourly.columns = [
        "timestamp",
        "news_sentiment_mean",
        "news_sentiment_std",
        "news_count",
        "news_fed_mentions",
        "news_inflation_mentions",
        "news_war_mentions",
        "news_china_mentions",
        "news_gold_mentions",
        "news_oil_mentions",
        "news_crisis_mentions",
        "news_bullish_mentions",
        "news_bearish_mentions",
        "news_regulation_mentions",
    ]

    # Fill NaN std with 0
    hourly["news_sentiment_std"] = hourly["news_sentiment_std"].fillna(0)

    # Create composite scores
    # Geopolitical risk score (higher = more risk = bearish)
    hourly["news_geopolitical_risk"] = (
        hourly["news_war_mentions"] * 0.4 +
        hourly["news_china_mentions"] * 0.2 +
        hourly["news_crisis_mentions"] * 0.3 +
        hourly["news_inflation_mentions"] * 0.1
    ).clip(0, 10) / 10  # Normalize to 0-1

    # Safe haven demand (gold + war = flight to safety, often bearish crypto)
    hourly["news_safe_haven"] = (
        hourly["news_gold_mentions"] * 0.5 +
        hourly["news_war_mentions"] * 0.3 +
        hourly["news_oil_mentions"] * 0.2
    ).clip(0, 10) / 10

    # Crypto sentiment score
    hourly["news_crypto_score"] = (
        hourly["news_sentiment_mean"] * 0.3 +
        (hourly["news_bullish_mentions"] - hourly["news_bearish_mentions"]) * 0.1 -
        hourly["news_crisis_mentions"] * 0.2 -
        hourly["news_regulation_mentions"] * 0.1 -
        hourly["news_geopolitical_risk"] * 0.2
    ).clip(-1, 1)

    # Overall news score
    hourly["news_score"] = hourly["news_crypto_score"]

    return hourly


def save_news_cache(articles):
    """Save fetched articles to cache file."""
    cache_data = {
        "fetched_at": datetime.now().isoformat(),
        "articles": articles
    }
    with open(NEWS_CACHE_FILE, "w") as f:
        json.dump(cache_data, f)
    print(f"[INFO] Saved {len(articles)} articles to cache")


def load_news_cache():
    """Load articles from cache if recent enough."""
    if not os.path.exists(NEWS_CACHE_FILE):
        return None

    try:
        with open(NEWS_CACHE_FILE, "r") as f:
            cache_data = json.load(f)

        fetched_at = datetime.fromisoformat(cache_data["fetched_at"])
        age_hours = (datetime.now() - fetched_at).total_seconds() / 3600

        # Cache valid for 24 hours
        if age_hours < 24:
            print(f"[INFO] Using cached news data ({age_hours:.1f}h old)")
            return cache_data["articles"]
        else:
            print(f"[INFO] Cache expired ({age_hours:.1f}h old), fetching fresh data")
            return None
    except Exception as e:
        print(f"[WARN] Failed to load cache: {e}")
        return None


def get_historical_news_features(days_back=30, use_cache=True):
    """
    Main function to get historical news features aligned by hour.
    Returns DataFrame with news features indexed by timestamp.
    """
    # Try cache first
    if use_cache:
        articles = load_news_cache()
    else:
        articles = None

    # Fetch if no cache
    if articles is None:
        articles = fetch_historical_news(days_back)
        if articles:
            save_news_cache(articles)

    if not articles:
        print("[WARN] No news data available, returning empty features")
        return pd.DataFrame()

    # Process and aggregate
    processed = process_articles(articles)
    hourly_features = aggregate_news_by_hour(processed)

    if not hourly_features.empty:
        hourly_features = hourly_features.set_index("timestamp")
        print(f"[INFO] Created news features for {len(hourly_features)} hours")

    return hourly_features


def align_news_with_candles(candle_df, news_df, timestamp_col="timestamp"):
    """
    Align news features with candle data by timestamp.
    Fills missing news hours with neutral values.
    """
    if news_df.empty:
        # Return neutral features
        candle_df["news_sentiment_mean"] = 0.0
        candle_df["news_sentiment_std"] = 0.0
        candle_df["news_count"] = 0
        candle_df["news_fed_mentions"] = 0
        candle_df["news_crisis_mentions"] = 0
        candle_df["news_bullish_mentions"] = 0
        candle_df["news_bearish_mentions"] = 0
        candle_df["news_score"] = 0.0
        return candle_df

    # Merge on timestamp
    if timestamp_col in candle_df.columns:
        candle_df[timestamp_col] = pd.to_datetime(candle_df[timestamp_col])
        candle_df = candle_df.set_index(timestamp_col)

    # Join news features
    aligned = candle_df.join(news_df, how="left")

    # Fill missing with neutral values
    fill_values = {
        "news_sentiment_mean": 0.0,
        "news_sentiment_std": 0.0,
        "news_count": 0,
        "news_fed_mentions": 0,
        "news_crisis_mentions": 0,
        "news_bullish_mentions": 0,
        "news_bearish_mentions": 0,
        "news_score": 0.0,
    }
    aligned = aligned.fillna(fill_values)

    return aligned.reset_index()


# Feature names for training
NEWS_FEATURE_NAMES = [
    # Base sentiment
    "news_sentiment_mean",
    "news_sentiment_std",
    "news_count",
    # Monetary policy
    "news_fed_mentions",
    "news_inflation_mentions",
    # Geopolitical
    "news_war_mentions",
    "news_china_mentions",
    # Commodities
    "news_gold_mentions",
    "news_oil_mentions",
    # Crisis & Crypto
    "news_crisis_mentions",
    "news_bullish_mentions",
    "news_bearish_mentions",
    "news_regulation_mentions",
    # Composite scores
    "news_geopolitical_risk",
    "news_safe_haven",
    "news_crypto_score",
    "news_score",
]


if __name__ == "__main__":
    # Test the module
    print("Fetching historical news...")
    news_features = get_historical_news_features(days_back=30)
    print(f"\nNews features shape: {news_features.shape}")
    if not news_features.empty:
        print(f"\nSample data:\n{news_features.head()}")
        print(f"\nFeature stats:\n{news_features.describe()}")
