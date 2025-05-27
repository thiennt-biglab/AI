# news_sentiment.py
import requests
from transformers import pipeline
from datetime import datetime, timedelta
from config import *

# === Load Sentiment Pipeline ===
sentiment_pipeline = pipeline("sentiment-analysis")

def fetch_news():
    url = NEWS_URL
    from_date = (datetime.utcnow() - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "q": QUERY,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 50,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json().get("articles", [])

def categorize_news(title):
    matched = []
    for topic, words in KEYWORDS.items():
        if any(w.lower() in title.lower() for w in words):
            matched.append(topic)
    return matched

def analyze_news():
    articles = fetch_news()
    scores = []

    for news in articles:
        title = news['title']
        sentiment = sentiment_pipeline(title[:512])[0]  # truncate long titles
        label = sentiment['label']
        confidence = float(sentiment['score'])
        categories = categorize_news(title)

        if not categories:
            continue

        impact_score = 0
        for cat in categories:
            if label == 'POSITIVE':
                impact_score += 0.2
            elif label == 'NEGATIVE':
                impact_score -= 0.3

        scores.append(impact_score)

    if not scores:
        return 0.0

    total = sum(scores)
    return max(-0.5, min(0.5, total))  # clamp between -0.5 to +0.5

# For debug usage:
if __name__ == "__main__":
    print(f"Impact score: {analyze_news():.3f}")
