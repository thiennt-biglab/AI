# data/fetch_geopolitical_news.py

import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def fetch_geopolitical_news(api_key: str, limit=30):
    url = f"https://newsapi.org/v2/everything?q=war OR conflict OR missile OR crisis&language=en&sortBy=publishedAt&pageSize={limit}&apiKey={api_key}"
    resp = requests.get(url)
    articles = resp.json().get("articles", [])

    rows = []
    for a in articles:
        text = f"{a['title']} - {a['description'] or ''}"
        score = analyzer.polarity_scores(text)['compound']
        rows.append({
            "time": pd.to_datetime(a['publishedAt']),
            "source": "NewsAPI",
            "text": text,
            "sentiment": score
        })
    
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    API_KEY = "your_newsapi_key"
    df = fetch_geopolitical_news(API_KEY)
    print(df[['time', 'sentiment', 'text']].head())