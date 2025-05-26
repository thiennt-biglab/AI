# data/fetch_news.py

import feedparser
import time
import tweepy
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import TWITTER_USERS, TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET, TWITTER_USERS, TWITTER_LIMIT_NEW, TWITTER_BEARER_TOKEN, NEWSAPI_KEY 

analyzer = SentimentIntensityAnalyzer()

def fetch_binance_news_from_newsapi(api_key: str, limit=100):
    url = f"https://newsapi.org/v2/everything?q=binance&language=en&pageSize={limit}&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])

    news = []
    for article in articles:
        title = article["title"]
        published = article["publishedAt"]
        score = analyzer.polarity_scores(title)["compound"]
        news.append({
            "time": pd.to_datetime(published),
            "source": "NewsAPI:Binance",
            "text": title,
            "sentiment": score
        })

    return pd.DataFrame(news)



# ---------------------------
# TWITTER API
# ---------------------------

def fetch_twitter_sentiment_safe(bearer_token, usernames, max_results):
    try:
        return fetch_twitter_sentiment(bearer_token, usernames, max_results)
    except tweepy.TooManyRequests:
        print("⚠️ Twitter rate limited — loading from cache")
        return pd.read_csv("cache/twitter.csv", parse_dates=["time"])


def fetch_twitter_sentiment(bearer_token, usernames=TWITTER_USERS, max_results=5):
    client = tweepy.Client(bearer_token=bearer_token)

    tweet_data = []

    for username in usernames:
        # Get user ID by username
        user = client.get_user(username=username)
        user_id = user.data.id

        # Fetch tweets
        tweets = client.get_users_tweets(id=user_id, max_results=max_results)
        if tweets.data:
            for tweet in tweets.data:
                text = tweet.text
                score = analyzer.polarity_scores(text)['compound']
                tweet_data.append({
                    'time': pd.Timestamp(tweet.created_at),
                    'source': f'Twitter:{username}',
                    'text': text,
                    'sentiment': score
                })
        time.sleep(10)

    return pd.DataFrame(tweet_data)

# ---------------------------
# MAIN TEST
# ---------------------------
if __name__ == "__main__":
    binance_df = fetch_binance_news_from_newsapi(api_key=NEWSAPI_KEY)
    twitter_df = fetch_twitter_sentiment(bearer_token=TWITTER_BEARER_TOKEN)
    combined = pd.concat([binance_df, twitter_df])
    print(combined[['time', 'source', 'sentiment']].sort_values('time', ascending=False).head(10))
