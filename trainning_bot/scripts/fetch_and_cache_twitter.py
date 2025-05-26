# scripts/fetch_and_cache_twitter.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.fetch_news import fetch_twitter_sentiment
import pandas as pd
from config import TWITTER_BEARER_TOKEN, TWITTER_USERS

print("🔁 Fetching tweets from Twitter API...")

df = fetch_twitter_sentiment(bearer_token=TWITTER_BEARER_TOKEN, usernames=TWITTER_USERS, max_results=30)
df.to_csv("cache/twitter.csv", index=False)

print("✅ Cached Twitter data to cache/twitter.csv")
