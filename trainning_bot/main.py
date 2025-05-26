from data.fetch_binance import fetch_ohlcv
from data.fetch_news import fetch_binance_news, fetch_twitter_sentiment
from data.fetch_geopolitical_news import fetch_geopolitical_news
from utils.feature_engineering import add_technical_indicators, merge_sentiment_to_price
import pandas as pd
from config import NEWSAPI_KEY, SYMBOL, TIMEFRAME, HISTORY_LIMIT

price_df = fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=HISTORY_LIMIT)
price_df = add_technical_indicators(price_df)

news_df = fetch_binance_news(limit=30)
twitter_df = fetch_twitter_sentiment(limit=30)
sentiment_df = pd.concat([news_df, twitter_df])

geo_df = fetch_geopolitical_news(NEWSAPI_KEY, limit=30)

price_df = merge_sentiment_to_price(price_df, sentiment_df, prefix='sentiment')
price_df = merge_sentiment_to_price(price_df, geo_df, prefix='geo_sentiment')

print(price_df[['close', 'rsi', 'ema', 'sentiment', 'geo_sentiment']].tail())
