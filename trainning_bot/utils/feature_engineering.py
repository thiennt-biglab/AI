# utils/feature_engineering.py

import pandas as pd
import ta

def add_technical_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['ema'] = ta.trend.EMAIndicator(df['close'], window=14).ema_indicator()
    df['macd'] = ta.trend.macd_diff(df['close'])
    df.dropna(inplace=True)
    return df

def merge_sentiment_to_price(price_df, sentiment_df, prefix='sentiment'):
    if sentiment_df.empty or 'time' not in sentiment_df.columns:
        price_df[prefix] = 0.0
        return price_df

    try:
        sentiment_df['time'] = pd.to_datetime(sentiment_df['time'], errors='coerce')
        sentiment_df.dropna(subset=['time'], inplace=True)
        sentiment_df['time'] = sentiment_df['time'].dt.tz_localize(None)
        sentiment_df = sentiment_df.set_index('time').sort_index()
    except Exception as e:
        print(f"⚠️ Error processing sentiment time column: {e}")
        price_df[prefix] = 0.0
        return price_df

    price_df = price_df.copy()
    price_df[prefix] = 0.0

    for idx in price_df.index:
        recent = sentiment_df[sentiment_df.index <= idx]
        if not recent.empty:
            avg = recent[-5:].sentiment.mean()
            price_df.loc[idx, prefix] = avg

    return price_df
