# Re-import necessary modules after code execution state reset
import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from binance.client import Client
from config import API_KEY, API_SECRET, SYMBOL, INTERVALS
from news_sentiment import analyze_news
from trader import safe_api_call

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# Ensure cache directory exists
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Interval mapping to milliseconds
def get_interval_ms(interval):
    mapping = {
        '1m': 60_000, '3m': 180_000, '5m': 300_000, '15m': 900_000,
        '30m': 1_800_000, '1h': 3_600_000, '2h': 7_200_000,
        '4h': 14_400_000, '1d': 86_400_000
    }
    return mapping[interval]

# Fetch Klines with backfill logic
def fetch_full_klines(symbol, interval, total_limit=10000):
    interval_ms = get_interval_ms(interval)
    end_time = int(time.time() * 1000)
    fetch_limit = 1000
    loops = total_limit // fetch_limit
    start_time = end_time - loops * fetch_limit * interval_ms
    all_klines = []

    for _ in range(loops):
        klines = safe_api_call(
            client.futures_klines,
            symbol=symbol,
            interval=interval,
            limit=fetch_limit,
            startTime=start_time,
            endTime=start_time + fetch_limit * interval_ms
        )
        if not klines:
            break
        all_klines += klines
        start_time += fetch_limit * interval_ms

    return all_klines

# Market indicators
def get_btc_dominance():
    try:
        r = requests.get('https://api.coingecko.com/api/v3/global').json()
        return r['data']['market_cap_percentage']['btc']
    except:
        return 0.0

def get_dxy():
    try:
        dxy = yf.download('DX-Y.NYB', period='1d', interval='1m')
        return dxy['Close'][-1]
    except:
        return 0.0

def get_funding_rate(symbol="PEOPLEUSDT"):
    try:
        url = f'https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1'
        r = requests.get(url).json()
        return float(r[0]['fundingRate'])
    except:
        return 0.0

# Feature extraction per timeframe
def fetch_features_multi_timeframe():
    all_dfs = []

    for interval in INTERVALS:
        cache_path = os.path.join(CACHE_DIR, f"{SYMBOL}_{interval}.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
        else:
            klines = fetch_full_klines(SYMBOL, interval, total_limit=10000)
            if not klines or len(klines) < 100:
                print(f"[ERROR] Klines for {interval} is too short or empty.")
                continue

            df = pd.DataFrame(klines, columns=[
                'timestamp','open','high','low','close','volume',
                'close_time','quote_asset_volume','num_trades',
                'taker_buy_base','taker_buy_quote','ignore'])

            df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            df.drop_duplicates(subset='timestamp', inplace=True)
            df.sort_values(by='timestamp', inplace=True)

            try:
                df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
                df['rsi_diff'] = df['rsi'].diff()
                macd = ta.trend.MACD(df['close'])
                df['macd_diff'] = macd.macd_diff()
                df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
                df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
                df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
                df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
                stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
                df['stoch_k'] = stoch.stoch()
                df['stoch_d'] = stoch.stoch_signal()
                df['mom'] = ta.momentum.ROCIndicator(df['close']).roc()
                bb = ta.volatility.BollingerBands(df['close'])
                df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()
                df['volume_change'] = df['volume'].pct_change()
                df['volume_ema'] = ta.trend.EMAIndicator(df['volume'], window=20).ema_indicator()
                df['volume_ratio'] = df['volume'] / df['volume_ema']
                df['candle_body'] = abs(df['close'] - df['open'])
                df['candle_range'] = df['high'] - df['low']
                df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
                df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
                df['body_to_range'] = df['candle_body'] / df['candle_range'].replace(0, np.nan)
                df['upper_to_range'] = df['upper_shadow'] / df['candle_range'].replace(0, np.nan)
                df['lower_to_range'] = df['lower_shadow'] / df['candle_range'].replace(0, np.nan)
                df.dropna(inplace=True)
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.to_csv(cache_path, index=False)
            except Exception as e:
                print(f"[ERROR] Failed to compute indicators for {interval}: {e}")
                continue

        required_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df[required_cols]
        df.columns = [f"{col}_{interval}" for col in df.columns]
        all_dfs.append(df.reset_index(drop=True))

    if not all_dfs:
        raise ValueError("[CRITICAL] All intervals failed. No data available for training or prediction.")

    min_len = min(len(df) for df in all_dfs)
    all_dfs = [df.iloc[-min_len:].reset_index(drop=True) for df in all_dfs]
    combined = pd.concat(all_dfs, axis=1)

    external = {
        'sentiment': analyze_news(),
        'btc_dominance': get_btc_dominance(),
        'dxy': get_dxy(),
        'funding_rate': get_funding_rate(SYMBOL)
    }
    for key, val in external.items():
        combined[key] = val

    return combined
