from binance.client import Client
import pandas as pd
import ta
import requests
import yfinance as yf
import numpy as np
from config import API_KEY, API_SECRET, SYMBOL, INTERVALS
from news_sentiment import analyze_news
from trader import safe_api_call

client = Client(API_KEY, API_SECRET)

def fetch_full_klines(symbol, interval, total_limit=5000):
    all_klines = []
    last_time = None

    while len(all_klines) < total_limit:
        fetch_limit = min(1000, total_limit - len(all_klines))
        klines = safe_api_call(
            client.futures_klines,
            symbol=symbol,
            interval=interval,
            limit=fetch_limit,
            endTime=last_time
        )

        if not klines:
            break
        all_klines = klines + all_klines
        last_time = klines[0][0] - 1
        if len(klines) < fetch_limit:
            break
    return all_klines

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

def fetch_features_multi_timeframe():
    all_dfs = []

    for interval in INTERVALS:
        klines = fetch_full_klines(SYMBOL, interval, total_limit=5000)

        if not klines or len(klines) < 100:
            print(f"[ERROR] Klines for {interval} is too short or empty.")
            continue

        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume',
            'close_time','quote_asset_volume','num_trades',
            'taker_buy_base','taker_buy_quote','ignore'])

        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})

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
            # Volume-based features
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ema'] = ta.trend.EMAIndicator(df['volume'], window=20).ema_indicator()
            df['volume_ratio'] = df['volume'] / df['volume_ema']

            # Candlestick body features
            df['candle_body'] = abs(df['close'] - df['open'])
            df['candle_range'] = df['high'] - df['low']
            df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']

            # Normalize w.r.t range to keep things scale-invariant
            df['body_to_range'] = df['candle_body'] / df['candle_range'].replace(0, np.nan)
            df['upper_to_range'] = df['upper_shadow'] / df['candle_range'].replace(0, np.nan)
            df['lower_to_range'] = df['lower_shadow'] / df['candle_range'].replace(0, np.nan)

        except Exception as e:
            print(f"[ERROR] Failed to compute indicators for {interval}: {e}")
            continue

        df.dropna(inplace=True)

        required_cols = [
            'rsi', 'rsi_diff', 'macd_diff', 'ema_20', 'ema_50', 'atr', 'volume',
            'cci', 'stoch_k', 'stoch_d', 'mom', 'bb_width',
            'volume_change', 'volume_ema', 'volume_ratio',
            'candle_body', 'candle_range', 'upper_shadow', 'lower_shadow',
            'body_to_range', 'upper_to_range', 'lower_to_range'
        ]

        if not all(col in df.columns for col in required_cols):
            print(f"[ERROR] Missing expected indicators in {interval}")
            continue

        df = df[required_cols]
        df.columns = [f"{col}_{interval}" for col in df.columns]
        all_dfs.append(df.reset_index(drop=True))

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

    if not all_dfs:
        raise ValueError("[CRITICAL] All intervals failed. No data available for training or prediction.")

    # Giữ số dòng đồng nhất
    min_len = min(len(df) for df in all_dfs)
    all_dfs = [df.iloc[-min_len:].reset_index(drop=True) for df in all_dfs]

    combined = pd.concat(all_dfs, axis=1)

    if combined.empty:
        raise ValueError("[CRITICAL] Combined DataFrame is empty after concat and dropna.")

    # External features (chỉ cần thêm 1 lần cho toàn bộ chuỗi)
    try:
        combined['sentiment'] = analyze_news()
        combined['btc_dominance'] = get_btc_dominance()
        combined['dxy'] = get_dxy()
        combined['funding_rate'] = get_funding_rate(SYMBOL)
    except Exception as e:
        print(f"[WARN] Failed to fetch external features: {e}")
        combined['sentiment'] = 0.0
        combined['btc_dominance'] = 0.0
        combined['dxy'] = 0.0
        combined['funding_rate'] = 0.0

    return combined
