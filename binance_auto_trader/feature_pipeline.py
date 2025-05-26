from binance.client import Client
import pandas as pd
import ta
from config import API_KEY, API_SECRET, SYMBOL, INTERVALS

client = Client(API_KEY, API_SECRET)

def fetch_full_klines(symbol, interval, total_limit=5000):
    all_klines = []
    last_time = None
    while len(all_klines) < total_limit:
        fetch_limit = min(1000, total_limit - len(all_klines))
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=fetch_limit,
            endTime=last_time
        )
        if not klines or len(klines) == 0:
            break
        all_klines = klines + all_klines
        last_time = klines[0][0] - 1
        if len(klines) < fetch_limit:
            break
    return all_klines

def fetch_features_multi_timeframe():
    all_dfs = []
    for interval in INTERVALS:
        klines = fetch_full_klines(SYMBOL, interval, total_limit=5000)
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume',
            'close_time','quote_asset_volume','num_trades',
            'taker_buy_base','taker_buy_quote','ignore']
        )
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd_diff'] = ta.trend.MACD(df['close']).macd_diff()
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        df['stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        df['stoch_d'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch_signal()
        df['mom'] = ta.momentum.ROCIndicator(df['close']).roc()
        df.dropna(inplace=True)
        df = df[[
            'rsi', 'macd_diff', 'ema_20', 'ema_50', 'atr', 'volume',
            'cci', 'stoch_k', 'stoch_d', 'mom']]
        df.columns = [f"{c}_{interval}" for c in df.columns]
        all_dfs.append(df.reset_index(drop=True))

    combined = pd.concat(all_dfs, axis=1).dropna().reset_index(drop=True)
    return combined
