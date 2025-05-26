# data/fetch_binance.py

import ccxt
import pandas as pd
from config import HISTORY_LIMIT, TIMEFRAME, SYMBOL

def fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=HISTORY_LIMIT):
    binance = ccxt.binance({
        'enableRateLimit': True
    })

    # Lấy dữ liệu giá
    ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # Chuyển sang DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df

if __name__ == "__main__":
    df = fetch_ohlcv()
    print(df.tail())
