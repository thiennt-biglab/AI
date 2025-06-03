import os
import time
import numpy as np
import pandas as pd
import ta
import joblib
from binance.client import Client
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from loss import custom_loss  # hoặc focal, risk_loss
from config import API_KEY, API_SECRET, SYMBOL

client = Client(API_KEY, API_SECRET)

# Fetch 1-day data for 1-minute interval
def fetch_1day_klines(symbol, interval='1m'):
    now = int(time.time() * 1000)
    one_day_ms = 24 * 60 * 60 * 1000
    start = now - one_day_ms
    klines = client.futures_klines(symbol=symbol, interval=interval, startTime=start, endTime=now, limit=1000)
    return klines

# Prepare and process data
def preprocess(klines):
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})

    # Indicators
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd_diff'] = ta.trend.MACD(df['close']).macd_diff()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df.dropna(inplace=True)

    # Labels
    future_return = df['ema_20'].shift(-3) / df['ema_20'] - 1
    labels = np.zeros(len(future_return))
    labels[future_return > 0.0025] = 1
    labels[future_return < -0.0025] = 2
    df = df[:-3]  # bỏ phần không có label
    labels = labels[:-3]

    return df, labels

# Create sequences for LSTM
def create_sequences(df, labels, sequence_len=30):
    X_seq, y_seq = [], []
    features = df.columns
    for i in range(len(df) - sequence_len):
        X_seq.append(df.iloc[i:i+sequence_len][features].values)
        y_seq.append(labels[i+sequence_len])
    return np.array(X_seq), np.array(y_seq)

# Main finetune routine
def finetune():
    klines = fetch_1day_klines(SYMBOL)
    df, labels = preprocess(klines)
    X_seq, y_seq = create_sequences(df, labels)

    if len(X_seq) == 0:
        print("[!] Not enough data to train.")
        return

    # Scale and encode
    scaler = joblib.load("scaler.pkl")
    X_seq = scaler.transform(X_seq.reshape(-1, X_seq.shape[2])).reshape(X_seq.shape)
    y_seq = to_categorical(y_seq, num_classes=3)

    # Load and train
    model = load_model("model.h5", custom_objects={'loss': custom_loss})
    model.fit(X_seq, y_seq, epochs=5, batch_size=32, validation_split=0.1)
    model.save("model_finetuned.h5")
    print("[✅] Fine-tuning complete and saved as model_finetuned.h5")

if __name__ == "__main__":
    finetune()
