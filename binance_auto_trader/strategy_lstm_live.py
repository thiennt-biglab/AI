# strategy_lstm_live.py
import numpy as np
import pandas as pd
import ta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from feature_pipeline import fetch_features_multi_timeframe

# Load model and scaler
model = load_model("lstm_model.keras")
scaler = joblib.load("scaler.pkl")  # optional if saved during training

# Constants
WINDOW = 30
FEATURES = [
    'rsi', 'macd_diff', 'ema_20', 'ema_50', 'atr', 'volume'
]

# You may need to match these to actual feature columns used during training
def extract_features_live(df, interval_label="5m"):
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd_diff'] = macd.macd_diff()
    df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df.dropna(inplace=True)
    df = df[['rsi', 'macd_diff', 'ema_20', 'ema_50', 'atr', 'volume']]
    df.columns = [f"{col}_{interval_label}" for col in df.columns]
    return df

def prepare_input(df_combined):
    X_raw = df_combined.values
    X_scaled = scaler.transform(X_raw)
    X_seq = []
    for i in range(len(X_scaled) - WINDOW, len(X_scaled)):
        X_seq.append(X_scaled[i-WINDOW+1:i+1])
    X_input = np.array(X_seq)
    return X_input[-1:]  # latest sequence

def lstm_based_action(df_combined):
    model = load_model("lstm_model.keras")
    scaler = joblib.load("scaler.pkl")
    
    df = fetch_features_multi_timeframe()
    X_raw = df.values[-30:]  # giữ đúng khung
    X_scaled = scaler.transform(X_raw)
    X_input = np.expand_dims(X_scaled, axis=0)

    proba = model.predict(X_input)[0]
    pred = np.argmax(proba)
    confidence = float(np.max(proba))

    if pred == 1:
        return 'LONG', confidence
    elif pred == 2:
        return 'SHORT', confidence
    return 'HOLD', confidence


def select_leverage(confidence):
    # Clamp confidence trong khoảng an toàn
    confidence = max(0.7, min(confidence, 0.98))
    # Tuyến tính từ 1x (0.7) → 15x (0.98)
    leverage = 1 + (confidence - 0.7) / (0.98 - 0.7) * (15 - 1)
    print(f"[INFO] leverage: {leverage}")
    return round(leverage, 1)


