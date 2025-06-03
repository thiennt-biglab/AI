# strategy_lstm_live.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from news_sentiment import analyze_news
from config import *

# Load model and scaler
model = load_model(FINE_TUNE_MODEL_PATH)
scaler = joblib.load("scaler.pkl")  # optional if saved during training

FEATURES = [f"{col}_{interval}" for interval in INTERVALS for col in [
    'rsi', 'rsi_diff', 'macd_diff', 'ema_20', 'ema_50', 'atr', 'volume',
    'cci', 'stoch_k', 'stoch_d', 'mom', 'bb_width',
    'volume_change', 'volume_ema', 'volume_ratio',
    'candle_body', 'candle_range', 'upper_shadow', 'lower_shadow',
    'body_to_range', 'upper_to_range', 'lower_to_range'
]] + ['sentiment', 'btc_dominance', 'dxy', 'funding_rate']

def preprocess_for_lstm(df: pd.DataFrame) -> np.ndarray:
    df = df.dropna()
    return scaler.transform(df)


def lstm_based_action(df_combined):
    X_raw = df_combined.iloc[-SEQ_LEN_MODEL:]  # Giữ nguyên DataFrame và tên cột
    X_scaled = scaler.transform(X_raw)
    X_input = np.expand_dims(X_scaled, axis=0)

    proba = model.predict(X_input)[0]
    pred = np.argmax(proba)
    confidence = float(np.max(proba))

    # Thêm ảnh hưởng từ tin tức
    try:
        impact_score = analyze_news()
    except Exception as e:
        print(f"[WARN] Failed to analyze news: {e}")
        impact_score = 0.0

    adjusted_conf = min(1.0, max(0.0, confidence + impact_score))

    if pred == 1:
        return 'LONG', adjusted_conf
    elif pred == 2:
        return 'SHORT', adjusted_conf
    return 'HOLD', adjusted_conf



def select_leverage(confidence):
    # Giới hạn confidence trong khoảng 0.6 - 0.98
    confidence = max(0.7, min(confidence, 0.98))
    # Tuyến tính: 0.6 → 1x, 0.98 → 20x
    leverage = 1 + (confidence - 0.7) / (0.98 - 0.7) * (20 - 1)
    print(f"[INFO] leverage: {leverage:.2f}")
    return round(leverage, 1)

def compute_min_leverage(balance, price, risk_percent, min_notional=5, max_leverage=20):
    if balance <= 0 or price <= 0:
        return max_leverage  # fallback nếu dữ liệu lỗi

    capital = balance * (risk_percent / 100)  # vốn có thể dùng
    required_qty = min_notional / price       # cần mua ít nhất bao nhiêu coin
    raw_leverage = (required_qty * price) / capital

    min_leverage = int(np.ceil(raw_leverage))
    min_leverage = max(1, min(min_leverage, max_leverage))  # clamp trong [1, max_leverage]
    return min_leverage

