# strategy_lstm_live.py
import numpy as np
import pandas as pd
import os
from config import *

# Lazy loading - don't load at import time
_model = None
_scaler = None

def _load_model_and_scaler():
    """Load model and scaler lazily with proper error handling."""
    global _model, _scaler

    if _model is None:
        from tensorflow.keras.models import load_model
        import joblib

        # Try to load best model first, then fine-tuned, then base model
        model_paths = [BEST_MODEL_PATH, FINE_TUNE_MODEL_PATH, TRAINED_MODE_PATH]

        for path in model_paths:
            if os.path.exists(path):
                try:
                    # Try loading with custom objects for FocalLoss
                    try:
                        from train_lstm_keras import FocalLoss
                        _model = load_model(path, custom_objects={'FocalLoss': FocalLoss})
                    except:
                        _model = load_model(path)
                    print(f"[INFO] Loaded model: {path}")
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load {path}: {e}")

        if _model is None:
            raise FileNotFoundError("No trained model found! Run: python train_lstm_keras.py")

        # Load scaler
        if os.path.exists("scaler.pkl"):
            _scaler = joblib.load("scaler.pkl")
            print("[INFO] Loaded scaler: scaler.pkl")
        else:
            raise FileNotFoundError("scaler.pkl not found! Run: python train_lstm_keras.py")

    return _model, _scaler

# News sentiment (optional)
def _get_news_impact():
    try:
        from news_sentiment import analyze_news
        return analyze_news()
    except Exception as e:
        print(f"[WARN] News analysis failed: {e}")
        return 0.0

def preprocess_for_lstm(df: pd.DataFrame) -> np.ndarray:
    """Preprocess dataframe for LSTM input."""
    _, scaler = _load_model_and_scaler()
    df = df.dropna()
    return scaler.transform(df)


def lstm_based_action(df_combined):
    """Get trading action from LSTM model."""
    model, scaler = _load_model_and_scaler()

    X_raw = df_combined.iloc[-SEQ_LEN_MODEL:]
    X_scaled = scaler.transform(X_raw)
    X_input = np.expand_dims(X_scaled, axis=0)

    proba = model.predict(X_input, verbose=0)[0]
    pred = np.argmax(proba)
    confidence = float(np.max(proba))

    # Add news sentiment influence
    impact_score = _get_news_impact()
    adjusted_conf = min(1.0, max(0.0, confidence + impact_score))

    if pred == 1:
        return 'LONG', adjusted_conf
    elif pred == 2:
        return 'SHORT', adjusted_conf
    return 'HOLD', adjusted_conf


def select_leverage(confidence):
    confidence = max(0.6, min(confidence, 0.98))
    # Hàm mũ để tăng nhanh về cuối
    leverage = 1 + ((confidence - 0.6) / (0.98 - 0.6)) ** 2 * (20 - 1)
    print(f"[INFO] leverage: {leverage:.2f}")
    return round(leverage, 3)

def compute_min_leverage(balance, price, risk_percent, min_notional=5, max_leverage=20):
    if balance <= 0 or price <= 0:
        return max_leverage  # fallback nếu dữ liệu lỗi

    capital = balance * (risk_percent / 100)  # vốn có thể dùng
    required_qty = min_notional / price       # cần mua ít nhất bao nhiêu coin
    raw_leverage = (required_qty * price) / capital

    min_leverage = int(np.ceil(raw_leverage))
    min_leverage = max(1, min(min_leverage, max_leverage))  # clamp trong [1, max_leverage]
    return min_leverage

