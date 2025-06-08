import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from config import *

# --- Cấu hình ---
LABELS = {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}
PREDICT_HORIZON = FUTURE_WINDOW
DATA_PATH = "lstm_data.pkl"
OUTPUT_CSV = "misclassified_trades.csv"

# --- Load dữ liệu ---
X_seq, y_seq, close_prices, high_prices, low_prices = joblib.load(DATA_PATH)

if X_seq.shape[1] > SEQ_LEN_MODEL:
    X_seq = X_seq[:, -SEQ_LEN_MODEL:, :]
elif X_seq.shape[1] < SEQ_LEN_MODEL:
    raise ValueError(f"X_seq có độ dài {X_seq.shape[1]} nhỏ hơn SEQ_LEN_MODEL={SEQ_LEN_MODEL}")

split_idx = int(len(X_seq) * 0.8)
X_test = X_seq[split_idx:]

y_test_raw = y_seq[split_idx:]
close_prices_test = close_prices[split_idx:]
y_test = to_categorical(y_test_raw, num_classes=3)

# --- Load mô hình ---
model = load_model(BEST_MODEL_PATH, compile=False)
y_pred_proba = model.predict(X_test)
y_pred_class = np.argmax(y_pred_proba, axis=1)

# --- Tìm dự đoán sai ---
def calculate_profit(pred_label, price_now, price_future):
    if pred_label == 1:  # LONG
        return (price_future - price_now) / price_now
    elif pred_label == 2:  # SHORT
        return (price_now - price_future) / price_now
    return 0

misclassified = []
for i, (true, pred) in enumerate(zip(y_test_raw, y_pred_class)):
    if true != pred:
        price_now = close_prices_test[i]
        price_future = close_prices_test[i + PREDICT_HORIZON] if i + PREDICT_HORIZON < len(close_prices_test) else None
        profit = calculate_profit(pred, price_now, price_future) if price_future is not None else None

        misclassified.append({
            "Index": i,
            "True Label": LABELS[int(true)],
            "Predicted Label": LABELS[int(pred)],
            "Price Now": price_now,
            f"Price in {PREDICT_HORIZON} Candles": price_future,
            "Simulated Profit": profit
        })

# --- Xuất CSV ---
df_misclassified = pd.DataFrame(misclassified)
df_misclassified.to_csv(OUTPUT_CSV, index=False)

# --- Hiển thị kết quả đầu ---
print(df_misclassified.head())
