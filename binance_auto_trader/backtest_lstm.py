import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from config import *

# Load model và dữ liệu đã lưu
model = load_model(FINE_TUNE_MODEL_PATH)
scaler = joblib.load("scaler.pkl")
X_seq, y_seq, close_prices, high_prices, low_prices = joblib.load("lstm_data.pkl")  # từ lúc training đã lưu

# Adjust sequence length for compatibility
EXPECTED_SEQ_LEN = model.input_shape[1]
if X_seq.shape[1] > EXPECTED_SEQ_LEN:
    X_seq = X_seq[:, -EXPECTED_SEQ_LEN:, :]
elif X_seq.shape[1] < EXPECTED_SEQ_LEN:
    pad_width = EXPECTED_SEQ_LEN - X_seq.shape[1]
    X_seq = np.pad(X_seq, ((0, 0), (pad_width, 0), (0, 0)), mode='edge')

# Dự đoán
y_pred_proba = model.predict(X_seq)
y_pred = np.argmax(y_pred_proba, axis=1)

# Phân loại
label_map = {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}
y_true_named = [label_map[y] for y in y_seq]
y_pred_named = [label_map[y] for y in y_pred]

# Hiển thị kết quả
print("=== Classification Report ===")
print(classification_report(y_true_named, y_pred_named, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true_named, y_pred_named, labels=['HOLD', 'LONG', 'SHORT'])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HOLD', 'LONG', 'SHORT'], yticklabels=['HOLD', 'LONG', 'SHORT'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for LSTM Model')
plt.tight_layout()
plt.savefig("confusion_matrix.png")

# === Backtest ===
y_pred = model.predict(X_seq, verbose=0)
y_class = np.argmax(y_pred, axis=1)


future_window = FUTURE_WINDOW + 10
capital = 60
TP = 0.01   # 1% Take Profit
SL = 0.015  # 0.5% Stop Loss
profits = []
FEE = 0.0007
wins = 0
total_trades = 0
fixed_trade_size = 0.5 * capital
for i in range(len(y_class) - future_window):
    pred = y_class[i]
    price_entry = close_prices[i]

    if pred == 1:  # LONG
        tp_price = price_entry * (1 + TP)
        sl_price = price_entry * (1 - SL)
        highs = high_prices[i+1:i+future_window+1]
        lows = low_prices[i+1:i+future_window+1]

        hit_tp = np.any(highs >= tp_price)
        hit_sl = np.any(lows <= sl_price)

        if hit_tp and (not hit_sl or np.argmax(highs >= tp_price) <= np.argmax(lows <= sl_price)):
            net = TP - FEE
            capital += fixed_trade_size * net
            profits.append(net)
            wins += 1
        elif hit_sl:
            net = -SL - FEE
            capital += fixed_trade_size * net
            profits.append(net)
        else:
            price_exit = close_prices[i + future_window]
            change = (price_exit - price_entry) / price_entry - FEE
            capital += fixed_trade_size * change
            profits.append(change)
            if change > 0:
                wins += 1
        total_trades += 1

    elif pred == 2:  # SHORT
        tp_price = price_entry * (1 - TP)
        sl_price = price_entry * (1 + SL)
        highs = high_prices[i+1:i+future_window+1]
        lows = low_prices[i+1:i+future_window+1]

        hit_tp = np.any(lows <= tp_price)
        hit_sl = np.any(highs >= sl_price)

        if hit_tp and (not hit_sl or np.argmax(lows <= tp_price) <= np.argmax(highs >= sl_price)):
            net = TP - FEE
            capital += fixed_trade_size * net
            profits.append(net)
            wins += 1
        elif hit_sl:
            net = -SL - FEE
            capital += fixed_trade_size * net
            profits.append(net)
        else:
            price_exit = close_prices[i + future_window]
            change = (price_entry - price_exit) / price_entry - FEE
            capital += fixed_trade_size * change
            profits.append(change)
            if change > 0:
                wins += 1
        total_trades += 1

profit = capital - 60
avg_trade = np.mean(profits) * 100 if profits else 0
std_trade = np.std(profits) * 100 if profits else 0
winrate = (wins / total_trades * 100) if total_trades > 0 else 0

print(f"Profit: ${profit:.2f} | Winrate: {winrate:.2f}% | Avg Trade: {avg_trade:.3f}% | Std: {std_trade:.3f}%")
