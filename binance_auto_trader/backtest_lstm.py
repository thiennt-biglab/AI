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
X_seq, y_seq, close_prices = joblib.load("lstm_data.pkl")  # từ lúc training đã lưu

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
capital = 1000
balance = capital
position = None
entry_price = 0
fee_rate = 0.0004  # 0.04%
equity_curve = [capital]
TP = 0.02  # 2%
SL = 0.01  # 1%
win_count = 0
lose_count = 0

i = 1
while i < len(X_seq) - 3:
    price_now = close_prices[i]
    pred = y_pred[i]

    # Bỏ qua tín hiệu yếu
    if np.max(y_pred_proba[i]) < 0.6:
        equity_curve.append(balance)
        i += 1
        continue

    if position is None:
        if pred == 1:  # LONG
            position = 'LONG'
            entry_price = price_now
        elif pred == 2:  # SHORT
            position = 'SHORT'
            entry_price = price_now
    else:
        exit = False
        for j in range(1, 4):
            future_price = close_prices[i + j]
            if position == 'LONG':
                change = (future_price - entry_price) / entry_price
            else:
                change = (entry_price - future_price) / entry_price

            if change >= TP:
                pnl = TP
                win_count += 1
                exit = True
                break
            elif change <= -SL:
                pnl = -SL
                lose_count += 1
                exit = True
                break

        if not exit:
            final_price = close_prices[i + 3]
            if position == 'LONG':
                pnl = (final_price - entry_price) / entry_price
            else:
                pnl = (entry_price - final_price) / entry_price
            if pnl > 0:
                win_count += 1
            else:
                lose_count += 1

        trade_value = balance
        fee = trade_value * fee_rate
        profit = trade_value * pnl
        balance += profit - fee  # chỉ tính 1 chiều phí để dễ phân tích
        position = None
        i += 3  # skip future bars đã dùng
        equity_curve.append(balance)
        continue

    equity_curve.append(balance)
    i += 1

final_profit = balance - capital
print(f"\n\U0001F9EA Final capital: ${balance:.2f} (profit: ${final_profit:.2f})")
print(f"✅ Tổng lệnh thắng: {win_count}, ❌ Lệnh thua: {lose_count}, 📉 Winrate: {win_count / (win_count + lose_count + 1e-9):.2%}")

# Vẽ Equity Curve
plt.figure(figsize=(10, 4))
plt.plot(equity_curve, label="Equity Curve")
plt.title("Equity Curve over Backtest")
plt.xlabel("Trades")
plt.ylabel("Capital ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("equity_curve.png")
