import numpy as np

from config import BEST_MODEL_FILE
from train_lstm_keras import focal_loss
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load model và dữ liệu đã lưu
focal = focal_loss(gamma=1.0, alpha=0.5)
model = load_model(BEST_MODEL_FILE, custom_objects={'loss': focal})
scaler = joblib.load("scaler.pkl")
X_seq, y_seq, close_prices = joblib.load("lstm_data.pkl")  # từ lúc training đã lưu

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

# Tham số: ngưỡng confidence
confidence_threshold = 0.6

capital = 60
balance = capital
position = None
entry_price = 0
fee_rate = 0.0004  # 0.04%
total_trades = 0
win_trades = 0
balances = []
trade_indices = []
leverage = 10

for i in range(1, len(X_seq)):
    price_now = close_prices[i]
    pred = y_pred[i]
    confidence = np.max(y_pred_proba[i])

    # Chỉ xử lý nếu độ tự tin cao hơn threshold
    if confidence < confidence_threshold:
        continue

    if position is None:
        if pred == 1:  # LONG
            position = 'LONG'
            entry_price = price_now
        elif pred == 2:  # SHORT
            position = 'SHORT'
            entry_price = price_now
    else:
        if (position == 'LONG' and pred != 1) or (position == 'SHORT' and pred != 2):

            trade_indices.append(i)

            trade_value = balance
            fee = trade_value * fee_rate

            if position == 'LONG':
                pnl = ((price_now - entry_price) / entry_price) * leverage
            else:
                pnl = ((entry_price - price_now) / entry_price) * leverage

            if pnl <= -1.0:
                print(f"[LIQUIDATED] {position} at step {i}")
                break

            profit = trade_value * pnl
            balance += profit - 2 * fee
            balances.append(balance)
            position = None

            total_trades += 1
            if profit > 0:
                win_trades += 1

final_profit = balance - capital
print(f"\n🧪 Final capital: ${balance:.2f} (profit: ${final_profit:.2f})")
print(f"Total trades: {total_trades}")
print(f"Win rate: {win_trades / total_trades:.2%}")

if trade_indices:
    total_steps = trade_indices[-1]
    print(f"Reached ${balance:.2f} in {total_steps} steps (candles)")
    print(f"Avg profit per trade: ${final_profit / total_trades:.2f}")
else:
    print("No trades executed.")

plt.plot(balances)
plt.title("Balance Over Time")
plt.xlabel("Trade #")
plt.ylabel("Balance ($)")
plt.grid()
plt.savefig("balance_over_time.png")



