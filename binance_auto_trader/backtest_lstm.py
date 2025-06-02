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

capital = 1000
balance = capital
position = None
entry_price = 0
fee_rate = 0.0004  # 0.04%

for i in range(1, len(X_seq)):
    price_now = close_prices[i]    # giả định close là [0]
    pred = y_pred[i]
    prev_pred = y_pred[i-1]


    if position is None:
        if pred == 1:  # LONG
            position = 'LONG'
            entry_price = price_now
        elif pred == 2:  # SHORT
            position = 'SHORT'
            entry_price = price_now
    else:
        if (position == 'LONG' and pred != 1) or (position == 'SHORT' and pred != 2):
            trade_value = balance
            fee = trade_value * fee_rate

            if position == 'LONG':
                pnl = (price_now - entry_price) / entry_price
            else:
                pnl = (entry_price - price_now) / entry_price

            profit = trade_value * pnl
            balance += profit - 2 * fee
            position = None

final_profit = balance - capital
print(f"\n🧪 Final capital: ${balance:.2f} (profit: ${final_profit:.2f})")

