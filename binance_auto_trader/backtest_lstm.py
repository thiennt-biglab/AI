# Re-import necessary packages after code state reset
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from config import *

# Redefine custom loss function
def custom_loss(y_true, y_pred):
    alpha_long = 3.0
    alpha_short = 3.0
    alpha_hold = 0.5
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    weight = (
            y_true[:, 0] * alpha_hold +
            y_true[:, 1] * alpha_long +
            y_true[:, 2] * alpha_short
    )
    cross_entropy = -K.sum(y_true * K.log(y_pred), axis=1)
    return weight * cross_entropy

# Load model and data
model = load_model(FINETUNED_MODEL_FILE, custom_objects={'loss': custom_loss})
scaler = joblib.load("./scaler.pkl")
X_seq, y_seq, close_prices = joblib.load("./lstm_data.pkl")

# Predict
y_pred_proba = model.predict(X_seq)
y_pred = np.argmax(y_pred_proba, axis=1)

# Classification labels
label_map = {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}
y_true_named = [label_map[y] for y in y_seq]
y_pred_named = [label_map[y] for y in y_pred]

# Classification report
report = classification_report(y_true_named, y_pred_named, digits=4, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Confusion matrix
cm = confusion_matrix(y_true_named, y_pred_named, labels=['HOLD', 'LONG', 'SHORT'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HOLD', 'LONG', 'SHORT'], yticklabels=['HOLD', 'LONG', 'SHORT'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for LSTM Model')
plt.tight_layout()
plt.savefig("./confusion_matrix.png")

# Backtest
confidence_threshold = 0.6
capital = 60
balance = capital
position = None
entry_price = 0
fee_rate = 0.0004
total_trades = 0
win_trades = 0
balances = []
trade_indices = []
leverage = 10

for i in range(1, len(X_seq)):
    price_now = close_prices[i]
    pred = y_pred[i]
    confidence = np.max(y_pred_proba[i])

    if confidence < confidence_threshold:
        continue

    if position is None:
        if pred == 1:
            position = 'LONG'
            entry_price = price_now
        elif pred == 2:
            position = 'SHORT'
            entry_price = price_now
    else:
        if (position == 'LONG' and pred != 1) or (position == 'SHORT' and pred != 2):
            trade_indices.append(i)
            trade_value = balance
            fee = trade_value * fee_rate

            pnl = ((price_now - entry_price) / entry_price) * leverage if position == 'LONG' else ((entry_price - price_now) / entry_price) * leverage

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
avg_profit_per_trade = final_profit / total_trades if total_trades > 0 else 0
win_rate = win_trades / total_trades if total_trades > 0 else 0
total_steps = trade_indices[-1] if trade_indices else 0

# Save balance plot
plt.figure()
plt.plot(balances)
plt.title("Balance Over Time")
plt.xlabel("Trade #")
plt.ylabel("Balance ($)")
plt.grid()
plt.savefig("./balance_over_time.png")

import json

results = {
    "Final Capital": round(balance, 2),
    "Profit": round(final_profit, 2),
    "Total Trades": total_trades,
    "Win Rate": f"{win_trades / total_trades:.2%}" if total_trades > 0 else "0.00%",
    "Avg Profit per Trade": round(final_profit / total_trades, 2) if total_trades > 0 else 0.0,
    "Total Steps": trade_indices[-1] if trade_indices else 0
}

print(json.dumps(results, indent=4))
