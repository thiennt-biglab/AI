import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load dữ liệu đã lưu
X_seq, y_seq, close_prices = joblib.load("lstm_data.pkl")
split_idx = int(len(X_seq) * 0.8)

X_test = X_seq[split_idx:]
y_test_raw = y_seq[split_idx:]
close_prices_test = close_prices[split_idx:]
y_test = to_categorical(y_test_raw, num_classes=3)

# Load mô hình đã huấn luyện
model = load_model("lstm_model.keras", compile=False)
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Tìm các dự đoán sai
misclassified = []
for i in range(len(y_test_raw)):
    if y_pred_class[i] != y_test_raw[i]:
        price_now = close_prices_test[i]
        if i + 3 < len(close_prices_test):
            price_future = close_prices_test[i + 3]
            if y_pred_class[i] == 1:  # LONG
                profit = (price_future - price_now) / price_now
            elif y_pred_class[i] == 2:  # SHORT
                profit = (price_now - price_future) / price_now
            else:  # HOLD
                profit = 0
        else:
            profit = None

        misclassified.append({
            "Index": i,
            "True Label": int(y_test_raw[i]),
            "Predicted Label": int(y_pred_class[i]),
            "Price Now": price_now,
            "Price in 3 Candles": price_future if i + 3 < len(close_prices_test) else None,
            "Simulated Profit": profit
        })

# Xuất ra CSV để debug dễ dàng nếu cần
df_misclassified = pd.DataFrame(misclassified)
df_misclassified.to_csv("misclassified_trades.csv", index=False)
print(df_misclassified.head())
