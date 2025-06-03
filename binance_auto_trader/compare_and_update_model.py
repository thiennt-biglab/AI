import pandas as pd
from config import *
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import os

# === Step 1: Load dữ liệu từ log ===
df = pd.read_csv("loss_analysis_log.csv")
df = df[df["reason"].isin(["STOP_LOSS_TRIGGERED", "AUTO-PROFIT"])]
df["label"] = df["reason"].map({"STOP_LOSS_TRIGGERED": 0, "AUTO-PROFIT": 1})

feature_cols = [col for col in df.columns if any(key in col for key in ['ema_', 'atr_', 'sentiment', 'btc_dominance', 'volume'])]
if not feature_cols:
    print("[ERROR] Không tìm thấy đặc trưng phù hợp.")
    exit(1)

X = df[feature_cols].copy()
y = df["label"]

# === Step 2: Chuẩn hóa dữ liệu ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42, stratify=y)

joblib.dump(scaler, "finetune_scaler.pkl")

# === Step 3: Load mô hình gốc ===
if not os.path.exists("best_model_active.h5"):
    print("[WARN] Không có mô hình gốc. Sẽ train mới từ đầu.")
    old_model = None
    old_metrics = {"acc": 0, "auc": 0, "loss": float("inf")}
else:
    old_model = load_model("best_model_active.h5")
    y_pred_old = old_model.predict(X_test).flatten()
    y_pred_bin_old = (y_pred_old > 0.5).astype(int)
    old_metrics = {
        "acc": accuracy_score(y_test, y_pred_bin_old),
        "auc": roc_auc_score(y_test, y_pred_old),
        "loss": log_loss(y_test, y_pred_old)
    }
    print(f"[INFO] Mô hình cũ → acc: {old_metrics['acc']:.4f}, auc: {old_metrics['auc']:.4f}, loss: {old_metrics['loss']:.4f}")

# === Step 4: Train mô hình mới ===
model = Sequential()
model.add(LSTM(64, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)

# === Step 5: Đánh giá mô hình mới ===
y_pred_new = model.predict(X_test).flatten()
y_pred_bin_new = (y_pred_new > 0.5).astype(int)

new_metrics = {
    "acc": accuracy_score(y_test, y_pred_bin_new),
    "auc": roc_auc_score(y_test, y_pred_new),
    "loss": log_loss(y_test, y_pred_new)
}

print(f"[INFO] Mô hình mới → acc: {new_metrics['acc']:.4f}, auc: {new_metrics['auc']:.4f}, loss: {new_metrics['loss']:.4f}")

# === Step 6: So sánh và ghi đè nếu tốt hơn ===
if new_metrics["auc"] > old_metrics["auc"] and new_metrics["acc"] >= old_metrics["acc"]:
    save_model(model, "best_model_active.h5")
    print("[✅] Mô hình mới TỐT HƠN → đã ghi đè mô hình chính thức.")
else:
    save_model(model, f"finetuned_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
    print("[ℹ️] Mô hình mới KHÔNG tốt hơn rõ rệt → đã lưu riêng để phân tích thêm.")
