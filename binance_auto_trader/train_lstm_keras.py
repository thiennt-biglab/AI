import numpy as np
from binance.client import Client
from config import API_KEY, API_SECRET, SYMBOL, INTERVALS, BEST_MODEL_PATH, SEQ_LEN_MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Add
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv1D, MaxPooling1D
import joblib
from collections import Counter
from feature_pipeline import fetch_features_multi_timeframe

client = Client(API_KEY, API_SECRET)

# === Training pipeline ===
df = fetch_features_multi_timeframe()
scaler = StandardScaler()
X = scaler.fit_transform(df)
joblib.dump(scaler, "scaler.pkl")

ema = df[f'ema_20_{INTERVALS[0]}'].values
atr = df['atr_5m'].values  # Cột ATR phải có trong feature_pipeline
labels = np.zeros(len(ema))
future_window = 5  # số nến tương lai để kiểm tra TP/SL

for i in range(len(ema) - future_window):
    entry_price = ema[i]
    if atr[i] / entry_price < 0.001:
        continue
    tp_ratio = atr[i] / entry_price * 1.5  # TP = 1.5 * ATR
    sl_ratio = atr[i] / entry_price * 1.0  # SL = 1.0 * ATR
    future_prices = ema[i+1:i+1+future_window]

    for price in future_prices:
        if price >= entry_price * (1 + tp_ratio):
            labels[i] = 1  # LONG
            break
        elif price <= entry_price * (1 - sl_ratio):
            labels[i] = 2  # SHORT
            break

print(Counter(labels.astype(int)))

# Loại bỏ các sample quá gần cuối vì không đủ future_window
valid_idx = np.arange(len(labels) - future_window)

X = X[valid_idx]
labels = labels[valid_idx]


X_seq, y_seq = [], []
window = SEQ_LEN_MODEL
for i in range(window, len(X)):
    X_seq.append(X[i - window:i])
    y_seq.append(labels[i])
close_prices = df['ema_20_5m'].values[window:]

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_train, X_test, y_train_raw, y_test_raw = train_test_split(X_seq, y_seq, test_size=0.2)

# Class weights
classes = np.unique(y_train_raw)
class_weights_array = compute_class_weight("balanced", classes=classes, y=y_train_raw)
class_weights = dict(zip(classes, class_weights_array))

# One-hot
y_train = to_categorical(y_train_raw, num_classes=3)
y_test = to_categorical(y_test_raw, num_classes=3)

print(dict(zip(["HOLD", "LONG", "SHORT"], np.bincount(y_seq.astype(int)))))

# === Model builders ===
def build_model_1():
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(128, return_sequences=True),
        Dropout(0.4),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=CategoricalCrossentropy(label_smoothing=0.05), metrics=['accuracy'])
    return model

def build_model_2():
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=CategoricalCrossentropy(label_smoothing=0.05), metrics=['accuracy'])
    return model

def build_model_gru():
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        GRU(64, return_sequences=True),
        Dropout(0.3),
        GRU(32),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=CategoricalCrossentropy(label_smoothing=0.05), metrics=['accuracy'])
    return model

def build_model_bilstm():
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=CategoricalCrossentropy(label_smoothing=0.05), metrics=['accuracy'])
    return model

def build_model_lstm_attention():
    from tensorflow.keras.models import Model
    from tensorflow.keras import layers

    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    attn_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn_output])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=CategoricalCrossentropy(label_smoothing=0.05), metrics=['accuracy'])
    return model

def build_cnn_lstm_model():
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_transformer_encoder_model():
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Input, Dense, Dropout, GlobalAveragePooling1D, Add

    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    x = Add()([inputs, x])
    x = LayerNormalization()(x)

    dense_output = Dense(inputs.shape[-1], activation='relu')(x)
    x = Add()([x, dense_output])
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

models = {
    "lstm_v1": build_model_1(),
    "lstm_v2": build_model_2(),
    "gru": build_model_gru(),
    "bilstm": build_model_bilstm(),
    "lstm_attn": build_model_lstm_attention(),
    "cnn_lstm": build_cnn_lstm_model(),
    "transformer_encoder": build_transformer_encoder_model()
}

if __name__ == "__main__":

    results = []
    best_model = None
    best_name = ""
    best_score = -np.inf

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        history = model.fit(
            X_train, y_train,
            epochs=300,
            batch_size=64,
            validation_split=0.1,
            class_weight=class_weights,
            callbacks=[
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            ],
            verbose=0
        )
        val_acc = max(history.history['val_accuracy'])
        val_loss = min(history.history['val_loss'])

        y_pred = model.predict(X_seq, verbose=0)
        y_class = np.argmax(y_pred, axis=1)

        close_prices = df[f"ema_20_{INTERVALS[0]}"].values
        close_prices = close_prices[-len(X_seq):]

        capital = 60
        TP = 0.01   # 1% Take Profit
        SL = 0.005  # 0.5% Stop Loss
        profits = []

        for i in range(len(y_class) - 3):
            pred = y_class[i]
            price_entry = close_prices[i]

            if pred == 1:  # LONG
                for j in range(1, 4):
                    price_now = close_prices[i + j]
                    change = (price_now - price_entry) / price_entry
                    if change >= TP:
                        capital *= (1 + TP)
                        profits.append(TP)
                        break
                    elif change <= -SL:
                        capital *= (1 - SL)
                        profits.append(-SL)
                        break
                else:
                    change = (close_prices[i + 3] - price_entry) / price_entry
                    capital *= (1 + change)
                    profits.append(change)

            elif pred == 2:  # SHORT
                for j in range(1, 4):
                    price_now = close_prices[i + j]
                    change = (price_entry - price_now) / price_entry
                    if change >= TP:
                        capital *= (1 + TP)
                        profits.append(TP)
                        break
                    elif change <= -SL:
                        capital *= (1 - SL)
                        profits.append(-SL)
                        break
                else:
                    change = (price_entry - close_prices[i + 3]) / price_entry
                    capital *= (1 + change)
                    profits.append(change)

        profit = capital - 60
        avg_trade = np.mean(profits) * 100 if profits else 0
        std_trade = np.std(profits) * 100 if profits else 0

        score = val_acc * 100 - val_loss * 10 + profit * 0.003 + avg_trade * 6 - std_trade * 4

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

        print(f"{name} | Val Accuracy: {val_acc:.4f} | Profit: ${profit:.2f} | Score: {score} | Avg Trade: {avg_trade:.3f}% | Std: {std_trade:.3f}%")

    if best_model:
        print(f"\n✅ Best model: {best_name} | Score: {best_score:.2f}")
    best_model.save(BEST_MODEL_PATH)
    print(f"✅ Model saved to best_model.keras")
