import numpy as np
from binance.client import Client
from config import *
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
close = df[f'close_{INTERVALS[0]}'].values
atr = df['atr_5m'].values
future_window = FUTURE_WINDOW  # số nến tương lai để kiểm tra TP/SL

window = SEQ_LEN_MODEL# mặc định là HOLD
labels = np.zeros(len(future_return))

high_prices = df[f"high_{INTERVALS[0]}"].values
low_prices = df[f"low_{INTERVALS[0]}"].values
close_prices = df[f"close_{INTERVALS[0]}"].values
volatility = df[f"atr_{INTERVALS[0]}"] / close_prices
threshold = np.clip(volatility * 1.5, 0.003, 0.008)  # giữ TP/SL nằm trong 0.3%–0.8%
trend_filter = df[f"ema_9_{INTERVALS[0]}"] > df[f"ema_20_{INTERVALS[0]}"]

# Gán nhãn dựa vào TP/SL trong tương lai
for i in range(len(df) - future_window):
    entry_price = close_prices[i]
    tp = entry_price * (1 + threshold[i])
    sl = entry_price * (1 - threshold[i])

    future_highs = high_prices[i+1:i+1+future_window]
    future_lows = low_prices[i+1:i+1+future_window]

    if trend_filter.iloc[i]:  # Ưu tiên LONG khi trend tăng
        if np.any(future_highs >= tp):
            labels[i] = 1  # LONG
        elif np.any(future_lows <= sl):
            labels[i] = 0  # HOLD
    else:  # Ưu tiên SHORT khi trend giảm
        if np.any(future_lows <= sl):
            labels[i] = 2  # SHORT
        elif np.any(future_highs >= tp):
            labels[i] = 0  # HOLD

# Loại bỏ phần cuối thiếu tương lai
labels = labels[:-future_window]

print(Counter(labels.astype(int)))

# Loại bỏ các sample quá gần cuối vì không đủ future_window
valid_idx = np.arange(len(labels) - future_window)

X = X[valid_idx]
labels = labels[valid_idx]


X_seq, y_seq = [], []

for i in range(window, len(X)):
    X_seq.append(X[i - window:i])
    y_seq.append(labels[i])

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
    "transformer_encoder": build_transformer_encoder_model()
}

joblib.dump((X_seq, y_seq, close, high_prices, low_prices), "lstm_data.pkl")

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
            verbose=1
        )
        val_acc = max(history.history['val_accuracy'])
        val_loss = min(history.history['val_loss'])

        y_pred = model.predict(X_seq, verbose=0)
        y_class = np.argmax(y_pred, axis=1)

        close_prices = close[-len(X_seq):]

        capital = 60
        TP = 0.01   # 1% Take Profit
        SL = 0.005  # 0.5% Stop Loss
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

        score = val_acc * 100 - val_loss * 10 + avg_trade * 4 - std_trade * 6 + winrate

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

        print(f"{name} | Val Accuracy: {val_acc:.4f} | Profit: ${profit:.2f} | Winrate: {winrate:.2f}% | "
              f"Score: {score:.2f} | Avg Trade: {avg_trade:.3f}% | Std: {std_trade:.3f}%")

    if best_model:
        print(f"\n✅ Best model: {best_name} | Score: {best_score:.2f}")
        best_model.save(BEST_MODEL_PATH)
        print(f"✅ Model saved to best_model.keras")
