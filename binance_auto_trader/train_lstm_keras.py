import numpy as np
from binance.client import Client
from config import API_KEY, API_SECRET, SYMBOL, INTERVALS
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
from feature_pipeline import fetch_features_multi_timeframe

client = Client(API_KEY, API_SECRET)

# === Training pipeline ===
df = fetch_features_multi_timeframe()
scaler = StandardScaler()
X = scaler.fit_transform(df)
joblib.dump(scaler, "scaler.pkl")

future_return = df[f'ema_20_{INTERVALS[0]}'].shift(-3) / df[f'ema_20_{INTERVALS[0]}'] - 1
labels = np.zeros(len(future_return))
labels[future_return > 0.003] = 1  # LONG
labels[future_return < -0.003] = 2  # SHORT

valid_idx = ~np.isnan(future_return)
X = X[valid_idx]
labels = labels[valid_idx]

X_seq, y_seq = [], []
window = 30
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
    "cnn_lstm": build_cnn_lstm_model()
}

if __name__ == "__main__":
    results = []

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

        y_pred = model.predict(X_seq, verbose=0)
        y_class = np.argmax(y_pred, axis=1)

        close_prices = df[f"ema_20_{INTERVALS[0]}"].values
        close_prices = close_prices[~np.isnan(future_return)][window:]

        capital = 1000
        for i in range(len(y_class) - 3):
            pred = y_class[i]
            price_now = close_prices[i]
            price_next = close_prices[i + 3]

            if pred == 1:  # LONG
                change = (price_next - price_now) / price_now
                capital *= (1 + change)
            elif pred == 2:  # SHORT
                change = (price_now - price_next) / price_now
                capital *= (1 + change)

        profit = capital - 1000
        print(f"{name} | Val Accuracy: {val_acc:.4f} | Profit: ${profit:.2f}")

        results.append({
            "name": name,
            "model": model,
            "val_acc": val_acc,
            "profit": profit,
            "history": history
        })

    # Normalize & compute final score
    min_profit = min(r["profit"] for r in results)
    max_profit = max(r["profit"] for r in results)

    def normalize_profit(p):
        return (p - min_profit) / (max_profit - min_profit) if max_profit > min_profit else 0.0

    acc_weight = 0.95  # 🎯 bạn có thể đổi sang 0.6, 0.3, v.v.

    for r in results:
        norm_profit = normalize_profit(r["profit"])
        r["combined_score"] = r["val_acc"] * acc_weight + norm_profit * (1 - acc_weight)

    best_result = max(results, key=lambda x: x["combined_score"])

    best_model = best_result["model"]
    best_name = best_result["name"]
    best_profit = best_result["profit"]
    best_score = best_result["combined_score"]
    best_history = best_result["history"]

    best_model.save('lstm_model.keras')
    joblib.dump((X_seq, y_seq, close_prices), "lstm_data.pkl")

    with open("best_model_name.txt", "w") as f:
        f.write(best_name)

    print(f"✅ Best model: {best_name} | Profit: ${best_profit:.2f} | Combined Score: {best_score:.4f}")