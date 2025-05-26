import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
from binance.client import Client
from config import API_KEY, API_SECRET, SYMBOL, INTERVALS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Input, Attention, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from feature_pipeline import fetch_features_multi_timeframe

client = Client(API_KEY, API_SECRET)

# === Training pipeline ===
df = fetch_features_multi_timeframe()
scaler = StandardScaler()
X = scaler.fit_transform(df.values)

joblib.dump(scaler, "scaler.pkl")

future_return = df[f'ema_20_{INTERVALS[0]}'].shift(-3) / df[f'ema_20_{INTERVALS[0]}'] - 1
labels = np.zeros(len(future_return))
labels[future_return > 0.0008] = 1  # LONG
labels[future_return < -0.0008] = 2  # SHORT

valid_idx = ~np.isnan(future_return)
X = X[valid_idx]
labels = labels[valid_idx]

X_seq = []
y_seq = []
window = 30
stride = 1
for i in range(window, len(X), stride):
    X_seq.append(X[i-window:i])
    y_seq.append(labels[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2)

unique, counts = np.unique(y_seq, return_counts=True)
print(dict(zip(["HOLD", "LONG", "SHORT"], counts)))

# compute class weights
classes = np.unique(y_train)
class_weights_array = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights = dict(zip(classes, class_weights_array))

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
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_lstm_attention():
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Input, Dense, Dropout
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
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
    

models = {
    "lstm_v1": build_model_1(),
    "lstm_v2": build_model_2(),
    "gru": build_model_gru(),
    "bilstm": build_model_bilstm(),
    "lstm_attn": build_model_lstm_attention()
}

if __name__ == "__main__":
    best_model = None
    best_score = -1

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
        print(f"{name} best val_acc: {val_acc:.4f}")
        if val_acc > best_score:
            best_score = val_acc
            best_model = model
            best_name = name
            best_history = history

    best_model.save('lstm_model.keras')
    joblib.dump((X_seq, y_seq), "lstm_data.pkl")

    with open("best_model_name.txt", "w") as f:
        f.write(best_name)
    print(f"✅ Best model '{best_name}' saved to lstm_model.keras with val_acc: {best_score:.4f}")

    plt.figure()
    plt.plot(best_history.history['accuracy'], label='Train Accuracy')
    plt.plot(best_history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Best Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')

    plt.figure()
    plt.plot(best_history.history['loss'], label='Train Loss')
    plt.plot(best_history.history['val_loss'], label='Val Loss')
    plt.title('Best Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
plt.savefig('loss_curve.png')
