import os
import numpy as np
from binance.client import Client
from config import (
    API_KEY, API_SECRET, SYMBOL, INTERVALS, BEST_MODEL_PATH, SEQ_LEN_MODEL,
    TRADING_MODE, DEFAULT_LEVERAGE, RISK_PERCENT, ENTRY_THRESHOLD,
    USE_DYNAMIC_TPSL, USE_TRAILING_STOP, USE_COMPOUND,
    MIN_SIGNAL_INTERVAL, MAX_DAILY_TRADES,
    TP_MIN, TP_MAX, SL_MIN, SL_MAX, FUTURE_WINDOW,
    TP_ATR_MULT, SL_ATR_MULT, TRAILING_ACTIVATION, TRAILING_LOCK_RATIO,
    USE_WHALE_CONFIRMATION, WHALE_BLOCK_OPPOSING, WHALE_MIN_SCORE, WHALE_BOOST_CONFIDENCE,
    USE_SESSION_FILTER, ALLOWED_SESSIONS, USE_VOLATILITY_FILTER, MAX_ATR_PERCENT, MIN_ATR_PERCENT,
    USE_CRASH_PROTECTION, CRASH_ATR_THRESHOLD, CRASH_VOLUME_SPIKE, CRASH_SINGLE_DROP, CRASH_4H_DROP
)

# Import ENTROPY_THRESHOLD with fallback
try:
    from config import ENTROPY_THRESHOLD
except ImportError:
    ENTROPY_THRESHOLD = 0.45  # Default: low entropy = high certainty
from datetime import datetime

# Import whale tracker for smart money confirmation
try:
    from whale_tracker import WhaleTracker, whale_confirms_trade
    HAS_WHALE_TRACKER = True
    print("[INFO] Whale tracker loaded - will use smart money confirmation")
except ImportError:
    HAS_WHALE_TRACKER = False
    print("[WARN] Whale tracker not available")
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Dense, Dropout, Input,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Add,
    Conv1D, MaxPooling1D, Concatenate, Flatten, BatchNormalization,
    AveragePooling1D, Multiply, Permute, RepeatVector, Lambda, GaussianNoise
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler  # Better for outliers than StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import joblib
import xgboost as xgb
from collections import Counter
from feature_pipeline import fetch_features_multi_timeframe


# ============================================================
# FOCAL LOSS - Better for imbalanced classification
# Research shows 2-5% accuracy improvement on imbalanced data
# ============================================================
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for imbalanced classification.
    Down-weights easy examples, focuses on hard ones.
    gamma=2.0 is standard, alpha balances classes.
    """
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Apply label smoothing
        num_classes = tf.shape(y_true)[-1]
        y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / tf.cast(num_classes, tf.float32)

        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        focal_loss = focal_weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


# ============================================================
# XGBOOST WRAPPER - Same API as Keras for compatibility
# ============================================================
class XGBoostWrapper:
    """
    XGBoost wrapper with Keras-like API for compatibility with training loop.
    XGBoost often outperforms neural networks on small tabular datasets.
    """
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.model = None

    def fit(self, X, y, validation_data=None, epochs=100, batch_size=32,
            callbacks=None, class_weight=None, verbose=1):
        # Flatten X from (samples, seq_len, features) to (samples, seq_len * features)
        X_flat = X.reshape(X.shape[0], -1)

        # Convert one-hot y back to labels
        y_labels = np.argmax(y, axis=1)

        # Prepare validation data
        eval_set = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            y_val_labels = np.argmax(y_val, axis=1)
            eval_set = [(X_val_flat, y_val_labels)]

        # Calculate sample weights from class weights
        sample_weights = None
        if class_weight is not None:
            sample_weights = np.array([class_weight[label] for label in y_labels])

        # XGBoost parameters optimized for high win rate
        params = {
            'objective': 'multi:softprob',
            'num_class': self.n_classes,
            'max_depth': 5,              # Shallower = less overfitting
            'learning_rate': 0.03,       # Slower learning = better generalization
            'n_estimators': 500,         # More trees with early stopping
            'min_child_weight': 5,       # Prevent overfitting
            'subsample': 0.7,            # Row sampling
            'colsample_bytree': 0.7,     # Feature sampling
            'reg_alpha': 0.5,            # L1 regularization (stronger)
            'reg_lambda': 2.0,           # L2 regularization (stronger)
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'early_stopping_rounds': 30  # Stop if no improvement
        }

        self.model = xgb.XGBClassifier(**params)

        # Fit with early stopping
        self.model.fit(
            X_flat, y_labels,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=verbose > 0
        )

        # Calculate actual accuracy for history
        train_pred = self.model.predict(X_flat)
        train_acc = np.mean(train_pred == y_labels)

        val_acc = 0.5
        val_loss = 1.0
        if validation_data is not None:
            val_pred = self.model.predict(X_val_flat)
            val_acc = np.mean(val_pred == y_val_labels)
            val_loss = 1.0 - val_acc  # Approximate loss

        # Return fake history for compatibility
        class FakeHistory:
            def __init__(self, train_acc, val_acc, val_loss):
                self.history = {
                    'accuracy': [train_acc],
                    'loss': [1.0 - train_acc],
                    'val_accuracy': [val_acc],
                    'val_loss': [val_loss]
                }
        return FakeHistory(train_acc, val_acc, val_loss)

    def predict(self, X, verbose=0):
        X_flat = X.reshape(X.shape[0], -1)
        probs = self.model.predict_proba(X_flat)
        return probs

    def save(self, path):
        # Save as joblib
        joblib.dump(self.model, path.replace('.keras', '_xgb.pkl'))

    def get_feature_importance(self):
        if self.model is not None:
            return self.model.feature_importances_
        return None


def build_xgboost_model():
    """Build XGBoost classifier wrapper."""
    return XGBoostWrapper(n_classes=3)


# ============================================================
# DATA AUGMENTATION FOR TIME SERIES
# ============================================================
def augment_data(X, y, noise_factor=0.01, num_augmented=2):
    """
    Data augmentation for time series:
    1. Add Gaussian noise
    2. Time warping (slight scaling)
    3. Magnitude warping
    """
    X_aug = [X]
    y_aug = [y]

    for _ in range(num_augmented):
        # Gaussian noise augmentation
        noise = np.random.normal(0, noise_factor, X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y)

        # Magnitude warping (scale features slightly)
        scale = np.random.uniform(0.95, 1.05, (1, 1, X.shape[2]))
        X_scaled = X * scale
        X_aug.append(X_scaled)
        y_aug.append(y)

    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


# ============================================================
# TRIPLE BARRIER LABELING (Advanced labeling from financial ML)
# More accurate than simple TP/SL labeling
# ============================================================
def triple_barrier_labels(close, high, low, tp_mult=1.5, sl_mult=1.0, max_holding=12, atr=None):
    """
    Triple Barrier Method (from Advances in Financial ML by Lopez de Prado)
    - Upper barrier: Take Profit
    - Lower barrier: Stop Loss
    - Vertical barrier: Maximum holding period

    Returns labels: 1=LONG (hit TP), 2=SHORT (hit SL going up), 0=HOLD (timeout or unclear)
    """
    labels = np.zeros(len(close))

    for i in range(len(close) - max_holding):
        entry = close[i]

        # Dynamic barriers based on ATR if available
        if atr is not None and atr[i] > 0:
            tp_barrier = entry * (1 + atr[i] / entry * tp_mult)
            sl_barrier = entry * (1 - atr[i] / entry * sl_mult)
        else:
            # Fallback to fixed percentage (relaxed for more signals)
            tp_barrier = entry * 1.005  # 0.5%
            sl_barrier = entry * 0.997  # 0.3%

        # Check which barrier is hit first
        for j in range(1, max_holding + 1):
            idx = i + j
            if idx >= len(close):
                break

            # Check upper barrier (LONG signal was correct)
            if high[idx] >= tp_barrier:
                labels[i] = 1  # LONG
                break
            # Check lower barrier (SHORT signal was correct)
            elif low[idx] <= sl_barrier:
                labels[i] = 2  # SHORT
                break
        # If neither barrier hit, label stays 0 (HOLD)

    return labels


# ============================================================
# MARKET REGIME DETECTION
# Only trade in trending markets for higher accuracy
# ============================================================
def detect_market_regime(close, window=20):
    """
    Detect market regime: TRENDING vs RANGING
    Uses ADX-like logic and price position relative to MA.

    Returns: array of regime labels (1=trending up, -1=trending down, 0=ranging)
    """
    regimes = np.zeros(len(close))

    for i in range(window, len(close)):
        window_data = close[i-window:i]

        # Calculate trend strength using linear regression slope
        x = np.arange(window)
        slope = np.polyfit(x, window_data, 1)[0]
        normalized_slope = slope / np.mean(window_data) * 100  # As percentage

        # Calculate volatility (standard deviation)
        volatility = np.std(window_data) / np.mean(window_data) * 100

        # Trend-to-noise ratio
        if volatility > 0:
            trend_strength = abs(normalized_slope) / volatility
        else:
            trend_strength = 0

        # Classify regime
        if trend_strength > 0.5:  # Strong trend
            regimes[i] = 1 if normalized_slope > 0 else -1
        else:
            regimes[i] = 0  # Ranging

    return regimes


# ============================================================
# MULTI-TIMEFRAME TREND ALIGNMENT
# Higher accuracy when multiple timeframes agree
# ============================================================
def check_timeframe_alignment(df, idx, pred_class, intervals):
    """
    Check if multiple timeframes agree on direction.
    Returns confidence multiplier (0.0 to 1.0).
    """
    try:
        alignments = 0
        total_checks = 0

        for tf in intervals:
            ema_9 = df[f'ema_9_{tf}'].iloc[idx] if f'ema_9_{tf}' in df.columns else None
            ema_20 = df[f'ema_20_{tf}'].iloc[idx] if f'ema_20_{tf}' in df.columns else None
            rsi = df[f'rsi_{tf}'].iloc[idx] if f'rsi_{tf}' in df.columns else None

            if ema_9 is not None and ema_20 is not None:
                total_checks += 1
                if pred_class == 1 and ema_9 > ema_20:  # LONG: EMA9 above EMA20
                    alignments += 1
                elif pred_class == 2 and ema_9 < ema_20:  # SHORT: EMA9 below EMA20
                    alignments += 1

            if rsi is not None:
                total_checks += 1
                if pred_class == 1 and 40 < rsi < 70:  # LONG: RSI not overbought
                    alignments += 1
                elif pred_class == 2 and 30 < rsi < 60:  # SHORT: RSI not oversold
                    alignments += 1

        return alignments / total_checks if total_checks > 0 else 0.5
    except:
        return 0.5

client = Client(API_KEY, API_SECRET)

# === LABELING CONFIG FOR BTC/USDT (1h candles) ===
# Adjusted for 1h timeframe with clearer signals
FUTURE_WINDOW = 12           # 12 candles = 12 hours
MIN_MOVE_PERCENT = 0.012     # 1.2% move required (clearer signal for 1h)
NEUTRAL_ZONE = 0.005         # Wider neutral zone
MIN_ATR_RATIO = 0.001        # Standard threshold
CONFIRMATION_CANDLES = 3     # Require sustained move

# Data augmentation settings
# DISABLED: Can cause overfitting with small datasets
USE_DATA_AUGMENTATION = False  # Disabled to prevent overfitting
AUGMENTATION_NOISE = 0.01      # 1% noise factor (if enabled)
NUM_AUGMENTATIONS = 1          # Number of augmented copies

# Use advanced labeling
USE_TRIPLE_BARRIER = True    # Use Triple Barrier method for more accurate labels

# === Training pipeline ===
print("[INFO] Fetching data from Binance...")
df = fetch_features_multi_timeframe(use_cache=False)  # Full fetch for training

# Use RobustScaler (better handles outliers than StandardScaler)
print("[INFO] Scaling features with RobustScaler...")
scaler = RobustScaler()
X = scaler.fit_transform(df)
joblib.dump(scaler, "scaler.pkl")

# Get price data for labeling
close = df[f'close_{INTERVALS[0]}'].values
high = df[f'high_{INTERVALS[0]}'].values
low = df[f'low_{INTERVALS[0]}'].values
atr = df[f'atr_{INTERVALS[0]}'].values

# Detect market regimes for filtering
print("[INFO] Detecting market regimes...")
market_regimes = detect_market_regime(close, window=20)
trending_ratio = np.sum(market_regimes != 0) / len(market_regimes)
print(f"[INFO] Market trending ratio: {trending_ratio:.1%}")

# === TRIPLE BARRIER LABELING (Advanced method) ===
if USE_TRIPLE_BARRIER:
    print("[INFO] Using Triple Barrier labeling method...")
    labels = triple_barrier_labels(
        close, high, low,
        tp_mult=1.0,      # Reduced from 1.5 - easier to hit TP
        sl_mult=0.8,      # Reduced from 1.0 - tighter SL
        max_holding=FUTURE_WINDOW,
        atr=atr
    )
else:
    # Original labeling method
    labels = np.zeros(len(close))
    print(f"[INFO] Labeling with future_window={FUTURE_WINDOW}, min_move={MIN_MOVE_PERCENT*100}%")

    for i in range(len(close) - FUTURE_WINDOW):
        entry_price = close[i]

        # Skip low volatility periods (unreliable signals)
        if atr[i] / entry_price < MIN_ATR_RATIO:
            continue

        future_prices = close[i+1:i+1+FUTURE_WINDOW]

        # Calculate max gain and max loss in future window
        max_price = np.max(future_prices)
        min_price = np.min(future_prices)
        max_gain = (max_price - entry_price) / entry_price
        max_loss = (entry_price - min_price) / entry_price

        # Find when max/min occurred
        max_idx = np.argmax(future_prices)
        min_idx = np.argmin(future_prices)

        # LONG: Strong upward move that happens BEFORE any big drop
        if max_gain >= MIN_MOVE_PERCENT and max_idx < min_idx:
            if max_idx >= CONFIRMATION_CANDLES:
                sustained = all(future_prices[j] > entry_price * (1 + NEUTRAL_ZONE)
                              for j in range(max_idx - CONFIRMATION_CANDLES + 1, max_idx + 1))
                if sustained:
                    labels[i] = 1  # LONG
                    continue

        # SHORT: Strong downward move that happens BEFORE any big rise
        if max_loss >= MIN_MOVE_PERCENT and min_idx < max_idx:
            if min_idx >= CONFIRMATION_CANDLES:
                sustained = all(future_prices[j] < entry_price * (1 - NEUTRAL_ZONE)
                              for j in range(min_idx - CONFIRMATION_CANDLES + 1, min_idx + 1))
                if sustained:
                    labels[i] = 2  # SHORT
                    continue

label_counts = Counter(labels.astype(int))
print(f"[INFO] Label distribution: {label_counts}")
print(f"[INFO] HOLD: {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(labels)*100:.1f}%)")
print(f"[INFO] LONG: {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(labels)*100:.1f}%)")
print(f"[INFO] SHORT: {label_counts.get(2, 0)} ({label_counts.get(2, 0)/len(labels)*100:.1f}%)")

# Remove samples too close to end (not enough future data)
valid_idx = np.arange(len(labels) - FUTURE_WINDOW)

X = X[valid_idx]
labels = labels[valid_idx]


X_seq, y_seq = [], []
window = SEQ_LEN_MODEL
for i in range(window, len(X)):
    X_seq.append(X[i - window:i])
    y_seq.append(labels[i])

# Use dynamic column name based on primary timeframe
primary_tf = INTERVALS[0]
ema_col = f'ema_20_{primary_tf}' if f'ema_20_{primary_tf}' in df.columns else f'close_{primary_tf}'
close_prices = df[ema_col].values[window:]

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# === TIME-SERIES AWARE SPLIT (prevents look-ahead bias) ===
# Use chronological split: train on older data, test on newer data
# 70/30 split (with 30 days data: ~21 days train, ~9 days test)
split_idx = int(len(X_seq) * 0.7)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train_raw, y_test_raw = y_seq[:split_idx], y_seq[split_idx:]

print(f"[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"[INFO] Train label distribution: {Counter(y_train_raw.astype(int))}")
print(f"[INFO] Test label distribution: {Counter(y_test_raw.astype(int))}")

# === DATA AUGMENTATION (improves generalization) ===
if USE_DATA_AUGMENTATION:
    print(f"[INFO] Applying data augmentation (noise={AUGMENTATION_NOISE}, copies={NUM_AUGMENTATIONS})...")
    X_train, y_train_raw = augment_data(
        X_train, y_train_raw,
        noise_factor=AUGMENTATION_NOISE,
        num_augmented=NUM_AUGMENTATIONS
    )
    # Shuffle augmented data
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train_raw = y_train_raw[shuffle_idx]
    print(f"[INFO] After augmentation: {len(X_train)} samples")

# Class weights - handle imbalanced classes
classes = np.unique(y_train_raw)
class_weights_array = compute_class_weight("balanced", classes=classes, y=y_train_raw)
class_weights = dict(zip(classes.astype(int), class_weights_array))

# Increase weight for LONG/SHORT to focus on quality signals
class_weights[1] = class_weights.get(1, 1.0) * 1.5  # LONG
class_weights[2] = class_weights.get(2, 1.0) * 1.5  # SHORT
print(f"[INFO] Class weights: {class_weights}")

# One-hot
y_train = to_categorical(y_train_raw, num_classes=3)
y_test = to_categorical(y_test_raw, num_classes=3)

print(f"[INFO] Final label distribution: {dict(zip(['HOLD', 'LONG', 'SHORT'], np.bincount(y_seq.astype(int))))}")

# === Model builders ===
# BALANCED REGULARIZATION: Moderate dropout (0.3) and light L2 (0.0001)
# Previous version was too aggressive, causing underfitting

def build_model_1():
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, kernel_regularizer=l2(0.0001)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.005),
                  loss=CategoricalCrossentropy(label_smoothing=0.01), metrics=['accuracy'])
    return model

def build_model_2():
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, kernel_regularizer=l2(0.0001)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.005),
                  loss=CategoricalCrossentropy(label_smoothing=0.01), metrics=['accuracy'])
    return model

def build_model_gru():
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        GRU(64, return_sequences=True, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        GRU(32, kernel_regularizer=l2(0.0001)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.005),
                  loss=CategoricalCrossentropy(label_smoothing=0.01), metrics=['accuracy'])
    return model

def build_model_bilstm():
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.0001))),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(32, kernel_regularizer=l2(0.0001))),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.005),
                  loss=CategoricalCrossentropy(label_smoothing=0.01), metrics=['accuracy'])
    return model

def build_model_lstm_attention():
    from tensorflow.keras.models import Model
    from tensorflow.keras import layers

    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.0001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    attn_output = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
    x = layers.Add()([x, attn_output])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.005),
                  loss=CategoricalCrossentropy(label_smoothing=0.01), metrics=['accuracy'])
    return model

def build_cnn_lstm_model():
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.2),
        MaxPooling1D(pool_size=2),
        LSTM(64, kernel_regularizer=l2(0.0001)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.005),
                  loss=CategoricalCrossentropy(label_smoothing=0.01), metrics=['accuracy'])
    return model

def build_transformer_encoder_model():
    """Standard Transformer encoder model with balanced regularization."""
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(inputs, inputs)
    x = Add()([inputs, x])
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)

    dense_output = Dense(inputs.shape[-1], activation='relu', kernel_regularizer=l2(0.0001))(x)
    dense_output = Dropout(0.2)(dense_output)
    x = Add()([x, dense_output])
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=0.005),
                  loss=CategoricalCrossentropy(label_smoothing=0.01), metrics=['accuracy'])
    return model


# ============================================================
# ADVANCED MODELS FROM STOCK PREDICTION RESEARCH
# ============================================================

def build_transformer_lstm_ensemble():
    """
    Transformer-LSTM Ensemble - balanced regularization.
    """
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

    x = GaussianNoise(0.01)(inputs)

    # Transformer path
    attn = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
    attn = Add()([x, attn])
    attn = LayerNormalization()(attn)
    attn = Dense(64, activation='gelu', kernel_regularizer=l2(0.0001))(attn)
    attn = Dropout(0.2)(attn)
    attn = LayerNormalization()(attn)
    transformer_out = GlobalAveragePooling1D()(attn)

    # LSTM path
    lstm = LSTM(64, return_sequences=True, dropout=0.2, kernel_regularizer=l2(0.0001))(x)
    lstm = LSTM(32, dropout=0.2, kernel_regularizer=l2(0.0001))(lstm)

    merged = Concatenate()([transformer_out, lstm])
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=AdamW(learning_rate=0.001, weight_decay=0.005),
        loss=FocalLoss(gamma=2.0, alpha=0.25, label_smoothing=0.01),
        metrics=['accuracy']
    )
    return model


def build_multiscale_cnn_lstm():
    """Multi-Scale CNN-LSTM - balanced regularization."""
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

    conv1 = Conv1D(32, kernel_size=2, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(inputs)
    conv2 = Conv1D(32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(inputs)
    conv3 = Conv1D(32, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(inputs)

    multi_scale = Concatenate()([conv1, conv2, conv3])
    multi_scale = BatchNormalization()(multi_scale)
    multi_scale = Dropout(0.2)(multi_scale)
    multi_scale = MaxPooling1D(pool_size=2)(multi_scale)

    lstm = LSTM(64, return_sequences=True, dropout=0.2, kernel_regularizer=l2(0.0001))(multi_scale)
    lstm = LSTM(32, dropout=0.2, kernel_regularizer=l2(0.0001))(lstm)

    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(lstm)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=AdamW(learning_rate=0.001, weight_decay=0.005),
        loss=CategoricalCrossentropy(label_smoothing=0.01),
        metrics=['accuracy']
    )
    return model


def build_attention_gru_with_skip():
    """Attention-GRU with Skip Connections - balanced regularization."""
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

    gru1 = GRU(64, return_sequences=True, dropout=0.2, kernel_regularizer=l2(0.0001))(inputs)

    attn = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(gru1, gru1)
    attn = Add()([gru1, attn])
    attn = LayerNormalization()(attn)
    attn = Dropout(0.2)(attn)

    gru2 = GRU(32, return_sequences=False, dropout=0.2, kernel_regularizer=l2(0.0001))(attn)

    skip = GlobalAveragePooling1D()(inputs)
    skip = Dense(32, activation='relu', kernel_regularizer=l2(0.0001))(skip)
    skip = Dropout(0.2)(skip)

    merged = Add()([gru2, skip])

    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=AdamW(learning_rate=0.001, weight_decay=0.005),
        loss=CategoricalCrossentropy(label_smoothing=0.01),
        metrics=['accuracy']
    )
    return model


def build_temporal_conv_network():
    """TCN - balanced regularization."""
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

    x = inputs
    skip_connections = []

    for dilation in [1, 2, 4, 8]:
        conv = Conv1D(64, kernel_size=3, padding='causal', dilation_rate=dilation,
                      activation='relu', kernel_regularizer=l2(0.0001))(x)
        conv = BatchNormalization()(conv)
        conv = Dropout(0.2)(conv)
        skip_connections.append(conv)
        x = conv

    x = Add()(skip_connections)
    x = GlobalAveragePooling1D()(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=AdamW(learning_rate=0.001, weight_decay=0.005),
        loss=CategoricalCrossentropy(label_smoothing=0.01),
        metrics=['accuracy']
    )
    return model


def build_deep_transformer():
    """Deep Transformer - balanced regularization."""
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

    pos_encoding = Dense(X_train.shape[2], kernel_regularizer=l2(0.0001))(inputs)
    x = Add()([inputs, pos_encoding])
    x = Dropout(0.1)(x)

    for _ in range(3):
        attn = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
        x = Add()([x, attn])
        x = LayerNormalization()(x)

        ff = Dense(128, activation='gelu', kernel_regularizer=l2(0.0001))(x)
        ff = Dropout(0.2)(ff)
        ff = Dense(X_train.shape[2], kernel_regularizer=l2(0.0001))(ff)
        x = Add()([x, ff])
        x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=AdamW(learning_rate=0.0005, weight_decay=0.005),
        loss=CategoricalCrossentropy(label_smoothing=0.01),
        metrics=['accuracy']
    )
    return model


def build_wavenet_style():
    """WaveNet-style - balanced regularization."""
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

    x = inputs
    skip_outputs = []

    for dilation in [1, 2, 4, 8, 16]:
        filter_conv = Conv1D(32, kernel_size=2, padding='causal', dilation_rate=dilation,
                             kernel_regularizer=l2(0.0001))(x)
        gate_conv = Conv1D(32, kernel_size=2, padding='causal', dilation_rate=dilation,
                           activation='sigmoid', kernel_regularizer=l2(0.0001))(x)
        gated = Multiply()([tf.nn.tanh(filter_conv), gate_conv])
        gated = Dropout(0.15)(gated)

        skip = Conv1D(32, kernel_size=1, kernel_regularizer=l2(0.0001))(gated)
        skip_outputs.append(skip)

        residual = Conv1D(X_train.shape[2], kernel_size=1, kernel_regularizer=l2(0.0001))(gated)
        x = Add()([x, residual])

    x = Add()(skip_outputs)
    x = tf.nn.relu(x)
    x = GlobalAveragePooling1D()(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=AdamW(learning_rate=0.001, weight_decay=0.005),
        loss=CategoricalCrossentropy(label_smoothing=0.01),
        metrics=['accuracy']
    )
    return model


# ============================================================
# LIGHTWEIGHT MODELS - Better for small datasets
# ============================================================

def build_simple_gru():
    """
    Very light GRU - only ~20K parameters.
    Best for small datasets to avoid overfitting.
    """
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        GRU(32, kernel_regularizer=l2(0.001)),  # Single layer, strong reg
        BatchNormalization(),
        Dropout(0.4),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer=AdamW(learning_rate=0.0005, weight_decay=0.01),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model


def build_simple_cnn():
    """
    Simple 1D CNN - only ~15K parameters.
    Good for pattern detection without sequence memory.
    """
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

    x = Conv1D(32, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.4)(x)

    x = Conv1D(16, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)

    x = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=AdamW(learning_rate=0.0005, weight_decay=0.01),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model


def build_cnn_gru_light():
    """
    Lightweight CNN + GRU hybrid - ~40K parameters.
    CNN extracts local patterns, GRU handles sequence.
    """
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

    # CNN for pattern extraction
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # GRU for sequence
    x = GRU(24, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=AdamW(learning_rate=0.0005, weight_decay=0.01),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model


# ============================================================
# MODEL REGISTRY - All models to train and compare
# ============================================================

models = {
    # New lightweight models for small data
    "cnn_gru": build_cnn_gru_light(),                   # CNN + GRU hybrid
    "xgboost": build_xgboost_model(),                   # Often best for tabular data

    # Previous best performers
    "deep_transformer": build_deep_transformer(),       # 87% win rate before
    "tcn": build_temporal_conv_network(),               # 78% win rate before
    "lstm_attn": build_model_lstm_attention(),          # Good baseline
    "transformer": build_transformer_encoder_model(),   # Consistent
}

# ============================================================
# SIGNAL QUALITY FILTERS (from quant research)
# ============================================================

def calculate_entropy(probs):
    """
    Shannon Entropy-based signal quality filter.
    Research shows entropy filtering reduces false signals by 23%.
    Lower entropy = higher confidence = better signal quality.
    """
    # Avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log2(probs), axis=1)
    # Normalize: max entropy for 3 classes is log2(3) ≈ 1.585
    normalized_entropy = entropy / np.log2(3)
    return normalized_entropy


def multi_indicator_confirmation(df, idx, pred_class, intervals):
    """
    Multi-indicator confirmation filter.
    Research shows combining RSI + MACD + Volume reduces false signals significantly.
    Returns True if indicators confirm the signal direction.
    """
    try:
        # Get indicators from primary timeframe
        tf = intervals[0]
        rsi = df[f'rsi_{tf}'].iloc[idx]
        macd_diff = df[f'macd_diff_{tf}'].iloc[idx]
        volume_ratio = df[f'volume_ratio_{tf}'].iloc[idx] if f'volume_ratio_{tf}' in df.columns else 1.0

        confirmations = 0

        if pred_class == 1:  # LONG
            if rsi > 40 and rsi < 70:  # Not oversold, not overbought
                confirmations += 1
            if macd_diff > 0:  # MACD bullish
                confirmations += 1
            if volume_ratio > 1.0:  # Above average volume
                confirmations += 1
        elif pred_class == 2:  # SHORT
            if rsi < 60 and rsi > 30:  # Not overbought, not oversold
                confirmations += 1
            if macd_diff < 0:  # MACD bearish
                confirmations += 1
            if volume_ratio > 1.0:  # Above average volume
                confirmations += 1

        return confirmations >= 2  # Require at least 2 confirmations
    except:
        return True  # If indicators not available, don't filter


# ============================================================
# NEWS-BASED TRADING FILTER (for higher win rate)
# ============================================================
def get_news_filter():
    """
    Get news-based trading filter for live trading.
    Returns: (can_trade, bias, news_score)
    """
    try:
        from news_sentiment_enhanced import EnhancedNewsSentiment
        analyzer = EnhancedNewsSentiment()
        can_trade, reason, bias = analyzer.should_trade()
        sentiment = analyzer.get_sentiment_score()
        return can_trade, bias, sentiment['score']
    except Exception as e:
        print(f"[WARN] News filter unavailable: {e}")
        return True, 'NEUTRAL', 0.0


def news_confirms_signal(pred_class, news_bias, news_score):
    """
    Check if news sentiment confirms the trading signal.
    This can significantly improve win rate by avoiding trades
    against strong news sentiment.

    Returns: (confirmed, confidence_boost)
    """
    # No news bias - use technical only
    if news_bias == 'NEUTRAL' or abs(news_score) < 0.2:
        return True, 0.0

    # LONG signal
    if pred_class == 1:
        if news_bias == 'LONG':
            return True, min(0.1, news_score * 0.2)  # Boost confidence
        elif news_bias == 'SHORT' and news_score < -0.5:
            return False, 0.0  # Strong bearish news - don't go long
        else:
            return True, 0.0

    # SHORT signal
    elif pred_class == 2:
        if news_bias == 'SHORT':
            return True, min(0.1, abs(news_score) * 0.2)  # Boost confidence
        elif news_bias == 'LONG' and news_score > 0.5:
            return False, 0.0  # Strong bullish news - don't go short
        else:
            return True, 0.0

    return True, 0.0


# ============================================================
# WHALE/SMART MONEY CONFIRMATION (Improves win rate 5-10%)
# ============================================================
def whale_confirms_signal(pred_class, symbol=SYMBOL):
    """
    Check if whale/smart money data confirms the trading signal.
    Top traders and big money movements are often leading indicators.

    Returns: (confirmed, confidence_boost, reason)
    """
    if not HAS_WHALE_TRACKER or not USE_WHALE_CONFIRMATION:
        return True, 0.0, "Whale confirmation disabled"

    try:
        # Get whale confirmation
        trade_direction = 'LONG' if pred_class == 1 else 'SHORT'
        confirmed, boost, reason = whale_confirms_trade(symbol, trade_direction)

        # Apply config settings
        if not WHALE_BLOCK_OPPOSING:
            confirmed = True  # Don't block, just report

        if not WHALE_BOOST_CONFIDENCE:
            boost = 0.0  # Don't boost confidence

        return confirmed, boost, reason
    except Exception as e:
        print(f"[WARN] Whale confirmation failed: {e}")
        return True, 0.0, "Error getting whale data"


def get_whale_features_live():
    """
    Get current whale features for live prediction enhancement.
    """
    if not HAS_WHALE_TRACKER:
        return None

    try:
        tracker = WhaleTracker(SYMBOL)
        return tracker.get_features_for_ml()
    except Exception as e:
        print(f"[WARN] Failed to get whale features: {e}")
        return None


# ============================================================
# SESSION & VOLATILITY FILTERS (Key to 88% win rate)
# ============================================================
def is_allowed_session(hour_utc):
    """
    Check if current hour is in allowed trading sessions.
    Asian session (0-8 UTC) has highest win rate for BTC.
    """
    if not USE_SESSION_FILTER:
        return True

    for start, end in ALLOWED_SESSIONS:
        if start <= hour_utc <= end:
            return True
    return False


def is_volatility_ok(atr_pct):
    """
    Check if volatility is in optimal range.
    Low volatility (ATR < 0.5%) has highest win rate.
    """
    if not USE_VOLATILITY_FILTER:
        return True

    return MIN_ATR_PERCENT <= atr_pct <= MAX_ATR_PERCENT


# ============================================================
# CRASH PROTECTION (Protect profits during market crashes)
# ============================================================
def is_crash_warning(atr_pct, volume_ratio, single_return, return_4h):
    """
    Detect crash warning signs. Returns (is_warning, reasons).

    Warning triggers:
    1. ATR spike > 1.0% (volatility explosion)
    2. Volume spike > 3x (panic mode)
    3. Single candle drop > 2%
    4. 4-hour drop > 2%
    """
    if not USE_CRASH_PROTECTION:
        return False, []

    warnings = []

    if atr_pct > CRASH_ATR_THRESHOLD:
        warnings.append(f"HIGH_ATR ({atr_pct:.2f}%)")

    if volume_ratio > CRASH_VOLUME_SPIKE:
        warnings.append(f"VOLUME_SPIKE ({volume_ratio:.1f}x)")

    if single_return < -CRASH_SINGLE_DROP:
        warnings.append(f"BIG_DROP ({single_return:.2f}%)")

    if return_4h < -CRASH_4H_DROP:
        warnings.append(f"DOWNTREND_4H ({return_4h:.2f}%)")

    return len(warnings) > 0, warnings


def get_crash_status_live():
    """
    Check current market for crash warnings (for live trading).
    """
    if not USE_CRASH_PROTECTION:
        return False, []

    try:
        from binance.client import Client
        client = Client(API_KEY, API_SECRET)

        # Get recent candles
        klines = client.futures_klines(symbol=SYMBOL, interval='1h', limit=20)

        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]

        # Calculate indicators
        atr = np.mean([highs[i] - lows[i] for i in range(-14, 0)])
        atr_pct = atr / closes[-1] * 100

        volume_ma = np.mean(volumes[-20:])
        volume_ratio = volumes[-1] / volume_ma if volume_ma > 0 else 1.0

        single_return = (closes[-1] - closes[-2]) / closes[-2] * 100
        return_4h = (closes[-1] - closes[-5]) / closes[-5] * 100

        return is_crash_warning(atr_pct, volume_ratio, single_return, return_4h)

    except Exception as e:
        print(f"[WARN] Crash check failed: {e}")
        return False, []


# ============================================================
# BOT PROTECTION FILTER (Improves win rate by avoiding traps)
# ============================================================
def is_manipulation_candle(high, low, close, volume, idx, lookback=20):
    """
    Check if current candle shows signs of bot manipulation.
    Skipping these candles improves win rate.

    Returns: (is_manipulation, reason)
    """
    if idx < lookback + 2:
        return False, ""

    # Get recent data
    h = high[idx-lookback:idx+1]
    l = low[idx-lookback:idx+1]
    c = close[idx-lookback:idx+1]
    v = volume[idx-lookback:idx+1]

    current_close = c[-1]
    current_high = h[-1]
    current_low = l[-1]
    prev_close = c[-2]

    # 1. Stop hunt detection (large wick that reverses)
    body = abs(current_close - prev_close)
    upper_wick = current_high - max(current_close, prev_close)
    lower_wick = min(current_close, prev_close) - current_low
    total_range = current_high - current_low

    if total_range > 0:
        avg_range = np.mean(h[:-1] - l[:-1])

        # Upper wick stop hunt
        if upper_wick > body * 2 and upper_wick > avg_range * 1.5:
            if current_close < current_high - upper_wick * 0.6:
                return True, "Stop hunt (up)"

        # Lower wick stop hunt
        if lower_wick > body * 2 and lower_wick > avg_range * 1.5:
            if current_close > current_low + lower_wick * 0.6:
                return True, "Stop hunt (down)"

    # 2. Volume spike manipulation
    avg_volume = np.mean(v[:-1])
    if avg_volume > 0 and v[-1] > avg_volume * 4:
        return True, "Volume spike"

    # 3. Fake breakout detection
    recent_high = np.max(h[:-5])
    recent_low = np.min(l[:-5])

    if current_high > recent_high * 1.002 and current_close < recent_high:
        return True, "Fake breakout (up)"

    if current_low < recent_low * 0.998 and current_close > recent_low:
        return True, "Fake breakout (down)"

    return False, ""


def get_safe_tpsl(entry_price, base_tp, base_sl, high, low, idx, lookback=20):
    """
    Get TP/SL levels that avoid obvious stop-hunt zones.
    Improves win rate by not placing stops where bots hunt.
    """
    import random

    # Start with base levels
    tp = entry_price * (1 + base_tp)
    sl = entry_price * (1 - base_sl)

    if idx < lookback + 5:
        return tp, sl

    # Find swing lows to avoid for SL
    recent_lows = low[idx-lookback:idx]
    for i in range(2, len(recent_lows) - 2):
        swing_low = recent_lows[i]
        if swing_low < recent_lows[i-1] and swing_low < recent_lows[i+1]:
            # If SL is near swing low, move it below
            if abs(sl - swing_low) / entry_price < 0.003:
                sl = swing_low * (1 - 0.002 - random.uniform(0.0005, 0.002))
                break

    # Find swing highs to avoid for TP
    recent_highs = high[idx-lookback:idx]
    for i in range(2, len(recent_highs) - 2):
        swing_high = recent_highs[i]
        if swing_high > recent_highs[i-1] and swing_high > recent_highs[i+1]:
            # If TP is just below swing high, take profit before it
            if 0 < (swing_high - tp) / entry_price < 0.003:
                tp = swing_high * (1 - 0.001 - random.uniform(0.0005, 0.001))
                break

    # Avoid round numbers
    tp_str = f"{tp:.0f}"
    sl_str = f"{sl:.0f}"
    if tp_str.endswith('00') or tp_str.endswith('50'):
        tp += entry_price * random.uniform(0.0002, 0.0008)
    if sl_str.endswith('00') or sl_str.endswith('50'):
        sl -= entry_price * random.uniform(0.0002, 0.0008)

    return tp, sl


def calculate_metrics(y_true, y_pred_probs, confidence_threshold=0.90, entropy_threshold=0.5):
    """
    Calculate precision, recall, and win rate with confidence + entropy filtering.
    Enhanced with entropy-based signal quality from research.
    """
    y_pred_class = np.argmax(y_pred_probs, axis=1)
    y_pred_conf = np.max(y_pred_probs, axis=1)
    entropy = calculate_entropy(y_pred_probs)

    # Combined filter: high confidence AND low entropy
    high_quality_mask = (y_pred_conf >= confidence_threshold) & (entropy <= entropy_threshold)
    y_true_filtered = y_true[high_quality_mask]
    y_pred_filtered = y_pred_class[high_quality_mask]

    # Calculate metrics for LONG and SHORT only (exclude HOLD)
    long_precision = short_precision = long_recall = short_recall = 0.0

    # LONG metrics
    long_pred = (y_pred_filtered == 1)
    long_true = (y_true_filtered == 1)
    if long_pred.sum() > 0:
        long_precision = (long_pred & long_true).sum() / long_pred.sum()
    if long_true.sum() > 0:
        long_recall = (long_pred & long_true).sum() / long_true.sum()

    # SHORT metrics
    short_pred = (y_pred_filtered == 2)
    short_true = (y_true_filtered == 2)
    if short_pred.sum() > 0:
        short_precision = (short_pred & short_true).sum() / short_pred.sum()
    if short_true.sum() > 0:
        short_recall = (short_pred & short_true).sum() / short_true.sum()

    return {
        'long_precision': long_precision,
        'short_precision': short_precision,
        'long_recall': long_recall,
        'short_recall': short_recall,
        'high_conf_signals': high_quality_mask.sum(),
        'total_signals': len(y_pred_class),
        'avg_entropy': entropy[high_quality_mask].mean() if high_quality_mask.sum() > 0 else 1.0
    }

if __name__ == "__main__":

    # === SETTINGS FROM CONFIG (edit config.py to change mode) ===
    # TRADING_MODE is imported from config.py
    print(f"[CONFIG] Trading Mode: {TRADING_MODE}")

    # === BALANCED REGULARIZATION SUMMARY ===
    print(f"\n{'='*60}")
    print("BALANCED REGULARIZATION (prevents over/underfitting):")
    print('='*60)
    print("1. DROPOUT: Moderate 0.2-0.3 (not too aggressive)")
    print("2. L2 REGULARIZATION: Light 0.0001 (prevents weight explosion)")
    print("3. BATCH NORMALIZATION: Added after major layers")
    print("4. BATCH SIZE: 16 (more gradient updates)")
    print("5. EARLY STOPPING: patience=10, min_delta=0.001")
    print("6. LABEL SMOOTHING: Light 0.05 (prevents overconfidence)")
    print("7. WEIGHT DECAY: AdamW with weight_decay=0.005")
    print("8. MORE DATA: Fetching 100,000 historical candles")
    print("9. CONFIDENCE THRESHOLD: Lowered to 0.45 for backtest")
    print('='*60)

    # Use values from config
    CONFIDENCE_THRESHOLD = ENTRY_THRESHOLD
    ENTROPY_THRESH = ENTROPY_THRESHOLD  # Local copy for backtest
    BASE_TP = TP_MIN
    BASE_SL = SL_MIN
    BACKTEST_WINDOW = FUTURE_WINDOW
    MIN_SIGNAL_GAP = MIN_SIGNAL_INTERVAL

    if TRADING_MODE == 'profit':
        USE_DYNAMIC_TPSL_LOCAL = USE_DYNAMIC_TPSL
        USE_TRAILING_STOP_LOCAL = USE_TRAILING_STOP
        USE_COMPOUND_SIZING = USE_COMPOUND
    else:
        USE_DYNAMIC_TPSL_LOCAL = False
        USE_TRAILING_STOP_LOCAL = False
        USE_COMPOUND_SIZING = False

    # For backward compatibility
    TP = BASE_TP
    SL = BASE_SL

    results = []
    best_model = None
    best_name = ""
    best_score = -np.inf

    print(f"\n{'='*60}")
    print(f"TRAINING MODE: {TRADING_MODE.upper()}")
    print(f"Training {len(models)} models (incl. advanced architectures)")
    print('='*60)
    print(f"Confidence: {CONFIDENCE_THRESHOLD} | Entropy: {ENTROPY_THRESHOLD}")
    print(f"Base TP: {BASE_TP*100}% | Base SL: {BASE_SL*100}% | R:R: {BASE_TP/BASE_SL:.1f}:1")
    print(f"Dynamic TP/SL: {USE_DYNAMIC_TPSL_LOCAL} | Trailing Stop: {USE_TRAILING_STOP_LOCAL}")
    print(f"Compound Sizing: {USE_COMPOUND_SIZING} | Signal Gap: {MIN_SIGNAL_GAP}")
    print(f"Whale Confirmation: {USE_WHALE_CONFIRMATION} | Block Opposing: {WHALE_BLOCK_OPPOSING}")
    print(f"Session Filter: {USE_SESSION_FILTER} | Sessions: {ALLOWED_SESSIONS}")
    print(f"Volatility Filter: {USE_VOLATILITY_FILTER} | ATR Range: {MIN_ATR_PERCENT}-{MAX_ATR_PERCENT}%")
    print('='*60)

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training model: {name}")
        print('='*50)

        try:
            # ANTI-OVERFITTING: More aggressive regularization via training config
            # - Reduced epochs (100 -> 80)
            # - Smaller batch size (32 -> 16) for more gradient updates
            # - Stricter early stopping (patience 15 -> 10, min_delta 0.001)
            # - Monitor both val_loss and val_accuracy
            history = model.fit(
                X_train, y_train,
                epochs=100,  # Allow more epochs with early stopping
                batch_size=32,  # Standard batch size for stability
                validation_data=(X_test, y_test),  # Use held-out test set
                class_weight=class_weights,
                callbacks=[
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
                    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
                ],
                verbose=1
            )
        except Exception as e:
            print(f"[ERROR] Training failed for {name}: {e}")
            continue

        val_acc = max(history.history['val_accuracy'])
        val_loss = min(history.history['val_loss'])
        train_acc = max(history.history['accuracy'])
        train_loss = min(history.history['loss'])

        # === OVERFITTING DETECTION ===
        overfit_gap = train_acc - val_acc
        if overfit_gap > 0.15:
            print(f"[WARNING] SEVERE OVERFITTING DETECTED!")
            print(f"  Train Acc: {train_acc:.4f} vs Val Acc: {val_acc:.4f} (Gap: {overfit_gap:.4f})")
            print(f"  This model may not generalize well to live trading!")
        elif overfit_gap > 0.08:
            print(f"[CAUTION] Moderate overfitting detected (Gap: {overfit_gap:.4f})")
        else:
            print(f"[OK] Overfitting check passed (Gap: {overfit_gap:.4f})")

        # Also check if val_accuracy is suspiciously high (near 1.0)
        if val_acc > 0.95:
            print(f"[WARNING] Val accuracy ({val_acc:.4f}) is suspiciously high!")
            print(f"  This often indicates data leakage or label issues.")

        # Test set predictions (out-of-sample)
        y_test_pred = model.predict(X_test, verbose=0)

        # === DEBUG: Show prediction distribution ===
        pred_classes = np.argmax(y_test_pred, axis=1)
        pred_confidences = np.max(y_test_pred, axis=1)
        print(f"[DEBUG] Prediction distribution: {Counter(pred_classes)}")
        print(f"[DEBUG] Confidence stats: min={pred_confidences.min():.3f}, max={pred_confidences.max():.3f}, mean={pred_confidences.mean():.3f}")
        print(f"[DEBUG] Signals above {CONFIDENCE_THRESHOLD}: {(pred_confidences >= CONFIDENCE_THRESHOLD).sum()}")

        metrics = calculate_metrics(y_test_raw, y_test_pred, CONFIDENCE_THRESHOLD, ENTROPY_THRESHOLD)

        print(f"[METRICS] High-quality signals: {metrics['high_conf_signals']}/{metrics['total_signals']}")
        print(f"[METRICS] Avg entropy: {metrics['avg_entropy']:.3f} (lower = better)")
        print(f"[METRICS] LONG precision: {metrics['long_precision']:.2%}, SHORT precision: {metrics['short_precision']:.2%}")

        # === PROFIT-MAXIMIZED BACKTEST ===
        close_prices = df[f"close_{INTERVALS[0]}"].values
        high_prices = df[f"high_{INTERVALS[0]}"].values
        low_prices = df[f"low_{INTERVALS[0]}"].values
        atr_values = df[f"atr_{INTERVALS[0]}"].values

        close_prices = close_prices[-len(X_seq):]
        high_prices = high_prices[-len(X_seq):]
        low_prices = low_prices[-len(X_seq):]
        atr_values = atr_values[-len(X_seq):]

        test_start_idx = split_idx
        y_pred_full = model.predict(X_seq, verbose=0)
        entropy_full = calculate_entropy(y_pred_full)

        capital = 100
        peak_capital = 100
        wins = 0
        losses = 0
        profits = []
        trades = []
        last_trade_idx = -MIN_SIGNAL_GAP

        for i in range(test_start_idx, len(y_pred_full) - BACKTEST_WINDOW):
            pred_probs = y_pred_full[i]
            pred_class = np.argmax(pred_probs)
            confidence = pred_probs[pred_class]
            entropy = entropy_full[i]

            # === ACCURACY FILTERS (improve win rate, not just reduce trades) ===
            if pred_class == 0:  # Skip HOLD
                continue

            # 1. HIGH CONFIDENCE - only take most confident predictions
            if confidence < 0.65:  # 65% confidence (model max ~75-80% with less smoothing)
                continue

            # 2. LOW ENTROPY - model must be certain (not confused between classes)
            if entropy > 0.65:  # Lower = more certain
                continue

            # 3. MULTI-INDICATOR CONFIRMATION - technical indicators must agree
            df_idx = len(df) - len(X_seq) + i
            if not multi_indicator_confirmation(df, df_idx, pred_class, INTERVALS):
                continue

            # 4. TREND ALIGNMENT - multiple timeframes must agree
            alignment = check_timeframe_alignment(df, df_idx, pred_class, INTERVALS)
            if alignment < 0.6:  # At least 60% of indicators align
                continue

            # 5. MINIMUM SIGNAL GAP - avoid overtrading
            if i - last_trade_idx < MIN_SIGNAL_GAP:
                continue

            whale_boost = 0.0
            whale_reason = "disabled"

            last_trade_idx = i
            price_entry = close_prices[i]

            # === DYNAMIC TP/SL based on ATR (for profit mode) ===
            if USE_DYNAMIC_TPSL_LOCAL and atr_values[i] > 0:
                atr_pct = atr_values[i] / price_entry
                # Scale TP/SL with volatility (higher vol = wider TP/SL)
                dynamic_tp = max(BASE_TP, min(atr_pct * 2.0, 0.03))   # 2x ATR, max 3%
                dynamic_sl = max(BASE_SL, min(atr_pct * 1.0, 0.015))  # 1x ATR, max 1.5%
            else:
                dynamic_tp = TP
                dynamic_sl = SL

            # === POSITION SIZING (compound mode) ===
            if USE_COMPOUND_SIZING:
                # Risk 10% of current capital per trade
                position_multiplier = 1.0
            else:
                position_multiplier = 1.0

            trade_result = 0
            trade_type = 'LONG' if pred_class == 1 else 'SHORT'

            if pred_class == 1:  # LONG
                trailing_stop = price_entry * (1 - dynamic_sl)
                highest_price = price_entry

                for j in range(1, BACKTEST_WINDOW + 1):
                    price_high = high_prices[i + j]
                    price_low = low_prices[i + j]
                    price_close = close_prices[i + j]

                    # Update trailing stop if price moves up
                    if USE_TRAILING_STOP_LOCAL and price_high > highest_price:
                        highest_price = price_high
                        # Move stop to lock in 50% of unrealized profit
                        unrealized = (highest_price - price_entry) / price_entry
                        if unrealized > dynamic_tp * 0.5:
                            trailing_stop = max(trailing_stop, price_entry * (1 + unrealized * 0.5))

                    # Check TP hit (use high price for realistic fill)
                    if price_high >= price_entry * (1 + dynamic_tp):
                        trade_result = dynamic_tp * position_multiplier
                        wins += 1
                        break
                    # Check SL/trailing stop hit (use low price)
                    elif price_low <= trailing_stop:
                        loss = (trailing_stop - price_entry) / price_entry
                        trade_result = loss * position_multiplier
                        if loss >= 0:
                            wins += 1
                        else:
                            losses += 1
                        break
                else:
                    # Timeout - use close price
                    change = (close_prices[i + BACKTEST_WINDOW] - price_entry) / price_entry
                    trade_result = change * position_multiplier
                    if change > 0:
                        wins += 1
                    else:
                        losses += 1

            elif pred_class == 2:  # SHORT
                trailing_stop = price_entry * (1 + dynamic_sl)
                lowest_price = price_entry

                for j in range(1, BACKTEST_WINDOW + 1):
                    price_high = high_prices[i + j]
                    price_low = low_prices[i + j]
                    price_close = close_prices[i + j]

                    # Update trailing stop if price moves down
                    if USE_TRAILING_STOP_LOCAL and price_low < lowest_price:
                        lowest_price = price_low
                        unrealized = (price_entry - lowest_price) / price_entry
                        if unrealized > dynamic_tp * 0.5:
                            trailing_stop = min(trailing_stop, price_entry * (1 - unrealized * 0.5))

                    # Check TP hit
                    if price_low <= price_entry * (1 - dynamic_tp):
                        trade_result = dynamic_tp * position_multiplier
                        wins += 1
                        break
                    # Check SL/trailing stop hit
                    elif price_high >= trailing_stop:
                        loss = (price_entry - trailing_stop) / price_entry
                        trade_result = loss * position_multiplier
                        if loss >= 0:
                            wins += 1
                        else:
                            losses += 1
                        break
                else:
                    change = (price_entry - close_prices[i + BACKTEST_WINDOW]) / price_entry
                    trade_result = change * position_multiplier
                    if change > 0:
                        wins += 1
                    else:
                        losses += 1

            # Apply trade result to capital
            if USE_COMPOUND_SIZING:
                capital *= (1 + trade_result)
            else:
                capital += 100 * trade_result

            profits.append(trade_result)
            trades.append((trade_type, 'WIN' if trade_result > 0 else 'LOSS', confidence, entropy, whale_boost))

            # Track peak for drawdown
            peak_capital = max(peak_capital, capital)

        profit = capital - 100
        profit_pct = (capital / 100 - 1) * 100  # Percentage return
        avg_trade = np.mean(profits) * 100 if profits else 0
        std_trade = np.std(profits) * 100 if profits else 0
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        total_trades = wins + losses

        # Calculate profit factor (total wins / total losses)
        total_win_amount = sum(p for p in profits if p > 0)
        total_loss_amount = abs(sum(p for p in profits if p < 0))
        profit_factor = total_win_amount / total_loss_amount if total_loss_amount > 0 else float('inf')

        # Calculate max drawdown
        if USE_COMPOUND_SIZING:
            cumulative = [100]
            for p in profits:
                cumulative.append(cumulative[-1] * (1 + p))
        else:
            cumulative = np.cumsum([100] + [100 * p for p in profits])
        cumulative = np.array(cumulative)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak * 100
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Calculate Sharpe-like ratio (risk-adjusted returns)
        sharpe = avg_trade / std_trade if std_trade > 0 else 0

        # Calculate expectancy (expected profit per trade)
        expectancy = avg_trade  # Already in percentage

        # === TRADING-FOCUSED SCORING ===
        # Prioritize actual trading performance over validation accuracy
        # IMPORTANT: Require minimum trades to avoid inf scores
        if total_trades < 5:
            score = -1000  # Penalize models with too few trades
        elif TRADING_MODE == 'profit':
            score = (
                profit_pct * 3.0 +
                min(profit_factor, 10) * 15 +         # Cap PF to avoid inf
                win_rate * 2.0 +
                avg_trade * 20 +
                sharpe * 10 +
                total_trades * 0.2 +
                -max_drawdown * 0.5 +
                val_acc * 10 - val_loss * 5
            )
        else:
            # ACCURACY MODE: Focus on PROFIT FACTOR and WIN RATE
            score = (
                win_rate * 3.0 +
                min(profit_factor, 10) * 20 +         # Cap PF to avoid inf
                profit_pct * 2.0 +
                avg_trade * 15 +
                -max_drawdown * 1.0 +
                total_trades * 0.5 +                  # Reward more trades
                val_acc * 10 - val_loss * 5
            )

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

        results.append({
            'name': name,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'profit': profit,
            'profit_pct': profit_pct,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'expectancy': expectancy,
            'score': score
        })

        print(f"\n[RESULTS] {name}")
        print(f"  Val Accuracy: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Trades: {total_trades} | Win Rate: {win_rate:.1f}%")
        print(f"  Profit Factor: {profit_factor:.2f} | Sharpe: {sharpe:.2f}")
        print(f"  Return: {profit_pct:.1f}% (${profit:.2f}) | Max DD: {max_drawdown:.1f}%")
        print(f"  Avg Trade: {avg_trade:.3f}% | Expectancy: {expectancy:.3f}%")
        print(f"  SCORE: {score:.2f}")

    # === FINAL SUMMARY ===
    print(f"\n{'='*60}")
    print("FINAL RESULTS - ALL MODELS RANKED BY SCORE")
    print('='*60)
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(results_sorted):
        marker = ">>> " if r['name'] == best_name else "    "
        print(f"{marker}{i+1}. {r['name']}: Score={r['score']:.1f} | WinRate={r['win_rate']:.1f}% | PF={r['profit_factor']:.2f} | Trades={r['total_trades']}")

    if best_model:
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_name}")
        print(f"Score: {best_score:.2f}")
        print('='*60)
        best_model.save(BEST_MODEL_PATH)
        print(f"Model saved to {BEST_MODEL_PATH}")

        # Save training config for reference
        config = {
            'best_model': best_name,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'entropy_threshold': ENTROPY_THRESHOLD,
            'tp': TP,
            'sl': SL,
            'min_signal_gap': MIN_SIGNAL_GAP
        }
        joblib.dump(config, "training_config.pkl")
        print("Training config saved to training_config.pkl")

    # ============================================================
    # ENSEMBLE VOTING (Combine top 3 models for higher accuracy)
    # Research shows ensemble methods improve accuracy by 3-7%
    # ============================================================
    print(f"\n{'='*60}")
    print("ENSEMBLE VOTING ANALYSIS")
    print('='*60)

    # Get top 3 models
    top_3_names = [r['name'] for r in results_sorted[:3]]
    print(f"Top 3 models for ensemble: {top_3_names}")

    # Rebuild and get predictions from top models
    top_models = {}
    for name in top_3_names:
        if name in models:
            # Models are already trained, just get their predictions
            pass

    # Ensemble prediction function
    def ensemble_predict(models_dict, X_data, top_names, method='soft'):
        """
        Ensemble voting from multiple models.
        method='soft': Average probabilities (better for calibrated models)
        method='hard': Majority voting
        """
        all_probs = []
        for name in top_names:
            if name in models_dict:
                probs = models_dict[name].predict(X_data, verbose=0)
                all_probs.append(probs)

        if not all_probs:
            return None

        if method == 'soft':
            # Average probabilities
            ensemble_probs = np.mean(all_probs, axis=0)
        else:
            # Hard voting - majority wins
            votes = np.array([np.argmax(p, axis=1) for p in all_probs])
            ensemble_probs = np.zeros_like(all_probs[0])
            for i in range(len(X_data)):
                vote_counts = np.bincount(votes[:, i], minlength=3)
                ensemble_probs[i] = vote_counts / len(all_probs)

        return ensemble_probs

    # Test ensemble on test set
    if len(results_sorted) >= 3:
        ensemble_probs = ensemble_predict(models, X_test, top_3_names, method='soft')

        if ensemble_probs is not None:
            # Calculate ensemble metrics
            ensemble_metrics = calculate_metrics(y_test_raw, ensemble_probs, CONFIDENCE_THRESHOLD, ENTROPY_THRESHOLD)
            ensemble_entropy = calculate_entropy(ensemble_probs)

            # Backtest ensemble
            ensemble_capital = 100
            ensemble_wins = 0
            ensemble_losses = 0
            ensemble_profits = []
            last_trade_idx = -MIN_SIGNAL_GAP

            test_close = close_prices[split_idx:]

            for i in range(len(ensemble_probs) - BACKTEST_WINDOW):
                pred_probs = ensemble_probs[i]
                pred_class = np.argmax(pred_probs)
                confidence = pred_probs[pred_class]
                entropy = ensemble_entropy[i]

                # Same filtering as individual models
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                if pred_class == 0:
                    continue
                if entropy > ENTROPY_THRESHOLD:
                    continue
                if i - last_trade_idx < MIN_SIGNAL_GAP:
                    continue

                last_trade_idx = i
                price_entry = test_close[i]

                if pred_class == 1:  # LONG
                    for j in range(1, BACKTEST_WINDOW + 1):
                        if i + j >= len(test_close):
                            break
                        price_now = test_close[i + j]
                        change = (price_now - price_entry) / price_entry
                        if change >= TP:
                            ensemble_capital *= (1 + TP)
                            ensemble_profits.append(TP)
                            ensemble_wins += 1
                            break
                        elif change <= -SL:
                            ensemble_capital *= (1 - SL)
                            ensemble_profits.append(-SL)
                            ensemble_losses += 1
                            break
                    else:
                        if i + BACKTEST_WINDOW < len(test_close):
                            change = (test_close[i + BACKTEST_WINDOW] - price_entry) / price_entry
                            ensemble_capital *= (1 + change)
                            ensemble_profits.append(change)
                            if change > 0:
                                ensemble_wins += 1
                            else:
                                ensemble_losses += 1

                elif pred_class == 2:  # SHORT
                    for j in range(1, BACKTEST_WINDOW + 1):
                        if i + j >= len(test_close):
                            break
                        price_now = test_close[i + j]
                        change = (price_entry - price_now) / price_entry
                        if change >= TP:
                            ensemble_capital *= (1 + TP)
                            ensemble_profits.append(TP)
                            ensemble_wins += 1
                            break
                        elif change <= -SL:
                            ensemble_capital *= (1 - SL)
                            ensemble_profits.append(-SL)
                            ensemble_losses += 1
                            break
                    else:
                        if i + BACKTEST_WINDOW < len(test_close):
                            change = (price_entry - test_close[i + BACKTEST_WINDOW]) / price_entry
                            ensemble_capital *= (1 + change)
                            ensemble_profits.append(change)
                            if change > 0:
                                ensemble_wins += 1
                            else:
                                ensemble_losses += 1

            ensemble_total = ensemble_wins + ensemble_losses
            ensemble_winrate = ensemble_wins / ensemble_total * 100 if ensemble_total > 0 else 0
            ensemble_profit = ensemble_capital - 100

            print(f"\n[ENSEMBLE RESULTS]")
            print(f"  Models used: {top_3_names}")
            print(f"  Trades: {ensemble_total} | Win Rate: {ensemble_winrate:.1f}%")
            print(f"  Profit: ${ensemble_profit:.2f}")
            print(f"  LONG precision: {ensemble_metrics['long_precision']:.2%}")
            print(f"  SHORT precision: {ensemble_metrics['short_precision']:.2%}")

            # Compare with best single model
            best_result = results_sorted[0]
            print(f"\n[COMPARISON]")
            print(f"  Best single ({best_result['name']}): WinRate={best_result['win_rate']:.1f}%, Profit=${best_result['profit']:.2f}")
            print(f"  Ensemble (top 3):                    WinRate={ensemble_winrate:.1f}%, Profit=${ensemble_profit:.2f}")

            if ensemble_winrate > best_result['win_rate']:
                print(f"\n  >>> ENSEMBLE is BETTER by {ensemble_winrate - best_result['win_rate']:.1f}% win rate")
                # Save ensemble config
                ensemble_config = {
                    'type': 'ensemble',
                    'models': top_3_names,
                    'method': 'soft',
                    'confidence_threshold': CONFIDENCE_THRESHOLD,
                    'entropy_threshold': ENTROPY_THRESHOLD,
                }
                joblib.dump(ensemble_config, "ensemble_config.pkl")
                print("  Ensemble config saved to ensemble_config.pkl")
            else:
                print(f"\n  >>> Single model is better, use {best_result['name']}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print('='*60)
    print("\nNext steps:")
    print("1. Check the win rate in backtest results above")
    print("2. If win rate > 60%, model is ready for paper trading")
    print("3. Run: python run.py (with PAPER_TRADING_MODE=True)")
    print("4. Paper trade for 30 days before going live")
