import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils import shuffle

from config import *

# === Step 1: Load data ===
df_misclassified = pd.read_csv("misclassified_trades.csv")
X_seq, y_seq, close_prices, high_prices, low_prices = joblib.load("lstm_data.pkl")

# Adjust input sequence length to match SEQ_LEN_MODEL for training
if X_seq.shape[1] > SEQ_LEN_MODEL:
    X_seq = X_seq[:, -SEQ_LEN_MODEL:, :]
elif X_seq.shape[1] < SEQ_LEN_MODEL:
    raise ValueError(f"X_seq có độ dài {X_seq.shape[1]} nhỏ hơn SEQ_LEN_MODEL={SEQ_LEN_MODEL}")

# === Step 2: Extract hard examples ===
hard_indices = df_misclassified["Index"].values
hard_indices = hard_indices[hard_indices < len(X_seq)]
X_hard = X_seq[hard_indices]
y_hard = y_seq[hard_indices]

# === Step 3: Extract soft borderline examples ===
model = load_model(BEST_MODEL_PATH, compile=False)
EXPECTED_SEQ_LEN = model.input_shape[1]

# Prepare padded input for prediction
if X_seq.shape[1] > EXPECTED_SEQ_LEN:
    X_seq_predict = X_seq[:, -EXPECTED_SEQ_LEN:, :]
elif X_seq.shape[1] < EXPECTED_SEQ_LEN:
    pad_width = EXPECTED_SEQ_LEN - X_seq.shape[1]
    X_seq_predict = np.pad(X_seq, ((0, 0), (pad_width, 0), (0, 0)), mode='edge')
else:
    X_seq_predict = X_seq

y_proba = model.predict(X_seq_predict, verbose=0)
confidences = np.max(y_proba, axis=1)
borderline_indices = np.where((confidences >= 0.4) & (confidences <= 0.7))[0]
X_soft = X_seq[borderline_indices]
y_soft = y_seq[borderline_indices]

# === Step 4: Repeat hard and borderline examples ===
X_hard_rep = np.repeat(X_hard, 3, axis=0)
y_hard_rep = np.repeat(y_hard, 3, axis=0)
X_soft_rep = np.repeat(X_soft, 2, axis=0)
y_soft_rep = np.repeat(y_soft, 2, axis=0)

# === Step 5: Combine all ===
X_augmented = np.concatenate([X_seq, X_hard_rep, X_soft_rep], axis=0)
y_augmented = np.concatenate([y_seq, y_hard_rep, y_soft_rep], axis=0)

# === Step 5.1: Pad augmented data to EXPECTED_SEQ_LEN ===
if X_augmented.shape[1] < EXPECTED_SEQ_LEN:
    pad_width = EXPECTED_SEQ_LEN - X_augmented.shape[1]
    X_augmented = np.pad(X_augmented, ((0, 0), (pad_width, 0), (0, 0)), mode='edge')
elif X_augmented.shape[1] > EXPECTED_SEQ_LEN:
    X_augmented = X_augmented[:, -EXPECTED_SEQ_LEN:, :]

# === Step 6: Create sample weights ===
sample_weights = np.ones(len(X_augmented))
sample_weights[-len(y_hard_rep)-len(y_soft_rep):-len(y_soft_rep)] *= 5.0
sample_weights[-len(y_soft_rep):] *= 3.0
y_aug_cat = to_categorical(y_augmented, num_classes=3)

# === Step 7: Shuffle the data ===
X_augmented, y_aug_cat, sample_weights = shuffle(
    X_augmented, y_aug_cat, sample_weights, random_state=42
)

# === Step 8: Fine-tune the model ===
model.compile(
    optimizer='adam',
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy'],
    weighted_metrics=[]
)

model.fit(
    X_augmented, y_aug_cat,
    epochs=200,
    batch_size=128,
    validation_split=0.1,
    sample_weight=sample_weights,
    callbacks=[
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, verbose=1),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
    ],
    verbose=2
)

# Save final model
model.save(FINE_TUNE_MODEL_PATH)
