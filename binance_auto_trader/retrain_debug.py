#retrain_debug.py
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from config import *

# === Step 1: Load data ===
df_misclassified = pd.read_csv("misclassified_trades.csv")
X_seq, y_seq, close_prices = joblib.load("lstm_data.pkl")

if X_seq.shape[1] > SEQ_LEN_MODEL:
    X_seq = X_seq[:, -SEQ_LEN_MODEL:, :]
elif X_seq.shape[1] < SEQ_LEN_MODEL:
    raise ValueError(f"X_seq có độ dài {X_seq.shape[1]} nhỏ hơn SEQ_LEN_MODEL={SEQ_LEN_MODEL}")

# === Step 2: Extract hard examples ===
hard_indices = df_misclassified["Index"].values
X_hard = X_seq[hard_indices]
y_hard = y_seq[hard_indices]

# === Step 3: Extract soft borderline examples ===
# Load model
model = load_model(BEST_MODEL_PATH, compile=False)

# Predict probabilities on all samples
y_proba = model.predict(X_seq, verbose=0)

# Compute confidence (max probability per sample)
confidences = np.max(y_proba, axis=1)

# Borderline: confidence between 0.4 and 0.7
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

# Create sample weights
sample_weights = np.ones(len(X_augmented))
sample_weights[-len(y_hard_rep)-len(y_soft_rep):-len(y_soft_rep)] *= 5.0  # hard
sample_weights[-len(y_soft_rep):] *= 3.0  # borderline

# One-hot encode labels
y_aug_cat = to_categorical(y_augmented, num_classes=3)

# === Step 6: Fine-tune the model ===
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    weighted_metrics=[]
)

model.fit(
    X_augmented, y_aug_cat,
    epochs=100,
    batch_size=128,
    validation_split=0.1,
    sample_weight=sample_weights,
    callbacks=[
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=1),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
    ],
    verbose=2
)

# Save final model
model.save(FINE_TUNE_MODEL_PATH)
