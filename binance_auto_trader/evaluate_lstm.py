import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Load best model name
try:
    with open("best_model_name.txt") as f:
        model_name = f.read().strip()
except FileNotFoundError:
    model_name = "UNKNOWN"

print(f"🎯 Evaluating best model: {model_name}")

# Load model and data
X, y = joblib.load("lstm_data.pkl")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: show class weights for debugging
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
print("📊 Computed Class Weights:")
for label, weight in zip(["HOLD", "LONG", "SHORT"], class_weights):
    print(f"{label}: {weight:.2f}")

# Evaluate model
model = load_model("lstm_model.keras")
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
labels = ["HOLD", "LONG", "SHORT"]
used_labels = sorted(np.unique(np.concatenate([y_test, y_pred_class])).astype(int))
active_labels = [labels[i] for i in used_labels]

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=active_labels).plot()
plt.title(f"Confusion Matrix of Model: {model_name}")
plt.savefig("confusion_matrix.png")
plt.show()

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_class, target_names=labels, zero_division=0))
