import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

DATA_DIR = "dataset"
FRAMES, FEATURES = 30, 126
X, labels = [], []

for label in sorted(os.listdir(DATA_DIR)):
    folder = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder):
        continue
    for name in os.listdir(folder):
        if name.endswith(".npy"):
            arr = np.load(os.path.join(folder, name))
            if arr.shape == (FRAMES, FEATURES):
                X.append(arr.astype(np.float32))
                labels.append(label)

if not X:
    raise SystemExit("No valid samples. Collect data first.")

X = np.asarray(X, dtype=np.float32)
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Prototype split only. For final research, split by signer/session groups.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

os.makedirs("models", exist_ok=True)
np.save("models/classes.npy", encoder.classes_)

# LSTM sequence model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(FRAMES, FEATURES)),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
]
model.fit(X_train, y_train, validation_split=0.2,
          epochs=100, batch_size=16, callbacks=callbacks)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print(f"LSTM test accuracy (prototype split): {acc:.4f}")
print(classification_report(y_test, pred, target_names=encoder.classes_))
np.save("outputs/confusion_matrix.npy", confusion_matrix(y_test, pred))
model.save("models/sign_lstm.keras")

# Random Forest baseline on mean + standard deviation over time
X_rf = np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)
Xtr, Xte, ytr, yte = train_test_split(
    X_rf, y, test_size=0.2, random_state=42, stratify=y
)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                            random_state=42, n_jobs=-1)
rf.fit(Xtr, ytr)
rf_pred = rf.predict(Xte)
print(f"Random Forest test accuracy (prototype split): {accuracy_score(yte, rf_pred):.4f}")
joblib.dump(rf, "models/random_forest.joblib")
print("Saved models and class labels.")
