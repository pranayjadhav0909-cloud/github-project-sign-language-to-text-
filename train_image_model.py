
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_DIR = Path("data/raw/dataset2")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

EPOCHS_INITIAL = 12
EPOCHS_FINE_TUNE = 8

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# ==========================================
# 2. LOAD DATASET FILE PATHS
# ==========================================

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}"
    )

class_names = sorted([
    folder.name
    for folder in DATA_DIR.iterdir()
    if folder.is_dir()
])

print("Classes:", class_names)
print("Total classes:", len(class_names))

if len(class_names) != 35:
    print("WARNING: Expected 35 classes. Check folder structure.")

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}

image_paths = []
labels = []

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for class_name in class_names:

    folder = DATA_DIR / class_name

    for file in folder.iterdir():

        if file.suffix.lower() in extensions:

            image_paths.append(str(file))
            labels.append(class_to_index[class_name])

image_paths = np.array(image_paths)
labels = np.array(labels)

print("\nTotal images:", len(image_paths))

# ==========================================
# 3. STRATIFIED DATA SPLIT
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print("\nDataset distribution:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(image, tf.float32)

    return image, label


def create_dataset(paths, targets, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, targets)
    )

    if training:
        dataset = dataset.shuffle(
            len(paths),
            seed=SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    return dataset.prefetch(AUTOTUNE)


train_ds = create_dataset(
    X_train, y_train, training=True
)

val_ds = create_dataset(
    X_val, y_val
)

test_ds = create_dataset(
    X_test, y_test
)

# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.04),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomContrast(0.10)
], name="data_augmentation")

# No horizontal flipping because it may
# change the meaning of hand configurations.

# ==========================================
# 6. LOAD MOBILENETV2
# ==========================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)

x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.30)(x)

x = tf.keras.layers.Dense(
    256,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.20)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# ==========================================
# 7. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7
    ),

    # Save the best-performing model
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_sign_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    ),

    # Automatically recover interrupted training
    tf.keras.callbacks.BackupAndRestore(
        backup_dir="models/training_backup"
    )
]


# ==========================================
# 8. INITIAL TRAINING
# ==========================================

print("\nSTARTING INITIAL TRAINING")

history_initial = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_INITIAL,
    callbacks=callbacks
)

# ==========================================
# 9. FINE-TUNING
# ==========================================

print("\nSTARTING FINE-TUNING")

base_model.trainable = True

# Freeze early layers and fine-tune later layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=callbacks
)

# ==========================================
# 10. LOAD BEST MODEL
# ==========================================

best_model = tf.keras.models.load_model(
    "models/best_sign_model.keras"
)

# ==========================================
# 11. TEST EVALUATION
# ==========================================

print("\nEVALUATING ON TEST DATA")

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

predictions = best_model.predict(test_ds, verbose=1)

y_pred = np.argmax(predictions, axis=1)

print("\nTest Accuracy:", round(test_accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    )
)

# ==========================================
# 12. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(class_names))
)

plt.figure(figsize=(18, 15))

import seaborn as sns

sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("ISL Recognition Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=200
)

plt.close()

# ==========================================
# 13. SAVE FINAL MODEL AND LABELS
# ==========================================

best_model.save("models/isl_mobilenetv2.keras")

with open(
    "models/classes.json",
    "w"
) as file:

    json.dump(class_names, file)

print("\nTRAINING COMPLETED")

print("Saved model: models/isl_mobilenetv2.keras")
print("Saved classes: models/classes.json")
print("Confusion matrix: outputs/confusion_matrix.png")

if test_accuracy >= 0.95:
    print("95%+ TEST ACCURACY TARGET ACHIEVED")
else:
    print("95% target not yet achieved.")
    print("Further optimization and evaluation required.")
