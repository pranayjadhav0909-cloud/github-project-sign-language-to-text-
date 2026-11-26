
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import defaultdict

MODEL_PATH = "models/isl_mobilenetv2.keras"
DATASET_PATH = Path("data/raw/dataset2")
CLASSES_PATH = "models/classes.json"

SAMPLES_PER_CLASS = 30

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

if isinstance(classes, dict):
    class_names = [
        name for name, index in
        sorted(classes.items(), key=lambda item: int(item[1]))
    ]
else:
    class_names = classes

results = defaultdict(lambda: [0, 0])
confusions = defaultdict(int)

extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

for label in class_names:

    folder = DATASET_PATH / label

    if not folder.exists():
        print(f"Missing folder: {label}")
        continue

    images = []

    for ext in extensions:
        images.extend(folder.glob(ext))

    images = sorted(images)[:SAMPLES_PER_CLASS]

    if not images:
        print(f"No images found: {label}")
        continue

    for image_path in images:

        image = tf.keras.utils.load_img(
            image_path,
            target_size=(224, 224),
            color_mode="rgb"
        )

        image = tf.keras.utils.img_to_array(image)
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image, verbose=0)[0]

        predicted_index = int(np.argmax(prediction))
        predicted_label = class_names[predicted_index]

        results[label][1] += 1

        if predicted_label == label:
            results[label][0] += 1
        else:
            confusions[(label, predicted_label)] += 1

print("\n========== CLASS-WISE ACCURACY ==========")

total_correct = 0
total_images = 0

for label in class_names:

    correct, total = results[label]

    if total == 0:
        continue

    accuracy = (correct / total) * 100

    total_correct += correct
    total_images += total

    print(
        f"{label:>3}: {correct:2}/{total:2} "
        f"= {accuracy:.2f}%"
    )

print("\n========== OVERALL DIAGNOSTIC ACCURACY ==========")

if total_images:
    print(
        f"{total_correct}/{total_images} = "
        f"{(total_correct / total_images) * 100:.2f}%"
    )

print("\n========== COMMON CONFUSIONS ==========")

for (actual, predicted), count in sorted(
    confusions.items(),
    key=lambda item: item[1],
    reverse=True
)[:20]:

    print(f"Actual {actual} predicted as {predicted}: {count}")
