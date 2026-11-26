
import cv2
import json
import numpy as np
import tensorflow as tf
from collections import deque, Counter

# =====================================
# 1. LOAD TRAINED MODEL
# =====================================

MODEL_PATH = "models/isl_mobilenetv2.keras"
CLASSES_PATH = "models/classes.json"

print("Loading ISL model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

# Support both list and dictionary class mappings
if isinstance(classes, dict):
    class_names = [
        name for name, index in
        sorted(classes.items(), key=lambda item: int(item[1]))
    ]
else:
    class_names = classes

print("Model loaded successfully!")
print("Classes:", class_names)

# =====================================
# 2. SETTINGS
# =====================================

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.50
STABLE_FRAMES = 6

prediction_history = deque(maxlen=10)

# =====================================
# 3. START WEBCAM
# =====================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("ISL Recognition Started")
print("Press Q to exit")

# =====================================
# 4. REAL-TIME RECOGNITION
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot read webcam frame.")
        break

    # Mirror webcam image
    frame = cv2.flip(frame, 1)

    # ---------------------------------
    # IMAGE PREPROCESSING
    # ---------------------------------

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image.astype(np.float32)

    # IMPORTANT:
    # Do not divide by 127.5 or subtract 1.
    # The trained model already contains
    # its own Rescaling layer.

    image = np.expand_dims(image, axis=0)

    # ---------------------------------
    # MODEL PREDICTION
    # ---------------------------------

    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    confidence = float(probabilities[predicted_index])

    raw_prediction = class_names[predicted_index]

    # ---------------------------------
    # CONFIDENCE CHECK
    # ---------------------------------

    if confidence >= CONFIDENCE_THRESHOLD:
        prediction_history.append(raw_prediction)
    else:
        prediction_history.clear()

    # ---------------------------------
    # STABLE PREDICTION
    # ---------------------------------

    stable_prediction = "UNKNOWN"

    if len(prediction_history) >= STABLE_FRAMES:

        most_common = Counter(
            prediction_history
        ).most_common(1)[0]

        label, count = most_common

        if count >= STABLE_FRAMES:
            stable_prediction = label

    # ---------------------------------
    # TOP 3 PREDICTIONS
    # ---------------------------------

    top_indices = np.argsort(probabilities)[-3:][::-1]

    top_text = []

    for index in top_indices:

        label = class_names[int(index)]

        score = probabilities[index] * 100

        top_text.append(
            f"{label}: {score:.1f}%"
        )

    # ---------------------------------
    # DISPLAY INFORMATION
    # ---------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 285),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        frame,
        f"Raw Prediction: {raw_prediction}",
        (15, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.2f}%",
        (15, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Stable Prediction: {stable_prediction}",
        (15, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    for i, text in enumerate(top_text):

        cv2.putText(
            frame,
            f"Top {i + 1}: {text}",
            (15, 195 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1
        )

    cv2.imshow(
        "ISL Real-Time Recognition",
        frame
    )

    # ---------------------------------
    # EXIT
    # ---------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =====================================
# 5. RELEASE RESOURCES
# =====================================

cap.release()
cv2.destroyAllWindows()

print("ISL Recognition Stopped")
