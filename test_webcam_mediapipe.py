
import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp

from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_PATH = "models/isl_mobilenetv2.keras"
CLASSES_PATH = "models/classes.json"
HAND_MODEL_PATH = "models/hand_landmarker.task"

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.50
STABLE_FRAMES = 6
PADDING = 40

CAMERA_INDEX = 0

prediction_history = deque(maxlen=10)


# ==================================================
# 2. LOAD CLASSIFICATION MODEL
# ==================================================

print("\nLoading MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as file:
    classes = json.load(file)

if isinstance(classes, dict):
    class_names = [
        name for name, index in
        sorted(classes.items(), key=lambda item: int(item[1]))
    ]
else:
    class_names = classes

if len(class_names) != model.output_shape[-1]:
    raise ValueError("Class count does not match model output.")

print("Classifier loaded!")
print("Number of classes:", len(class_names))
print("Available classes:", class_names)


# ==================================================
# 3. LOAD MEDIAPIPE HAND DETECTOR
# ==================================================

print("\nLoading MediaPipe hand detector...")

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    options
)

print("MediaPipe loaded successfully!")


# ==================================================
# 4. START WEBCAM
# ==================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    detector.close()
    raise RuntimeError("Cannot open webcam.")

print("\nISL MediaPipe Recognition Started")
print("Press Q to exit")


# ==================================================
# 5. MAIN WEBCAM LOOP
# ==================================================

crop_window_created = False

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Unable to read webcam frame.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        display_frame = frame.copy()

        prediction_text = "NO HAND DETECTED"
        raw_prediction = "NONE"
        confidence = 0.0

        top_predictions = []

        # ==========================================
        # 6. MEDIAPIPE HAND DETECTION
        # ==========================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = detector.detect(mp_image)

        hand_count = len(results.hand_landmarks)

        # ==========================================
        # 7. PROCESS DETECTED HANDS
        # ==========================================

        if hand_count > 0:

            all_x = []
            all_y = []

            for hand_landmarks in results.hand_landmarks:

                for point in hand_landmarks:

                    px = int(point.x * width)
                    py = int(point.y * height)

                    all_x.append(px)
                    all_y.append(py)

                    cv2.circle(
                        display_frame,
                        (px, py),
                        3,
                        (0, 0, 255),
                        -1
                    )

            # ======================================
            # 8. COMBINED HAND BOUNDING BOX
            # ======================================

            x1 = max(0, min(all_x) - PADDING)
            y1 = max(0, min(all_y) - PADDING)

            x2 = min(width, max(all_x) + PADDING)
            y2 = min(height, max(all_y) + PADDING)

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # ======================================
            # 9. CROP COMPLETE HAND REGION
            # ======================================

            hand_crop = frame[y1:y2, x1:x2]

            if hand_crop.size > 0:

                # Show actual cropped image
                cv2.imshow(
                    "Hand Crop Sent to Model",
                    hand_crop
                )

                crop_window_created = True

                # ==================================
                # 10. IMAGE PREPROCESSING
                # ==================================

                crop_rgb = cv2.cvtColor(
                    hand_crop,
                    cv2.COLOR_BGR2RGB
                )

                resized = cv2.resize(
                    crop_rgb,
                    (IMG_SIZE, IMG_SIZE)
                )

                image = resized.astype(np.float32)

                # Model already contains Rescaling.
                # Do not normalize again.

                image = np.expand_dims(
                    image,
                    axis=0
                )

                # ==================================
                # 11. MOBILE NET PREDICTION
                # ==================================

                probabilities = model.predict(
                    image,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(probabilities)
                )

                raw_prediction = class_names[
                    predicted_index
                ]

                confidence = float(
                    probabilities[predicted_index]
                )

                # ==================================
                # 12. CONFIDENCE FILTER
                # ==================================

                if confidence >= CONFIDENCE_THRESHOLD:

                    prediction_history.append(
                        raw_prediction
                    )

                else:

                    prediction_history.clear()

                prediction_text = "UNKNOWN"

                # ==================================
                # 13. STABLE PREDICTION
                # ==================================

                if len(prediction_history) >= STABLE_FRAMES:

                    label, count = Counter(
                        prediction_history
                    ).most_common(1)[0]

                    if count >= STABLE_FRAMES:
                        prediction_text = label

                # ==================================
                # 14. TOP THREE PREDICTIONS
                # ==================================

                top_indices = np.argsort(
                    probabilities
                )[-3:][::-1]

                for index in top_indices:

                    label = class_names[int(index)]

                    score = float(
                        probabilities[index]
                    ) * 100

                    top_predictions.append(
                        f"{label}: {score:.2f}%"
                    )

        else:

            prediction_history.clear()

            prediction_text = "NO HAND DETECTED"
            raw_prediction = "NONE"
            confidence = 0.0

        # ==========================================
        # 15. DISPLAY INFORMATION
        # ==========================================

        cv2.rectangle(
            display_frame,
            (0, 0),
            (width, 255),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            display_frame,
            f"Hands Detected: {hand_count}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"Raw Prediction: {raw_prediction}",
            (15, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Stable Prediction: {prediction_text}",
            (15, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Confidence: {confidence * 100:.2f}%",
            (15, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        for i, text in enumerate(top_predictions):

            cv2.putText(
                display_frame,
                f"Top {i + 1}: {text}",
                (15, 205 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1
            )

        # ==========================================
        # 16. SHOW WEBCAM
        # ==========================================

        cv2.imshow(
            "ISL MediaPipe Recognition",
            display_frame
        )

        # ==========================================
        # 17. EXIT
        # ==========================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    # ==========================================
    # 18. CLEANUP
    # ==========================================

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    print("\nISL Recognition Stopped.")
