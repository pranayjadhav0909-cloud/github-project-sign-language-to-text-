
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. CONFIGURATION
# ==========================================

HAND_MODEL_PATH = "models/hand_landmarker.task"

CAMERA_INDEX = 0
MAX_HANDS = 2

DETECTION_CONFIDENCE = 0.35
PRESENCE_CONFIDENCE = 0.35

# Extra space around detected hands
PADDING_RATIO = 0.25


# ==========================================
# 2. LOAD MEDIAPIPE
# ==========================================

print("Loading MediaPipe...")

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=DETECTION_CONFIDENCE,
    min_hand_presence_confidence=PRESENCE_CONFIDENCE,
    min_tracking_confidence=0.35
)

detector = vision.HandLandmarker.create_from_options(
    options
)

print("MediaPipe loaded successfully!")


# ==========================================
# 3. START CAMERA
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX)

# Request a higher camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    detector.close()
    raise RuntimeError("Cannot open webcam.")

print("Hand detection diagnostic started.")
print("Press Q to exit.")

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot read webcam.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # ======================================
        # 4. MEDIAPIPE DETECTION
        # ======================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = detector.detect(mp_image)

        hand_count = len(results.hand_landmarks)

        display = frame.copy()

        all_x = []
        all_y = []

        # ======================================
        # 5. INDIVIDUAL HAND BOXES
        # ======================================

        for hand_number, landmarks in enumerate(
            results.hand_landmarks, start=1
        ):

            hand_x = [
                int(point.x * width)
                for point in landmarks
            ]

            hand_y = [
                int(point.y * height)
                for point in landmarks
            ]

            x1 = max(0, min(hand_x))
            y1 = max(0, min(hand_y))

            x2 = min(width - 1, max(hand_x))
            y2 = min(height - 1, max(hand_y))

            # Individual hand padding
            pad_x = int((x2 - x1) * PADDING_RATIO) + 15
            pad_y = int((y2 - y1) * PADDING_RATIO) + 15

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)

            x2 = min(width - 1, x2 + pad_x)
            y2 = min(height - 1, y2 + pad_y)

            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

            # Draw individual hand box
            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Hand {hand_number}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Draw landmarks
            for point in landmarks:

                px = int(point.x * width)
                py = int(point.y * height)

                cv2.circle(
                    display,
                    (px, py),
                    3,
                    (0, 0, 255),
                    -1
                )

        # ======================================
        # 6. COMBINED TWO-HAND CROP
        # ======================================

        if hand_count > 0:

            x1 = max(0, min(all_x))
            y1 = max(0, min(all_y))

            x2 = min(width, max(all_x))
            y2 = min(height, max(all_y))

            # Draw combined bounding box
            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )

            combined_crop = frame[y1:y2, x1:x2]

            if combined_crop.size > 0:

                crop_h, crop_w = combined_crop.shape[:2]

                # ==================================
                # 7. SQUARE PREVIEW WITHOUT STRETCHING
                # ==================================

                side = max(crop_w, crop_h)

                square_crop = np.zeros(
                    (side, side, 3),
                    dtype=np.uint8
                )

                offset_x = (side - crop_w) // 2
                offset_y = (side - crop_h) // 2

                square_crop[
                    offset_y:offset_y + crop_h,
                    offset_x:offset_x + crop_w
                ] = combined_crop

                cv2.imshow(
                    "Combined Gesture Crop",
                    combined_crop
                )

                cv2.imshow(
                    "Square Gesture Preview",
                    square_crop
                )

        # ======================================
        # 8. DISPLAY STATUS
        # ======================================

        cv2.putText(
            display,
            f"Hands Detected: {hand_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Press Q to exit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ISL Hand Detection Diagnostic",
            display
        )

        # ======================================
        # 9. EXIT
        # ======================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    print("Diagnostic stopped.")
