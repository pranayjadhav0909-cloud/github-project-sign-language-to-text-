
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

# Configure MediaPipe
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
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

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Hand detection started")
print("Press Q to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    results = detector.detect(mp_image)

    h, w, _ = frame.shape

    hand_count = len(results.hand_landmarks)

    # Draw detected hands
    for hand_landmarks in results.hand_landmarks:

        x_coordinates = [
            int(point.x * w)
            for point in hand_landmarks
        ]

        y_coordinates = [
            int(point.y * h)
            for point in hand_landmarks
        ]

        x1 = max(0, min(x_coordinates) - 20)
        y1 = max(0, min(y_coordinates) - 20)

        x2 = min(w, max(x_coordinates) + 20)
        y2 = min(h, max(y_coordinates) + 20)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # Draw hand landmarks
        for point in hand_landmarks:

            px = int(point.x * w)
            py = int(point.y * h)

            cv2.circle(
                frame,
                (px, py),
                4,
                (0, 0, 255),
                -1
            )

    cv2.putText(
        frame,
        f"Hands Detected: {hand_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "ISL Hand Detection Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
