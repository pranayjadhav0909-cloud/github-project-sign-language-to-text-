
import argparse
import os
import time
import cv2
import mediapipe as mp
import numpy as np

from src.features import (
    extract_hand_features,
    draw_hand_landmarks
)

MODEL_PATH = "models/hand_landmarker.task"

parser = argparse.ArgumentParser()

parser.add_argument("--label", required=True)
parser.add_argument("--samples", type=int, default=60)
parser.add_argument("--frames", type=int, default=30)

args = parser.parse_args()

label = args.label.strip().upper().replace(" ", "_")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Hand model missing: models/hand_landmarker.task"
    )

# Folder for numerical landmark data
save_dir = os.path.join("dataset", label)
os.makedirs(save_dir, exist_ok=True)

# Folder for actual webcam images
image_dir = os.path.join("dataset_images", label)
os.makedirs(image_dir, exist_ok=True)

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    detector.close()
    raise RuntimeError("Could not open webcam.")

sample_no = len([
    x for x in os.listdir(save_dir)
    if x.endswith(".npy")
])

timestamp_ms = 0

print("ISL DATASET COLLECTION")
print("Label:", label)
print("Numerical data folder:", save_dir)
print("Image folder:", image_dir)
print("Press Q to stop.")

try:

    while sample_no < args.samples:

        seq = []
        preview_frame = None

        # Preparation countdown
        for remaining in (3, 2, 1):

            ok, frame = cap.read()

            if not ok:
                continue

            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                f"Prepare {label}: {remaining}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("ISL Dataset Collection", frame)

            if cv2.waitKey(1000) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        # Record 30 frames
        while len(seq) < args.frames:

            ok, frame = cap.read()

            if not ok:
                continue

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms += 33

            result = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )

            landmarks = result.hand_landmarks

            # Draw hand landmarks
            draw_hand_landmarks(frame, landmarks)

            # Extract numerical features
            features = extract_hand_features(landmarks)

            seq.append(features)

            # Save one representative image
            # from the middle of the recording
            if len(seq) == args.frames // 2:
                preview_frame = frame.copy()

            cv2.putText(
                frame,
                f"Label: {label} | Sample: "
                f"{sample_no + 1}/{args.samples}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Frames: {len(seq)}/{args.frames} | Q = Quit",
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("ISL Dataset Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        # Save numerical landmark sequence
        data_path = os.path.join(
            save_dir,
            f"sample_{sample_no:05d}.npy"
        )

        np.save(
            data_path,
            np.asarray(seq, dtype=np.float32)
        )

        # Save representative webcam image
        if preview_frame is not None:

            image_path = os.path.join(
                image_dir,
                f"sample_{sample_no:05d}.jpg"
            )

            cv2.imwrite(image_path, preview_frame)

            print("Saved image:", image_path)

        else:
            print("Warning: No preview image captured.")

        sample_no += 1

        print("Saved numerical data:", data_path)

        time.sleep(0.2)

finally:

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

print("Collection finished.")
print("Total samples:", sample_no)
print("Numerical data:", save_dir)
print("Webcam images:", image_dir)
