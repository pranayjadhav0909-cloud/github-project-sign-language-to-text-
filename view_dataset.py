
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Select the dataset label and sample
LABEL = "A"
SAMPLE_NUMBER = 1
FRAME_NUMBER = 0

file_path = (
    f"dataset/{LABEL}/sample_{SAMPLE_NUMBER:05d}.npy"
)

if not os.path.exists(file_path):
    print("Dataset file not found:", file_path)
    raise SystemExit

# Load the saved dataset
data = np.load(file_path)

print("Dataset shape:", data.shape)

# Select one frame and reshape into 2 hands,
# 21 landmarks, and 3 coordinates (x, y, z).
frame = data[FRAME_NUMBER].reshape(2, 21, 3)

# Connections between hand landmarks
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

plt.figure(figsize=(7, 7))

hand_found = False

for hand_index, hand in enumerate(frame):

    # Ignore an empty hand
    if np.all(hand == 0):
        continue

    hand_found = True
    x = hand[:, 0]
    y = hand[:, 1]

    plt.scatter(x, y, s=35)

    for start, end in connections:
        plt.plot(
            [x[start], x[end]],
            [y[start], y[end]],
            linewidth=2
        )

    # Show landmark numbers
    for i in range(21):
        plt.text(x[i], y[i], str(i), fontsize=8)

if not hand_found:
    print("No hand landmarks detected in this frame.")

plt.title(
    f"ISL Dataset - {LABEL} "
    f"| Sample {SAMPLE_NUMBER} "
    f"| Frame {FRAME_NUMBER}"
)

plt.xlabel("X coordinates")
plt.ylabel("Y coordinates")
plt.gca().invert_yaxis()
plt.axis("equal")
plt.grid(True)
plt.show()
