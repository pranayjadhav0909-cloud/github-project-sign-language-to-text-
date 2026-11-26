
from pathlib import Path
from PIL import Image, ImageDraw
import random
import math

DATASET_PATH = Path("data/raw/dataset2")
OUTPUT_PATH = Path("outputs/problem_classes_preview.jpg")

problem_classes = [
    "2", "3", "8", "9",
    "D", "H", "J", "K", "X", "Y"
]

SAMPLES_PER_CLASS = 5
IMAGE_SIZE = (150, 150)

random.seed(42)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

columns = 5
rows = len(problem_classes) * 1

cell_width = 180
cell_height = 205

sheet = Image.new(
    "RGB",
    (columns * cell_width, rows * cell_height),
    "white"
)

draw = ImageDraw.Draw(sheet)

for index, label in enumerate(problem_classes):

    folder = DATASET_PATH / label

    files = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        files.extend(folder.glob(ext))

    if not files:
        print("No images found:", label)
        continue

    selected = random.sample(
        files,
        min(SAMPLES_PER_CLASS, len(files))
    )

    row = index

    draw.text(
        (10, row * cell_height + 5),
        f"Actual class: {label}",
        fill="black"
    )

    for col, image_path in enumerate(selected):

        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail(IMAGE_SIZE)

            x = col * cell_width + 10
            y = row * cell_height + 35

            sheet.paste(image, (x, y))

            draw.text(
                (x, y + 153),
                image_path.name,
                fill="black"
            )

        except Exception as error:
            print("Error:", image_path, error)

sheet.save(OUTPUT_PATH)

print("Preview saved:", OUTPUT_PATH)
