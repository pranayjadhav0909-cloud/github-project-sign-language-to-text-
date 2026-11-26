
import os
from pathlib import Path
from PIL import Image

ROOT = Path("data/raw")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

dataset_classes = {}

for dataset_name in ["dataset1", "dataset2"]:

    root = ROOT / dataset_name

    print("\n" + "=" * 55)
    print("CHECKING:", dataset_name)
    print("=" * 55)

    if not root.exists():
        print("Dataset folder missing:", root)
        continue

    class_counts = {}
    corrupt_images = 0
    total_images = 0
    image_sizes = {}

    for class_folder in sorted(root.iterdir()):

        if not class_folder.is_dir():
            continue

        label = class_folder.name

        files = [
            f for f in class_folder.iterdir()
            if f.suffix.lower() in EXTENSIONS
        ]

        valid_count = 0

        for file in files:
            try:
                with Image.open(file) as img:
                    img.verify()

                with Image.open(file) as img:
                    size = img.size

                valid_count += 1
                image_sizes[size] = image_sizes.get(size, 0) + 1

            except Exception:
                corrupt_images += 1
                print("Corrupt:", file)

        class_counts[label] = valid_count
        total_images += valid_count

    dataset_classes[dataset_name] = class_counts

    print("Classes:", sorted(class_counts))
    print("Total valid images:", total_images)
    print("Corrupt images:", corrupt_images)

    print("\nImages per class:")

    for label, count in sorted(class_counts.items()):
        print(label, ":", count)

    print("\nMost common image sizes:")

    for size, count in sorted(
        image_sizes.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]:
        print(size, ":", count)

# Compare the class names
if len(dataset_classes) == 2:

    first = set(dataset_classes["dataset1"])
    second = set(dataset_classes["dataset2"])

    print("\n" + "=" * 55)
    print("CLASS COMPARISON")
    print("=" * 55)

    print("Common classes:", sorted(first & second))
    print("Only in dataset1:", sorted(first - second))
    print("Only in dataset2:", sorted(second - first))

print("\nDataset validation completed.")
