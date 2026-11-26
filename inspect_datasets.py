
import os

ROOT = "data/raw"

IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
)

for dataset_name in ["dataset1", "dataset2"]:

    dataset_path = os.path.join(ROOT, dataset_name)

    print("\n" + "=" * 55)
    print("DATASET:", dataset_name)
    print("=" * 55)

    if not os.path.exists(dataset_path):
        print("Dataset folder not found:", dataset_path)
        continue

    total_images = 0

    for current_root, dirs, files in os.walk(dataset_path):

        image_files = [
            f for f in files
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

        if image_files:
            total_images += len(image_files)
            relative_path = os.path.relpath(
                current_root, dataset_path
            )

            print(
                f"{relative_path}: "
                f"{len(image_files)} images"
            )

            print("  Example:", image_files[0])

        other_files = [
            f for f in files
            if f.lower().endswith(
                (".csv", ".json", ".txt")
            )
        ]

        for filename in other_files:
            print("Metadata file:", filename)

    print("\nTotal images:", total_images)
