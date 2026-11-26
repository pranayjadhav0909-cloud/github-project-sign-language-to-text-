
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import math

ROOT = Path("data/raw")
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for dataset_name in ["dataset1", "dataset2"]:

    dataset_path = ROOT / dataset_name

    if not dataset_path.exists():
        print("Folder not found:", dataset_path)
        continue

    class_folders = sorted(
        [p for p in dataset_path.iterdir() if p.is_dir()],
        key=lambda p: p.name
    )

    print(f"\n{dataset_name}")
    print("Classes:", len(class_folders))

    images_to_show = []

    for folder in class_folders:

        images = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in EXTENSIONS
        ])

        print(f"{folder.name}: {len(images)} images")

        if images:
            images_to_show.append((folder.name, images[0]))

    if not images_to_show:
        print("No images found.")
        continue

    columns = 6
    rows = math.ceil(len(images_to_show) / columns)

    fig, axes = plt.subplots(
        rows, columns,
        figsize=(15, rows * 2.5)
    )

    axes = axes.flatten()

    for ax, (label, image_path) in zip(axes, images_to_show):

        try:
            image = Image.open(image_path).convert("RGB")
            ax.imshow(image)
            ax.set_title(f"Class: {label}")
            ax.axis("off")

        except Exception as error:
            print("Could not open:", image_path, error)
            ax.set_title("Invalid image")
            ax.axis("off")

    for ax in axes[len(images_to_show):]:
        ax.axis("off")

    plt.suptitle(
        f"{dataset_name} - Dataset Preview",
        fontsize=16
    )

    plt.tight_layout()

    output_path = OUTPUT / f"{dataset_name}_preview.jpg"

    plt.savefig(output_path, dpi=120)
    plt.close()

    print("Preview saved:", output_path)

print("\nDataset preview completed.")
