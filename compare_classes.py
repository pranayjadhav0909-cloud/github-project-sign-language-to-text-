
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path("data/raw")

classes = ["A", "B", "C", "D", "E", "F"]

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

fig, axes = plt.subplots(
    len(classes), 2,
    figsize=(9, 20)
)

for row, label in enumerate(classes):

    for col, dataset in enumerate(["dataset1", "dataset2"]):

        folder = ROOT / dataset / label

        images = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in extensions
        ])

        ax = axes[row, col]

        if not images:
            ax.set_title(f"{dataset} - Missing")
            ax.axis("off")
            continue

        image = Image.open(images[0]).convert("RGB")

        ax.imshow(image)
        ax.set_title(f"{dataset} - Class {label}")
        ax.axis("off")

plt.suptitle(
    "ISL Dataset Comparison: Same Classes",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "outputs/class_comparison.jpg",
    dpi=120
)

plt.show()

print("Comparison saved to outputs/class_comparison.jpg")
