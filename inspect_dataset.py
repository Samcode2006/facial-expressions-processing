
from pathlib import Path
from collections import Counter
import random

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

random.seed(42)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

train_dir = DATA_DIR / "train"
test_dir = DATA_DIR / "test"

if not train_dir.is_dir() or not test_dir.is_dir():
    raise FileNotFoundError(
        "Expected data/raw/train and data/raw/test folders."
    )

classes = sorted(
    folder.name for folder in train_dir.iterdir()
    if folder.is_dir()
)

print("Expression classes:", classes)

# PART 1: Display example images
fig, axes = plt.subplots(
    len(classes), 5,
    figsize=(12, 2 * len(classes)),
    squeeze=False
)

for row, emotion in enumerate(classes):
    folder = train_dir / emotion

    files = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }
    ]

    chosen = random.sample(files, min(5, len(files)))

    for col, ax in enumerate(axes[row]):
        ax.axis("off")

        if col < len(chosen):
            try:
                with Image.open(chosen[col]) as image:
                    ax.imshow(image.convert("RGB"))
                if col == 0:
                    ax.set_ylabel(emotion)
            except (OSError, UnidentifiedImageError):
                ax.set_title("Unreadable")

plt.suptitle("Facial Expression Dataset - Training Samples")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "sample_images.png",
    dpi=120,
    bbox_inches="tight"
)
plt.show()
plt.close(fig)

# PART 2: Fast image readability and dimension audit
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ("train", "test"):
    split_dir = DATA_DIR / split

    image_paths = [
        p
        for emotion_dir in split_dir.iterdir()
        if emotion_dir.is_dir()
        for p in emotion_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    readable = 0
    unreadable = []
    dimensions = Counter()

    print(f"\n--- {split.upper()} ---")
    print(f"Images found: {len(image_paths)}")

    for index, path in enumerate(image_paths, start=1):
        try:
            with Image.open(path) as image:
                image.load()
                dimensions[image.size] += 1

            readable += 1

        except (OSError, ValueError, UnidentifiedImageError):
            unreadable.append(str(path))

        if index % 2000 == 0 or index == len(image_paths):
            print(f"Checked {index}/{len(image_paths)}")

    print(f"Readable images: {readable}")
    print(f"Unreadable images: {len(unreadable)}")
    print("Most common dimensions:", dimensions.most_common(10))

    if unreadable:
        output = REPORT_DIR / f"{split}_unreadable.txt"
        output.write_text(
            "\n".join(unreadable),
            encoding="utf-8"
        )

print("\nInspection complete!")
print(f"Reports saved to: {REPORT_DIR}")
