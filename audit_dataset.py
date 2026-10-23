
from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

# Project paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

# Common facial expression labels
EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def sha256_file(path):
    """Hash file contents to identify exact duplicate files."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_image_files():
    """Inspect images stored in folders."""
    files = [
        path for path in DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    records = []
    hashes = Counter()

    print(f"\nImage files found: {len(files)}")

    for index, path in enumerate(files, start=1):
        record = {
            "path": str(path.relative_to(DATA_DIR)),
            "extension": path.suffix.lower(),
            "label_candidate": path.parent.name,
            "width": None,
            "height": None,
            "mode": None,
            "readable": False,
            "sha256": None,
            "error": None,
        }

        try:
            # Verify the file, then reopen it to inspect its dimensions.
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                record["width"], record["height"] = image.size
                record["mode"] = image.mode

            record["readable"] = True
            record["sha256"] = sha256_file(path)
            hashes[record["sha256"]] += 1

        except (OSError, ValueError, UnidentifiedImageError) as error:
            record["error"] = str(error)

        records.append(record)

        if index % 500 == 0:
            print(f"Checked {index}/{len(files)} images")

    frame = pd.DataFrame(records)
    frame.to_csv(REPORT_DIR / "image_inventory.csv", index=False)

    if not frame.empty:
        readable = frame[frame["readable"]]

        print(f"Readable images: {len(readable)}")
        print(f"Unreadable images: {(~frame['readable']).sum()}")

        print("\nFile formats:")
        print(frame["extension"].value_counts().to_string())

        print("\nImage modes:")
        print(readable["mode"].value_counts().to_string())

        print("\nMost common image dimensions:")
        dimensions = (
            readable.groupby(["width", "height"])
            .size()
            .sort_values(ascending=False)
            .head(15)
        )
        print(dimensions.to_string())

        duplicate_groups = sum(
            count > 1 for count in hashes.values()
        )
        duplicate_files_beyond_first = sum(
            count - 1 for count in hashes.values() if count > 1
        )

        print(f"\nExact duplicate hash groups: {duplicate_groups}")
        print(
            "Files beyond first copy in those groups: "
            f"{duplicate_files_beyond_first}"
        )

        # Treat parent folder names as candidate labels only.
        # Do not assume they are correct without inspecting the dataset.
        label_counts = frame["label_candidate"].value_counts()

        label_counts.to_csv(
            REPORT_DIR / "folder_label_counts.csv",
            header=["count"]
        )

        if len(label_counts) > 1:
            label_counts.plot(kind="bar", figsize=(10, 5))
            plt.title("Image count by parent folder")
            plt.xlabel("Parent folder")
            plt.ylabel("Number of files")
            plt.tight_layout()
            plt.savefig(
                REPORT_DIR / "folder_label_distribution.png",
                dpi=150
            )
            plt.close()

    return len(files)


def audit_csv_files():
    """Inspect CSV files without assuming their schemas."""
    csv_files = list(DATA_DIR.rglob("*.csv"))
    print(f"\nCSV files found: {len(csv_files)}")

    summaries = []

    for path in csv_files:
        print(f"\n--- {path.relative_to(DATA_DIR)} ---")

        try:
            # Read only a small sample first to inspect the structure.
            sample = pd.read_csv(path, nrows=5)
            print("Columns:", list(sample.columns))
            print(sample.head(3).to_string(index=False))

            # FER2013 pixel strings are large, so avoid printing them all.
            full = pd.read_csv(path, low_memory=False)

            print(f"Rows: {len(full)}")
            print(f"Columns: {len(full.columns)}")

            record = {
                "file": str(path.relative_to(DATA_DIR)),
                "rows": len(full),
                "columns": len(full.columns),
                "column_names": ", ".join(map(str, full.columns)),
            }

            # Find likely label columns, but never assume an
            # unfamiliar dataset uses FER2013's numeric mapping.
            label_columns = [
                column for column in full.columns
                if str(column).strip().lower()
                in {"emotion", "label", "expression", "class", "usage"}
            ]

            for column in label_columns:
                print(f"\nValue counts for '{column}':")
                print(
                    full[column]
                    .value_counts(dropna=False)
                    .head(30)
                    .to_string()
                )

            # FER2013 commonly stores one row per image, with
            # 2304 pixel values in a space-separated string.
            lower_columns = {
                str(column).strip().lower(): column
                for column in full.columns
            }

            if {"emotion", "pixels"}.issubset(lower_columns):
                emotion_col = lower_columns["emotion"]
                pixels_col = lower_columns["pixels"]

                print("\nPossible FER2013-format CSV detected.")

                labels = pd.to_numeric(
                    full[emotion_col], errors="coerce"
                )

                print("Numeric emotion-label counts:")
                print(labels.value_counts(dropna=False).sort_index())

                pixel_lengths = full[pixels_col].dropna().map(
                    lambda value: len(str(value).split())
                )

                print("Pixel value counts per row:")
                print(pixel_lengths.value_counts().head(10).to_string())

                print(
                    "Rows with 2304 pixel values:",
                    int((pixel_lengths == 2304).sum())
                )

            summaries.append(record)

        except (OSError, ValueError, pd.errors.ParserError) as error:
            print(f"Could not parse CSV: {error}")

    pd.DataFrame(summaries).to_csv(
        REPORT_DIR / "csv_inventory.csv", index=False
    )

    return len(csv_files)


def main():
    if not DATA_DIR.exists():
        print(f"Dataset directory does not exist: {DATA_DIR}")
        print("Create data/raw and extract your dataset there.")
        return

    all_files = [
        path for path in DATA_DIR.rglob("*") if path.is_file()
    ]

    if not all_files:
        print("No files found. Download and extract your dataset first.")
        return

    print("=" * 60)
    print("FACIAL EXPRESSION DATASET AUDIT")
    print("=" * 60)
    print(f"Dataset directory: {DATA_DIR}")
    print(f"Total files: {len(all_files)}")

    image_count = audit_image_files()
    csv_count = audit_csv_files()

    summary = {
        "dataset_directory": str(DATA_DIR),
        "total_files": len(all_files),
        "image_files_found": image_count,
        "csv_files_found": csv_count,
        "audit_completed": True,
    }

    with (REPORT_DIR / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print(f"Reports saved in: {REPORT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
