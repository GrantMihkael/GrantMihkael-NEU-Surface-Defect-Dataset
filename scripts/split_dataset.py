import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def list_class_folders(cleaned_dir: Path):
    """Return class folders from a class-based dataset directory."""
    return sorted([p for p in cleaned_dir.iterdir() if p.is_dir()])


def directory_contains_images(path: Path) -> bool:
    for child in path.iterdir():
        if child.is_file() and is_image_file(child):
            return True
    return False


def collect_class_to_images(cleaned_dir: Path):
    """
    Support both layouts:
    1) data/cleaned/<class>/<images>
    2) data/cleaned/<dataset>/<class>/<images>
    """
    class_to_images = defaultdict(list)
    top_folders = list_class_folders(cleaned_dir)

    for top in top_folders:
        subdirs = [p for p in top.iterdir() if p.is_dir()]
        nested_class_dirs = [p for p in subdirs if directory_contains_images(p)]

        if nested_class_dirs:
            for class_dir in nested_class_dirs:
                class_name = class_dir.name
                images = [p for p in class_dir.rglob("*") if p.is_file() and is_image_file(p)]
                class_to_images[class_name].extend(images)
            continue

        if directory_contains_images(top):
            class_name = top.name
            images = [p for p in top.rglob("*") if p.is_file() and is_image_file(p)]
            class_to_images[class_name].extend(images)

    return dict(class_to_images)


def split_items(items, train_ratio: float, val_ratio: float, seed: int):
    """Create deterministic train/val/test splits using a fixed random seed."""
    rng = random.Random(seed)
    data = list(items)
    rng.shuffle(data)

    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]
    return train, val, test


def copy_files(file_paths, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in file_paths:
        dst = dst_dir / src.name
        suffix = 1
        while dst.exists():
            dst = dst_dir / f"{src.stem}_{suffix}{src.suffix}"
            suffix += 1
        shutil.copy2(src, dst)
        copied += 1
    return copied


def prepare_split_folders(out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)


def create_splits(cleaned_dir: Path, out_dir: Path, train_ratio: float, val_ratio: float, seed: int):
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    class_to_images = collect_class_to_images(cleaned_dir)
    if not class_to_images:
        raise FileNotFoundError(f"No class images found in {cleaned_dir}")

    prepare_split_folders(out_dir)

    summary_rows = []
    split_totals = defaultdict(int)

    for class_name in sorted(class_to_images.keys()):
        images = sorted(class_to_images[class_name])
        if not images:
            continue

        train_items, val_items, test_items = split_items(images, train_ratio, val_ratio, seed)

        n_train = copy_files(train_items, out_dir / "train" / class_name)
        n_val = copy_files(val_items, out_dir / "val" / class_name)
        n_test = copy_files(test_items, out_dir / "test" / class_name)

        split_totals["train"] += n_train
        split_totals["val"] += n_val
        split_totals["test"] += n_test

        summary_rows.append(
            {
                "class_name": class_name,
                "train": n_train,
                "val": n_val,
                "test": n_test,
                "total": n_train + n_val + n_test,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("class_name").reset_index(drop=True)
    totals_row = {
        "class_name": "TOTAL",
        "train": int(summary_df["train"].sum()) if not summary_df.empty else 0,
        "val": int(summary_df["val"].sum()) if not summary_df.empty else 0,
        "test": int(summary_df["test"].sum()) if not summary_df.empty else 0,
        "total": int(summary_df["total"].sum()) if not summary_df.empty else 0,
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([totals_row])], ignore_index=True)

    print("\nSplit summary table:\n")
    print(summary_df.to_string(index=False))

    return summary_df, dict(split_totals)


def parse_args():
    parser = argparse.ArgumentParser(description="Create reproducible train/val/test splits")
    parser.add_argument("--cleaned-dir", default="data/cleaned", help="Input cleaned class folders")
    parser.add_argument("--out-dir", default="data/splits", help="Output split folder")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--summary-csv",
        default="outputs/split_summary_table.csv",
        help="Where to save split summary table",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cleaned_dir = Path(args.cleaned_dir)
    out_dir = Path(args.out_dir)
    summary_csv = Path(args.summary_csv)

    summary_df, _ = create_splits(
        cleaned_dir=cleaned_dir,
        out_dir=out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()