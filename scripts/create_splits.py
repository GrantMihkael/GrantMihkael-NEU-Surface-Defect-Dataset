import argparse
import csv
import json
import random
import shutil
from pathlib import Path


# Requirement tag: train/val/test split creation with no leakage.
# Images are copied into disjoint split folders, preventing path overlap across splits.
def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def split_list(items, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    data = list(items)
    rng.shuffle(data)

    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    return train, val, test


def create_split_folders(base_dir: Path):
    for split in ["train", "val", "test"]:
        (base_dir / split).mkdir(parents=True, exist_ok=True)


def clear_existing_split_data(base_dir: Path):
    if not base_dir.exists():
        return
    for split_dir in [base_dir / "train", base_dir / "val", base_dir / "test"]:
        if split_dir.exists():
            shutil.rmtree(split_dir)


def copy_to_split(files, split_name, class_name, out_dir: Path):
    target_class_dir = out_dir / split_name / class_name
    target_class_dir.mkdir(parents=True, exist_ok=True)

    copied_paths = []
    for src in files:
        dst = target_class_dir / src.name
        suffix = 1
        while dst.exists():
            dst = target_class_dir / f"{src.stem}_{suffix}{src.suffix}"
            suffix += 1
        shutil.copy2(src, dst)
        copied_paths.append(dst)
    return copied_paths


def directory_contains_images(path: Path):
    for child in path.iterdir():
        if child.is_file() and is_image_file(child):
            return True
    return False


def collect_class_images(cleaned_dir: Path):
    """
    Supports both cleaned layouts:
    1) data/cleaned/<class>/<images>
    2) data/cleaned/<dataset>/<class>/<images>

    Returns:
    - class_to_images: dict[str, list[Path]]
    - class_sources: dict[str, list[str]]
    """
    class_to_images = {}
    class_sources = {}

    top_dirs = [p for p in cleaned_dir.iterdir() if p.is_dir()]
    if not top_dirs:
        raise FileNotFoundError(f"No folders found in cleaned dataset: {cleaned_dir}")

    for top_dir in sorted(top_dirs):
        subdirs = [p for p in top_dir.iterdir() if p.is_dir()]

        nested_class_dirs = [p for p in subdirs if directory_contains_images(p)]
        if nested_class_dirs:
            dataset_name = top_dir.name
            for class_dir in sorted(nested_class_dirs):
                class_name = class_dir.name
                images = [p for p in class_dir.rglob("*") if p.is_file() and is_image_file(p)]
                if not images:
                    continue
                class_to_images.setdefault(class_name, []).extend(images)
                class_sources.setdefault(class_name, set()).add(dataset_name)
            continue

        if directory_contains_images(top_dir):
            class_name = top_dir.name
            images = [p for p in top_dir.rglob("*") if p.is_file() and is_image_file(p)]
            if images:
                class_to_images.setdefault(class_name, []).extend(images)
                class_sources.setdefault(class_name, set()).add("direct")

    if not class_to_images:
        raise FileNotFoundError(
            "No class images found. Expected cleaned layout: "
            "data/cleaned/<class>/<images> or data/cleaned/<dataset>/<class>/<images>."
        )

    class_sources = {k: sorted(list(v)) for k, v in class_sources.items()}
    return class_to_images, class_sources


def create_splits(cleaned_dir: Path, splits_dir: Path, train_ratio: float, val_ratio: float, seed: int):
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("Invalid ratios. Ensure train_ratio + val_ratio < 1.0")

    clear_existing_split_data(splits_dir)
    create_split_folders(splits_dir)

    manifest_rows = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    class_split_counts = {}
    class_source_datasets = {}

    class_to_images, class_sources = collect_class_images(cleaned_dir)

    for class_name in sorted(class_to_images.keys()):
        images = class_to_images[class_name]
        class_source_datasets[class_name] = class_sources.get(class_name, [])

        train, val, test = split_list(images, train_ratio, val_ratio, seed)
        class_split_counts[class_name] = {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }

        for split_name, files in [("train", train), ("val", val), ("test", test)]:
            copied = copy_to_split(files, split_name, class_name, splits_dir)
            split_counts[split_name] += len(copied)
            for path in copied:
                manifest_rows.append(
                    {
                        "filepath": str(path.as_posix()),
                        "label": class_name,
                        "split": split_name,
                    }
                )

    manifest_path = splits_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "total_per_split": split_counts,
        "per_class_split_counts": class_split_counts,
        "class_source_datasets": class_source_datasets,
        "manifest": str(manifest_path),
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Create reproducible train/val/test splits")
    parser.add_argument("--cleaned-dir", default="data/cleaned")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-path", default="metrics/split_summary.json")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = create_splits(
        cleaned_dir=Path(args.cleaned_dir),
        splits_dir=Path(args.splits_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Split creation complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
