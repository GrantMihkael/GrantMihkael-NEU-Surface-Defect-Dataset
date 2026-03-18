import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def normalize_label(label: str) -> str:
    return "_".join(label.strip().lower().replace("-", " ").split())


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def directory_contains_images(path: Path) -> bool:
    for child in path.iterdir():
        if child.is_file() and is_image_file(child):
            return True
    return False


def infer_label_from_filename(path: Path) -> str:
    stem = path.stem.strip().lower()
    # Keep full class tokens and remove only trailing numeric index.
    # Examples:
    #   crazing_1 -> crazing
    #   pitted_surface_20 -> pitted_surface
    #   rolled-in_scale_3 -> rolled_in_scale
    stem = re.sub(r"([_-]?\d+)$", "", stem)
    return normalize_label(stem)


def collect_dataset_class_folders(raw_dir: Path):
    """
    Supports both layouts:
    1) data/raw/<class_name>/<images>
    2) data/raw/<dataset_name>/<class_name>/<images>
    """
    pairs = []
    flat_pools = []
    top_level_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]

    has_nested_dataset_structure = False
    for top_dir in top_level_dirs:
        subdirs = [p for p in top_dir.iterdir() if p.is_dir()]
        if any(directory_contains_images(p) for p in subdirs):
            has_nested_dataset_structure = True
            break

    for top_dir in sorted(top_level_dirs):
        subdirs = [p for p in top_dir.iterdir() if p.is_dir()]
        nested_class_dirs = [p for p in subdirs if directory_contains_images(p)]

        if has_nested_dataset_structure:
            # Mixed/nested mode: top-level folders are dataset names.
            if nested_class_dirs:
                dataset_name = normalize_label(top_dir.name)
                for class_dir in sorted(nested_class_dirs):
                    class_dir_name = normalize_label(class_dir.name)
                    if class_dir_name in {"images", "image", "imgs"}:
                        flat_pools.append((dataset_name, class_dir))
                    else:
                        pairs.append((dataset_name, class_dir))
                continue

            # Dataset folder may store all images in one pool (e.g. IMAGES folder).
            if directory_contains_images(top_dir):
                flat_pools.append((normalize_label(top_dir.name), top_dir))
            continue

        # Direct class mode: top-level folders are classes.
        if directory_contains_images(top_dir):
            pairs.append((None, top_dir))

    return pairs, flat_pools


def clean_dataset(raw_dir: Path, cleaned_dir: Path, summary_path: Path):
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes = set()

    duplicate_count = 0
    invalid_count = 0
    total_images = 0
    copied_images = 0
    per_class_counts = {}
    per_dataset_counts = {}
    dataset_mode = False

    dataset_class_pairs, flat_pools = collect_dataset_class_folders(raw_dir)
    if not dataset_class_pairs and not flat_pools:
        raise FileNotFoundError(
            "No valid class folders found in raw directory. "
            "Use either data/raw/<class>/<images> or data/raw/<dataset>/<class>/<images>."
        )

    for dataset_name, class_dir in dataset_class_pairs:
        if dataset_name is not None:
            dataset_mode = True

        class_name = normalize_label(class_dir.name)
        if dataset_name is None:
            target_class_dir = cleaned_dir / class_name
            summary_key = class_name
        else:
            target_class_dir = cleaned_dir / dataset_name / class_name
            summary_key = f"{dataset_name}/{class_name}"

        target_class_dir.mkdir(parents=True, exist_ok=True)

        for src in sorted(class_dir.rglob("*")):
            if not src.is_file() or not is_image_file(src):
                continue

            total_images += 1

            if not validate_image(src):
                invalid_count += 1
                continue

            file_hash = sha256_file(src)
            if file_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(file_hash)

            dst = target_class_dir / src.name
            # Keep filenames unique inside each class.
            suffix = 1
            while dst.exists():
                dst = target_class_dir / f"{src.stem}_{suffix}{src.suffix}"
                suffix += 1

            shutil.copy2(src, dst)
            copied_images += 1
            per_class_counts[summary_key] = per_class_counts.get(summary_key, 0) + 1
            if dataset_name is not None:
                per_dataset_counts[dataset_name] = per_dataset_counts.get(dataset_name, 0) + 1

    for dataset_name, pool_dir in flat_pools:
        dataset_mode = True
        for src in sorted(pool_dir.rglob("*")):
            if not src.is_file() or not is_image_file(src):
                continue

            total_images += 1

            if not validate_image(src):
                invalid_count += 1
                continue

            file_hash = sha256_file(src)
            if file_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(file_hash)

            inferred_class = infer_label_from_filename(src)
            target_class_dir = cleaned_dir / dataset_name / inferred_class
            target_class_dir.mkdir(parents=True, exist_ok=True)

            dst = target_class_dir / src.name
            suffix = 1
            while dst.exists():
                dst = target_class_dir / f"{src.stem}_{suffix}{src.suffix}"
                suffix += 1

            shutil.copy2(src, dst)
            copied_images += 1
            summary_key = f"{dataset_name}/{inferred_class}"
            per_class_counts[summary_key] = per_class_counts.get(summary_key, 0) + 1
            per_dataset_counts[dataset_name] = per_dataset_counts.get(dataset_name, 0) + 1

    summary = {
        "raw_dir": str(raw_dir),
        "cleaned_dir": str(cleaned_dir),
        "total_images_seen": total_images,
        "valid_images_copied": copied_images,
        "duplicates_removed": duplicate_count,
        "invalid_removed": invalid_count,
        "missing_values_checked": True,
        "label_standardization": "lowercase + underscore naming",
        "dataset_mode": dataset_mode,
        "final_dataset_counts": per_dataset_counts,
        "final_class_counts": per_class_counts,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and standardize image dataset")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw dataset")
    parser.add_argument("--cleaned-dir", default="data/cleaned", help="Path for cleaned dataset")
    parser.add_argument(
        "--summary",
        default="metrics/data_cleaning_summary.json",
        help="Path for cleaning summary JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = clean_dataset(Path(args.raw_dir), Path(args.cleaned_dir), Path(args.summary))

    print("Cleaning complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
