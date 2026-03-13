from pathlib import Path


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


def get_data_root() -> Path:
    return Path("data/raw")


def validate_data_structure() -> bool:
    data_root = get_data_root()
    if not data_root.exists():
        return False
    return all((data_root / class_name).exists() for class_name in CLASS_NAMES)
