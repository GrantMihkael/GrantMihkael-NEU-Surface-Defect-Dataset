from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        start_index = index
        while True:
            path, _ = self.samples[index]
            try:
                image, target = super().__getitem__(index)
                return image, target, path
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as err:
                print(f"Skipping unreadable image: {path} ({err})")
                index = (index + 1) % len(self.samples)
                if index == start_index:
                    raise RuntimeError("No readable images found in dataset.") from err


def build_transforms(img_size: int, use_augmentation: bool):
    base = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
    ]

    aug = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
    ] if use_augmentation else []

    tail = [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]

    return transforms.Compose(base + aug + tail)


def build_dataloaders(splits_dir: str, img_size: int, batch_size: int, num_workers: int, use_augmentation: bool):
    root = Path(splits_dir)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Expected split folders at data/splits/... with train, val, test")

    train_tf = build_transforms(img_size=img_size, use_augmentation=use_augmentation)
    eval_tf = build_transforms(img_size=img_size, use_augmentation=False)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_ds = ImageFolderWithPaths(test_dir, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.classes
