from pathlib import Path


def main() -> None:
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Download the NEU Surface Defect Database manually from the selected Kaggle source.")
    print("After download, extract the dataset into: data/raw/")
    print("Expected class folders:")
    print("- crazing")
    print("- inclusion")
    print("- patches")
    print("- pitted_surface")
    print("- rolled_in_scale")
    print("- scratches")


if __name__ == "__main__":
    main()
