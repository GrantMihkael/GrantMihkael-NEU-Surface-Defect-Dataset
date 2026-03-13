from data_pipeline import validate_data_structure


def main() -> None:
	if not validate_data_structure():
		print("Dataset structure not found. Run: python data/download_dataset.py")
		return

	print("Training scaffold ready. Implement dataloaders and training loop in v0.9.")


if __name__ == "__main__":
	main()
