from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

build_transforms = importlib.import_module("defect_cv.data").build_transforms
build_model = importlib.import_module("defect_cv.models").build_model
utils_mod = importlib.import_module("defect_cv.utils")
load_config = utils_mod.load_config
ensure_dir = utils_mod.ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Predict defect class for image input")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--image", help="Path to one image file")
    parser.add_argument("--input-dir", help="Path to folder of images")
    parser.add_argument("--checkpoint", help="Optional checkpoint override")
    parser.add_argument("--class-names-file", help="Optional JSON file containing class_names")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-json", help="Optional output json file for predictions")
    return parser.parse_args()


def _load_class_names(config: dict, args) -> list[str]:
    if args.class_names_file:
        payload = json.loads(Path(args.class_names_file).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(x) for x in payload]
        if isinstance(payload, dict) and "class_names" in payload:
            return [str(x) for x in payload["class_names"]]
        raise ValueError("class-names-file must be a JSON list or an object with class_names")

    tables_dir = Path(config["results_dir"]) / "tables"
    train_summary_path = tables_dir / f"{config['run_name']}_train_summary.json"
    if train_summary_path.exists():
        data = json.loads(train_summary_path.read_text(encoding="utf-8"))
        if "class_names" in data:
            return [str(x) for x in data["class_names"]]

    metrics_path = tables_dir / f"{config['run_name']}_metrics.json"
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "class_names" in data:
            return [str(x) for x in data["class_names"]]

    raise FileNotFoundError(
        "Could not resolve class names. Run training/evaluation first, or pass --class-names-file."
    )


def _load_model(config: dict, checkpoint_path: Path, class_names: list[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model_name"], num_classes=len(class_names)).to(device)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def _predict_one(model, device, transform, image_path: Path, class_names: list[str], top_k: int):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(top_k, probs.shape[0])
    scores, indices = torch.topk(probs, k=k)

    topk = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        topk.append(
            {
                "class_index": int(idx),
                "class_name": class_names[int(idx)],
                "confidence": float(score),
            }
        )

    return {
        "image_path": str(image_path),
        "predicted_class": topk[0]["class_name"],
        "predicted_confidence": topk[0]["confidence"],
        "top_k": topk,
    }


def _iter_images(input_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    args = parse_args()
    if not args.image and not args.input_dir:
        raise ValueError("Provide --image or --input-dir")

    config = load_config(args.config)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(config["checkpoint_dir"]) / f"{config['run_name']}_best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    class_names = _load_class_names(config, args)
    transform = build_transforms(img_size=int(config["img_size"]), use_augmentation=False)
    model, device = _load_model(config, checkpoint, class_names)

    predictions = []
    if args.image:
        predictions.append(
            _predict_one(model, device, transform, Path(args.image), class_names, top_k=int(args.top_k))
        )
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        for image_path in _iter_images(input_dir):
            try:
                predictions.append(
                    _predict_one(model, device, transform, image_path, class_names, top_k=int(args.top_k))
                )
            except Exception as err:
                predictions.append(
                    {
                        "image_path": str(image_path),
                        "error": str(err),
                    }
                )

    print(f"Predictions generated: {len(predictions)}")
    for row in predictions[:5]:
        if "error" in row:
            print(f"{row['image_path']} -> ERROR: {row['error']}")
        else:
            print(
                f"{row['image_path']} -> {row['predicted_class']} "
                f"({row['predicted_confidence']:.4f})"
            )

    output_json = Path(args.output_json) if args.output_json else None
    if output_json is None and args.input_dir:
        output_json = Path(config["results_dir"]) / "analysis" / f"{config['run_name']}_image_predictions.json"

    if output_json:
        ensure_dir(output_json.parent)
        output_json.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
        print(f"Saved predictions: {output_json}")


if __name__ == "__main__":
    main()
