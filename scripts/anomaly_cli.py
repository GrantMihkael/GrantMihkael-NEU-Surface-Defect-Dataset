import argparse
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import matplotlib
matplotlib.use('TkAgg')  # Force Windows-compatible backend
import matplotlib.pyplot as plt


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:  # pragma: no cover
    tk = None
    filedialog = None


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def load_multiclass_metrics(path: Path) -> tuple[list[str], np.ndarray, str | None]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    class_names = payload.get("class_names")
    cm = payload.get("confusion_matrix")
    model_name = payload.get("model")

    if not isinstance(class_names, list) or not class_names:
        raise ValueError("metrics JSON must include a non-empty class_names list")
    if not isinstance(cm, list) or not cm:
        raise ValueError("metrics JSON must include a non-empty confusion_matrix")

    matrix = np.array(cm, dtype=np.int64)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion_matrix must be square")
    if matrix.shape[0] != len(class_names):
        raise ValueError("confusion_matrix size must match class_names length")

    return class_names, matrix, model_name


def binary_confusion_from_multiclass(class_names: list[str], cm: np.ndarray, normal_class: str) -> dict[str, int]:
    if normal_class not in class_names:
        raise ValueError(f"normal class '{normal_class}' not found in class_names")

    normal_idx = class_names.index(normal_class)

    tn = int(cm[normal_idx, normal_idx])
    fp = int(cm[normal_idx, :].sum() - tn)
    fn = int(cm[:, normal_idx].sum() - tn)
    tp = int(cm.sum() - tn - fp - fn)

    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def compute_binary_metrics(binary_cm: dict[str, int]) -> dict[str, float | int]:
    tn = binary_cm["tn"]
    fp = binary_cm["fp"]
    fn = binary_cm["fn"]
    tp = binary_cm["tp"]

    total = tn + fp + fn + tp
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)

    return {
        "total_samples": int(total),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy_binary": _safe_div(tp + tn, total),
        "anomaly_precision": precision,
        "anomaly_recall": recall,
        "anomaly_f1": _safe_div(2 * precision * recall, precision + recall),
        "specificity_normal": _safe_div(tn, tn + fp),
        "false_alarm_rate": _safe_div(fp, fp + tn),
        "miss_rate": _safe_div(fn, fn + tp),
    }


def build_model_from_state_dict(state_dict: dict, num_classes: int):
    keys = set(state_dict.keys())
    if any(k.startswith("features.") for k in keys):
        model = SimpleCNN(num_classes=num_classes)
        model_name = "simplecnn"
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model_name = "resnet18"

    model.load_state_dict(state_dict)
    return model, model_name


def build_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def generate_gradcam(
    image_path: Path,
    model,
    device,
    transform,
    img_size: int,
    class_names: list[str],
    target_class_idx: int,
):
    """Generate Grad-CAM visualization for an image."""
    try:
        # Load and prepare image
        image = Image.open(image_path).convert("L")
        image_resized = image.resize((img_size, img_size))
        original_np = np.array(image_resized)
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Register hooks
        feature_maps = []
        gradients = []
        
        def forward_hook(module, input, output):
            feature_maps.clear()
            feature_maps.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            gradients.clear()
            gradients.append(grad_output[0].detach())
        
        # Get target layer and register hooks
        is_simplecnn = any(k.startswith("features.") for k in model.state_dict().keys())
        target_layer = model.features[6] if is_simplecnn else model.layer4[-1].conv2
        
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        # Forward pass
        model.zero_grad()
        output = model(input_tensor)
        
        # Backward pass for target class
        model.zero_grad()
        target_score = output[0, target_class_idx]
        target_score.backward()
        
        # Compute Grad-CAM
        if not feature_maps or not gradients:
            print("Warning: No feature maps or gradients captured")
            return original_np, None, None, None
            
        feature_map = feature_maps[0].cpu().numpy()[0]
        gradient = gradients[0].cpu().numpy()[0]
        
        weights = np.mean(gradient, axis=(1, 2))
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * feature_map[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (img_size, img_size))
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        original_bgr = cv2.cvtColor(original_np, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)
        
        forward_handle.remove()
        backward_handle.remove()
        
        return original_np, cam, overlay, heatmap
    
    except Exception as e:
        print(f"Error in Grad-CAM: {e}")
        return None, None, None, None


def save_and_show_gradcam(
    image_path: Path,
    original_np,
    cam,
    overlay,
    heatmap,
    class_name: str,
    confidence: float,
    output_dir: Path = None,
    show_plot: bool = True,
):
    """Save and display Grad-CAM visualization."""
    if cam is None or overlay is None:
        print("⚠ Warning: Could not generate Grad-CAM heatmap.")
        return False
    
    try:
        # Prepare output directory
        if output_dir is None:
            output_dir = Path("experiments/gradcam")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        print("🖼 Creating visualization...")
        fig = plt.figure(figsize=(15, 5))
        
        # Original image
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(original_np, cmap="gray")
        ax1.set_title("Original Image", fontsize=12, fontweight="bold")
        ax1.axis("off")
        
        # Heatmap
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(cam, cmap="jet")
        ax2.set_title("Grad-CAM Heatmap\n(Red = Model Focus)", fontsize=12, fontweight="bold")
        ax2.axis("off")
        
        # Overlay with prediction
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        title = f"Prediction: {class_name}\nConfidence: {confidence:.2%}"
        ax3.set_title(title, fontsize=12, fontweight="bold", color="darkgreen")
        ax3.axis("off")
        
        plt.tight_layout()
        
        # Save figure
        image_stem = image_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figure_path = output_dir / f"{image_stem}_{timestamp}_gradcam_figure.png"
        overlay_path = output_dir / f"{image_stem}_{timestamp}_gradcam_overlay.png"
        
        plt.savefig(figure_path, dpi=150, bbox_inches="tight")
        cv2.imwrite(str(overlay_path), overlay)
        
        print(f"✓ Saved: {figure_path}")
        print(f"✓ Saved: {overlay_path}")
        
        if show_plot:
            print("📊 Displaying visualization window...")
            plt.show()
        
        plt.close()
        return True
        
    except Exception as e:
        print(f"⚠ Error saving visualization: {e}")
        plt.close()
        return False


def predict_one(
    image_path: Path,
    model,
    device,
    transform,
    class_names: list[str],
    normal_class: str,
    top_k: int,
):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {image_path.suffix}")

    image = Image.open(image_path).convert("L")
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
                "class_name": class_names[int(idx)],
                "confidence": float(score),
            }
        )

    pred_class = topk[0]["class_name"]
    status = "NORMAL" if pred_class == normal_class else "ANOMALY"

    return {
        "image_path": str(image_path),
        "status": status,
        "predicted_class": pred_class,
        "predicted_confidence": float(topk[0]["confidence"]),
        "top_k": topk,
    }


def print_anomaly_metrics(metrics: dict[str, float | int], normal_class: str):
    print("=" * 70)
    print("ANOMALY DETECTION SUMMARY")
    print(f"Normal class: {normal_class}")
    print("=" * 70)
    print(f"Samples          : {metrics['total_samples']}")
    print(f"Accuracy (binary): {metrics['accuracy_binary']:.4f}")
    print(f"Anomaly precision: {metrics['anomaly_precision']:.4f}")
    print(f"Anomaly recall   : {metrics['anomaly_recall']:.4f}")
    print(f"Anomaly F1       : {metrics['anomaly_f1']:.4f}")
    print(f"False alarm rate : {metrics['false_alarm_rate']:.4f}")
    print(f"Miss rate        : {metrics['miss_rate']:.4f}")
    print(f"TP/TN/FP/FN      : {metrics['tp']}/{metrics['tn']}/{metrics['fp']}/{metrics['fn']}")
    print("=" * 70)


def print_prediction(result: dict):
    print("-" * 70)
    print(f"Image      : {result['image_path']}")
    print(f"Status     : {result['status']}")
    print(f"Class      : {result['predicted_class']}")
    print(f"Confidence : {result['predicted_confidence']:.4f}")
    print("Top-K:")
    for row in result["top_k"]:
        print(f"  - {row['class_name']}: {row['confidence']:.4f}")
    print("-" * 70)


def pick_image_file() -> Path | None:
    if tk is None or filedialog is None:
        print("Warning: file picker is unavailable (tkinter not installed in this environment).")
        return None

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Select image for anomaly prediction",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
    except Exception as err:
        print(f"Warning: could not open file picker: {err}")
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass

    if not selected:
        return None
    return Path(selected)


def prompt_image_path() -> Path | None:
    raw = input("Paste/drag image path (or press Enter to cancel): ").strip()
    if not raw:
        return None
    return Path(raw.strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive CLI for anomaly metrics display and image prediction"
    )
    parser.add_argument(
        "--model-path",
        default="models/submission/cnn_baseline_best.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--metrics-json",
        default="metrics/cnn_baseline_metrics.json",
        help="Multiclass metrics JSON with class_names and confusion_matrix",
    )
    parser.add_argument(
        "--normal-class",
        default="clean_sample",
        help="Class considered normal; all other classes are anomalies",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--top-k", type=int, default=3, help="How many classes to print")
    parser.add_argument(
        "--image",
        help="Optional image path for one-shot prediction; omit for interactive loop",
    )
    parser.add_argument(
        "--pick-image",
        action="store_true",
        help="Open file picker for one-shot prediction (no manual path typing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metrics_path = Path(args.metrics_json)
    model_path = Path(args.model_path)

    class_names, multiclass_cm, _ = load_multiclass_metrics(metrics_path)
    binary_cm = binary_confusion_from_multiclass(class_names, multiclass_cm, args.normal_class)
    anomaly_metrics = compute_binary_metrics(binary_cm)
    print_anomaly_metrics(anomaly_metrics, args.normal_class)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    model, model_name = build_model_from_state_dict(state_dict, len(class_names))
    model = model.to(device)
    model.eval()
    transform = build_transform(args.img_size)

    print(f"Loaded model: {model_name} from {model_path}")
    print("● Grad-CAM visualization enabled - heatmaps will be saved for each prediction\n")

    if args.image and args.pick_image:
        raise ValueError("Use either --image or --pick-image, not both")

    if args.image or args.pick_image:
        image_path = Path(args.image) if args.image else pick_image_file()
        if image_path is None:
            image_path = prompt_image_path()
            if image_path is None:
                print("No file selected. Exiting.")
                return

        result = predict_one(
            image_path=image_path,
            model=model,
            device=device,
            transform=transform,
            class_names=class_names,
            normal_class=args.normal_class,
            top_k=args.top_k,
        )
        print_prediction(result)
        
        # Generate and display Grad-CAM
        print("\n🔍 Generating Grad-CAM visualization...")
        try:
            pred_class_idx = class_names.index(result["predicted_class"])
            original_np, cam, overlay, heatmap = generate_gradcam(
                image_path=image_path,
                model=model,
                device=device,
                transform=transform,
                img_size=args.img_size,
                class_names=class_names,
                target_class_idx=pred_class_idx,
            )
            save_and_show_gradcam(
                image_path=image_path,
                original_np=original_np,
                cam=cam,
                overlay=overlay,
                heatmap=heatmap,
                class_name=result["predicted_class"],
                confidence=result["predicted_confidence"],
                output_dir=Path("experiments/gradcam"),
                show_plot=True,
            )
        except Exception as err:
            print(f"⚠ Warning: Could not generate Grad-CAM: {err}")
            import traceback
            traceback.print_exc()
        
        return

    print("Type an image path and press Enter.")
    print("Press Enter on empty input to open file picker.")
    print("Type q to quit.")
    print()

    while True:
        try:
            raw = input("image> ").strip()
        except EOFError:
            print("\nExiting anomaly CLI.")
            break

        if raw.lower() in {"q", "quit", "exit"}:
            print("Exiting anomaly CLI.")
            break
        if not raw:
            picked = pick_image_file()
            if picked is None:
                picked = prompt_image_path()
                if picked is None:
                    print("No file selected.")
                    continue
            candidate = picked
        else:
            candidate = Path(raw.strip('"').strip("'"))

        try:
            result = predict_one(
                image_path=candidate,
                model=model,
                device=device,
                transform=transform,
                class_names=class_names,
                normal_class=args.normal_class,
                top_k=args.top_k,
            )
            print_prediction(result)
            
            # Generate and display Grad-CAM
            print("\n🔍 Generating Grad-CAM visualization...")
            try:
                pred_class_idx = class_names.index(result["predicted_class"])
                original_np, cam, overlay, heatmap = generate_gradcam(
                    image_path=candidate,
                    model=model,
                    device=device,
                    transform=transform,
                    img_size=args.img_size,
                    class_names=class_names,
                    target_class_idx=pred_class_idx,
                )
                save_and_show_gradcam(
                    image_path=candidate,
                    original_np=original_np,
                    cam=cam,
                    overlay=overlay,
                    heatmap=heatmap,
                    class_name=result["predicted_class"],
                    confidence=result["predicted_confidence"],
                    output_dir=Path("experiments/gradcam"),
                    show_plot=True,
                )
            except Exception as err:
                print(f"⚠ Warning: Could not generate Grad-CAM: {err}")
                import traceback
                traceback.print_exc()
                
        except Exception as err:
            print(f"Error: {err}")


if __name__ == "__main__":
    main()
