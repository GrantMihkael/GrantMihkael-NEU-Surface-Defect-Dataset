import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models

# =========================
# CONFIG
# =========================
MODEL_PATH = "best_defect_model.pth"
IMAGE_PATH = "sample.jpg"
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]


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


def load_class_names(class_names_file: str):
    p = Path(class_names_file)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            names = data.get("class_names", [])
            if isinstance(names, list) and names:
                return names
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return CLASS_NAMES


def build_model_from_state_dict(state_dict, class_names):
    keys = set(state_dict.keys())

    if any(k.startswith("features.") for k in keys):
        model = SimpleCNN(num_classes=len(class_names))
        target_layer = model.features[6]
        arch = "simplecnn"
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        target_layer = model.layer4[-1].conv2
        arch = "resnet18"

    model.load_state_dict(state_dict)
    return model, target_layer, arch


def parse_args():
    parser = argparse.ArgumentParser(description="Run Grad-CAM prediction and save outputs")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to model checkpoint")
    parser.add_argument("--image-path", default=IMAGE_PATH, help="Path to input image")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="Input resize dimension")
    parser.add_argument(
        "--output-dir",
        default="experiments/gradcam",
        help="Directory where Grad-CAM images will be saved",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib window (useful in headless/report runs)",
    )
    parser.add_argument(
        "--class-names-file",
        default="metrics/cnn_baseline_metrics.json",
        help="JSON file containing class_names array",
    )
    return parser.parse_args()


args = parse_args()
MODEL_PATH = args.model_path
IMAGE_PATH = args.image_path
IMG_SIZE = args.img_size
# Automatically create timestamped subdirectory for each run
base_output_dir = Path(args.output_dir)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = base_output_dir / timestamp
output_dir.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = load_class_names(args.class_names_file)

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
if not Path(IMAGE_PATH).exists():
    raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# =========================
# LOAD MODEL
# =========================
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model, target_layer, model_arch = build_model_from_state_dict(state_dict, CLASS_NAMES)
model = model.to(DEVICE)
model.eval()

# =========================
# HOOKS
# =========================
feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# =========================
# LOAD IMAGE
# =========================
image = Image.open(IMAGE_PATH).convert("L")
image_resized = image.resize((IMG_SIZE, IMG_SIZE))
original = np.array(image_resized)

input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# =========================
# PREDICT
# =========================
output = model(input_tensor)
pred_class = output.argmax(dim=1).item()
pred_label = CLASS_NAMES[pred_class]

# =========================
# BACKPROP FOR GRAD-CAM
# =========================
model.zero_grad()
output[0, pred_class].backward()

feature_map = feature_maps[0].detach().cpu().numpy()[0]
gradient = gradients[0].detach().cpu().numpy()[0]

weights = np.mean(gradient, axis=(1, 2))
cam = np.zeros(feature_map.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * feature_map[i]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

if cam.max() != 0:
    cam = cam / cam.max()

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
overlay = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)

# =========================
# SHOW RESULTS
# =========================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cam, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"Prediction: {pred_label}")
plt.axis("off")

plt.tight_layout()

image_stem = Path(IMAGE_PATH).stem
figure_path = output_dir / f"{image_stem}_gradcam_figure.png"
overlay_path = output_dir / f"{image_stem}_gradcam_overlay.png"
heatmap_path = output_dir / f"{image_stem}_gradcam_heatmap.png"

plt.savefig(figure_path, dpi=150)

if not args.no_show:
    plt.show()
else:
    plt.close()

cv2.imwrite(str(overlay_path), overlay)
cv2.imwrite(str(heatmap_path), np.uint8(255 * cam))

print(f"Predicted Class: {pred_label}")
print(f"Model architecture: {model_arch}")
print(f"Saved figure: {figure_path}")
print(f"Saved overlay: {overlay_path}")
print(f"Saved heatmap: {heatmap_path}")