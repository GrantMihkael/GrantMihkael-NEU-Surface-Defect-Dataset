import torch
from torch import nn


class DefectClassifier(nn.Module):
	def __init__(self, num_classes: int = 6) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(32 * 50 * 50, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, num_classes),
		)

	def forward(self, images: torch.Tensor) -> torch.Tensor:
		features = self.features(images)
		return self.classifier(features)
