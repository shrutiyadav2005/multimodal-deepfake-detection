"""
CNN-based Image Deepfake Detection Model.
Utilizes transfer learning (ResNet18 / EfficientNet) with custom binary classification head.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models

from configs.config import config
from utils.device import get_device


class ImageDeepfakeDetector(nn.Module):
    """
    ResNet-based deepfake image detector for distinguishing REAL (0) vs FAKE (1).
    """

    def __init__(
        self,
        architecture: str = "resnet18",
        pretrained_backbone: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.architecture = architecture
        self.num_classes = config.image.num_classes

        # Set safe cache directory inside /tmp to prevent permission errors
        os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")

        # Load backbone with graceful offline fallback
        if architecture == "resnet18":
            try:
                weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
                self.backbone = models.resnet18(weights=weights)
            except Exception as e:
                # Fallback to randomly initialized architecture if offline / sandbox blocks download
                self.backbone = models.resnet18(weights=None)
            in_features = self.backbone.fc.in_features
            # Replace final classification head
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate * 0.7),
                nn.Linear(256, self.num_classes)
            )
            self.target_layer = self.backbone.layer4[-1]
        elif architecture == "resnet50":
            try:
                weights = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
                self.backbone = models.resnet50(weights=weights)
            except Exception as e:
                self.backbone = models.resnet50(weights=None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate * 0.7),
                nn.Linear(512, self.num_classes)
            )
            self.target_layer = self.backbone.layer4[-1]
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 3, H, W).
        Returns:
            torch.Tensor: Unnormalized logits of shape (B, 2).
        """
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes softmax probabilities.
        Returns:
            torch.Tensor: Probabilities of shape (B, 2) where [:, 0] is Real, [:, 1] is Fake.
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def get_target_layer_for_cam(self) -> nn.Module:
        """
        Returns the last convolutional layer used for Grad-CAM activation mapping.
        """
        return self.target_layer

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ) -> Tuple["ImageDeepfakeDetector", bool]:
        """
        Loads model weights from checkpoint if available, or initializes cleanly with backbone.

        Returns:
            Tuple[ImageDeepfakeDetector, bool]: (model, is_trained_checkpoint_loaded)
        """
        device = device or get_device()
        model = cls(architecture=config.image.architecture, pretrained_backbone=True)
        model.to(device)
        model.eval()

        if checkpoint_path is None:
            checkpoint_path = config.checkpoints_dir / config.image.checkpoint_name

        path = Path(checkpoint_path)
        if path.exists() and path.is_file():
            try:
                state_dict = torch.load(path, map_location=device)
                # Handle full checkpoint dict vs bare state_dict
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                model.load_state_dict(state_dict)
                return model, True
            except Exception as e:
                print(f"[Warning] Failed to load checkpoint {path}: {e}. Running with initialized backbone.")
                return model, False

        return model, False
