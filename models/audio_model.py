"""
CNN-based Audio Deepfake Detection Model.
Processes log-Mel spectrogram representations to detect synthetic speech,
voice cloning, and acoustic artifacts.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from configs.config import config
from utils.device import get_device


class AudioDeepfakeDetector(nn.Module):
    """
    2D CNN architecture for Mel-spectrogram spoofing and synthetic speech detection.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Hierarchical Feature Extraction Backbone
        self.features = nn.Sequential(
            # Stage 1: Initial frequency/temporal texture extraction
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: Acoustic spectral pattern modeling
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: Higher-order synthetic artifact detection
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 4: High-level representation
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spectrogram tensor of shape (B, 1, n_mels, time_steps).
        Returns:
            torch.Tensor: Logits of shape (B, 2).
        """
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Softmax probability estimation.
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ) -> Tuple["AudioDeepfakeDetector", bool]:
        """
        Loads trained checkpoint if available; falls back to initialized model.
        """
        device = device or get_device()
        model = cls()
        model.to(device)
        model.eval()

        if checkpoint_path is None:
            checkpoint_path = config.checkpoints_dir / config.audio.checkpoint_name

        path = Path(checkpoint_path)
        if path.exists() and path.is_file():
            try:
                state_dict = torch.load(path, map_location=device)
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                model.load_state_dict(state_dict)
                return model, True
            except Exception as e:
                print(f"[Warning] Failed to load audio checkpoint {path}: {e}")
                return model, False

        return model, False
