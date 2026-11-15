"""
Video Deepfake Detection Model.
Processes sampled video frames through a spatial feature backbone
and applies temporal aggregation to detect manipulation across the video sequence.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np

from configs.config import config
from models.image_model import ImageDeepfakeDetector
from utils.device import get_device


class VideoDeepfakeDetector(nn.Module):
    """
    Temporal aggregation model for video manipulation forensics.
    """

    def __init__(
        self,
        image_model: Optional[ImageDeepfakeDetector] = None,
        aggregation_method: str = "top_k_mean",
        suspicious_threshold: float = 0.50,
        top_k_ratio: float = 0.25
    ):
        super().__init__()
        self.image_model = image_model or ImageDeepfakeDetector()
        self.aggregation_method = aggregation_method
        self.suspicious_threshold = suspicious_threshold
        self.top_k_ratio = top_k_ratio

    def forward(self, frames_tensor: torch.Tensor) -> Dict[str, Union[float, str, List[float], List[int]]]:
        """
        Inference over extracted video frames.

        Args:
            frames_tensor: Tensor of shape (N, 3, H, W) where N is number of sampled frames.

        Returns:
            Dict containing video-level verdict, confidence, per-frame scores, and suspicious indices.
        """
        # Ensure evaluation mode
        self.image_model.eval()

        with torch.no_grad():
            logits = self.image_model(frames_tensor)
            probs = torch.softmax(logits, dim=1)  # (N, 2)
            frame_fake_probs = probs[:, 1].cpu().numpy().tolist()  # (N,)

        num_frames = len(frame_fake_probs)
        if num_frames == 0:
            return {
                "label": "REAL",
                "fake_probability": 0.0,
                "real_probability": 1.0,
                "frame_probabilities": [],
                "suspicious_frame_indices": [],
                "peak_suspicious_index": 0,
                "aggregation_method": self.aggregation_method
            }

        # Temporal Aggregation
        fake_probs_arr = np.array(frame_fake_probs)
        if self.aggregation_method == "max":
            agg_fake_prob = float(np.max(fake_probs_arr))
        elif self.aggregation_method == "mean":
            agg_fake_prob = float(np.mean(fake_probs_arr))
        elif self.aggregation_method == "top_k_mean":
            # Sort descending and average top k% most suspicious frames
            k = max(1, int(np.ceil(num_frames * self.top_k_ratio)))
            top_k_values = np.sort(fake_probs_arr)[::-1][:k]
            agg_fake_prob = float(np.mean(top_k_values))
        else:
            agg_fake_prob = float(np.mean(fake_probs_arr))

        # Clamp between [0.0, 1.0]
        agg_fake_prob = max(0.0, min(1.0, agg_fake_prob))
        real_prob = 1.0 - agg_fake_prob

        # Flag frames exceeding threshold
        suspicious_indices = [
            i for i, p in enumerate(frame_fake_probs) if p >= self.suspicious_threshold
        ]
        peak_idx = int(np.argmax(fake_probs_arr))
        label = "FAKE" if agg_fake_prob >= self.suspicious_threshold else "REAL"

        return {
            "label": label,
            "fake_probability": round(agg_fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "frame_probabilities": [round(p, 4) for p in frame_fake_probs],
            "suspicious_frame_indices": suspicious_indices,
            "peak_suspicious_index": peak_idx,
            "aggregation_method": self.aggregation_method
        }
