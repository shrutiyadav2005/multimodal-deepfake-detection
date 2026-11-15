"""
Video preprocessing and intelligent frame sampling pipeline.
Extracts frames uniformly across time, handles corrupt or oversize videos,
and coordinates with image preprocessing.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from configs.config import config
from preprocessing.image_preprocessing import ImagePreprocessor
from utils.helpers import validate_file_path


class VideoPreprocessor:
    """
    Intelligently samples frames and extracts metadata from video files.
    """

    def __init__(
        self,
        frame_sample_count: int = 16,
        max_size_mb: float = 100.0,
        image_preprocessor: Optional[ImagePreprocessor] = None
    ):
        self.frame_sample_count = frame_sample_count
        self.max_size_mb = max_size_mb
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()

    def inspect_video(self, video_path: Union[str, Path]) -> Dict[str, Union[int, float, str]]:
        """
        Validates video file size and extracts basic metadata (fps, total frames, duration, resolution).
        """
        path = validate_file_path(video_path, allowed_extensions=config.video.supported_extensions)
        size_mb = path.stat().st_size / (1024 * 1024)

        if size_mb > self.max_size_mb:
            raise ValueError(
                f"Video file size ({size_mb:.2f} MB) exceeds maximum allowed limit ({self.max_size_mb} MB)."
            )

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {path}. File may be corrupted or codec unsupported.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        cap.release()

        if total_frames <= 0:
            raise ValueError(f"Video file contains no readable frames: {path}")

        return {
            "path": str(path),
            "size_mb": round(size_mb, 2),
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_sec": round(duration, 2),
        }

    def sample_frames(
        self,
        video_path: Union[str, Path],
        crop_face: bool = True
    ) -> Tuple[torch.Tensor, List[np.ndarray], List[float], Dict[str, Union[int, float, str]]]:
        """
        Extracts uniformly distributed frames across the video duration.

        Returns:
            Tuple:
                - frames_tensor: Tensor of shape (N, 3, H, W) normalized for classifier.
                - raw_frames: List of N RGB numpy arrays (H, W, 3) for display.
                - timestamps: List of timestamps (seconds) for each sampled frame.
                - metadata: Dict with video specifications.
        """
        metadata = self.inspect_video(video_path)
        total_frames = int(metadata["total_frames"])
        fps = float(metadata["fps"])
        path = metadata["path"]

        # Calculate uniform indices across the timeline
        sample_count = min(self.frame_sample_count, total_frames)
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        cap = cv2.VideoCapture(str(path))
        tensors = []
        raw_frames = []
        timestamps = []

        current_idx = 0
        target_idx_set = set(indices)

        while cap.isOpened() and len(raw_frames) < sample_count:
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx in target_idx_set:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor, display_rgb = self.image_preprocessor.preprocess(
                    rgb_frame,
                    crop_face=crop_face,
                    is_training=False
                )
                tensors.append(tensor)
                raw_frames.append(display_rgb)
                timestamp = current_idx / fps if fps > 0 else 0.0
                timestamps.append(round(timestamp, 2))

            current_idx += 1

        cap.release()

        if len(tensors) == 0:
            raise ValueError(f"Could not extract any valid frames from {video_path}")

        frames_tensor = torch.cat(tensors, dim=0)  # Shape: (N, 3, H, W)
        return frames_tensor, raw_frames, timestamps, metadata
