"""
Global configuration for Multimodal Deepfake Detection & Explainable Content Verification.
Defines parameters for preprocessing, model architectures, training, inference, and explainability.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

# Set safe writable cache paths in sandbox environment
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure all primary runtime output directories exist
for directory in [CHECKPOINTS_DIR, OUTPUTS_DIR, PLOTS_DIR, REPORTS_DIR, SAMPLE_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ImageConfig:
    """Configuration for Image Deepfake Detection."""
    supported_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
    image_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    architecture: str = "resnet18"  # "resnet18", "resnet50", or "efficientnet_b0"
    num_classes: int = 2  # 0: REAL, 1: FAKE
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    checkpoint_name: str = "image_detector.pth"


@dataclass
class VideoConfig:
    """Configuration for Video Deepfake Detection."""
    supported_extensions: Tuple[str, ...] = (".mp4", ".mov", ".avi")
    frame_sample_count: int = 16  # Uniformly sample N frames across duration
    max_video_size_mb: float = 100.0  # Safe handling limit
    suspicious_threshold: float = 0.50  # Prob > threshold is flagged suspicious
    aggregation_method: str = "top_k_mean"  # "mean", "max", "top_k_mean"
    top_k_ratio: float = 0.25  # Use top 25% highest fake scores for conservative aggregation
    frame_size: Tuple[int, int] = (224, 224)
    checkpoint_name: str = "video_detector.pth"


@dataclass
class AudioConfig:
    """Configuration for Audio Deepfake Detection."""
    supported_extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac")
    sample_rate: int = 16000
    duration: float = 4.0  # Fixed 4 seconds duration
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    f_min: int = 20
    f_max: int = 8000
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 5
    checkpoint_name: str = "audio_detector.pth"


@dataclass
class MultimodalConfig:
    """Configuration for Multimodal Fusion & Audio-Visual Consistency."""
    visual_weight: float = 0.55
    audio_weight: float = 0.45
    inconsistency_threshold: float = 0.35  # Absolute delta |P_video - P_audio| to trigger alert
    checkpoint_name: str = "multimodal_fusion.pth"


@dataclass
class BenchmarkGoals:
    """
    Projected target benchmarks for future optimization (per portfolio requirements).
    Actual measured metrics will always be recorded separately without hardcoding.
    """
    target_accuracy: float = 0.92
    target_f1: float = 0.90
    target_precision: float = 0.91
    target_recall: float = 0.89


@dataclass
class Config:
    """Root configuration object."""
    project_name: str = "Multimodal Deepfake Detection & Explainable Content Verification"
    version: str = "1.0.0"
    base_dir: Path = BASE_DIR
    checkpoints_dir: Path = CHECKPOINTS_DIR
    outputs_dir: Path = OUTPUTS_DIR
    plots_dir: Path = PLOTS_DIR
    reports_dir: Path = REPORTS_DIR
    sample_data_dir: Path = SAMPLE_DATA_DIR
    seed: int = 42

    image: ImageConfig = field(default_factory=ImageConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    benchmarks: BenchmarkGoals = field(default_factory=BenchmarkGoals)


config = Config()
