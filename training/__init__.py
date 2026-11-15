"""
Training modules and synthetic dataset generator.
"""

from training.dataset_generator import generate_all_samples, generate_synthetic_dataset
from training.train_image import train_image_model, DeepfakeImageDataset
from training.train_audio import train_audio_model, DeepfakeAudioDataset
from training.train_video import evaluate_video_pipeline

__all__ = [
    "generate_all_samples",
    "generate_synthetic_dataset",
    "train_image_model",
    "DeepfakeImageDataset",
    "train_audio_model",
    "DeepfakeAudioDataset",
    "evaluate_video_pipeline",
]
