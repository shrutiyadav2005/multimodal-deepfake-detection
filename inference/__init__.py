"""
Inference pipelines for image, video, and audio deepfake detection.
"""

from inference.image_inference import ImageInferencePipeline
from inference.video_inference import VideoInferencePipeline
from inference.audio_inference import AudioInferencePipeline

__all__ = [
    "ImageInferencePipeline",
    "VideoInferencePipeline",
    "AudioInferencePipeline",
]
