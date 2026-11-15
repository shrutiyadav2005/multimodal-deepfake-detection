"""
Deepfake detection model architectures.
"""

from models.image_model import ImageDeepfakeDetector
from models.video_model import VideoDeepfakeDetector
from models.audio_model import AudioDeepfakeDetector
from models.multimodal_model import MultimodalVerifier

__all__ = [
    "ImageDeepfakeDetector",
    "VideoDeepfakeDetector",
    "AudioDeepfakeDetector",
    "MultimodalVerifier",
]
