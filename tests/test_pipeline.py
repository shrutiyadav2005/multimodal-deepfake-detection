"""
Automated Unit and Integration Test Suite.
Validates Preprocessing, Models, Explainability (Grad-CAM), Multimodal Consistency,
Evaluation Metrics, and Error Handling.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe cache paths
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")

import pytest
import numpy as np
import torch
import cv2

from configs.config import config
from utils.device import get_device, get_device_name, set_seed
from utils.helpers import validate_file_path, format_file_size, calculate_processing_time
from preprocessing.image_preprocessing import ImagePreprocessor
from preprocessing.video_preprocessing import VideoPreprocessor
from preprocessing.audio_preprocessing import AudioPreprocessor
from models.image_model import ImageDeepfakeDetector
from models.video_model import VideoDeepfakeDetector
from models.audio_model import AudioDeepfakeDetector
from models.multimodal_model import MultimodalVerifier
from explainability.gradcam import GradCAM
from evaluation.metrics import compute_classification_metrics
from inference.image_inference import ImageInferencePipeline
from inference.video_inference import VideoInferencePipeline
from inference.audio_inference import AudioInferencePipeline


# ---------------------------------------------------------------------
# 1. Hardware Device & Helper Utilities Tests
# ---------------------------------------------------------------------
def test_device_selection():
    device = get_device()
    assert isinstance(device, torch.device)
    name = get_device_name(device)
    assert isinstance(name, str) and len(name) > 0


def test_seed_reproducibility():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_file_validation():
    # Valid file
    sample_img = config.sample_data_dir / "images" / "sample_real.jpg"
    assert sample_img.exists()
    validated = validate_file_path(sample_img, allowed_extensions=config.image.supported_extensions)
    assert validated == sample_img

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        validate_file_path("non_existent_file.jpg")

    # Invalid extension
    with pytest.raises(ValueError):
        validate_file_path(sample_img, allowed_extensions=(".mp4", ".mov"))


def test_helpers_formatting():
    assert format_file_size(500) == "500 B"
    assert format_file_size(2048) == "2.00 KB"
    assert format_file_size(5 * 1024 * 1024) == "5.00 MB"


# ---------------------------------------------------------------------
# 2. Preprocessing Tests
# ---------------------------------------------------------------------
def test_image_preprocessing():
    preprocessor = ImagePreprocessor(image_size=(224, 224))
    sample_img = config.sample_data_dir / "images" / "sample_real.jpg"

    tensor, display_rgb = preprocessor.preprocess(sample_img, crop_face=False)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert isinstance(display_rgb, np.ndarray)
    assert display_rgb.shape == (224, 224, 3)

    # Test face detection & crop
    crop_tensor, crop_display = preprocessor.preprocess(sample_img, crop_face=True)
    assert crop_tensor.shape == (1, 3, 224, 224)


def test_video_preprocessing():
    sample_vid = config.sample_data_dir / "videos" / "sample_real.mp4"
    preprocessor = VideoPreprocessor(frame_sample_count=8)

    meta = preprocessor.inspect_video(sample_vid)
    assert meta["total_frames"] > 0
    assert meta["duration_sec"] > 0

    frames_tensor, raw_frames, timestamps, meta_out = preprocessor.sample_frames(sample_vid, crop_face=False)
    assert frames_tensor.shape[0] <= 8
    assert frames_tensor.shape[1:] == (3, 224, 224)
    assert len(raw_frames) == frames_tensor.shape[0]
    assert len(timestamps) == len(raw_frames)


def test_audio_preprocessing():
    sample_aud = config.sample_data_dir / "audio" / "sample_real.wav"
    preprocessor = AudioPreprocessor(sample_rate=16000, duration=3.0, n_mels=128)

    tensor, mel_db, meta = preprocessor.preprocess(sample_aud)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 4
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 1
    assert tensor.shape[2] == 128  # n_mels
    assert mel_db.shape[0] == 128
    assert meta["sample_rate"] == 16000


# ---------------------------------------------------------------------
# 3. Model Architecture Tests
# ---------------------------------------------------------------------
def test_image_model_forward():
    model = ImageDeepfakeDetector(architecture="resnet18", pretrained_backbone=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)
    probs = model.predict_proba(x)
    assert probs.shape == (2, 2)
    assert torch.allclose(torch.sum(probs, dim=1), torch.ones(2), atol=1e-5)


def test_gradcam_generation():
    model = ImageDeepfakeDetector(architecture="resnet18", pretrained_backbone=False)
    target_layer = model.get_target_layer_for_cam()
    gradcam = GradCAM(model, target_layer)

    x = torch.randn(1, 3, 224, 224)
    cam = gradcam.generate_heatmap(x, target_class=1)
    assert cam.shape == (224, 224)
    assert np.min(cam) >= 0.0 and np.max(cam) <= 1.0

    dummy_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    overlay = gradcam.overlay_heatmap_on_image(dummy_rgb, cam)
    assert overlay.shape == (224, 224, 3)


def test_video_model_aggregation():
    img_model = ImageDeepfakeDetector(architecture="resnet18", pretrained_backbone=False)
    vid_model = VideoDeepfakeDetector(image_model=img_model, aggregation_method="top_k_mean")

    x = torch.randn(10, 3, 224, 224)
    res = vid_model(x)
    assert "label" in res
    assert "fake_probability" in res
    assert 0.0 <= res["fake_probability"] <= 1.0
    assert len(res["frame_probabilities"]) == 10


def test_audio_model_forward():
    model = AudioDeepfakeDetector()
    x = torch.randn(2, 1, 128, 120)
    out = model(x)
    assert out.shape == (2, 2)
    probs = model.predict_proba(x)
    assert probs.shape == (2, 2)
    assert torch.allclose(torch.sum(probs, dim=1), torch.ones(2), atol=1e-5)


def test_multimodal_consistency():
    verifier = MultimodalVerifier()
    # Consistent real
    res_real = verifier.evaluate_consistency(0.1, 0.15)
    assert res_real["overall_verdict"] == "REAL"
    assert res_real["consistency_score"] >= 0.85
    assert not res_real["is_inconsistent"]

    # Inconsistent (Face swap: Fake video, Real audio)
    res_swap = verifier.evaluate_consistency(0.85, 0.10)
    assert res_swap["overall_verdict"] == "SUSPICIOUS_VISUAL"
    assert res_swap["is_inconsistent"]

    # Inconsistent (Voice clone: Real video, Fake audio)
    res_clone = verifier.evaluate_consistency(0.12, 0.90)
    assert res_clone["overall_verdict"] == "SUSPICIOUS_AUDIO"
    assert res_clone["is_inconsistent"]


# ---------------------------------------------------------------------
# 4. Evaluation Metrics Tests
# ---------------------------------------------------------------------
def test_evaluation_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]

    metrics = compute_classification_metrics(y_true, y_pred, y_scores)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]

    # Empty inputs edge case
    empty_m = compute_classification_metrics([], [])
    assert empty_m["accuracy"] == 0.0


# ---------------------------------------------------------------------
# 5. End-to-End Inference Pipelines Tests
# ---------------------------------------------------------------------
def test_image_inference_pipeline():
    pipe = ImageInferencePipeline()
    sample_img = config.sample_data_dir / "images" / "sample_real.jpg"
    res = pipe.analyze(sample_img)
    assert res["verdict"] in ["REAL", "FAKE"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["gradcam_overlay"].shape == (224, 224, 3)
    assert res["processing_time_sec"] > 0


def test_video_inference_pipeline():
    pipe = VideoInferencePipeline(frame_sample_count=8)
    sample_vid = config.sample_data_dir / "videos" / "sample_real.mp4"
    res = pipe.analyze(sample_vid)
    assert res["verdict"] in ["REAL", "FAKE"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert "timeline_image" in res
    assert res["total_processing_time_sec"] > 0


def test_audio_inference_pipeline():
    pipe = AudioInferencePipeline()
    sample_aud = config.sample_data_dir / "audio" / "sample_real.wav"
    res = pipe.analyze(sample_aud)
    assert res["verdict"] in ["REAL", "FAKE"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert "spectrogram_image" in res
    assert res["processing_time_sec"] > 0
