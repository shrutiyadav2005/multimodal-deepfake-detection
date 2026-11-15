"""
Image Deepfake Inference Pipeline.
End-to-end execution: validation -> preprocessing -> detection -> Grad-CAM generation -> explanation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import time
import numpy as np
import torch
from PIL import Image

from configs.config import config
from models.image_model import ImageDeepfakeDetector
from preprocessing.image_preprocessing import ImagePreprocessor
from explainability.gradcam import GradCAM
from explainability.visualizations import generate_explainability_summary
from utils.device import get_device, get_device_name
from utils.helpers import calculate_processing_time, validate_file_path


class ImageInferencePipeline:
    """
    Production-grade inference pipeline for single image forensics.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or get_device()
        self.preprocessor = ImagePreprocessor()
        self.model, self.checkpoint_loaded = ImageDeepfakeDetector.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self.device
        )
        self.model.to(self.device)
        self.model.eval()

        # Initialize Grad-CAM
        target_layer = self.model.get_target_layer_for_cam()
        self.gradcam = GradCAM(self.model, target_layer=target_layer)

    def analyze(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        crop_face: bool = True
    ) -> Dict[str, any]:
        """
        Executes end-to-end image verification.

        Returns:
            Dict containing verdict, confidence, Grad-CAM overlays, timing, and explanations.
        """
        start_time = time.time()

        # Preprocess
        tensor, display_rgb = self.preprocessor.preprocess(
            image_input,
            crop_face=crop_face,
            is_training=False
        )
        tensor = tensor.to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            real_prob = float(probs[0])
            fake_prob = float(probs[1])

        verdict = "FAKE" if fake_prob >= 0.50 else "REAL"
        confidence = fake_prob if verdict == "FAKE" else real_prob

        # Grad-CAM heatmap generation for class 1 (FAKE artifacts)
        cam_heatmap = self.gradcam.generate_heatmap(tensor, target_class=1)
        cam_overlay = self.gradcam.overlay_heatmap_on_image(display_rgb, cam_heatmap)

        latency = calculate_processing_time(start_time)

        explanation = generate_explainability_summary(
            modality="image",
            verdict=verdict,
            fake_prob=fake_prob,
            extra_details={"face_detected": crop_face}
        )

        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "original_image": display_rgb,
            "gradcam_heatmap": cam_heatmap,
            "gradcam_overlay": cam_overlay,
            "processing_time_sec": latency,
            "hardware_device": get_device_name(self.device),
            "checkpoint_loaded": self.checkpoint_loaded,
            "explanation": explanation
        }
