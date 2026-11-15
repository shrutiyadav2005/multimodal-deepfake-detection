"""
Video Deepfake Inference Pipeline.
End-to-end execution: sampling -> frame-level classification -> temporal aggregation
-> suspicious frame Grad-CAM -> timeline plot -> report.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import time
import numpy as np
import torch

from configs.config import config
from models.image_model import ImageDeepfakeDetector
from models.video_model import VideoDeepfakeDetector
from preprocessing.video_preprocessing import VideoPreprocessor
from explainability.gradcam import GradCAM
from explainability.visualizations import plot_video_timeline_to_image, generate_explainability_summary
from utils.device import get_device, get_device_name
from utils.helpers import calculate_processing_time, validate_file_path


class VideoInferencePipeline:
    """
    Production-grade inference pipeline for video deepfake detection.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        frame_sample_count: int = 16,
        aggregation_method: str = "top_k_mean",
        device: Optional[torch.device] = None
    ):
        self.device = device or get_device()
        self.image_model, self.checkpoint_loaded = ImageDeepfakeDetector.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self.device
        )
        self.image_model.to(self.device)
        self.image_model.eval()

        self.video_detector = VideoDeepfakeDetector(
            image_model=self.image_model,
            aggregation_method=aggregation_method,
            suspicious_threshold=config.video.suspicious_threshold,
            top_k_ratio=config.video.top_k_ratio
        )

        self.preprocessor = VideoPreprocessor(frame_sample_count=frame_sample_count)
        self.gradcam = GradCAM(self.image_model, target_layer=self.image_model.get_target_layer_for_cam())

    def analyze(
        self,
        video_path: Union[str, Path],
        crop_face: bool = True
    ) -> Dict[str, any]:
        """
        Executes end-to-end video forensic analysis.
        """
        start_time = time.time()

        # Frame extraction & preprocessing
        t0 = time.time()
        frames_tensor, raw_frames, timestamps, meta = self.preprocessor.sample_frames(
            video_path,
            crop_face=crop_face
        )
        sampling_time = round(time.time() - t0, 3)

        # Video-level temporal classification
        t1 = time.time()
        frames_tensor = frames_tensor.to(self.device)
        pred = self.video_detector(frames_tensor)
        inference_time = round(time.time() - t1, 3)

        total_latency = calculate_processing_time(start_time)
        fps = round(len(raw_frames) / total_latency, 2) if total_latency > 0 else 0.0

        # Grad-CAM on peak suspicious frame
        peak_idx = pred["peak_suspicious_index"]
        peak_frame_tensor = frames_tensor[peak_idx : peak_idx + 1]
        peak_raw_frame = raw_frames[peak_idx]

        peak_cam_heatmap = self.gradcam.generate_heatmap(peak_frame_tensor, target_class=1)
        peak_cam_overlay = self.gradcam.overlay_heatmap_on_image(peak_raw_frame, peak_cam_heatmap)

        # Suspicious frame details
        suspicious_frames_data = []
        for idx in pred["suspicious_frame_indices"]:
            suspicious_frames_data.append({
                "frame_index": idx,
                "timestamp_sec": timestamps[idx] if idx < len(timestamps) else 0.0,
                "fake_probability": pred["frame_probabilities"][idx],
                "image": raw_frames[idx]
            })

        # Timeline plot
        timeline_img = plot_video_timeline_to_image(
            frame_probabilities=pred["frame_probabilities"],
            timestamps=timestamps,
            threshold=config.video.suspicious_threshold
        )

        verdict = pred["label"]
        confidence = pred["fake_probability"] if verdict == "FAKE" else pred["real_probability"]

        explanation = generate_explainability_summary(
            modality="video",
            verdict=verdict,
            fake_prob=pred["fake_probability"],
            extra_details={
                "suspicious_frames_count": len(pred["suspicious_frame_indices"]),
                "total_frames_sampled": len(raw_frames),
                "aggregation_method": pred["aggregation_method"]
            }
        )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "fake_probability": pred["fake_probability"],
            "real_probability": pred["real_probability"],
            "metadata": meta,
            "frame_probabilities": pred["frame_probabilities"],
            "timestamps": timestamps,
            "suspicious_frames": suspicious_frames_data,
            "peak_frame_index": peak_idx,
            "peak_frame_timestamp": timestamps[peak_idx] if peak_idx < len(timestamps) else 0.0,
            "peak_raw_frame": peak_raw_frame,
            "peak_cam_overlay": peak_cam_overlay,
            "timeline_image": timeline_img,
            "sampling_time_sec": sampling_time,
            "inference_time_sec": inference_time,
            "total_processing_time_sec": total_latency,
            "throughput_fps": fps,
            "hardware_device": get_device_name(self.device),
            "checkpoint_loaded": self.checkpoint_loaded,
            "explanation": explanation
        }
