"""
Audio Deepfake Inference Pipeline.
End-to-end execution: audio loading -> Mel-spectrogram extraction -> CNN classification
-> spectrogram visualization -> explanation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import time
import numpy as np
import torch

from configs.config import config
from models.audio_model import AudioDeepfakeDetector
from preprocessing.audio_preprocessing import AudioPreprocessor
from explainability.visualizations import plot_spectrogram_to_image, generate_explainability_summary
from utils.device import get_device, get_device_name
from utils.helpers import calculate_processing_time, validate_file_path


class AudioInferencePipeline:
    """
    Production-grade inference pipeline for audio spoofing and synthetic voice forensics.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or get_device()
        self.preprocessor = AudioPreprocessor(
            sample_rate=config.audio.sample_rate,
            duration=config.audio.duration,
            n_mels=config.audio.n_mels
        )
        self.model, self.checkpoint_loaded = AudioDeepfakeDetector.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self.device
        )
        self.model.to(self.device)
        self.model.eval()

    def analyze(self, audio_path: Union[str, Path]) -> Dict[str, any]:
        """
        Executes end-to-end audio deepfake analysis.
        """
        start_time = time.time()

        # Feature extraction
        tensor, mel_db, meta = self.preprocessor.preprocess(audio_path)
        tensor = tensor.to(self.device)

        # Classification
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            real_prob = float(probs[0])
            fake_prob = float(probs[1])

        verdict = "FAKE" if fake_prob >= 0.50 else "REAL"
        confidence = fake_prob if verdict == "FAKE" else real_prob

        # Visualization
        spectrogram_img = plot_spectrogram_to_image(
            mel_db=mel_db,
            sr=config.audio.sample_rate,
            hop_length=config.audio.hop_length,
            title=f"Mel-Spectrogram - Verdict: {verdict} ({confidence * 100:.1f}%)"
        )

        latency = calculate_processing_time(start_time)

        explanation = generate_explainability_summary(
            modality="audio",
            verdict=verdict,
            fake_prob=fake_prob,
            extra_details=meta
        )

        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "metadata": meta,
            "spectrogram_image": spectrogram_img,
            "processing_time_sec": latency,
            "hardware_device": get_device_name(self.device),
            "checkpoint_loaded": self.checkpoint_loaded,
            "explanation": explanation
        }
