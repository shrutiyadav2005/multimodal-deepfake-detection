"""
Multimodal Deepfake Verification and Audio-Visual Consistency Engine.
Fuses independent video and audio forensic predictions, calculates cross-modal consistency,
and provides transparent technical explanations of observed incongruities.
"""

from typing import Dict, Optional, Union
import numpy as np
import torch

from configs.config import config


class MultimodalVerifier:
    """
    Multimodal fusion and cross-modal consistency evaluator.
    Combines independent visual and audio forensic streams to verify integrity.
    """

    def __init__(
        self,
        visual_weight: float = 0.55,
        audio_weight: float = 0.45,
        inconsistency_threshold: float = 0.35
    ):
        self.visual_weight = visual_weight
        self.audio_weight = audio_weight
        self.inconsistency_threshold = inconsistency_threshold

        # Normalize weights to sum to 1.0
        total_w = self.visual_weight + self.audio_weight
        self.visual_weight /= total_w
        self.audio_weight /= total_w

    def evaluate_consistency(
        self,
        video_fake_prob: float,
        audio_fake_prob: float
    ) -> Dict[str, Union[float, str, bool]]:
        """
        Calculates cross-modal consistency and fused multi-modal integrity verdict.

        Args:
            video_fake_prob: Assessed manipulation probability from video stream [0, 1].
            audio_fake_prob: Assessed manipulation probability from audio stream [0, 1].

        Returns:
            Dict containing fused scores, consistency index, anomaly flag, and diagnosis.
        """
        # Multimodal Late Fusion
        fused_fake_prob = (
            self.visual_weight * video_fake_prob + self.audio_weight * audio_fake_prob
        )
        fused_real_prob = 1.0 - fused_fake_prob

        # Cross-Modal Discrepancy
        inconsistency_delta = abs(video_fake_prob - audio_fake_prob)
        consistency_score = 1.0 - inconsistency_delta
        is_inconsistent = inconsistency_delta >= self.inconsistency_threshold

        # Multi-modal Diagnostic Verdict
        v_thresh = config.video.suspicious_threshold
        v_flag = video_fake_prob >= v_thresh
        a_flag = audio_fake_prob >= v_thresh

        if v_flag and a_flag:
            diagnosis = (
                "Dual-Modality Deepfake: Both visual facial stream and acoustic speech stream "
                "exhibit high probabilities of synthetic generation or manipulation."
            )
            overall_verdict = "FAKE"
        elif v_flag and not a_flag:
            diagnosis = (
                "Visual Manipulation / Face-Swap: The visual frames show strong indicators of "
                "manipulation while the audio track appears acoustic-authentic, suggesting face-swapping "
                "or facial reenactment on authentic source audio."
            )
            overall_verdict = "SUSPICIOUS_VISUAL"
        elif not v_flag and a_flag:
            diagnosis = (
                "Audio Spoofing / Voice Clone: The acoustic stream shows significant synthetic or voice "
                "conversion artifacts while the video frames appear unmanipulated, suggesting voice cloning "
                "or audio track dubbing."
            )
            overall_verdict = "SUSPICIOUS_AUDIO"
        else:
            diagnosis = (
                "Congruent Authentic Media: Both visual and acoustic modalities align with natural, "
                "unmanipulated media characteristics."
            )
            overall_verdict = "REAL"

        return {
            "overall_verdict": overall_verdict,
            "fused_fake_probability": round(fused_fake_prob, 4),
            "fused_real_probability": round(fused_real_prob, 4),
            "video_fake_probability": round(video_fake_prob, 4),
            "audio_fake_probability": round(audio_fake_prob, 4),
            "inconsistency_delta": round(inconsistency_delta, 4),
            "consistency_score": round(consistency_score, 4),
            "is_inconsistent": is_inconsistent,
            "diagnosis": diagnosis,
            "limitations_notice": (
                "Note on Methodology: The consistency score assesses forensic congruence between "
                "visual and acoustic authenticity streams. It detects modality mismatch (e.g., cloned audio "
                "dubbed over real video) but does not compute sub-second phonetic viseme-to-phoneme lip-sync "
                "correlation."
            )
        }
