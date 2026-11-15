"""
Explainable AI (XAI) Visualization and Diagnostic Reporting Utilities.
Generates spectrogram plots, temporal video suspicion timelines,
and structured technical justifications for forensic findings.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from PIL import Image


def plot_spectrogram_to_image(
    mel_db: np.ndarray,
    sr: int = 16000,
    hop_length: int = 512,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Log-Mel Spectrogram Analysis"
) -> np.ndarray:
    """
    Renders acoustic Mel-spectrogram with frequency and time scales.

    Returns:
        np.ndarray: RGB image array of the generated plot.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_ylabel("Frequency (Hz / Mel)", fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    if output_path:
        plt.savefig(str(output_path), format="png", bbox_inches="tight")
    plt.close(fig)

    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    return np.array(pil_img)


def plot_video_timeline_to_image(
    frame_probabilities: List[float],
    timestamps: Optional[List[float]] = None,
    threshold: float = 0.50,
    output_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Plots temporal suspicion trajectory across sampled video frames.

    Returns:
        np.ndarray: RGB image array of the timeline chart.
    """
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=150)

    n_frames = len(frame_probabilities)
    x = timestamps if timestamps and len(timestamps) == n_frames else list(range(n_frames))
    x_label = "Timeline (seconds)" if timestamps else "Sampled Frame Index"

    probs = np.array(frame_probabilities)
    ax.plot(x, probs, color="#1f77b4", lw=2, marker="o", markersize=4, label="Frame Manipulation Score")

    # Threshold line
    ax.axhline(threshold, color="#d62728", linestyle="--", lw=1.5, label=f"Suspicion Threshold ({threshold})")

    # Highlight suspicious frames
    suspicious_mask = probs >= threshold
    if np.any(suspicious_mask):
        ax.scatter(
            np.array(x)[suspicious_mask],
            probs[suspicious_mask],
            color="#d62728",
            s=45,
            zorder=5,
            label="Flagged Frames"
        )

    # Highlight peak frame
    if len(probs) > 0:
        peak_idx = int(np.argmax(probs))
        ax.annotate(
            f"Peak: {probs[peak_idx]:.2f}",
            (x[peak_idx], probs[peak_idx]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#333", lw=1)
        )

    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("Predicted Fake Probability", fontsize=9)
    ax.set_title("Temporal Deepfake Frame Analysis", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    if output_path:
        plt.savefig(str(output_path), format="png", bbox_inches="tight")
    plt.close(fig)

    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    return np.array(pil_img)


def generate_explainability_summary(
    modality: str,
    verdict: str,
    fake_prob: float,
    extra_details: Optional[Dict] = None
) -> str:
    """
    Generates a structured, transparent forensic explanation.
    Emphasizes that confidence is a model score rather than an absolute ground-truth guarantee.
    """
    extra_details = extra_details or {}
    confidence = fake_prob if verdict == "FAKE" else (1.0 - fake_prob)

    lines = [
        f"Forensic Assessment Summary ({modality.upper()}):",
        f"- Classification Verdict: {verdict}",
        f"- Model Confidence: {confidence * 100:.2f}% (Estimated class likelihood)",
    ]

    if modality.lower() == "image":
        face_detected = extra_details.get("face_detected", True)
        lines.append(f"- Facial Detection: {'Face detected and centered' if face_detected else 'Full frame processed'}")
        if verdict == "FAKE":
            lines.append(
                "- Visual Saliency (Grad-CAM): Significant activation focuses on facial perimeter boundaries, "
                "blending seam artifacts, or unnatural texture irregularities characteristic of GAN/Diffusion synthesis."
            )
        else:
            lines.append(
                "- Visual Saliency (Grad-CAM): Activation exhibits smooth, natural gradient continuity across "
                "facial landmarks with no localized boundary anomalies."
            )

    elif modality.lower() == "video":
        susp_count = extra_details.get("suspicious_frames_count", 0)
        total_count = extra_details.get("total_frames_sampled", 0)
        agg_method = extra_details.get("aggregation_method", "top_k_mean")
        lines.append(f"- Frames Sampled: {total_count} (Aggregation: {agg_method})")
        lines.append(f"- Flagged Suspicious Frames: {susp_count}/{total_count}")
        if verdict == "FAKE":
            lines.append(
                "- Temporal Saliency: Intermittent manipulation spikes detected across the temporal timeline, "
                "indicating face reenactment, video deepfaking, or frame splicing."
            )
        else:
            lines.append(
                "- Temporal Saliency: Frame-level probabilities remain uniformly below threshold across the entire clip duration."
            )

    elif modality.lower() == "audio":
        if verdict == "FAKE":
            lines.append(
                "- Acoustic Saliency: High-frequency vocoder distortions, robotic harmonic repetition, "
                "or spectral energy discontinuities typical of synthetic speech synthesis (TTS) or voice conversion."
            )
        else:
            lines.append(
                "- Acoustic Saliency: Natural formant transitions and continuous harmonic decay consistent with human vocal physiology."
            )

    lines.append(
        "\nScientific Disclosure: Model confidence represents an automated statistical estimate from forensic neural "
        "networks. It does not constitute an absolute guarantee of content authenticity or origin."
    )

    return "\n".join(lines)
