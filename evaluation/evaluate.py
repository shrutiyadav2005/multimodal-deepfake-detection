"""
Comprehensive evaluation script for Multimodal Deepfake Detection.
Evaluates Image, Audio, and Video modalities, compares measured metrics against
portfolio benchmark goals, benchmarks runtime latency, and generates publication plots.
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import config
from models.image_model import ImageDeepfakeDetector
from models.audio_model import AudioDeepfakeDetector
from models.video_model import VideoDeepfakeDetector
from preprocessing.image_preprocessing import ImagePreprocessor
from preprocessing.audio_preprocessing import AudioPreprocessor
from preprocessing.video_preprocessing import VideoPreprocessor
from training.train_image import DeepfakeImageDataset
from training.train_audio import DeepfakeAudioDataset
from evaluation.metrics import compute_classification_metrics
from utils.device import get_device, get_device_name
from utils.helpers import save_json


def evaluate_image_modality(device: torch.device) -> Dict[str, any]:
    """Evaluates image classifier on the test split."""
    test_dir = config.base_dir / "data" / "processed" / "images" / "test"
    if not test_dir.exists():
        return {}

    preprocessor = ImagePreprocessor()
    dataset = DeepfakeImageDataset(test_dir, preprocessor=preprocessor, is_training=False)
    if len(dataset) == 0:
        return {}

    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model, loaded = ImageDeepfakeDetector.load_from_checkpoint(device=device)

    y_true = []
    y_pred = []
    y_scores = []
    latencies = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            t0 = time.time()
            outputs = model(inputs)
            latencies.append((time.time() - t0) / inputs.size(0))

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_scores.extend(probs[:, 1].cpu().numpy().tolist())

    metrics = compute_classification_metrics(y_true, y_pred, y_scores)
    metrics["checkpoint_loaded"] = loaded
    metrics["avg_inference_latency_ms"] = round(float(np.mean(latencies)) * 1000, 2)
    return metrics


def evaluate_audio_modality(device: torch.device) -> Dict[str, any]:
    """Evaluates audio classifier on the test split."""
    test_dir = config.base_dir / "data" / "processed" / "audio" / "test"
    if not test_dir.exists():
        return {}

    preprocessor = AudioPreprocessor()
    dataset = DeepfakeAudioDataset(test_dir, preprocessor=preprocessor)
    if len(dataset) == 0:
        return {}

    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model, loaded = AudioDeepfakeDetector.load_from_checkpoint(device=device)

    y_true = []
    y_pred = []
    y_scores = []
    latencies = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            t0 = time.time()
            outputs = model(inputs)
            latencies.append((time.time() - t0) / inputs.size(0))

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_scores.extend(probs[:, 1].cpu().numpy().tolist())

    metrics = compute_classification_metrics(y_true, y_pred, y_scores)
    metrics["checkpoint_loaded"] = loaded
    metrics["avg_inference_latency_ms"] = round(float(np.mean(latencies)) * 1000, 2)
    return metrics


def benchmark_video_processing(device: torch.device) -> Dict[str, any]:
    """Benchmarks actual video sampling, frame inference, and aggregation throughput."""
    sample_video = config.sample_data_dir / "videos" / "sample_real.mp4"
    if not sample_video.exists():
        from training.dataset_generator import generate_all_samples
        generate_all_samples()

    img_model, _ = ImageDeepfakeDetector.load_from_checkpoint(device=device)
    video_detector = VideoDeepfakeDetector(image_model=img_model)
    video_detector.to(device)
    preprocessor = VideoPreprocessor(frame_sample_count=16)

    t0 = time.time()
    frames_tensor, raw_frames, timestamps, meta = preprocessor.sample_frames(sample_video)
    sampling_time = time.time() - t0

    t1 = time.time()
    frames_tensor = frames_tensor.to(device)
    result = video_detector(frames_tensor)
    inference_time = time.time() - t1

    total_time = sampling_time + inference_time
    fps_throughput = len(raw_frames) / total_time if total_time > 0 else 0.0

    return {
        "video_path": sample_video.name,
        "video_duration_sec": meta["duration_sec"],
        "frames_sampled": len(raw_frames),
        "frame_sampling_time_sec": round(sampling_time, 3),
        "model_inference_time_sec": round(inference_time, 3),
        "total_processing_time_sec": round(total_time, 3),
        "throughput_fps": round(fps_throughput, 2),
        "verdict": result["label"],
        "fake_probability": result["fake_probability"]
    }


def plot_confusion_matrix(cm: List[List[int]], title: str, output_path: Path) -> None:
    """Generates and saves a high-resolution confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        cbar=False
    )
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("Actual Ground Truth", fontsize=10)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close()


def plot_roc_curve(roc_curve_data: Dict[str, List[float]], roc_auc: float, output_path: Path) -> None:
    """Plots and saves the ROC-AUC curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    if roc_curve_data:
        plt.plot(roc_curve_data["fpr"], roc_curve_data["tpr"], color="#2b5c8f", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close()


def plot_benchmark_comparison(measured_metrics: Dict[str, float], output_path: Path) -> None:
    """Compares actual measured metrics against target goals."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["accuracy", "f1", "precision", "recall"]
    labels = ["Accuracy", "F1-Score", "Precision", "Recall"]

    targets = [
        config.benchmarks.target_accuracy,
        config.benchmarks.target_f1,
        config.benchmarks.target_precision,
        config.benchmarks.target_recall
    ]

    measured = [measured_metrics.get(k, 0.0) for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width/2, measured, width, label="Measured Actual", color="#1f77b4")
    plt.bar(x + width/2, targets, width, label="Target Benchmark", color="#aec7e8", alpha=0.7)

    plt.ylabel("Score", fontsize=10)
    plt.title("Model Performance: Measured vs Target Benchmarks", fontsize=12, fontweight="bold")
    plt.xticks(x, labels, fontsize=10)
    plt.ylim([0, 1.1])
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(labels)):
        plt.text(x[i] - width/2, measured[i] + 0.02, f"{measured[i]:.2f}", ha="center", fontsize=9)
        plt.text(x[i] + width/2, targets[i] + 0.02, f"{targets[i]:.2f}", ha="center", fontsize=9, color="#555")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close()


def run_full_evaluation() -> Dict[str, any]:
    """Runs end-to-end evaluation, generates report, and saves visualization plots."""
    device = get_device()
    print("=" * 65)
    print(f"MULTIMODAL DEEPFAKE EVALUATION RUN - [{get_device_name(device)}]")
    print("=" * 65)

    image_results = evaluate_image_modality(device)
    audio_results = evaluate_audio_modality(device)
    video_bench = benchmark_video_processing(device)

    print("\n--- Image Detection Test Metrics ---")
    for k, v in image_results.items():
        if k not in ["confusion_matrix", "roc_curve"]:
            print(f"  {k:<28}: {v}")

    print("\n--- Audio Detection Test Metrics ---")
    for k, v in audio_results.items():
        if k not in ["confusion_matrix", "roc_curve"]:
            print(f"  {k:<28}: {v}")

    print("\n--- Video Processing Runtime Benchmark ---")
    for k, v in video_bench.items():
        print(f"  {k:<28}: {v}")

    # Generate and save plots
    if "confusion_matrix" in image_results:
        plot_confusion_matrix(
            image_results["confusion_matrix"],
            "Image Detector: Confusion Matrix",
            config.plots_dir / "confusion_matrix_image.png"
        )
    if "roc_curve" in image_results and image_results["roc_curve"]:
        plot_roc_curve(
            image_results["roc_curve"],
            image_results.get("roc_auc", 0.0),
            config.plots_dir / "roc_curve_image.png"
        )
    if image_results:
        plot_benchmark_comparison(
            image_results,
            config.plots_dir / "benchmark_comparison.png"
        )

    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_device": get_device_name(device),
        "target_benchmarks": {
            "accuracy": config.benchmarks.target_accuracy,
            "f1": config.benchmarks.target_f1,
            "precision": config.benchmarks.target_precision,
            "recall": config.benchmarks.target_recall
        },
        "image_modality": image_results,
        "audio_modality": audio_results,
        "video_benchmark": video_bench
    }

    report_path = config.reports_dir / "full_evaluation_report.json"
    save_json(full_report, report_path)
    print(f"\n[✔] Full evaluation report saved to: {report_path}")
    print(f"[✔] Visualization plots saved to: {config.plots_dir}")
    print("=" * 65)
    return full_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Deepfake System")
    parser.parse_args()
    run_full_evaluation()


if __name__ == "__main__":
    main()
