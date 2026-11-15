"""
Training and validation script for Video Deepfake Detection.
Evaluates spatial-temporal video pipeline across benchmark sequences,
verifying frame sampling, aggregation methods, and video-level classification.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List
import torch

from configs.config import config
from models.image_model import ImageDeepfakeDetector
from models.video_model import VideoDeepfakeDetector
from preprocessing.video_preprocessing import VideoPreprocessor
from utils.device import get_device, get_device_name
from utils.helpers import save_json


def evaluate_video_pipeline(
    video_dir: Path,
    aggregation_method: str = "top_k_mean"
) -> Dict[str, float]:
    """
    Evaluates video detection accuracy across sample video files.
    """
    device = get_device()
    print(f"[*] Evaluating Video Pipeline on: {get_device_name(device)}")

    # Load image detector
    img_model, loaded = ImageDeepfakeDetector.load_from_checkpoint(device=device)
    if loaded:
        print("  [✔] Loaded trained image detector checkpoint.")
    else:
        print("  [!] Running with pretrained feature backbone.")

    video_detector = VideoDeepfakeDetector(
        image_model=img_model,
        aggregation_method=aggregation_method
    )
    video_detector.to(device)

    preprocessor = VideoPreprocessor(frame_sample_count=config.video.frame_sample_count)

    video_paths = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
    if not video_paths:
        print(f"[!] No video files found in {video_dir}. Generating samples...")
        from training.dataset_generator import generate_all_samples
        generate_all_samples()
        video_paths = list(video_dir.glob("*.mp4"))

    correct = 0
    total = 0
    results = []

    for vid_path in video_paths:
        # Determine ground truth from filename convention (contains 'real' or 'fake')
        is_real = "real" in vid_path.name.lower()
        true_label = "REAL" if is_real else "FAKE"

        try:
            frames_tensor, _, _, meta = preprocessor.sample_frames(vid_path, crop_face=True)
            frames_tensor = frames_tensor.to(device)
            pred = video_detector(frames_tensor)

            is_correct = (pred["label"] == true_label)
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "file": vid_path.name,
                "true_label": true_label,
                "predicted_label": pred["label"],
                "fake_probability": pred["fake_probability"],
                "suspicious_frames_count": len(pred["suspicious_frame_indices"]),
                "duration_sec": meta["duration_sec"],
                "correct": is_correct
            })
            print(f"  {vid_path.name:<20}: True={true_label:<4} Pred={pred['label']:<4} (Prob: {pred['fake_probability']:.2f})")
        except Exception as e:
            print(f"  [Error] Processing {vid_path.name}: {e}")

    accuracy = correct / total if total > 0 else 0.0
    summary = {
        "total_videos": total,
        "correct_predictions": correct,
        "accuracy": round(accuracy, 4),
        "results": results
    }

    report_path = config.reports_dir / "video_evaluation_summary.json"
    save_json(summary, report_path)
    print(f"[✔] Video evaluation completed. Accuracy: {accuracy * 100:.2f}%. Report: {report_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Video Deepfake Pipeline")
    parser.add_argument("--video-dir", type=str, default=str(config.sample_data_dir / "videos"))
    parser.add_argument("--aggregation", type=str, default="top_k_mean", choices=["mean", "max", "top_k_mean"])
    args = parser.parse_args()

    evaluate_video_pipeline(
        video_dir=Path(args.video_dir),
        aggregation_method=args.aggregation
    )


if __name__ == "__main__":
    main()
