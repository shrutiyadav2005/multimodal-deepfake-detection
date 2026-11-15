"""
Dataset management script.
Provides verification and download guidelines for FaceForensics++, Celeb-DF, and ASVspoof.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import config


def check_dataset_status() -> None:
    """Checks and displays availability of local datasets."""
    print("=" * 60)
    print("DATASET STATUS CHECK")
    print("=" * 60)

    datasets = {
        "Sample Data": config.sample_data_dir,
        "FaceForensics++ (Video)": config.base_dir / "data" / "FaceForensics",
        "Celeb-DF (Video)": config.base_dir / "data" / "Celeb-DF",
        "ASVspoof (Audio)": config.base_dir / "data" / "ASVspoof",
        "Processed Data": config.base_dir / "data" / "processed",
    }

    for name, path in datasets.items():
        exists = path.exists() and (len(list(path.iterdir())) > 0 if path.is_dir() else True)
        status = "FOUND" if exists else "NOT FOUND (Download or generate required)"
        print(f"[{'✔' if exists else '✖'}] {name:<25}: {status}")
        if exists:
            print(f"    Path: {path}")

    print("=" * 60)


def print_acquisition_guide(dataset_name: str) -> None:
    """Prints step-by-step acquisition instructions for a dataset."""
    guides = {
        "faceforensics": """
FaceForensics++ Acquisition Guide:
1. Visit the official GitHub: https://github.com/ondyari/FaceForensics
2. Submit the academic access request form to receive download credentials.
3. Use the official download script provided by TUM:
   python download-FaceForensics.py ./data/FaceForensics -d all -c c23 -t videos
4. Ensure files are structured as data/FaceForensics/original_sequences and manipulated_sequences.
""",
        "celeb-df": """
Celeb-DF (v2) Acquisition Guide:
1. Visit https://github.com/danmohaha/celeb-deepfakeforensics
2. Sign the agreement form for academic research.
3. Download the Celeb-DF-v2.zip dataset.
4. Extract into ./data/Celeb-DF/ with Celeb-real, Celeb-synthesis, YouTube-real.
""",
        "asvspoof": """
ASVspoof 2019 Audio Acquisition Guide:
1. Open access dataset hosted at Edinburgh DataShare:
   https://datashare.ed.ac.uk/handle/10283/3336
2. Download LA.zip (Logical Access containing synthetic & spoofed speech).
3. Unzip into ./data/ASVspoof/LA/.
"""
    }

    key = dataset_name.lower()
    if key in guides:
        print(guides[key])
    else:
        print(f"Unknown dataset '{dataset_name}'. Available: {list(guides.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Acquisition & Verification Manager")
    parser.add_argument("--check", action="store_true", help="Check local dataset availability")
    parser.add_argument("--guide", type=str, choices=["faceforensics", "celeb-df", "asvspoof"],
                        help="Display acquisition instructions for a specific dataset")
    args = parser.parse_args()

    if args.guide:
        print_acquisition_guide(args.guide)
    else:
        check_dataset_status()


if __name__ == "__main__":
    main()
