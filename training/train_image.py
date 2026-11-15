"""
Training script for Image Deepfake Detection Model.
Includes dataset loading, class imbalance weighting, validation tracking,
and reproducible checkpoint saving.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from configs.config import config
from models.image_model import ImageDeepfakeDetector
from preprocessing.image_preprocessing import ImagePreprocessor
from utils.device import get_device, get_device_name, set_seed
from utils.helpers import save_json


class DeepfakeImageDataset(Dataset):
    """
    PyTorch Dataset for Real and Fake facial images.
    Organized as:
      root_dir/
        real/*.jpg|png
        fake/*.jpg|png
    """

    def __init__(self, root_dir: Path, preprocessor: ImagePreprocessor, is_training: bool = False):
        self.root_dir = Path(root_dir)
        self.preprocessor = preprocessor
        self.is_training = is_training
        self.samples: List[Tuple[Path, int]] = []

        real_dir = self.root_dir / "real"
        fake_dir = self.root_dir / "fake"

        if real_dir.exists():
            for p in real_dir.iterdir():
                if p.suffix.lower() in config.image.supported_extensions:
                    self.samples.append((p, 0))  # 0: REAL

        if fake_dir.exists():
            for p in fake_dir.iterdir():
                if p.suffix.lower() in config.image.supported_extensions:
                    self.samples.append((p, 1))  # 1: FAKE

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        tensor, _ = self.preprocessor.preprocess(
            img_path,
            crop_face=True,
            is_training=self.is_training
        )
        return tensor.squeeze(0), label


def train_image_model(
    data_dir: Path,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    checkpoint_out: Path = None,
    seed: int = 42
) -> Dict[str, List[float]]:
    """
    Executes training loop with validation tracking and checkpoint saving.
    """
    set_seed(seed)
    device = get_device()
    print(f"[*] Training Image Detector on: {get_device_name(device)}")

    checkpoint_out = checkpoint_out or (config.checkpoints_dir / config.image.checkpoint_name)
    preprocessor = ImagePreprocessor()

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_dataset = DeepfakeImageDataset(train_dir, preprocessor=preprocessor, is_training=True)
    val_dataset = DeepfakeImageDataset(val_dir, preprocessor=preprocessor, is_training=False)

    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {train_dir}. Please run dataset generation first.")

    # Calculate class weights to handle imbalance
    labels = [s[1] for s in train_dataset.samples]
    num_real = labels.count(0)
    num_fake = labels.count(1)
    print(f"[*] Class Distribution: Real={num_real}, Fake={num_fake}")

    weights = torch.tensor([
        1.0 / max(num_real, 1),
        1.0 / max(num_fake, 1)
    ], dtype=torch.float).to(device)
    weights = weights / weights.sum()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else None

    # Model, Loss, Optimizer
    model = ImageDeepfakeDetector(architecture=config.image.architecture, pretrained_backbone=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.image.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))

        # Validation
        val_loss, val_acc = 0.0, 0.0
        if val_loader and len(val_dataset) > 0:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_running_loss += loss.item() * inputs.size(0)
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)

            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total
            history["val_loss"].append(round(val_loss, 4))
            history["val_acc"].append(round(val_acc, 4))

            print(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "architecture": config.image.architecture
                }, str(checkpoint_out))
                print(f"  --> Saved new best checkpoint to {checkpoint_out}")
        else:
            print(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": train_loss,
                "val_acc": train_acc,
                "architecture": config.image.architecture
            }, str(checkpoint_out))

    # Save training metrics history
    history_file = config.reports_dir / "image_training_history.json"
    save_json(history, history_file)
    print(f"[✔] Image model training completed. History saved to {history_file}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Image Deepfake Detector")
    parser.add_argument("--data-dir", type=str, default=str(config.base_dir / "data" / "processed" / "images"))
    parser.add_argument("--epochs", type=int, default=config.image.epochs)
    parser.add_argument("--batch-size", type=int, default=config.image.batch_size)
    parser.add_argument("--lr", type=float, default=config.image.learning_rate)
    args = parser.parse_args()

    train_image_model(
        data_dir=Path(args.data_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == "__main__":
    main()
