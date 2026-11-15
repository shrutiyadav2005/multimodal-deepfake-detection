"""
Synthetic sample generator and dataset partition utility.
Creates self-contained real and manipulated samples (images, videos, audio)
for zero-setup testing, training demonstration, and local experimentation.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import soundfile as sf

from configs.config import config
from utils.device import set_seed


def generate_face_like_image(is_fake: bool = False, seed: int = 42) -> np.ndarray:
    """
    Generates a synthetic portrait-like visual image with distinct natural vs artifact signatures.

    Args:
        is_fake: If True, introduces deepfake-like boundary seams and color shifts.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: RGB image array of size (256, 256, 3) in uint8.
    """
    rng = np.random.RandomState(seed)
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Background gradient
    for y in range(h):
        img[y, :, 0] = int(30 + 30 * (y / h))
        img[y, :, 1] = int(40 + 40 * (y / h))
        img[y, :, 2] = int(50 + 50 * (y / h))

    center = (w // 2, h // 2)
    # Head contour
    cv2.ellipse(img, center, (65, 85), 0, 0, 360, (215, 175, 145), -1)

    # Eyes
    left_eye = (center[0] - 25, center[1] - 15)
    right_eye = (center[0] + 25, center[1] - 15)
    cv2.circle(img, left_eye, 8, (255, 255, 255), -1)
    cv2.circle(img, right_eye, 8, (255, 255, 255), -1)
    cv2.circle(img, left_eye, 4, (45, 30, 20), -1)
    cv2.circle(img, right_eye, 4, (45, 30, 20), -1)

    # Mouth
    cv2.ellipse(img, (center[0], center[1] + 35), (20, 8), 0, 0, 180, (180, 70, 70), -1)

    # Smooth natural skin blending
    img = cv2.GaussianBlur(img, (5, 5), 1.0)

    if is_fake:
        # Inject manipulation artifacts:
        # 1. Blending boundary seam around the face
        cv2.ellipse(img, center, (67, 87), 0, 0, 360, (160, 120, 100), 2)
        # 2. Localized chromatic aberration / color mismatch in face region
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(face_mask, center, (60, 80), 0, 0, 360, 255, -1)
        noise = rng.normal(0, 18, (h, w, 3)).astype(np.int16)
        img_int = img.astype(np.int16)
        for c in range(3):
            img_int[:, :, c] = np.where(face_mask == 255, img_int[:, :, c] + noise[:, :, c] + 15, img_int[:, :, c])
        img = np.clip(img_int, 0, 255).astype(np.uint8)
        # 3. High-frequency checkerboard / compression artifact
        for i in range(center[1] - 30, center[1] + 30, 4):
            img[i, center[0] - 30 : center[0] + 30, 0] = np.clip(
                img[i, center[0] - 30 : center[0] + 30, 0] + 25, 0, 255
            )

    return img


def generate_sample_video(output_path: Path, is_fake: bool = False, fps: int = 25, duration_sec: int = 2) -> None:
    """
    Generates an MP4 video sequence simulating real or manipulated video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = 256, 256
    out = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))

    total_frames = fps * duration_sec
    for i in range(total_frames):
        # Base frame
        frame = generate_face_like_image(is_fake=False, seed=i)
        # Subtle movement
        shift = int(5 * np.sin(2 * np.pi * i / fps))
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        frame = cv2.warpAffine(frame, M, (w, h))

        if is_fake:
            # Deepfake flicker: certain frames exhibit boundary artifacts and facial distortion
            if i % 6 == 0 or i % 6 == 1:
                frame = generate_face_like_image(is_fake=True, seed=i + 100)
                frame = cv2.warpAffine(frame, M, (w, h))

        # Convert RGB to BGR for OpenCV writer
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr)

    out.release()


def generate_sample_audio(
    output_path: Path, is_fake: bool = False, sample_rate: int = 16000, duration: float = 3.0
) -> None:
    """
    Generates synthetic speech-like acoustic signals. Real has continuous harmonic structure;
    Fake exhibits spectral discontinuities and robotic vocoder buzzing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    if not is_fake:
        # Natural fundamental voice pitch (F0 ~ 130 Hz) + natural harmonics (260 Hz, 390 Hz, 520 Hz)
        f0 = 130 + 10 * np.sin(2 * np.pi * 1.5 * t)  # Pitch modulation
        wave = 0.5 * np.sin(2 * np.pi * f0 * t)
        wave += 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        wave += 0.12 * np.sin(2 * np.pi * 3 * f0 * t)
        # Formant envelope
        envelope = np.exp(-t / 1.5) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t))
        wave *= envelope
    else:
        # Deepfake vocoder artifacts: harsh square wave harmonics, phase discontinuities, robotic buzz
        f0 = 160  # Monotone pitch typical of rudimentary TTS
        wave = 0.4 * np.sign(np.sin(2 * np.pi * f0 * t))
        # Add high-frequency vocoder distortion buzz (3500 Hz & 6000 Hz)
        wave += 0.25 * np.sin(2 * np.pi * 3500 * t)
        # Periodic synthetic phase jump
        wave = np.where(np.sin(2 * np.pi * 8 * t) > 0.8, -wave, wave)

    # Normalize to [-0.9, 0.9]
    max_val = np.max(np.abs(wave)) + 1e-6
    wave = (wave / max_val) * 0.9

    sf.write(str(output_path), wave.astype(np.float32), sample_rate)


def generate_all_samples() -> None:
    """Generates all default sample assets in data/sample/."""
    sample_dir = config.sample_data_dir
    img_dir = sample_dir / "images"
    vid_dir = sample_dir / "videos"
    aud_dir = sample_dir / "audio"

    for d in [img_dir, vid_dir, aud_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[*] Generating verified sample images...")
    real_img = generate_face_like_image(is_fake=False, seed=1)
    fake_img = generate_face_like_image(is_fake=True, seed=2)
    cv2.imwrite(str(img_dir / "sample_real.jpg"), cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(img_dir / "sample_fake.jpg"), cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR))

    print("[*] Generating verified sample videos...")
    generate_sample_video(vid_dir / "sample_real.mp4", is_fake=False, duration_sec=2)
    generate_sample_video(vid_dir / "sample_fake.mp4", is_fake=True, duration_sec=2)

    print("[*] Generating verified sample audio...")
    generate_sample_audio(aud_dir / "sample_real.wav", is_fake=False, duration=3.0)
    generate_sample_audio(aud_dir / "sample_fake.wav", is_fake=True, duration=3.0)

    print("[✔] Sample assets created successfully in data/sample/.")


def generate_synthetic_dataset(
    num_samples_per_class: int = 40,
    split_ratios: tuple = (0.7, 0.15, 0.15)
) -> None:
    """
    Generates a full synthetic benchmark dataset partitioned into train/val/test splits
    for images and audio, allowing reproducible local training out of the box.
    """
    set_seed(42)
    base = config.base_dir / "data" / "processed"

    n_train = int(num_samples_per_class * split_ratios[0])
    n_val = int(num_samples_per_class * split_ratios[1])
    n_test = num_samples_per_class - n_train - n_val

    splits = [
        ("train", 0, n_train),
        ("val", n_train, n_train + n_val),
        ("test", n_train + n_val, num_samples_per_class)
    ]

    print(f"[*] Generating dataset: {num_samples_per_class} samples/class ({n_train} train, {n_val} val, {n_test} test)")

    for split_name, start_idx, end_idx in splits:
        img_real_dir = base / "images" / split_name / "real"
        img_fake_dir = base / "images" / split_name / "fake"
        aud_real_dir = base / "audio" / split_name / "real"
        aud_fake_dir = base / "audio" / split_name / "fake"

        for d in [img_real_dir, img_fake_dir, aud_real_dir, aud_fake_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for i in range(start_idx, end_idx):
            # Images
            r_img = generate_face_like_image(is_fake=False, seed=1000 + i)
            f_img = generate_face_like_image(is_fake=True, seed=2000 + i)
            cv2.imwrite(str(img_real_dir / f"real_{i:04d}.jpg"), cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(img_fake_dir / f"fake_{i:04d}.jpg"), cv2.cvtColor(f_img, cv2.COLOR_RGB2BGR))

            # Audio
            generate_sample_audio(aud_real_dir / f"real_{i:04d}.wav", is_fake=False, duration=3.0)
            generate_sample_audio(aud_fake_dir / f"fake_{i:04d}.wav", is_fake=True, duration=3.0)

    print("[✔] Processed training/validation/test datasets generated in data/processed/.")


if __name__ == "__main__":
    generate_all_samples()
    generate_synthetic_dataset(num_samples_per_class=40)
