"""
Image preprocessing and face extraction pipeline.
Supports loading, validation, face detection cropping, resizing, and PyTorch tensor transformations.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from configs.config import config
from utils.helpers import validate_file_path


class ImagePreprocessor:
    """
    Handles image loading, face detection cropping, and PyTorch normalization.
    """

    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.mean = config.image.mean
        self.std = config.image.std

        # Load OpenCV Haar cascade for face detection with graceful fallback
        self.face_cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            try:
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self.face_cascade = cascade
            except Exception:
                self.face_cascade = None

        # PyTorch Evaluation / Inference Pipeline
        self.eval_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        # Training Augmentation Pipeline
        self.train_transform = transforms.Compose([
            transforms.Resize((int(self.image_size[0] * 1.1), int(self.image_size[1] * 1.1))),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def load_image_rgb(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Loads an image from file path, numpy array, or PIL Image into RGB format.

        Returns:
            np.ndarray: Image array of shape (H, W, 3) in uint8 RGB format.
        """
        if isinstance(image_input, (str, Path)):
            path = validate_file_path(image_input, allowed_extensions=config.image.supported_extensions)
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"OpenCV could not decode image: {path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            return image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def detect_and_crop_face(
        self,
        rgb_image: np.ndarray,
        margin_ratio: float = 0.2
    ) -> Tuple[np.ndarray, bool]:
        """
        Detects primary face in image using Haar Cascades and crops with margin.
        If no face is detected, returns original image gracefully.

        Args:
            rgb_image: Input RGB numpy array.
            margin_ratio: Fractional margin padding around detected face box.

        Returns:
            Tuple[np.ndarray, bool]: (Cropped RGB image or original, whether face was detected).
        """
        if self.face_cascade is None:
            return rgb_image, False

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(faces) == 0:
            return rgb_image, False

        # Select largest face by bounding box area
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        img_h, img_w, _ = rgb_image.shape

        margin_w = int(w * margin_ratio)
        margin_h = int(h * margin_ratio)

        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img_w, x + w + margin_w)
        y2 = min(img_h, y + h + margin_h)

        face_crop = rgb_image[y1:y2, x1:x2]
        return face_crop, True

    def preprocess(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        crop_face: bool = True,
        is_training: bool = False
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Full preprocessing pipeline.

        Args:
            image_input: File path, numpy array, or PIL image.
            crop_face: Whether to detect and crop face region first.
            is_training: Whether to use data augmentations.

        Returns:
            Tuple[torch.Tensor, np.ndarray]:
                - Tensor of shape (1, 3, H, W) normalized for model input.
                - Raw RGB numpy image of size (H, W, 3) suitable for Grad-CAM overlay.
        """
        rgb_image = self.load_image_rgb(image_input)

        if crop_face:
            processed_rgb, _ = self.detect_and_crop_face(rgb_image)
        else:
            processed_rgb = rgb_image

        # Resize for display / Grad-CAM target
        display_rgb = cv2.resize(processed_rgb, self.image_size)

        pil_img = Image.fromarray(display_rgb)
        transform = self.train_transform if is_training else self.eval_transform
        tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension: (1, 3, H, W)

        return tensor, display_rgb
