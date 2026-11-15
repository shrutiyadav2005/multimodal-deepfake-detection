"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation.
Visualizes localized spatial regions contributing to deepfake classification decisions.
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """
    Computes Grad-CAM saliency heatmaps for convolutional neural networks.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] contains the gradient of target score w.r.t layer output
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1
    ) -> np.ndarray:
        """
        Generates 2D normalized Grad-CAM activation heatmap for the target class.

        Args:
            input_tensor: Normalized tensor of shape (1, 3, H, W).
            target_class: Integer class index (default: 1 for FAKE).

        Returns:
            np.ndarray: 2D float array in range [0, 1] matching input spatial dimensions (H, W).
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass (enable gradient computation for CAM)
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)

        # Select target class score
        score = logits[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Global Average Pooling of gradients across spatial dimensions (H, W)
        # gradients shape: (1, Channels, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Linear combination of activations weighted by channel importances
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)

        # Apply ReLU to retain features that positively correlate with the target class
        cam = torch.relu(cam)

        # Convert to numpy array
        cam_np = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize heatmap to match input tensor resolution (H, W)
        _, _, h, w = input_tensor.shape
        heatmap_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)
        return heatmap_resized

    @staticmethod
    def overlay_heatmap_on_image(
        rgb_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlays normalized Grad-CAM heatmap onto RGB image.

        Args:
            rgb_image: RGB image uint8 array of shape (H, W, 3).
            heatmap: 2D float array [0, 1] of shape (H, W).
            alpha: Blending weight for heatmap overlay.
            colormap: OpenCV colormap constant.

        Returns:
            np.ndarray: Blended RGB image uint8 array of shape (H, W, 3).
        """
        # Ensure image and heatmap match resolution
        h, w = rgb_image.shape[:2]
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h))

        # Convert normalized float [0, 1] to uint8 [0, 255]
        heatmap_uint8 = np.uint8(255 * heatmap)

        # Apply colormap (generates BGR)
        heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend
        blended = cv2.addWeighted(rgb_image, 1.0 - alpha, heatmap_color_rgb, alpha, 0)
        return blended
