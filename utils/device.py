"""
Device management and reproducibility utilities.
Ensures seamless execution across Apple Silicon (MPS), NVIDIA GPUs (CUDA), and CPU.
"""

import os
import random
import numpy as np
import torch
from typing import Tuple


def get_device() -> torch.device:
    """
    Automatically selects the best available compute hardware.
    Priority: CUDA -> Apple Silicon MPS -> CPU.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name(device: torch.device = None) -> str:
    """
    Returns a human-readable description of the compute device.
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device.type == "mps":
        return "Apple Silicon (MPS - Metal Performance Shaders)"
    else:
        return "CPU (Standard Host Execution)"


def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic seeds across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed (int): The integer random seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        # MPS specific deterministic behavior
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass
