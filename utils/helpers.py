"""
General helper functions for file handling, validation, timing, and formatting.
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional


def validate_file_path(file_path: Union[str, Path], allowed_extensions: Optional[Tuple[str, ...]] = None) -> Path:
    """
    Validates that a file exists, is a regular file, and has an allowed extension.

    Args:
        file_path: Path to the target file.
        allowed_extensions: Optional tuple of valid extensions (e.g. ('.jpg', '.png')).

    Returns:
        Path: Resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is not a file or has unsupported extension.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    if allowed_extensions:
        ext = path.suffix.lower()
        allowed_lower = tuple(e.lower() for e in allowed_extensions)
        if ext not in allowed_lower:
            raise ValueError(
                f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(allowed_lower)}"
            )

    return path


def format_file_size(size_bytes: int) -> str:
    """
    Converts a file size in bytes to a human-readable string.
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def calculate_processing_time(start_time: float) -> float:
    """
    Calculates elapsed time in seconds with high precision.
    """
    return round(time.time() - start_time, 4)


def save_json(data: Any, file_path: Union[str, Path]) -> None:
    """
    Safely serializes and saves data as formatted JSON.
    """
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Safely reads JSON data from file.
    """
    path = validate_file_path(file_path, allowed_extensions=(".json",))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_confidence(prob: float) -> str:
    """
    Formats probability as a clean percentage string.
    """
    return f"{prob * 100:.2f}%"
