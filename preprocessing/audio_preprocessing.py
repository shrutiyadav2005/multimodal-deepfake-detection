"""
Audio preprocessing and Mel-spectrogram feature extraction pipeline.
Transforms raw audio waveforms (.wav, .mp3) into normalized log-Mel spectrogram tensors.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import librosa
import soundfile as sf

from configs.config import config
from utils.helpers import validate_file_path


class AudioPreprocessor:
    """
    Loads audio, normalizes sample rate and length, and generates Mel-spectrogram tensors.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 4.0,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        f_min: int = 20,
        f_max: int = 8000,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int, float]:
        """
        Loads and resamples audio file to target sample rate.

        Returns:
            Tuple: (waveform_1d, sample_rate, duration_sec)
        """
        path = validate_file_path(audio_path, allowed_extensions=config.audio.supported_extensions)
        try:
            waveform, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        except Exception as e:
            # Fallback using soundfile if librosa default backend encounters format quirks
            try:
                waveform, sr = sf.read(str(path))
                if waveform.ndim > 1:
                    waveform = np.mean(waveform, axis=1)
                if sr != self.sample_rate:
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
                    sr = self.sample_rate
            except Exception as inner_e:
                raise ValueError(f"Failed to decode audio file {path}: {e} / {inner_e}")

        raw_duration = len(waveform) / sr if sr > 0 else 0.0
        return waveform, sr, raw_duration

    def fix_length(self, waveform: np.ndarray) -> np.ndarray:
        """
        Pads with silence or center-crops waveform to exact target length.
        """
        if len(waveform) < self.target_length:
            padding = self.target_length - len(waveform)
            pad_left = padding // 2
            pad_right = padding - pad_left
            return np.pad(waveform, (pad_left, pad_right), mode="constant")
        elif len(waveform) > self.target_length:
            start = (len(waveform) - self.target_length) // 2
            return waveform[start : start + self.target_length]
        return waveform

    def compute_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Calculates log-Mel spectrogram in decibels (dB) from 1D waveform.

        Returns:
            np.ndarray: Spectrogram array of shape (n_mels, time_steps).
        """
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db

    def normalize_spectrogram(self, mel_db: np.ndarray) -> np.ndarray:
        """
        Zero-mean unit-variance standard scaler normalization.
        """
        mean = np.mean(mel_db)
        std = np.std(mel_db) + 1e-6
        return (mel_db - mean) / std

    def preprocess(
        self, audio_path: Union[str, Path]
    ) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Union[float, int]]]:
        """
        Full audio preprocessing pipeline.

        Returns:
            Tuple:
                - tensor: PyTorch tensor of shape (1, 1, n_mels, time_steps)
                - raw_mel_db: Numpy 2D array of Mel spectrogram for visualization
                - metadata: Audio properties (duration, sample_rate, length)
        """
        waveform, sr, raw_duration = self.load_audio(audio_path)
        fixed_waveform = self.fix_length(waveform)
        mel_db = self.compute_mel_spectrogram(fixed_waveform)
        normalized_mel = self.normalize_spectrogram(mel_db)

        # PyTorch input shape: (batch_size=1, channels=1, height=n_mels, width=time_steps)
        tensor = torch.from_numpy(normalized_mel).float().unsqueeze(0).unsqueeze(0)

        metadata = {
            "sample_rate": sr,
            "raw_duration_sec": round(raw_duration, 2),
            "target_duration_sec": self.duration,
            "n_mels": self.n_mels,
            "time_steps": mel_db.shape[1],
        }

        return tensor, mel_db, metadata

    def extract_audio_from_video(
        self, video_path: Union[str, Path], output_wav_path: Union[str, Path]
    ) -> Path:
        """
        Extracts audio track from video file and saves as uncompressed 16-bit WAV.
        Uses librosa or moviepy/subprocess.
        """
        in_path = validate_file_path(video_path, allowed_extensions=config.video.supported_extensions)
        out_path = Path(output_wav_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Librosa uses audioread which handles container formats (.mp4, .mov, etc.)
            waveform, sr = librosa.load(str(in_path), sr=self.sample_rate, mono=True)
            sf.write(str(out_path), waveform, sr)
            return out_path
        except Exception as e:
            raise RuntimeError(
                f"Audio extraction from video failed ({in_path.name}): {e}. "
                "Ensure video contains a valid audio stream."
            )
