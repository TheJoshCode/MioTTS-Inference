from __future__ import annotations

import io
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import soundfile as sf
import torch
from miocodec.util import load_audio


@contextmanager
def _temp_audio_file(data: bytes, suffix: str = ".wav") -> Generator[Path, None, None]:
    """Create a temporary file with audio data that is automatically cleaned up."""
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            path = Path(tmp.name)
        yield path
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def load_reference_audio_bytes(data: bytes, sample_rate: int) -> torch.Tensor:
    with _temp_audio_file(data, suffix=".wav") as path:
        return load_audio(str(path), sample_rate=sample_rate)


def load_reference_audio_path(path: str, sample_rate: int) -> torch.Tensor:
    return load_audio(path, sample_rate=sample_rate)


def write_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    audio = ensure_1d(audio)
    if audio.dtype not in (torch.float32, torch.float64):
        audio = audio.float()
    buffer = io.BytesIO()
    sf.write(buffer, audio.cpu().numpy(), sample_rate, format="WAV")
    return buffer.getvalue()


def ensure_1d(audio: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 2 and audio.shape[0] == 1:
        return audio.squeeze(0)
    if audio.dim() == 1:
        return audio
    return audio.flatten()


def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return audio
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("torchaudio is required for resampling.") from exc
    if audio.is_cuda:
        audio = audio.cpu()
    if audio.dtype not in (torch.float32, torch.float64):
        audio = audio.float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    resampled = resampler(audio)
    return resampled.squeeze(0)
