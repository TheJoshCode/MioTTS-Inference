from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .audio import ensure_1d, resample_audio

logger = logging.getLogger(__name__)

_ASR_SAMPLE_RATE = 16000


@dataclass
class ASRConfig:
    model_id: str
    device: str
    compute_type: str
    batch_size: int
    language: str


class ASRService:
    def __init__(self, config: ASRConfig) -> None:
        self._config = config
        self._pipeline = None

    def load(self) -> None:
        try:
            from transformers import pipeline
        except Exception as exc:
            raise ImportError("transformers is required for ASR scoring.") from exc
        logger.info("Loading ASR pipeline: %s", self._config.model_id)
        device_index = _resolve_device_index(self._config.device)
        torch_dtype = _resolve_torch_dtype(self._config.compute_type)
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self._config.model_id,
            device=device_index,
            torch_dtype=torch_dtype,
        )
        logger.debug(
            "ASR config: device=%s device_index=%s dtype=%s batch_size=%d language=%s",
            self._config.device,
            device_index,
            torch_dtype,
            self._config.batch_size,
            self._config.language,
        )

    @property
    def pipeline(self):
        if self._pipeline is None:
            raise RuntimeError("ASR pipeline is not loaded.")
        return self._pipeline

    def transcribe_batch(
        self,
        audio_list: Iterable[torch.Tensor],
        sample_rate: int,
        language: str | None = None,
    ) -> list[str]:
        audio_list = list(audio_list)
        total = len(audio_list)
        if total == 0:
            return []
        if self._config.batch_size <= 0:
            batch_size = total
        else:
            batch_size = min(self._config.batch_size, total)
        logger.debug(
            "ASR transcribe_batch: total=%d batch_size=%d (config=%d)",
            total,
            batch_size,
            self._config.batch_size,
        )
        audio_np_list = [_prepare_audio(audio, sample_rate) for audio in audio_list]
        lang = _resolve_language(language, self._config.language)
        generate_kwargs = _build_generate_kwargs(lang)
        outputs = self.pipeline(
            audio_np_list,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
        )
        if isinstance(outputs, dict):
            return [_extract_text(outputs)]
        return [_extract_text(item) for item in outputs]


def _prepare_audio(audio: torch.Tensor, sample_rate: int) -> np.ndarray:
    audio = ensure_1d(audio)
    if sample_rate != _ASR_SAMPLE_RATE:
        audio = resample_audio(audio, sample_rate, _ASR_SAMPLE_RATE)
    if audio.is_cuda:
        audio = audio.cpu()
    audio = audio.detach().to(torch.float32)
    return audio.numpy()


def _resolve_language(requested: str | None, default_lang: str) -> str | None:
    if requested in {"ja", "en"}:
        return requested
    if default_lang in {"ja", "en"}:
        return default_lang
    return None


def _resolve_device_index(device: str) -> int:
    if not device or device == "cpu":
        return -1
    if device.startswith("cuda"):
        if ":" in device:
            try:
                return int(device.split(":", 1)[1])
            except ValueError:
                return 0
        return 0
    return -1


def _resolve_torch_dtype(compute_type: str) -> Any | None:
    try:
        import torch
    except Exception:
        return None
    if not compute_type:
        return None
    value = compute_type.lower()
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    return None


def _build_generate_kwargs(language: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"task": "transcribe"}
    if language == "ja":
        kwargs["language"] = "japanese"
    elif language == "en":
        kwargs["language"] = "english"
    return kwargs


def _extract_text(output: Any) -> str:
    if isinstance(output, dict):
        text = output.get("text", "")
        return text.strip()
    if isinstance(output, list) and output:
        text = output[0].get("text", "") if isinstance(output[0], dict) else str(output[0])
        return str(text).strip()
    return str(output).strip()
