"""
iris/adapters/stt/elevenlabs_stt.py
-----------------------------------
ElevenLabs Scribe Speech-to-Text Adapter.

Uses ElevenLabs Scribe API to accurately transcribe audio, including Indian English
accents, natural phrasing, and technical terms.
"""

from __future__ import annotations

import io
import logging
import wave
from typing import Any

from iris.adapters.stt.base import STTAdapter

logger = logging.getLogger(__name__)


class ElevenLabsSTT(STTAdapter):
    """
    Speech-to-text adapter using ElevenLabs Scribe model.
    """

    def __init__(self, api_key: str, model_id: str = "scribe_v1") -> None:
        if not api_key:
            raise ValueError("ElevenLabs API key is required for ElevenLabsSTT.")
        self._api_key = api_key
        self._model_id = model_id
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return f"ElevenLabs Scribe ({self._model_id})"

    def _get_client(self) -> Any:
        if self._client is None:
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=self._api_key)
        return self._client

    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        if not pcm_bytes or len(pcm_bytes) < 400:
            return ""

        client = self._get_client()

        # Wrap PCM in a standard in-memory WAV buffer
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        wav_buf.seek(0)

        logger.info("Transcribing audio via ElevenLabs STT (%d bytes)", len(pcm_bytes))
        try:
            result = client.speech_to_text.convert(
                file=wav_buf,
                model_id=self._model_id,
            )
            text = getattr(result, "text", "") or ""
            return text.strip()
        except Exception as exc:
            logger.error("ElevenLabs STT transcription failed: %s", exc)
            raise RuntimeError(f"ElevenLabs STT transcription error: {exc}") from exc

