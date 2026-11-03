"""
iris/adapters/tts/elevenlabs_tts.py
-----------------------------------
ElevenLabs REST Text-to-Speech Adapter.

Uses stateless REST /v1/text-to-speech conversion.
Does NOT run ElevenLabs conversational agent runtime or WebSocket loops.
Leaves LLM reasoning, dialog state, and audio capture entirely to IRIS.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from iris.adapters.tts.base import TTSAdapter

logger = logging.getLogger(__name__)


class ElevenLabsTTS(TTSAdapter):
    """
    Stateless ElevenLabs REST TTS adapter.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str = "ThT5KcBeYPX3keUQqHPh",  # Dorothy (British female)
        model_id: str = "eleven_flash_v2_5",
    ) -> None:
        if not api_key:
            raise ValueError("ElevenLabs API key is required for ElevenLabsTTS.")
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._client: Any = None

    @property
    def voice_name(self) -> str:
        return f"ElevenLabs ({self._voice_id})"

    def _get_client(self) -> Any:
        if self._client is None:
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=self._api_key)
        return self._client

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        clean_text = text.strip()
        if not clean_text:
            return np.zeros(0, dtype=np.int16), 24000

        client = self._get_client()
        logger.info(
            "Synthesizing speech via ElevenLabs REST (voice=%s, model=%s)",
            self._voice_id,
            self._model_id,
        )

        try:
            audio_stream = client.text_to_speech.convert(
                voice_id=self._voice_id,
                text=clean_text,
                model_id=self._model_id,
                output_format="pcm_24000",
            )
            pcm_bytes = b"".join(audio_stream)
            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
            return samples, 24000
        except Exception as exc:
            logger.error("ElevenLabs TTS failed: %s", exc)
            raise RuntimeError(f"ElevenLabs TTS synthesis failed: {exc}") from exc

