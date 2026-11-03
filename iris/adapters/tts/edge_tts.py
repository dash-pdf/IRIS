"""
iris/adapters/tts/edge_tts.py
-----------------------------
High-Speed Cloud Neural Text-to-Speech Adapter using Microsoft Edge TTS.

Features:
- Free cloud neural speech synthesis without API keys or subscriptions.
- Exceptional British female voice options (en-GB-SoniaNeural, en-GB-LibbyNeural, en-GB-MaisieNeural).
- Low latency (~250ms - 350ms) streaming playback.
- Decoupled from agent runtime and LLM providers.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
from collections.abc import Callable
from typing import Any

import edge_tts
import numpy as np
import soundfile as sf

from iris.adapters.tts.base import TTSAdapter
from iris.audio.player import AudioPlayer

logger = logging.getLogger(__name__)

# British female neural voices available in Edge TTS
EDGE_BRITISH_FEMALE_VOICES = {
    "en-GB-SoniaNeural": "British Female - Sonia (Refined, Warm)",
    "en-GB-LibbyNeural": "British Female - Libby (Crisp, Contemporary)",
    "en-GB-MaisieNeural": "British Female - Maisie (Young, Conversational)",
}
DEFAULT_EDGE_VOICE = "en-GB-SoniaNeural"


class EdgeTTS(TTSAdapter):
    """
    Edge TTS adapter providing fast cloud neural synthesis with British female voices.
    """

    def __init__(
        self,
        voice_name: str = DEFAULT_EDGE_VOICE,
        rate: str = "+0%",
        player: AudioPlayer | None = None,
    ) -> None:
        self._voice_name = self._validate_voice(voice_name)
        self.rate = rate
        self._player = player or AudioPlayer()

    def _validate_voice(self, voice: str) -> str:
        clean = voice.strip()
        if clean in EDGE_BRITISH_FEMALE_VOICES:
            return clean
        # Match case-insensitively or short aliases
        lower_map = {k.lower(): k for k in EDGE_BRITISH_FEMALE_VOICES}
        if clean.lower() in lower_map:
            return lower_map[clean.lower()]
        if clean.lower() in ("sonia", "gb-sonia"):
            return "en-GB-SoniaNeural"
        if clean.lower() in ("libby", "gb-libby"):
            return "en-GB-LibbyNeural"
        if clean.lower() in ("maisie", "gb-maisie"):
            return "en-GB-MaisieNeural"

        logger.warning(
            "Requested voice '%s' is not in Edge British female presets. Defaulting to '%s'.",
            voice,
            DEFAULT_EDGE_VOICE,
        )
        return DEFAULT_EDGE_VOICE

    @property
    def voice_name(self) -> str:
        return self._voice_name

    @voice_name.setter
    def voice_name(self, value: str) -> None:
        self._voice_name = self._validate_voice(value)

    async def _async_synthesize(
        self,
        text: str,
        voice: str,
        rate: str,
    ) -> tuple[np.ndarray, int]:
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
        audio_buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_bytes = audio_buffer.getvalue()
        if not audio_bytes:
            return np.zeros(0, dtype=np.float32), 24000

        audio_buffer.seek(0)
        samples, sample_rate = sf.read(audio_buffer, dtype="float32")
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples, dtype=np.float32)
        return samples, sample_rate

    def generate_audio(
        self,
        text: str,
        voice: str | None = None,
        rate: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize text into a numpy array (float32) and sample rate.
        """
        clean_text = text.strip()
        if not clean_text:
            return np.zeros(0, dtype=np.float32), 24000

        target_voice = self._validate_voice(voice) if voice else self._voice_name
        target_rate = rate or self.rate

        try:
            # Run async synthesis safely from sync context
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Run in separate thread if event loop is already active
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._async_synthesize(clean_text, target_voice, target_rate),
                    )
                    return future.result()
            else:
                return asyncio.run(
                    self._async_synthesize(clean_text, target_voice, target_rate)
                )
        except Exception as exc:
            logger.error("Edge TTS synthesis failed: %s", exc)
            raise RuntimeError(f"Edge TTS error: {exc}") from exc

    def speak(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        blocking: bool = False,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        """
        Synthesize text and play audio through speakers.
        """
        rate = f"+{int((speed - 1.0) * 100)}%" if speed and speed != 1.0 else self.rate
        samples, sample_rate = self.generate_audio(text, voice=voice, rate=rate)
        if len(samples) == 0:
            if on_done:
                on_done()
            return

        self._player.play(samples, sample_rate=sample_rate, on_done=on_done)
        if blocking:
            self._player.wait()

    def stop(self) -> None:
        """Immediately abort any active audio playback."""
        self._player.stop()

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Satisfies the TTSAdapter base contract."""
        return self.generate_audio(text)

