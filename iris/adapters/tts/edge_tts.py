"""
iris/adapters/tts/edge_tts.py
-----------------------------
High-Speed Cloud Neural Text-to-Speech Adapter using Microsoft Edge TTS,
with automatic resilient native macOS speech synthesis fallback.

Features:
- Primary: Free Microsoft Edge cloud neural speech synthesis without API keys.
- Exceptional British female voice options (en-GB-SoniaNeural, en-GB-LibbyNeural, en-GB-MaisieNeural).
- Fallback: Automatic native macOS offline TTS (MacOSTTS) if Edge TTS encounters network
  drops, timeouts, or HTTP 403 errors.
- Clean utterance isolation: Incomplete buffers from failed attempts are discarded before
  falling back, preventing duplicate or partial playback.
- Provider recovery: Automatically retries Edge TTS on subsequent turns with failure tracking.
- Secure logging: Sensitive URLs, tokens, and query parameters are redacted from error logs.
- Decoupled from agent runtime and LLM providers.
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
import threading
from collections.abc import Callable
from typing import Any

import edge_tts
import numpy as np
import soundfile as sf

from iris.adapters.tts.base import TTSAdapter
from iris.audio.player import AudioPlayer
from iris.core.security import sanitize_sensitive_text

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
    Edge TTS adapter providing fast cloud neural synthesis with British female voices
    and automatic failover to native macOS speech synthesis.
    """

    def __init__(
        self,
        voice_name: str = DEFAULT_EDGE_VOICE,
        rate: str = "+0%",
        player: AudioPlayer | None = None,
        fallback_tts: TTSAdapter | None = None,
        fallback_voice: str = "Flo (English (UK))",
    ) -> None:
        self._voice_name = self._validate_voice(voice_name)
        self.rate = rate
        self._player = player or AudioPlayer()
        self._consecutive_failures = 0
        self._last_used_fallback = False

        if fallback_tts is not None:
            self._fallback_tts: TTSAdapter | None = fallback_tts
        elif sys.platform == "darwin":
            try:
                from iris.adapters.tts.macos_tts import MacOSTTS

                self._fallback_tts = MacOSTTS(voice_name=fallback_voice)
            except Exception as exc:
                logger.warning("Could not initialize native macOS fallback TTS: %s", exc)
                self._fallback_tts = None
        else:
            self._fallback_tts = None

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

    @property
    def consecutive_failures(self) -> int:
        """Returns the number of sequential Edge TTS synthesis failures."""
        return self._consecutive_failures

    @property
    def fallback_tts(self) -> TTSAdapter | None:
        """Returns the configured fallback TTS adapter, if any."""
        return self._fallback_tts

    @property
    def using_fallback(self) -> bool:
        """Indicates whether the most recent synthesis was performed by the fallback adapter."""
        return self._last_used_fallback

    def _sanitize_error(self, err_msg: str) -> str:
        """
        Strip query strings, tokens, and credentials from error logs.
        """
        return sanitize_sensitive_text(err_msg)

    async def _async_synthesize(
        self,
        text: str,
        voice: str,
        rate: str,
        timeout_seconds: float = 12.0,
    ) -> tuple[np.ndarray, int]:
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
        audio_buffer = io.BytesIO()

        async with asyncio.timeout(timeout_seconds):
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

    def _synthesize_edge(
        self,
        text: str,
        voice: str,
        rate: str,
    ) -> tuple[np.ndarray, int]:
        """Run async Edge TTS synthesis from synchronous context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self._async_synthesize(text, voice, rate),
                )
                return future.result()
        else:
            return asyncio.run(
                self._async_synthesize(text, voice, rate)
            )

    def generate_audio(
        self,
        text: str,
        voice: str | None = None,
        rate: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize text into a numpy array (float32) and sample rate.
        Attempts Edge TTS first; on failure, discards any partial buffer and
        falls back cleanly to native macOS TTS.
        """
        clean_text = text.strip()
        if not clean_text:
            return np.zeros(0, dtype=np.float32), 24000

        target_voice = self._validate_voice(voice) if voice else self._voice_name
        target_rate = rate or self.rate

        # Attempt primary Edge TTS synthesis first (provider recovery on every turn)
        try:
            samples, sample_rate = self._synthesize_edge(clean_text, target_voice, target_rate)
            if len(samples) == 0:
                raise RuntimeError("Edge TTS returned empty audio buffer")

            if self._consecutive_failures > 0:
                logger.info(
                    "Edge TTS recovered successfully after %d consecutive failures.",
                    self._consecutive_failures,
                )
            self._consecutive_failures = 0
            self._last_used_fallback = False
            return samples, sample_rate

        except Exception as exc:
            self._consecutive_failures += 1
            safe_error = self._sanitize_error(str(exc))
            logger.warning(
                "Edge TTS synthesis failed (consecutive failures: %d): %s. Falling back to native macOS TTS.",
                self._consecutive_failures,
                safe_error,
            )

            # Clean fallback to native macOS TTS: re-synthesize full utterance
            if self._fallback_tts is not None:
                try:
                    fb_samples, fb_sr = self._fallback_tts.synthesize(clean_text)
                    if len(fb_samples) == 0:
                        raise RuntimeError("Fallback TTS returned empty audio buffer")
                    self._last_used_fallback = True
                    logger.info(
                        "Successfully synthesized utterance using macOS fallback TTS ('%s')",
                        getattr(self._fallback_tts, "voice_name", "macOS"),
                    )
                    return fb_samples, fb_sr
                except Exception as fb_exc:
                    safe_fb = self._sanitize_error(str(fb_exc))
                    logger.error("macOS fallback TTS failed after Edge TTS failure: %s", safe_fb)
                    raise RuntimeError(f"Both Edge TTS and fallback TTS failed: {safe_fb}") from fb_exc
            else:
                raise RuntimeError(f"Edge TTS error: {safe_error}") from exc

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
        Buffers synthesis fully before initiating playback to prevent partial/duplicate speech.
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
        if self._fallback_tts and hasattr(self._fallback_tts, "stop"):
            self._fallback_tts.stop()

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Satisfies the TTSAdapter base contract."""
        return self.generate_audio(text)
